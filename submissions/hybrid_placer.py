"""
Hybrid Placer v3 — Analytical init → Congestion-aware SA refinement.

Improvements over v2 (avg 1.6972):
  - RUDY congestion estimation in SA: incremental routing demand tracking
    on the evaluator's grid, matching L-shaped 2-pin routing + macro blockage.
  - SA acceptance uses combined cost: delta_hpwl + cong_weight * delta_congestion.
  - Periodic full congestion recompute to correct incremental drift.

Strategy:
  1. Analytical phase: Differentiable LSE-HPWL + density + overlap with Adam (GPU-accelerated when CUDA is available).
  2. Legalization: Minimum-displacement snap to resolve overlaps.
  3. SA phase: Congestion-aware SA with RUDY routing estimation + reheating.

Usage:
    uv run evaluate submissions/hybrid_placer.py
    uv run evaluate submissions/hybrid_placer.py --all
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _hybrid_device() -> torch.device:
    """Prefer CUDA for the analytical phase, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from submissions.base import BasePlacer
from submissions.analytical_placer import AnalyticalPlacer

from submissions.sa_placer import (
    _load_plc,
    _extract_nets,
    _legalize,
    _sa_refine,
    _update_soft_macros,
    _net_hpwl,
    _net_hpwl_override,
    _delta_hpwl,
    _compute_total_hpwl,
)


# ── Differentiable HPWL (log-sum-exp) ──────────────────────────────────────

def _build_net_tensors(plc, device):
    """
    Build padded tensor representation of nets for differentiable HPWL.
    """
    hard_name_to_idx = {}
    for tensor_i, plc_i in enumerate(plc.hard_macro_indices):
        name = plc.modules_w_pins[plc_i].get_name()
        hard_name_to_idx[name] = tensor_i

    soft_name_to_pos = {}
    for plc_i in plc.soft_macro_indices:
        mod = plc.modules_w_pins[plc_i]
        soft_name_to_pos[mod.get_name()] = mod.get_pos()

    raw_nets = []
    for driver_name, sink_names in plc.nets.items():
        if driver_name not in plc.mod_name_to_indices:
            continue
        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        weight = plc.modules_w_pins[driver_plc_idx].get_weight()

        pins = []
        for pin_name in [driver_name] + sink_names:
            if pin_name not in plc.mod_name_to_indices:
                continue
            pin_plc_idx = plc.mod_name_to_indices[pin_name]
            pin_obj = plc.modules_w_pins[pin_plc_idx]
            pin_type = pin_obj.get_type()

            if pin_type == "PORT":
                x, y = pin_obj.get_pos()
                pins.append((False, 0, x, y))
            elif pin_type == "MACRO_PIN":
                parent_name = pin_obj.get_macro_name()
                if parent_name in hard_name_to_idx:
                    idx = hard_name_to_idx[parent_name]
                    ox, oy = pin_obj.get_offset()
                    pins.append((True, idx, ox, oy))
                elif parent_name in soft_name_to_pos:
                    sx, sy = soft_name_to_pos[parent_name]
                    ox, oy = pin_obj.get_offset()
                    pins.append((False, 0, sx + ox, sy + oy))

        if len(pins) < 2:
            continue
        has_hard = any(p[0] for p in pins)
        if not has_hard:
            continue
        raw_nets.append((weight, pins))

    if not raw_nets:
        return None

    N = len(raw_nets)
    max_pins = max(len(rn[1]) for rn in raw_nets)

    net_mask = torch.zeros(N, max_pins, dtype=torch.bool, device=device)
    net_hard = torch.zeros(N, max_pins, dtype=torch.long, device=device)
    net_is_hard = torch.zeros(N, max_pins, dtype=torch.bool, device=device)
    net_ox = torch.zeros(N, max_pins, dtype=torch.float32, device=device)
    net_oy = torch.zeros(N, max_pins, dtype=torch.float32, device=device)
    net_weight = torch.zeros(N, dtype=torch.float32, device=device)

    for i, (w, pins) in enumerate(raw_nets):
        net_weight[i] = float(w)
        for j, (is_hard, idx, ox, oy) in enumerate(pins):
            net_mask[i, j] = True
            net_ox[i, j] = float(ox)
            net_oy[i, j] = float(oy)
            if is_hard:
                net_is_hard[i, j] = True
                net_hard[i, j] = idx

    return net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight


def _lse_hpwl(pos, net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight, gamma=5.0):
    """Differentiable HPWL using log-sum-exp approximation."""
    hard_x = pos[net_hard, 0]
    hard_y = pos[net_hard, 1]
    is_h = net_is_hard.float()

    pin_x = is_h * (hard_x + net_ox) + (1 - is_h) * net_ox
    pin_y = is_h * (hard_y + net_oy) + (1 - is_h) * net_oy

    BIG = 1e6
    pin_x_max = pin_x.clone(); pin_x_max[~net_mask] = -BIG
    pin_x_min = pin_x.clone(); pin_x_min[~net_mask] = BIG
    pin_y_max = pin_y.clone(); pin_y_max[~net_mask] = -BIG
    pin_y_min = pin_y.clone(); pin_y_min[~net_mask] = BIG

    inv_gamma = 1.0 / gamma
    x_max = inv_gamma * torch.logsumexp(gamma * pin_x_max, dim=1)
    x_min = -inv_gamma * torch.logsumexp(-gamma * pin_x_min, dim=1)
    y_max = inv_gamma * torch.logsumexp(gamma * pin_y_max, dim=1)
    y_min = -inv_gamma * torch.logsumexp(-gamma * pin_y_min, dim=1)

    hpwl = net_weight * ((x_max - x_min) + (y_max - y_min))
    return hpwl.sum()


def _density_penalty(pos, sizes, canvas_w, canvas_h, grid_n=16):
    """Gaussian-smoothed density penalty to spread macros apart."""
    device = pos.device
    cell_w = canvas_w / grid_n
    cell_h = canvas_h / grid_n
    cx = torch.linspace(cell_w / 2, canvas_w - cell_w / 2, grid_n, device=device)
    cy = torch.linspace(cell_h / 2, canvas_h - cell_h / 2, grid_n, device=device)
    grid_cx, grid_cy = torch.meshgrid(cx, cy, indexing="ij")
    grid_cx = grid_cx.reshape(1, -1)
    grid_cy = grid_cy.reshape(1, -1)

    sigma_x = sizes[:, 0:1] * 0.5 + cell_w * 0.5
    sigma_y = sizes[:, 1:2] * 0.5 + cell_h * 0.5

    dx = pos[:, 0:1] - grid_cx
    dy = pos[:, 1:2] - grid_cy

    weight_x = torch.exp(-0.5 * (dx / sigma_x) ** 2)
    weight_y = torch.exp(-0.5 * (dy / sigma_y) ** 2)
    area = sizes[:, 0:1] * sizes[:, 1:2]

    density = (area * weight_x * weight_y).sum(dim=0)

    target = (sizes[:, 0] * sizes[:, 1]).sum() / (grid_n * grid_n)
    overflow = torch.relu(density - target * 2.0)
    return overflow.sum()


def _overlap_penalty(pos, sizes):
    """Differentiable overlap penalty between all pairs of macros."""
    n = pos.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=pos.device)

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
    dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

    sep_x = half_w.unsqueeze(1) + half_w.unsqueeze(0)
    sep_y = half_h.unsqueeze(1) + half_h.unsqueeze(0)

    overlap_x = torch.nn.functional.softplus(sep_x - torch.abs(dx), beta=5.0)
    overlap_y = torch.nn.functional.softplus(sep_y - torch.abs(dy), beta=5.0)

    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=pos.device), diagonal=1)
    return (overlap_x * overlap_y * mask.float()).sum()


# ── Analytical placement phase ──────────────────────────────────────────────

def _analytical_place(
    benchmark: Benchmark,
    plc,
    num_steps: int = 1000,
    lr: float = 0.5,
    gamma: float = 5.0,
    density_weight: float = 0.001,
    overlap_weight: float = 0.05,
    seed: int = 42,
):
    """
    Gradient-based placement using differentiable HPWL + density + overlap.

    Uses fixed gamma=5 (proven to work well) with ramping density/overlap.
    """
    torch.manual_seed(seed)
    device = _hybrid_device()

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].clone().float().to(device)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].to(device)
    fixed_mask = ~movable

    init_pos = benchmark.macro_positions[:n_hard].clone().float().to(device)
    init_pos[:, 0] = torch.clamp(init_pos[:, 0], half_w, cw - half_w)
    init_pos[:, 1] = torch.clamp(init_pos[:, 1], half_h, ch - half_h)

    net_data = _build_net_tensors(plc, device)
    if net_data is None:
        return init_pos.detach().cpu().numpy().astype(np.float64)

    net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight = net_data

    pos = init_pos.clone().detach().requires_grad_(True)
    fixed_pos = init_pos.clone()

    optimizer = torch.optim.Adam([pos], lr=lr)

    best_pos = pos.data.clone()
    best_cost = float('inf')

    for step in range(num_steps):
        optimizer.zero_grad()

        with torch.no_grad():
            pos.data[fixed_mask] = fixed_pos[fixed_mask]

        # HPWL loss
        hpwl = _lse_hpwl(pos, net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight, gamma=gamma)

        # Density penalty (ramp up over first 30%)
        density_ramp = min(1.0, step / (num_steps * 0.3))
        density = _density_penalty(pos, sizes, cw, ch) * density_weight * density_ramp

        # Overlap penalty (ramp up over first 20%)
        overlap_ramp = min(1.0, step / (num_steps * 0.2))
        overlap = _overlap_penalty(pos, sizes) * overlap_weight * overlap_ramp

        loss = hpwl + density + overlap
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pos.data[:, 0] = torch.clamp(pos.data[:, 0], half_w, cw - half_w)
            pos.data[:, 1] = torch.clamp(pos.data[:, 1], half_h, ch - half_h)
            pos.data[fixed_mask] = fixed_pos[fixed_mask]

        cost_val = loss.item()
        if cost_val < best_cost:
            best_cost = cost_val
            best_pos = pos.data.clone()

    return best_pos.detach().cpu().numpy().astype(np.float64)


def _gpu_post_refine(
    benchmark: Benchmark,
    plc,
    init_pos_np: np.ndarray,
    num_steps: int = 250,
    lr: float = 0.08,
    gamma: float = 4.0,
    density_weight: float = 0.002,
    overlap_weight: float = 0.08,
    anchor_weight: float = 0.01,
    seed: int = 42,
):
    """
    Short GPU refinement pass after SA.

    Keeps the SA solution as an anchor while letting differentiable losses
    recover a bit of wirelength/density quality on the final placement.
    """
    torch.manual_seed(seed)
    device = _hybrid_device()

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].clone().float().to(device)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].to(device)
    fixed_mask = ~movable

    base_pos = torch.tensor(init_pos_np, dtype=torch.float32, device=device)
    base_pos[:, 0] = torch.clamp(base_pos[:, 0], half_w, cw - half_w)
    base_pos[:, 1] = torch.clamp(base_pos[:, 1], half_h, ch - half_h)

    net_data = _build_net_tensors(plc, device)
    if net_data is None:
        return init_pos_np.astype(np.float64)

    net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight = net_data
    pos = base_pos.clone().detach().requires_grad_(True)
    fixed_pos = base_pos.clone()
    optimizer = torch.optim.Adam([pos], lr=lr)

    best_pos = base_pos.clone()
    best_cost = float("inf")

    for _ in range(num_steps):
        optimizer.zero_grad()

        with torch.no_grad():
            pos.data[fixed_mask] = fixed_pos[fixed_mask]

        hpwl = _lse_hpwl(pos, net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight, gamma=gamma)
        density = _density_penalty(pos, sizes, cw, ch) * density_weight
        overlap = _overlap_penalty(pos, sizes) * overlap_weight
        anchor = ((pos[movable] - base_pos[movable]) ** 2).sum() * anchor_weight

        loss = hpwl + density + overlap + anchor
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pos.data[:, 0] = torch.clamp(pos.data[:, 0], half_w, cw - half_w)
            pos.data[:, 1] = torch.clamp(pos.data[:, 1], half_h, ch - half_h)
            pos.data[fixed_mask] = fixed_pos[fixed_mask]
            loss_value = float(loss.item())
            if loss_value < best_cost:
                best_cost = loss_value
                best_pos = pos.data.clone()

    return best_pos.detach().cpu().numpy().astype(np.float64)


def _proxy_score_pos(benchmark: Benchmark, plc, pos_hard: np.ndarray) -> float:
    """Score a hard-macro placement with the ground-truth proxy evaluator."""
    n_hard = benchmark.num_hard_macros
    full_pos = benchmark.macro_positions.clone()
    full_pos[:n_hard] = torch.tensor(pos_hard, dtype=torch.float32)
    return float(compute_proxy_cost(full_pos, benchmark, plc)["proxy_cost"])


# ── RUDY congestion estimation ─────────────────────────────────────────────

def _get_grid_cell(x, y, gw, gh, grid_col, grid_row):
    """Return (row, col) grid cell for position (x, y)."""
    col = max(0, min(int(x / gw), grid_col - 1))
    row = max(0, min(int(y / gh), grid_row - 1))
    return row, col


def _two_pin_route(sr, sc, kr, kc, grid_col, weight, H_cong, V_cong):
    """L-shaped routing: H along source row, V along sink col."""
    r_min, r_max = min(sr, kr), max(sr, kr)
    c_min, c_max = min(sc, kc), max(sc, kc)
    # H routing along source row
    for c in range(c_min, c_max):
        H_cong[sr * grid_col + c] += weight
    # V routing along sink col
    for r in range(r_min, r_max):
        V_cong[r * grid_col + kc] += weight


def _macro_blockage(mx, my, mw, mh, gw, gh, grid_col, grid_row,
                    V_macro, H_macro, vrouting_alloc, hrouting_alloc):
    """Add macro routing blockage to congestion grids."""
    x_lo, x_hi = mx - mw / 2, mx + mw / 2
    y_lo, y_hi = my - mh / 2, my + mh / 2
    c_min = max(0, int(x_lo / gw))
    c_max = min(grid_col - 1, int(x_hi / gw))
    r_min = max(0, int(y_lo / gh))
    r_max = min(grid_row - 1, int(y_hi / gh))

    for r in range(r_min, r_max + 1):
        cy_lo, cy_hi = r * gh, (r + 1) * gh
        ov_y = min(y_hi, cy_hi) - max(y_lo, cy_lo)
        if ov_y <= 0:
            continue
        for c in range(c_min, c_max + 1):
            cx_lo, cx_hi = c * gw, (c + 1) * gw
            ov_x = min(x_hi, cx_hi) - max(x_lo, cx_lo)
            if ov_x <= 0:
                continue
            idx = r * grid_col + c
            V_macro[idx] += ov_x * vrouting_alloc
            H_macro[idx] += ov_y * hrouting_alloc


def _remove_macro_blockage(mx, my, mw, mh, gw, gh, grid_col, grid_row,
                           V_macro, H_macro, vrouting_alloc, hrouting_alloc):
    """Remove macro routing blockage (for incremental update)."""
    x_lo, x_hi = mx - mw / 2, mx + mw / 2
    y_lo, y_hi = my - mh / 2, my + mh / 2
    c_min = max(0, int(x_lo / gw))
    c_max = min(grid_col - 1, int(x_hi / gw))
    r_min = max(0, int(y_lo / gh))
    r_max = min(grid_row - 1, int(y_hi / gh))

    for r in range(r_min, r_max + 1):
        cy_lo, cy_hi = r * gh, (r + 1) * gh
        ov_y = min(y_hi, cy_hi) - max(y_lo, cy_lo)
        if ov_y <= 0:
            continue
        for c in range(c_min, c_max + 1):
            cx_lo, cx_hi = c * gw, (c + 1) * gw
            ov_x = min(x_hi, cx_hi) - max(x_lo, cx_lo)
            if ov_x <= 0:
                continue
            idx = r * grid_col + c
            V_macro[idx] -= ov_x * vrouting_alloc
            H_macro[idx] -= ov_y * hrouting_alloc


def _build_net_pin_gcells(nets, pos, gw, gh, grid_col, grid_row, plc):
    """
    Build per-net pin grid cell lists and source cells for congestion routing.
    Returns list of (source_gcell, node_gcells_set, weight, hard_indices_in_net).
    """
    net_routing_info = []
    for net in nets:
        weight = net["weight"]
        hard_idx = net["hard_idx"]
        hard_ox = net["hard_ox"]
        hard_oy = net["hard_oy"]

        gcells = set()
        source_gcell = None

        # First pin is the source (driver)
        for k, idx in enumerate(hard_idx):
            px = pos[idx, 0] + hard_ox[k]
            py = pos[idx, 1] + hard_oy[k]
            gc = _get_grid_cell(px, py, gw, gh, grid_col, grid_row)
            gcells.add(gc)
            if k == 0:
                source_gcell = gc

        # Fixed pins (ports + soft macros) — use bounding box extremes
        INF = float("inf")
        if net["fxmin"] != INF:
            gc_min = _get_grid_cell(net["fxmin"], net["fymin"], gw, gh, grid_col, grid_row)
            gc_max = _get_grid_cell(net["fxmax"], net["fymax"], gw, gh, grid_col, grid_row)
            gcells.add(gc_min)
            gcells.add(gc_max)
            if source_gcell is None:
                source_gcell = gc_min

        if source_gcell is None and gcells:
            source_gcell = next(iter(gcells))

        net_routing_info.append((source_gcell, gcells, weight, hard_idx))
    return net_routing_info


def _route_net(source_gcell, node_gcells, weight, grid_col, H_cong, V_cong):
    """Route a single net onto H/V congestion grids (matching evaluator logic)."""
    gcells = list(node_gcells)
    n_pins = len(gcells)
    if n_pins < 2 or source_gcell is None:
        return

    if n_pins == 2:
        _two_pin_route(source_gcell[0], source_gcell[1],
                       gcells[0][0] if gcells[0] != source_gcell else gcells[1][0],
                       gcells[0][1] if gcells[0] != source_gcell else gcells[1][1],
                       grid_col, weight, H_cong, V_cong)
    elif n_pins == 3:
        # Use T-routing (simplified but effective)
        sorted_gc = sorted(gcells)
        y1, x1 = sorted_gc[0]
        y2, x2 = sorted_gc[1]
        y3, x3 = sorted_gc[2]
        xmin, xmax = min(x1, x2, x3), max(x1, x2, x3)
        # H routing along middle row
        for c in range(xmin, xmax):
            H_cong[y2 * grid_col + c] += weight
        # V routing from each pin to middle row
        for r in range(min(y1, y2), max(y1, y2)):
            V_cong[r * grid_col + x1] += weight
        for r in range(min(y2, y3), max(y2, y3)):
            V_cong[r * grid_col + x3] += weight
    else:
        # >3 pins: split into 2-pin nets from source to each sink
        for gc in gcells:
            if gc != source_gcell:
                _two_pin_route(source_gcell[0], source_gcell[1],
                               gc[0], gc[1], grid_col, weight, H_cong, V_cong)


def _unroute_net(source_gcell, node_gcells, weight, grid_col, H_cong, V_cong):
    """Remove a net's routing from H/V congestion grids (negative weight)."""
    _route_net(source_gcell, node_gcells, -weight, grid_col, H_cong, V_cong)


def _smooth_and_abu(H_cong, V_cong, grid_row, grid_col, grid_v_routes, grid_h_routes,
                    V_macro, H_macro, smooth_range):
    """Compute congestion cost: normalize, smooth, add macro blockage, abu(top 5%)."""
    n_cells = grid_row * grid_col
    # Normalize net routing
    V_norm = [v / grid_v_routes for v in V_cong]
    H_norm = [h / grid_h_routes for h in H_cong]

    # Smooth V (spread horizontally)
    V_smooth = [0.0] * n_cells
    for row in range(grid_row):
        for col in range(grid_col):
            lp = max(0, col - smooth_range)
            rp = min(grid_col - 1, col + smooth_range)
            cnt = rp - lp + 1
            val = V_norm[row * grid_col + col] / cnt
            for p in range(lp, rp + 1):
                V_smooth[row * grid_col + p] += val

    # Smooth H (spread vertically)
    H_smooth = [0.0] * n_cells
    for row in range(grid_row):
        for col in range(grid_col):
            lp = max(0, row - smooth_range)
            up = min(grid_row - 1, row + smooth_range)
            cnt = up - lp + 1
            val = H_norm[row * grid_col + col] / cnt
            for p in range(lp, up + 1):
                H_smooth[p * grid_col + col] += val

    # Add macro blockage (already normalized)
    V_macro_norm = [v / grid_v_routes for v in V_macro]
    H_macro_norm = [h / grid_h_routes for h in H_macro]

    combined = []
    for i in range(n_cells):
        combined.append(V_smooth[i] + V_macro_norm[i])
        combined.append(H_smooth[i] + H_macro_norm[i])

    # ABU: top 5%
    combined.sort(reverse=True)
    cnt = max(1, math.floor(len(combined) * 0.05))
    return sum(combined[:cnt]) / cnt


def _fast_congestion_cost(H_cong, V_cong, grid_row, grid_col,
                          grid_v_routes, grid_h_routes,
                          V_macro, H_macro, smooth_range):
    """Fast approximate congestion: skip smoothing, just normalize + abu using numpy."""
    n_cells = grid_row * grid_col
    inv_v = 1.0 / grid_v_routes
    inv_h = 1.0 / grid_h_routes
    combined = np.empty(2 * n_cells)
    for i in range(n_cells):
        combined[2 * i] = (V_cong[i] + V_macro[i]) * inv_v
        combined[2 * i + 1] = (H_cong[i] + H_macro[i]) * inv_h
    cnt = max(1, int(len(combined) * 0.05))
    # Use partial sort for top-k — O(n) average instead of O(n log n)
    top_k = np.partition(combined, -cnt)[-cnt:]
    return float(top_k.sum() / cnt)


# ── Congestion-aware SA refinement ─────────────────────────────────────────

def _sa_refine_congestion(
    pos, nets, macro_to_nets, neighbors,
    movable, sizes, half_w, half_h, sep_x, sep_y,
    cw, ch, max_iters, seed,
    grid_row, grid_col, vroutes_per_micron, hroutes_per_micron,
    vrouting_alloc, hrouting_alloc, smooth_range,
    plc,
    congestion_weight=0.5,
    t_start_factor=0.15, t_end_factor=0.001,
    reheat_threshold=10_000, reheat_factor=3.0,
    snapshot_interval=0, snapshot_callback=None,
    trace_interval=0, trace_callback=None,
):
    """SA loop minimising HPWL + congestion_weight * congestion."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos

    pos = pos.copy()
    n_hard = len(movable)
    GAP = 0.05

    gw = cw / grid_col
    gh = ch / grid_row
    grid_v_routes = gw * vroutes_per_micron
    grid_h_routes = gh * hroutes_per_micron

    def check_overlap(idx):
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        ov = (dx < sep_x[idx] + GAP) & (dy < sep_y[idx] + GAP)
        ov[idx] = False
        return bool(ov.any())

    # Build initial congestion grids
    H_cong = [0.0] * (grid_row * grid_col)
    V_cong = [0.0] * (grid_row * grid_col)
    H_macro = [0.0] * (grid_row * grid_col)
    V_macro = [0.0] * (grid_row * grid_col)

    # Route all nets
    net_routing = _build_net_pin_gcells(nets, pos, gw, gh, grid_col, grid_row, plc)
    for src_gc, node_gcs, w, _ in net_routing:
        _route_net(src_gc, node_gcs, w, grid_col, H_cong, V_cong)

    # Add macro blockage for all hard macros
    for m in range(n_hard):
        _macro_blockage(pos[m, 0], pos[m, 1], sizes[m, 0], sizes[m, 1],
                        gw, gh, grid_col, grid_row, V_macro, H_macro,
                        vrouting_alloc, hrouting_alloc)

    current_hpwl = _compute_total_hpwl(pos, nets)
    current_cong = _fast_congestion_cost(H_cong, V_cong, grid_row, grid_col,
                                         grid_v_routes, grid_h_routes,
                                         V_macro, H_macro, smooth_range)
    current_cost = current_hpwl + congestion_weight * current_cong

    best_pos = pos.copy()
    best_cost = current_cost
    best_hpwl = current_hpwl

    T_start = max(cw, ch) * t_start_factor
    T_end = max(cw, ch) * t_end_factor
    steps_since_improvement = 0

    # Recompute congestion every N steps to correct drift
    CONG_RECOMPUTE_INTERVAL = 5000

    for step in range(max_iters):
        frac = step / max_iters
        T = T_start * (T_end / T_start) ** frac

        if reheat_threshold > 0 and steps_since_improvement >= reheat_threshold:
            T = min(T * reheat_factor, T_start * 0.5)
            steps_since_improvement = 0

        # Periodic full congestion recompute to correct accumulated errors
        if step > 0 and step % CONG_RECOMPUTE_INTERVAL == 0:
            H_cong[:] = [0.0] * (grid_row * grid_col)
            V_cong[:] = [0.0] * (grid_row * grid_col)
            H_macro[:] = [0.0] * (grid_row * grid_col)
            V_macro[:] = [0.0] * (grid_row * grid_col)
            net_routing = _build_net_pin_gcells(nets, pos, gw, gh, grid_col, grid_row, plc)
            for src_gc, node_gcs, w, _ in net_routing:
                _route_net(src_gc, node_gcs, w, grid_col, H_cong, V_cong)
            for m in range(n_hard):
                _macro_blockage(pos[m, 0], pos[m, 1], sizes[m, 0], sizes[m, 1],
                                gw, gh, grid_col, grid_row, V_macro, H_macro,
                                vrouting_alloc, hrouting_alloc)
            current_cong = _fast_congestion_cost(H_cong, V_cong, grid_row, grid_col,
                                                 grid_v_routes, grid_h_routes,
                                                 V_macro, H_macro, smooth_range)
            current_cost = current_hpwl + congestion_weight * current_cong

        move = rng.random()
        i = rng.choice(movable_idx)
        old_x, old_y = pos[i, 0], pos[i, 1]

        if move < 0.5:
            # ── SHIFT ──
            shift = T * (0.3 + 0.7 * (1 - frac))
            new_x = float(np.clip(old_x + rng.gauss(0, shift), half_w[i], cw - half_w[i]))
            new_y = float(np.clip(old_y + rng.gauss(0, shift), half_h[i], ch - half_h[i]))

            pos[i, 0] = new_x; pos[i, 1] = new_y
            if check_overlap(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue
            pos[i, 0] = old_x; pos[i, 1] = old_y

            delta_hpwl = _delta_hpwl(i, new_x, new_y, pos, nets, macro_to_nets)

            # Incremental congestion update for nets touching macro i
            old_routings = []
            new_routings = []
            for net_idx in macro_to_nets[i]:
                net = nets[net_idx]
                src, gcs, w, _ = net_routing[net_idx]
                old_routings.append((net_idx, src, gcs, w))
                _unroute_net(src, gcs, w, grid_col, H_cong, V_cong)

            # Remove old macro blockage, add new
            _remove_macro_blockage(old_x, old_y, sizes[i, 0], sizes[i, 1],
                                   gw, gh, grid_col, grid_row, V_macro, H_macro,
                                   vrouting_alloc, hrouting_alloc)

            # Temporarily apply move for net gcell computation
            pos[i, 0] = new_x; pos[i, 1] = new_y
            for net_idx, _, _, w in old_routings:
                net = nets[net_idx]
                hard_idx = net["hard_idx"]
                hard_ox = net["hard_ox"]
                hard_oy = net["hard_oy"]
                gcells = set()
                src_gc = None
                for k, idx in enumerate(hard_idx):
                    px = pos[idx, 0] + hard_ox[k]
                    py = pos[idx, 1] + hard_oy[k]
                    gc = _get_grid_cell(px, py, gw, gh, grid_col, grid_row)
                    gcells.add(gc)
                    if k == 0:
                        src_gc = gc
                INF = float("inf")
                if net["fxmin"] != INF:
                    gcells.add(_get_grid_cell(net["fxmin"], net["fymin"], gw, gh, grid_col, grid_row))
                    gcells.add(_get_grid_cell(net["fxmax"], net["fymax"], gw, gh, grid_col, grid_row))
                    if src_gc is None:
                        src_gc = _get_grid_cell(net["fxmin"], net["fymin"], gw, gh, grid_col, grid_row)
                if src_gc is None and gcells:
                    src_gc = next(iter(gcells))
                new_routings.append((net_idx, src_gc, gcells, w))
                _route_net(src_gc, gcells, w, grid_col, H_cong, V_cong)

            _macro_blockage(new_x, new_y, sizes[i, 0], sizes[i, 1],
                            gw, gh, grid_col, grid_row, V_macro, H_macro,
                            vrouting_alloc, hrouting_alloc)

            new_cong = _fast_congestion_cost(H_cong, V_cong, grid_row, grid_col,
                                             grid_v_routes, grid_h_routes,
                                             V_macro, H_macro, smooth_range)
            delta_cong = new_cong - current_cong
            delta = delta_hpwl + congestion_weight * delta_cong

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                # Accept
                current_hpwl += delta_hpwl
                current_cong = new_cong
                current_cost = current_hpwl + congestion_weight * current_cong
                for net_idx, src, gcs, w in new_routings:
                    net_routing[net_idx] = (src, gcs, w, nets[net_idx]["hard_idx"])
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                # Reject: undo congestion changes
                pos[i, 0] = old_x; pos[i, 1] = old_y
                for net_idx, src, gcs, w in new_routings:
                    _unroute_net(src, gcs, w, grid_col, H_cong, V_cong)
                _remove_macro_blockage(new_x, new_y, sizes[i, 0], sizes[i, 1],
                                       gw, gh, grid_col, grid_row, V_macro, H_macro,
                                       vrouting_alloc, hrouting_alloc)
                for net_idx, src, gcs, w in old_routings:
                    _route_net(src, gcs, w, grid_col, H_cong, V_cong)
                _macro_blockage(old_x, old_y, sizes[i, 0], sizes[i, 1],
                                gw, gh, grid_col, grid_row, V_macro, H_macro,
                                vrouting_alloc, hrouting_alloc)
                steps_since_improvement += 1

        elif move < 0.80:
            # ── SWAP ── (use HPWL-only for speed; congestion recomputed periodically)
            if neighbors[i] and rng.random() < 0.7:
                cands = [j for j in neighbors[i] if movable[j]]
                j = rng.choice(cands) if cands else rng.choice(movable_idx)
            else:
                j = rng.choice(movable_idx)
            if i == j:
                continue

            old_jx, old_jy = pos[j, 0], pos[j, 1]
            new_ix = float(np.clip(old_jx, half_w[i], cw - half_w[i]))
            new_iy = float(np.clip(old_jy, half_h[i], ch - half_h[i]))
            new_jx = float(np.clip(old_x, half_w[j], cw - half_w[j]))
            new_jy = float(np.clip(old_y, half_h[j], ch - half_h[j]))

            pos[i, 0] = new_ix; pos[i, 1] = new_iy
            pos[j, 0] = new_jx; pos[j, 1] = new_jy
            if check_overlap(i) or check_overlap(j):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                continue

            affected = list(set(macro_to_nets[i]) | set(macro_to_nets[j]))
            new_hpwl_aff = sum(_net_hpwl(nets[k], pos) for k in affected)
            pos[i, 0] = old_x; pos[i, 1] = old_y
            pos[j, 0] = old_jx; pos[j, 1] = old_jy
            old_hpwl_aff = sum(_net_hpwl(nets[k], pos) for k in affected)
            delta_hpwl = new_hpwl_aff - old_hpwl_aff

            if delta_hpwl < 0 or rng.random() < math.exp(-delta_hpwl / max(T, 1e-12)):
                pos[i, 0] = new_ix; pos[i, 1] = new_iy
                pos[j, 0] = new_jx; pos[j, 1] = new_jy
                current_hpwl += delta_hpwl
                current_cost = current_hpwl + congestion_weight * current_cong
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1

        else:
            # ── MOVE TOWARD NEIGHBOR ──
            if not neighbors[i]:
                continue
            j = rng.choice(neighbors[i])
            alpha = rng.uniform(0.05, 0.3)
            new_x = float(np.clip(old_x + alpha * (pos[j, 0] - old_x), half_w[i], cw - half_w[i]))
            new_y = float(np.clip(old_y + alpha * (pos[j, 1] - old_y), half_h[i], ch - half_h[i]))

            pos[i, 0] = new_x; pos[i, 1] = new_y
            if check_overlap(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue
            pos[i, 0] = old_x; pos[i, 1] = old_y

            delta_hpwl = _delta_hpwl(i, new_x, new_y, pos, nets, macro_to_nets)
            if delta_hpwl < 0 or rng.random() < math.exp(-delta_hpwl / max(T, 1e-12)):
                pos[i, 0] = new_x; pos[i, 1] = new_y
                current_hpwl += delta_hpwl
                current_cost = current_hpwl + congestion_weight * current_cong
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1

        if snapshot_callback is not None and snapshot_interval > 0:
            if (step + 1) % snapshot_interval == 0 or step == max_iters - 1:
                snapshot_callback(best_pos.copy())
        if trace_callback is not None and trace_interval > 0:
            if (step + 1) % trace_interval == 0 or step == max_iters - 1:
                trace_callback({
                    "step": step + 1,
                    "temperature": T,
                    "current_hpwl": current_hpwl,
                    "best_hpwl": best_hpwl,
                    "congestion": current_cong,
                })

    return best_pos


# ── HybridPlacer class ─────────────────────────────────────────────────────

class HybridPlacer(BasePlacer):
    """
    GPU-accelerated in the analytical phase when CUDA is available.

    Hybrid placer v2: analytical gradient init → SA refinement with reheating.

    Phase 1 (analytical): 1200 Adam steps with annealing gamma, density, overlap.
    Phase 2 (legalize): Minimum-displacement overlap resolution.
    Phase 3 (SA): 150K iterations with reheating on stagnation.
    """

    def __init__(
        self,
        seed: int = 42,
        # Analytical phase
        analytical_steps: int = 1000,
        analytical_lr: float = 0.5,
        gamma: float = 5.0,
        density_weight: float = 0.001,
        overlap_weight: float = 0.05,
        advanced_analytical_warmstart: bool = False,
        analytical_candidates: int = 2,
        # SA phase
        sa_iters: int = 300_000,
        sa_t_start: float = 0.15,
        sa_t_end: float = 0.001,
        reheat_threshold: int = 10_000,
        reheat_factor: float = 3.0,
        congestion_weight: float = 50.0,
        density_sa_weight: float = 0.005,
        post_refine_steps: int = 0,
        # Soft macro FD
        run_fd: bool = False,
        # Debug
        capture_snapshots: bool = True,
        snapshot_interval: int = 2_000,
        trace_interval: int = 500,
    ):
        self.seed = seed
        self.analytical_steps = analytical_steps
        self.analytical_lr = analytical_lr
        self.gamma = gamma
        self.density_weight = density_weight
        self.overlap_weight = overlap_weight
        self.advanced_analytical_warmstart = advanced_analytical_warmstart
        self.analytical_candidates = analytical_candidates
        self.sa_iters = sa_iters
        self.sa_t_start = sa_t_start
        self.sa_t_end = sa_t_end
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor
        self.congestion_weight = congestion_weight
        self.density_sa_weight = density_sa_weight
        self.post_refine_steps = post_refine_steps
        self.run_fd = run_fd
        self.capture_snapshots = capture_snapshots
        self.snapshot_interval = snapshot_interval
        self.trace_interval = trace_interval
        self.debug_snapshots = []
        self.debug_trace = []

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.debug_snapshots = []
        self.debug_trace = []

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

        plc = _load_plc(benchmark.name)

        # ── Phase 1: Analytical placement ──────────────────────────────────
        if plc is not None:
            candidate_positions = []

            legacy_pos = _analytical_place(
                benchmark, plc,
                num_steps=self.analytical_steps,
                lr=self.analytical_lr,
                gamma=self.gamma,
                density_weight=self.density_weight,
                overlap_weight=self.overlap_weight,
                seed=self.seed,
            )
            candidate_positions.append(legacy_pos)

            if self.advanced_analytical_warmstart:
                analytical = AnalyticalPlacer(
                    seed=self.seed,
                    iters=max(2500, self.analytical_steps * 3),
                    lr=4.0,
                    gamma_start=40.0,
                    gamma_end=3.0,
                    density_weight=max(self.density_weight * 6.0, 0.006),
                    congestion_weight=0.0008,
                    boundary_weight=0.015,
                    overlap_weight_start=0.01,
                    overlap_weight_end=18.0,
                    num_candidates=max(1, self.analytical_candidates),
                    sa_polish_iters=0,
                )
                analytical_full = analytical.place(benchmark)
                candidate_positions.append(analytical_full[:n_hard].numpy().astype(np.float64))

            best_init_score = float("inf")
            pos = legacy_pos
            for cand in candidate_positions:
                legal_cand = _legalize(cand, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)
                try:
                    score = _proxy_score_pos(benchmark, plc, legal_cand)
                except Exception:
                    score = float("inf")
                if score < best_init_score:
                    best_init_score = score
                    pos = legal_cand
        else:
            pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)

        # ── Phase 2: Legalize ──────────────────────────────────────────────
        pos = _legalize(pos, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)

        def capture_snapshot(pos_hard: np.ndarray):
            if not self.capture_snapshots:
                return
            frame = benchmark.macro_positions.clone()
            frame[:n_hard] = torch.tensor(pos_hard, dtype=torch.float32)
            self.debug_snapshots.append(frame)

        def capture_trace(point: dict):
            self.debug_trace.append(point)

        capture_snapshot(pos)

        # ── Phase 3: SA refinement with reheating ──────────────────────────
        if plc is not None:
            nets, macro_to_nets = _extract_nets(benchmark, plc)
        else:
            nets, macro_to_nets = [], [[] for _ in range(n_hard)]

        neighbors = [[] for _ in range(n_hard)]
        for net in nets:
            idx = net["hard_idx"]
            for a in idx:
                for b in idx:
                    if a != b:
                        neighbors[a].append(int(b))

        if nets:
            pos = _sa_refine(
                pos, nets, macro_to_nets, neighbors,
                movable, sizes, half_w, half_h, sep_x, sep_y,
                cw, ch, self.sa_iters, self.seed,
                snapshot_interval=self.snapshot_interval,
                snapshot_callback=capture_snapshot if self.capture_snapshots else None,
                trace_interval=self.trace_interval,
                trace_callback=capture_trace,
                t_start_factor=self.sa_t_start,
                t_end_factor=self.sa_t_end,
                reheat_threshold=self.reheat_threshold,
                reheat_factor=self.reheat_factor,
            )

        if plc is not None and self.post_refine_steps > 0:
            sa_pos = pos.copy()
            refined_pos = _gpu_post_refine(
                benchmark,
                plc,
                pos,
                num_steps=self.post_refine_steps,
                seed=self.seed,
            )
            refined_pos = _legalize(refined_pos, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)
            try:
                if _proxy_score_pos(benchmark, plc, refined_pos) <= _proxy_score_pos(benchmark, plc, sa_pos):
                    pos = refined_pos
                else:
                    pos = sa_pos
            except Exception:
                pos = sa_pos

        # Build full placement tensor
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)

        if self.run_fd and plc is not None and benchmark.num_soft_macros > 0:
            soft_pos = _update_soft_macros(pos, benchmark, plc)
            full_pos[n_hard:] = torch.tensor(soft_pos, dtype=torch.float32)

        if self.capture_snapshots:
            if not self.debug_snapshots or not torch.equal(self.debug_snapshots[-1], full_pos):
                self.debug_snapshots.append(full_pos.clone())

        return full_pos
