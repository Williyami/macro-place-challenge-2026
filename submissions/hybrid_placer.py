"""
Hybrid Placer v8 — Multi-candidate analytical → pre-SA GPU smooth → SA → GPU refine
                   → Greedy refine → flip → GPU polish → fine greedy → micro GPU polish.

Improvements over v7 (avg 1.4828):
  - Pre-SA GPU smoothing (100 steps): removes legalization artifacts before SA starts,
    giving SA a cleaner starting point
  - Post-first-greedy flip: a second _greedy_flip after the main greedy passes catches
    new optimal orientations that greedy position shifts may have unlocked
  - Second greedy cycle with fine scales (0.001–0.05): 3 passes after GPU polish to
    capture sub-pixel improvements the coarser first pass (0.004–0.12) missed
  - Final micro GPU polish (100 steps, lr=0.02, tight anchor=0.1): smooths remaining
    interaction artifacts from the second greedy cycle

Strategy:
  1. Analytical phase: 7 candidates × 2500 Adam steps with gamma annealing,
     quadratic init, cosine LR, HPWL + density + congestion + repulsion + overlap.
  2. Legalization: Minimum-displacement snap to resolve overlaps.
  3. Pre-SA GPU smoothing: 100-step differentiable pass (tight anchor) to remove
     legalization displacement artifacts before SA.
  4. SA phase: HPWL + density co-optimization with reheating (300K iters).
  5. Greedy flip: Try mirror orientations for each macro to reduce HPWL.
  6. GPU refine: Cosine-scheduled differentiable pass with congestion-aware loss (500 steps).
  7. Greedy refine: Connectivity-sorted evaluator-guided coordinate descent (5 passes,
     scales 0.004–0.12).
  8. Post-greedy flip: Second orientation flip after greedy has repositioned macros.
  9. GPU polish: Short differentiable pass after greedy to remove interaction artifacts
     (200 steps, lr=0.04).
  10. Fine greedy: 3-pass coordinate descent with fine scales (0.001–0.05).
  11. Micro GPU polish: 100-step tight-anchor polish after fine greedy.
  12. Soft recenter: Move soft macros to weighted pin centroids.

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
import torch.nn.functional as F


def _hybrid_device() -> torch.device:
    """Prefer CUDA for the analytical phase, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from submissions.base import BasePlacer

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
    _greedy_flip,
    _build_density_grid,
    _density_cost_from_grid,
    _macro_cell_overlaps,
    _proxy_local_search,
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


def _density_penalty(pos, sizes, canvas_w, canvas_h, grid_n=32):
    """Evaluator-matched density penalty using sigmoid bin membership.

    Penalizes top-10% densest cells (matching the evaluator's formula).
    Uses softplus overlap computation for smooth gradients.
    """
    device = pos.device
    cell_w = canvas_w / grid_n
    cell_h = canvas_h / grid_n
    grid_area = cell_w * cell_h

    n = pos.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=device)

    # Grid cell boundaries
    bin_x_lo = torch.linspace(0, canvas_w - cell_w, grid_n, device=device)
    bin_x_hi = bin_x_lo + cell_w
    bin_y_lo = torch.linspace(0, canvas_h - cell_h, grid_n, device=device)
    bin_y_hi = bin_y_lo + cell_h

    # Macro boundaries
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    mac_x_lo = pos[:, 0] - half_w
    mac_x_hi = pos[:, 0] + half_w
    mac_y_lo = pos[:, 1] - half_h
    mac_y_hi = pos[:, 1] + half_h

    steep = 10.0 / max(cell_w, cell_h)

    # [N, G] overlap in each dimension
    ov_x = F.softplus(
        torch.min(mac_x_hi.unsqueeze(1), bin_x_hi.unsqueeze(0))
        - torch.max(mac_x_lo.unsqueeze(1), bin_x_lo.unsqueeze(0)),
        beta=steep,
    )
    ov_y = F.softplus(
        torch.min(mac_y_hi.unsqueeze(1), bin_y_hi.unsqueeze(0))
        - torch.max(mac_y_lo.unsqueeze(1), bin_y_lo.unsqueeze(0)),
        beta=steep,
    )

    # Density per cell: sum of overlap areas / grid_area → [G_x, G_y]
    density = torch.einsum("ni,nj->ij", ov_x, ov_y) / grid_area

    # Penalize top 10% (matching evaluator)
    flat = density.flatten()
    k = max(1, flat.numel() // 10)
    top_vals = flat.topk(k).values

    total_area = (sizes[:, 0] * sizes[:, 1]).sum()
    target = float(total_area / (canvas_w * canvas_h)) * 1.35
    excess = torch.clamp(top_vals - target, min=0)
    return excess.pow(2).sum() + 0.5 * top_vals.pow(2).mean()


def _congestion_penalty(pos, net_mask, net_hard, net_is_hard, net_ox, net_oy,
                        net_weight, canvas_w, canvas_h, grid_n=32):
    """Differentiable RUDY congestion proxy: net bounding-box routing demand.

    Estimates per-cell routing demand from net bounding boxes using sigmoid
    membership functions. Penalizes cells that exceed a headroom threshold,
    matching the evaluator's ABU-style congestion metric.
    """
    device = pos.device
    cell_w = canvas_w / grid_n
    cell_h = canvas_h / grid_n

    # Compute pin positions
    hard_x = pos[net_hard, 0]
    hard_y = pos[net_hard, 1]
    is_h = net_is_hard.float()

    pin_x = is_h * (hard_x + net_ox) + (1 - is_h) * net_ox
    pin_y = is_h * (hard_y + net_oy) + (1 - is_h) * net_oy

    BIG = 1e6
    pin_x_active = pin_x.clone(); pin_x_active[~net_mask] = BIG
    pin_x_active2 = pin_x.clone(); pin_x_active2[~net_mask] = -BIG
    pin_y_active = pin_y.clone(); pin_y_active[~net_mask] = BIG
    pin_y_active2 = pin_y.clone(); pin_y_active2[~net_mask] = -BIG

    net_xmin = pin_x_active.min(dim=1).values
    net_xmax = pin_x_active2.max(dim=1).values
    net_ymin = pin_y_active.min(dim=1).values
    net_ymax = pin_y_active2.max(dim=1).values

    # Grid cell centers
    cx = torch.linspace(cell_w / 2, canvas_w - cell_w / 2, grid_n, device=device)
    cy = torch.linspace(cell_h / 2, canvas_h - cell_h / 2, grid_n, device=device)

    sharpness = 4.0 / max(cell_w, cell_h)

    # Sigmoid membership: is grid cell center inside the net bounding box?
    in_x = (torch.sigmoid(sharpness * (cx.unsqueeze(0) - net_xmin.unsqueeze(1)))
            * torch.sigmoid(sharpness * (net_xmax.unsqueeze(1) - cx.unsqueeze(0))))
    in_y = (torch.sigmoid(sharpness * (cy.unsqueeze(0) - net_ymin.unsqueeze(1)))
            * torch.sigmoid(sharpness * (net_ymax.unsqueeze(1) - cy.unsqueeze(0))))

    # Weighted routing demand per cell
    demand = torch.einsum("ni,nj,n->ij", in_x, in_y, net_weight)

    # Target: average demand with headroom
    total_demand = (net_weight * ((net_xmax - net_xmin) / canvas_w)
                    * ((net_ymax - net_ymin) / canvas_h)).sum()
    target = total_demand / (grid_n * grid_n) * 2.0

    excess = torch.clamp(demand - target, min=0)
    return (excess ** 2).sum()


def _halo_sizes(sizes, area_scale=0.10, base_scale=0.03):
    """Inflate macro sizes to reserve routing channels during optimization."""
    area = sizes[:, 0] * sizes[:, 1]
    area_norm = area / area.max().clamp(min=1e-6)
    halo = base_scale + area_scale * area_norm.sqrt()
    scale = 1.0 + halo.unsqueeze(1)
    return sizes * scale


def _overlap_penalty(pos, sizes, movable=None):
    """Differentiable overlap penalty between all pairs of macros."""
    if movable is not None:
        idx = torch.where(movable)[0]
        if len(idx) < 2:
            return torch.tensor(0.0, device=pos.device)
        mp, ms = pos[idx], sizes[idx]
    else:
        mp, ms = pos, sizes
    n = mp.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=pos.device)

    half_w = ms[:, 0] / 2
    half_h = ms[:, 1] / 2

    dx = mp[:, 0].unsqueeze(1) - mp[:, 0].unsqueeze(0)
    dy = mp[:, 1].unsqueeze(1) - mp[:, 1].unsqueeze(0)

    sep_x = half_w.unsqueeze(1) + half_w.unsqueeze(0)
    sep_y = half_h.unsqueeze(1) + half_h.unsqueeze(0)

    ov_x = torch.clamp(sep_x - dx.abs(), min=0)
    ov_y = torch.clamp(sep_y - dy.abs(), min=0)

    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=pos.device), diagonal=1)
    return (ov_x * ov_y * mask.float()).sum()


def _repulsion_penalty(pos, sizes, movable):
    """Electrostatic repulsion between movable macros for global spreading."""
    mov_idx = torch.where(movable)[0]
    n = len(mov_idx)
    if n < 2:
        return torch.tensor(0.0, device=pos.device)

    mp = pos[mov_idx]
    ms = sizes[mov_idx]
    areas = ms[:, 0] * ms[:, 1]

    dx = mp[:, 0].unsqueeze(1) - mp[:, 0].unsqueeze(0)
    dy = mp[:, 1].unsqueeze(1) - mp[:, 1].unsqueeze(0)

    min_sep_x = (ms[:, 0].unsqueeze(1) + ms[:, 0].unsqueeze(0)) / 2
    min_sep_y = (ms[:, 1].unsqueeze(1) + ms[:, 1].unsqueeze(0)) / 2
    min_dist = (min_sep_x ** 2 + min_sep_y ** 2).sqrt()

    dist = (dx ** 2 + dy ** 2 + 1e-4).sqrt()

    charge = areas.unsqueeze(1) * areas.unsqueeze(0)
    area_norm = charge.max().clamp(min=1e-6)

    mask = torch.triu(torch.ones(n, n, device=pos.device, dtype=torch.bool), diagonal=1)
    repulsion = (charge / area_norm) * torch.clamp(min_dist * 1.5 - dist, min=0) / min_dist.clamp(min=1e-6)
    return repulsion[mask].sum()


def _quadratic_init(nets_sa, n_hard, movable_np, fixed_pos_np, cw, ch):
    """Quadratic placement init from net connectivity (weighted Laplacian solve)."""
    movable_idx = np.where(movable_np)[0]
    n_mov = len(movable_idx)
    if n_mov == 0:
        return fixed_pos_np.copy()

    mov_map = np.full(n_hard, -1, dtype=np.int64)
    for new_i, old_i in enumerate(movable_idx):
        mov_map[old_i] = new_i

    L = np.zeros((n_mov, n_mov))
    bx = np.zeros(n_mov)
    by = np.zeros(n_mov)
    INF = float("inf")

    for net in nets_sa:
        idx = net["hard_idx"]
        w = net["weight"]
        has_fixed = net["fxmin"] != INF
        n_pins = len(idx) + (1 if has_fixed else 0)
        if n_pins < 2:
            continue
        clique_w = w / (n_pins - 1)

        mov_in_net = [(mov_map[k], k) for k in idx if mov_map[k] >= 0]

        for a_i in range(len(mov_in_net)):
            ma, _ = mov_in_net[a_i]
            for b_i in range(a_i + 1, len(mov_in_net)):
                mb, _ = mov_in_net[b_i]
                L[ma, mb] -= clique_w
                L[mb, ma] -= clique_w
                L[ma, ma] += clique_w
                L[mb, mb] += clique_w

        for mi, orig_i in mov_in_net:
            for k in idx:
                if mov_map[k] < 0:
                    L[mi, mi] += clique_w
                    bx[mi] += clique_w * fixed_pos_np[k, 0]
                    by[mi] += clique_w * fixed_pos_np[k, 1]
            if has_fixed:
                fx = (net["fxmin"] + net["fxmax"]) / 2
                fy = (net["fymin"] + net["fymax"]) / 2
                L[mi, mi] += clique_w
                bx[mi] += clique_w * fx
                by[mi] += clique_w * fy

    anchor_w = np.max(np.diag(L)) * 0.001 + 1e-6
    for i in range(n_mov):
        L[i, i] += anchor_w
        bx[i] += anchor_w * cw / 2
        by[i] += anchor_w * ch / 2

    try:
        sol_x = np.linalg.solve(L, bx)
        sol_y = np.linalg.solve(L, by)
    except np.linalg.LinAlgError:
        sol_x = np.full(n_mov, cw / 2)
        sol_y = np.full(n_mov, ch / 2)

    result = fixed_pos_np.copy()
    for new_i, old_i in enumerate(movable_idx):
        result[old_i, 0] = np.clip(sol_x[new_i], 0, cw)
        result[old_i, 1] = np.clip(sol_y[new_i], 0, ch)
    return result


# ── Analytical placement phase ──────────────────────────────────────────────

def _analytical_place(
    benchmark: Benchmark,
    plc,
    nets_sa,
    num_steps: int = 2000,
    lr: float = 4.0,
    gamma_start: float = 40.0,
    gamma_end: float = 3.0,
    density_weight: float = 0.006,
    overlap_weight_start: float = 0.01,
    overlap_weight_end: float = 18.0,
    congestion_weight: float = 0.001,
    repulsion_weight: float = 0.6,
    halo_area_scale: float = 0.12,
    halo_base_scale: float = 0.04,
    use_quadratic_init: bool = True,
    seed: int = 42,
):
    """
    Gradient-based placement with gamma annealing, cosine LR, and multi-objective.

    v5 improvements over v4:
      - Gamma annealing (40→3): starts global, refines local
      - Cosine LR schedule for better convergence
      - Quadratic placement initialization from net connectivity
      - Electrostatic repulsion for global spreading
      - Overlap weight annealing (0.01→18.0) — gentle early, strong late
      - Gradient clipping for stability
      - 2000 steps (up from 1500)
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

    # Halo-inflated sizes for density/overlap/repulsion
    eff_sizes = _halo_sizes(sizes, area_scale=halo_area_scale, base_scale=halo_base_scale)

    init_pos = benchmark.macro_positions[:n_hard].clone().float().to(device)
    init_pos[:, 0] = torch.clamp(init_pos[:, 0], half_w, cw - half_w)
    init_pos[:, 1] = torch.clamp(init_pos[:, 1], half_h, ch - half_h)

    # Quadratic init from connectivity (better than initial.plc)
    if use_quadratic_init and nets_sa:
        movable_np = benchmark.get_movable_mask()[:n_hard].numpy()
        init_np = init_pos.detach().cpu().numpy().astype(np.float64)
        quad_np = _quadratic_init(nets_sa, n_hard, movable_np, init_np, cw, ch)
        init_pos = torch.tensor(quad_np, dtype=torch.float32, device=device)
        init_pos[:, 0] = torch.clamp(init_pos[:, 0], half_w, cw - half_w)
        init_pos[:, 1] = torch.clamp(init_pos[:, 1], half_h, ch - half_h)

    net_data = _build_net_tensors(plc, device)
    if net_data is None:
        return init_pos.detach().cpu().numpy().astype(np.float64)

    net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight = net_data

    pos = init_pos.clone().detach().requires_grad_(True)
    fixed_pos = init_pos[fixed_mask].clone()

    optimizer = torch.optim.Adam([pos], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr * 0.01,
    )

    best_pos = pos.data.clone()
    best_cost = float('inf')

    for step in range(num_steps):
        optimizer.zero_grad()

        with torch.no_grad():
            pos.data[fixed_mask] = fixed_pos

        frac = step / max(num_steps - 1, 1)

        # Gamma annealing: global → local
        gamma = gamma_start * (gamma_end / gamma_start) ** frac

        # HPWL loss
        hpwl = _lse_hpwl(pos, net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight, gamma=gamma)

        # Density penalty (progressive ramp: 0.1x → 1.0x)
        density_ramp = 0.1 + 0.9 * min(1.0, frac * 2.5)
        density = _density_penalty(pos, eff_sizes, cw, ch) * density_weight * density_ramp

        # Congestion penalty (ramp up over first 40%)
        cong_ramp = min(1.0, frac / 0.4)
        congestion = _congestion_penalty(
            pos, net_mask, net_hard, net_is_hard, net_ox, net_oy,
            net_weight, cw, ch,
        ) * congestion_weight * cong_ramp

        # Overlap weight annealing: gentle early, strong late
        ov_weight = overlap_weight_start * (
            overlap_weight_end / max(overlap_weight_start, 1e-12)
        ) ** frac
        overlap = _overlap_penalty(pos, eff_sizes, movable) * ov_weight

        # Electrostatic repulsion for global spreading
        repulsion = _repulsion_penalty(pos, eff_sizes, movable) * repulsion_weight * density_ramp

        loss = hpwl + density + congestion + overlap + repulsion
        loss.backward()

        # Gradient clipping for stability
        if pos.grad is not None:
            pos.grad[fixed_mask] = 0.0
            torch.nn.utils.clip_grad_norm_([pos], max_norm=max(cw, ch) * 2)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pos.data[:, 0] = torch.clamp(pos.data[:, 0], half_w, cw - half_w)
            pos.data[:, 1] = torch.clamp(pos.data[:, 1], half_h, ch - half_h)
            pos.data[fixed_mask] = fixed_pos

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
    congestion_weight: float = 0.001,
    overlap_weight: float = 0.08,
    anchor_weight: float = 0.01,
    seed: int = 42,
):
    """
    Short GPU refinement pass after SA.

    Keeps the SA solution as an anchor while letting differentiable losses
    (HPWL + density + congestion + overlap) fine-tune positions.
    The congestion term helps reduce routing hotspots that SA misses.
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr * 0.05,
    )

    best_pos = base_pos.clone()
    best_cost = float("inf")

    for _ in range(num_steps):
        optimizer.zero_grad()

        with torch.no_grad():
            pos.data[fixed_mask] = fixed_pos[fixed_mask]

        hpwl = _lse_hpwl(pos, net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight, gamma=gamma)
        density = _density_penalty(pos, sizes, cw, ch) * density_weight
        congestion = _congestion_penalty(
            pos, net_mask, net_hard, net_is_hard, net_ox, net_oy,
            net_weight, cw, ch,
        ) * congestion_weight
        overlap = _overlap_penalty(pos, sizes) * overlap_weight
        anchor = ((pos[movable] - base_pos[movable]) ** 2).sum() * anchor_weight

        loss = hpwl + density + congestion + overlap + anchor
        loss.backward()

        # Clip grads for stability
        if pos.grad is not None:
            pos.grad[fixed_mask] = 0.0
            torch.nn.utils.clip_grad_norm_([pos], max_norm=max(cw, ch))

        optimizer.step()
        scheduler.step()

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


# ── Evaluator-guided greedy refinement ─────────────────────────────────────

def _greedy_refine_proxy(
    pos: np.ndarray,
    benchmark,
    plc,
    movable: np.ndarray,
    sizes: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    cw: float,
    ch: float,
    max_passes: int = 5,
    scales: tuple = (0.004, 0.012, 0.03, 0.065, 0.12),
    connectivity_weights: np.ndarray | None = None,
    macro_pull_targets: np.ndarray | None = None,
):
    """
    Greedy coordinate descent using the actual proxy evaluator.

    For each movable macro, tries small perturbations in 8 directions at
    multiple scales.  Accepts the position that gives the lowest *real*
    proxy cost (WL + 0.5*density + 0.5*congestion).

    Macros are processed highest-connectivity-first so that the most
    impactful moves happen early in each pass, improving subsequent evals.

    This directly optimises the target metric, closing the gap that SA
    (which only tracks HPWL + density) leaves on the table — especially
    for congestion.
    """
    GAP = 0.05
    n_hard = len(movable)
    best_pos = pos.copy()
    best_score = _proxy_score_pos(benchmark, plc, best_pos)

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    movable_idx = np.where(movable)[0].copy()
    # Sort by connectivity descending — high-degree macros have most impact per eval
    if connectivity_weights is not None:
        conn_order = np.argsort(-connectivity_weights[movable_idx])
        movable_idx_sorted = movable_idx[conn_order]
    else:
        movable_idx_sorted = movable_idx

    for pass_num in range(max_passes):
        improved = False
        # Alternate: connectivity order on even passes, shuffled on odd passes
        if pass_num % 2 == 0:
            macro_order = movable_idx_sorted
        else:
            macro_order = movable_idx.copy()
            np.random.shuffle(macro_order)

        for i in macro_order:
            old_x, old_y = best_pos[i, 0], best_pos[i, 1]
            best_local_score = best_score
            best_local_x, best_local_y = old_x, old_y
            candidate_positions = []

            for scale in scales:
                dx = cw * scale
                dy = ch * scale
                for ax, ay in directions:
                    new_x = float(np.clip(old_x + ax * dx, half_w[i], cw - half_w[i]))
                    new_y = float(np.clip(old_y + ay * dy, half_h[i], ch - half_h[i]))
                    if new_x != old_x or new_y != old_y:
                        candidate_positions.append((new_x, new_y))

            if macro_pull_targets is not None:
                tx, ty = macro_pull_targets[i]
                for alpha in (0.35, 0.65, 1.0):
                    new_x = float(np.clip(old_x + alpha * (tx - old_x), half_w[i], cw - half_w[i]))
                    new_y = float(np.clip(old_y + alpha * (ty - old_y), half_h[i], ch - half_h[i]))
                    if new_x != old_x or new_y != old_y:
                        candidate_positions.append((new_x, new_y))
                    if new_x != old_x:
                        candidate_positions.append((new_x, old_y))
                    if new_y != old_y:
                        candidate_positions.append((old_x, new_y))

            seen = set()
            deduped_positions = []
            for cand_x, cand_y in candidate_positions:
                key = (round(cand_x, 6), round(cand_y, 6))
                if key in seen:
                    continue
                seen.add(key)
                deduped_positions.append((cand_x, cand_y))

            for new_x, new_y in deduped_positions:
                best_pos[i, 0] = new_x
                best_pos[i, 1] = new_y
                ddx = np.abs(best_pos[i, 0] - best_pos[:, 0])
                ddy = np.abs(best_pos[i, 1] - best_pos[:, 1])
                ov = (ddx < sep_x[i] + GAP) & (ddy < sep_y[i] + GAP)
                ov[i] = False
                if ov.any():
                    best_pos[i, 0] = old_x
                    best_pos[i, 1] = old_y
                    continue

                score = _proxy_score_pos(benchmark, plc, best_pos)
                if score < best_local_score:
                    best_local_score = score
                    best_local_x, best_local_y = new_x, new_y

                best_pos[i, 0] = old_x
                best_pos[i, 1] = old_y

            if best_local_score < best_score - 1e-6:
                best_pos[i, 0] = best_local_x
                best_pos[i, 1] = best_local_y
                best_score = best_local_score
                improved = True

                refine_dx = max(abs(best_local_x - old_x) * 0.5, cw * 0.001)
                refine_dy = max(abs(best_local_y - old_y) * 0.5, ch * 0.001)
                for ax, ay in directions:
                    cand_x = float(np.clip(best_local_x + ax * refine_dx, half_w[i], cw - half_w[i]))
                    cand_y = float(np.clip(best_local_y + ay * refine_dy, half_h[i], ch - half_h[i]))
                    if cand_x == best_pos[i, 0] and cand_y == best_pos[i, 1]:
                        continue
                    best_pos[i, 0] = cand_x
                    best_pos[i, 1] = cand_y
                    ddx = np.abs(best_pos[i, 0] - best_pos[:, 0])
                    ddy = np.abs(best_pos[i, 1] - best_pos[:, 1])
                    ov = (ddx < sep_x[i] + GAP) & (ddy < sep_y[i] + GAP)
                    ov[i] = False
                    if ov.any():
                        best_pos[i, 0] = best_local_x
                        best_pos[i, 1] = best_local_y
                        continue
                    score = _proxy_score_pos(benchmark, plc, best_pos)
                    if score < best_score - 1e-6:
                        best_score = score
                        best_local_x, best_local_y = cand_x, cand_y
                    best_pos[i, 0] = best_local_x
                    best_pos[i, 1] = best_local_y

        if not improved:
            break

    return best_pos


def _compute_macro_pull_targets(
    pos: np.ndarray,
    nets: list[dict],
    macro_to_nets: list[list[int]],
    movable: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
) -> np.ndarray:
    """Weighted connected-pin centroids used to bias greedy search proposals."""
    n_hard = len(movable)
    targets = pos.copy()
    canvas_center = np.array([cw / 2.0, ch / 2.0], dtype=np.float64)

    for i in range(n_hard):
        if not movable[i]:
            continue
        x_sum = 0.0
        y_sum = 0.0
        w_sum = 0.0
        for net_idx in macro_to_nets[i]:
            net = nets[net_idx]
            weight = float(net["weight"])
            for k, j in enumerate(net["hard_idx"]):
                if j == i:
                    continue
                x_sum += weight * (pos[j, 0] + net["hard_ox"][k])
                y_sum += weight * (pos[j, 1] + net["hard_oy"][k])
                w_sum += weight
            if net["fxmin"] != float("inf"):
                fx = 0.5 * (net["fxmin"] + net["fxmax"])
                fy = 0.5 * (net["fymin"] + net["fymax"])
                x_sum += weight * fx
                y_sum += weight * fy
                w_sum += weight

        if w_sum <= 1e-9:
            tx, ty = canvas_center
        else:
            tx = x_sum / w_sum
            ty = y_sum / w_sum
        targets[i, 0] = float(np.clip(tx, half_w[i], cw - half_w[i]))
        targets[i, 1] = float(np.clip(ty, half_h[i], ch - half_h[i]))

    return targets


# ── Fast soft macro recentering ────────────────────────────────────────────

def _recenter_soft_macros(pos_hard: np.ndarray, benchmark, plc):
    """
    Place each soft macro at the weighted centroid of its connected pin positions.

    After hard macros are placed, soft macro positions affect wirelength and
    congestion.  Moving them to the centroid of their net partners is a cheap
    improvement over the initial.plc positions.
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    # Build soft macro -> connected pin positions
    soft_positions = []
    for si, plc_idx in enumerate(benchmark.soft_macro_indices):
        mod = plc.modules_w_pins[plc_idx]
        name = mod.get_name()
        sw = mod.get_width()
        sh = mod.get_height()

        # Collect positions of all pins connected to this soft macro
        cx_sum, cy_sum, w_sum = 0.0, 0.0, 0.0
        pin_map = getattr(plc, '_macro_pin_map', {})
        for pin_idx in pin_map.get(name, []):
            pin = plc.modules_w_pins[pin_idx]
            if not pin.get_sink():
                continue
            for sink_list in pin.get_sink().values():
                for sink_name in sink_list:
                    if sink_name not in plc.mod_name_to_indices:
                        continue
                    sink_idx = plc.mod_name_to_indices[sink_name]
                    sink_obj = plc.modules_w_pins[sink_idx]
                    sx, sy = sink_obj.get_pos()
                    w = pin.get_weight() if pin.get_weight() > 0 else 1.0
                    cx_sum += sx * w
                    cy_sum += sy * w
                    w_sum += w

        if w_sum > 0:
            cx = np.clip(cx_sum / w_sum, sw / 2, cw - sw / 2)
            cy = np.clip(cy_sum / w_sum, sh / 2, ch - sh / 2)
            soft_positions.append((cx, cy))
        else:
            soft_positions.append(mod.get_pos())

    return np.array(soft_positions, dtype=np.float64) if soft_positions else np.empty((0, 2))


# ── HybridPlacer class ─────────────────────────────────────────────────────

class HybridPlacer(BasePlacer):
    """
    Hybrid placer v8: multi-candidate analytical → pre-SA GPU smooth → SA →
      GPU refine → greedy → flip → GPU polish → fine greedy → micro GPU polish.

    Phase 1  (analytical): 7 diverse candidates × 2500 Adam steps.
    Phase 2  (legalize): Minimum-displacement overlap resolution.
    Phase 3  (pre-SA GPU smooth): 100-step tight-anchor GPU pass to remove legalization
               displacement artifacts before SA.
    Phase 4  (SA): 300K iterations with HPWL + density co-optimization + reheating.
    Phase 5  (flip): Greedy pin orientation flipping for HPWL reduction.
    Phase 6  (GPU refine): Cosine-scheduled differentiable pass with congestion loss (500 steps).
    Phase 7  (greedy refine): Connectivity-sorted evaluator-guided coordinate descent
               (5 passes, scales 0.004–0.12).
    Phase 8  (post-greedy flip): Second orientation flip after greedy repositioning.
    Phase 9  (GPU polish): Short differentiable pass after greedy (200 steps, lr=0.04).
    Phase 10 (fine greedy): 3-pass coordinate descent with fine scales (0.001–0.05).
    Phase 11 (micro GPU polish): 100-step tight-anchor polish after fine greedy.
    Phase 12 (soft recenter): Move soft macros to weighted pin centroids.

    v8 changes vs v7 (avg 1.4828):
      - Pre-SA GPU smoothing (100 steps, tight anchor) to remove legalization artifacts
      - Post-first-greedy flip: second orientation flip after greedy repositioning
      - Fine greedy: 3 passes at scales (0.001, 0.003, 0.008, 0.02, 0.05) after GPU polish
      - Micro GPU polish: 100-step low-LR tight-anchor pass after fine greedy
    """

    def __init__(
        self,
        seed: int = 42,
        # Analytical phase
        analytical_steps: int = 2500,
        analytical_candidates: int = 7,
        # Pre-SA GPU smoothing (removes legalization artifacts)
        pre_sa_smoothing_steps: int = 100,
        # SA phase
        sa_iters: int = 300_000,
        congestion_sa_iters: int = 40_000,
        sa_t_start: float = 0.15,
        sa_t_end: float = 0.001,
        reheat_threshold: int = 10_000,
        reheat_factor: float = 3.0,
        sa_density_boost: float = 1.0,
        post_refine_steps: int = 500,
        post_greedy_polish_steps: int = 200,
        # Greedy refinement
        greedy_passes: int = 5,
        # Fine greedy (second cycle, small scales) + micro polish
        second_greedy_passes: int = 3,
        second_greedy_polish_steps: int = 100,
        # Soft macro FD
        run_fd: bool = False,
        # Debug
        capture_snapshots: bool = True,
        snapshot_interval: int = 2_000,
        trace_interval: int = 500,
    ):
        self.seed = seed
        self.analytical_steps = analytical_steps
        self.analytical_candidates = analytical_candidates
        self.pre_sa_smoothing_steps = pre_sa_smoothing_steps
        self.sa_iters = sa_iters
        self.congestion_sa_iters = congestion_sa_iters
        self.sa_t_start = sa_t_start
        self.sa_t_end = sa_t_end
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor
        self.sa_density_boost = sa_density_boost
        self.post_refine_steps = post_refine_steps
        self.post_greedy_polish_steps = post_greedy_polish_steps
        self.greedy_passes = greedy_passes
        self.second_greedy_passes = second_greedy_passes
        self.second_greedy_polish_steps = second_greedy_polish_steps
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

        # Extract nets early — needed for quadratic init and SA
        if plc is not None:
            nets, macro_to_nets = _extract_nets(benchmark, plc)
        else:
            nets, macro_to_nets = [], [[] for _ in range(n_hard)]

        # ── Phase 1: Multi-candidate analytical placement ─────────────────
        if plc is not None:
            # Different density/halo/repulsion configs explore diverse strategies
            candidate_configs = [
                # Balanced (original best)
                dict(density_weight=0.006, congestion_weight=0.001,
                     repulsion_weight=0.6, halo_area=0.12, halo_base=0.04,
                     gamma_start=40.0, gamma_end=3.0),
                # High density + congestion pressure (spread macros aggressively)
                dict(density_weight=0.012, congestion_weight=0.002,
                     repulsion_weight=1.0, halo_area=0.16, halo_base=0.05,
                     gamma_start=40.0, gamma_end=3.0),
                # Low pressure (trust HPWL, minimal spreading)
                dict(density_weight=0.004, congestion_weight=0.0008,
                     repulsion_weight=0.3, halo_area=0.08, halo_base=0.03,
                     gamma_start=40.0, gamma_end=3.0),
                # Congestion-heavy + gentle gamma (less aggressive local pinning)
                dict(density_weight=0.008, congestion_weight=0.005,
                     repulsion_weight=0.8, halo_area=0.20, halo_base=0.06,
                     gamma_start=20.0, gamma_end=4.0),
                # Wirelength-focused + aggressive gamma (tight local optimum)
                dict(density_weight=0.002, congestion_weight=0.0004,
                     repulsion_weight=0.15, halo_area=0.05, halo_base=0.02,
                     gamma_start=60.0, gamma_end=2.0),
                # Medium density + very gentle gamma (smooth global landscape)
                dict(density_weight=0.005, congestion_weight=0.0015,
                     repulsion_weight=0.5, halo_area=0.10, halo_base=0.035,
                     gamma_start=15.0, gamma_end=5.0),
                # High repulsion + mid gamma (maximally spread initial solution)
                dict(density_weight=0.010, congestion_weight=0.003,
                     repulsion_weight=1.5, halo_area=0.18, halo_base=0.055,
                     gamma_start=30.0, gamma_end=3.0),
            ][:max(1, self.analytical_candidates)]

            best_init_score = float("inf")
            pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)

            for ci, cfg in enumerate(candidate_configs):
                cand_pos = _analytical_place(
                    benchmark, plc, nets,
                    num_steps=self.analytical_steps,
                    lr=4.0,
                    gamma_start=cfg.get("gamma_start", 40.0),
                    gamma_end=cfg.get("gamma_end", 3.0),
                    density_weight=cfg["density_weight"],
                    congestion_weight=cfg["congestion_weight"],
                    repulsion_weight=cfg["repulsion_weight"],
                    halo_area_scale=cfg["halo_area"],
                    halo_base_scale=cfg["halo_base"],
                    use_quadratic_init=True,
                    seed=self.seed + ci * 997,
                )
                legal_cand = _legalize(cand_pos, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)
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

        # ── Phase 3: Pre-SA GPU smoothing ─────────────────────────────────
        # Remove legalization displacement artifacts before SA for a cleaner start.
        if plc is not None and self.pre_sa_smoothing_steps > 0:
            pre_smooth_pos = pos.copy()
            smoothed = _gpu_post_refine(
                benchmark, plc, pos,
                num_steps=self.pre_sa_smoothing_steps,
                lr=0.02,
                anchor_weight=0.05,
                seed=self.seed,
            )
            smoothed = _legalize(smoothed, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)
            try:
                if _proxy_score_pos(benchmark, plc, smoothed) <= _proxy_score_pos(benchmark, plc, pre_smooth_pos):
                    pos = smoothed
            except Exception:
                pass

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
        neighbors = [set() for _ in range(n_hard)]
        for net in nets:
            idx = net["hard_idx"]
            for a in idx:
                for b in idx:
                    if a != b:
                        neighbors[a].add(int(b))
        neighbors = [list(s) for s in neighbors]

        # ── Density weight calibration (SA co-optimises HPWL + density) ───
        grid_col = grid_row = 0
        density_w = 0.0
        if plc is not None and nets:
            try:
                grid_col, grid_row = plc.grid_col, plc.grid_row
                if grid_col > 0 and grid_row > 0:
                    from macro_place.objective import _set_placement
                    full_init = benchmark.macro_positions.clone()
                    full_init[:n_hard] = torch.tensor(pos, dtype=torch.float32)
                    _set_placement(plc, full_init, benchmark)
                    wl_norm = plc.get_cost()
                    raw_hpwl = _compute_total_hpwl(pos, nets)
                    if wl_norm > 1e-10 and raw_hpwl > 1e-10:
                        density_w = 0.5 * raw_hpwl / wl_norm
                        density_w *= self.sa_density_boost
            except Exception:
                grid_col = grid_row = 0
                density_w = 0.0

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
                density_weight=density_w,
                grid_col=grid_col,
                grid_row=grid_row,
                benchmark=benchmark,
            )

        # ── Short congestion-focused SA cleanup ────────────────────────────
        if plc is not None and nets and self.congestion_sa_iters > 0:
            try:
                vroutes_per_micron = float(getattr(plc, "vroutes_per_micron", getattr(benchmark, "vroutes_per_micron", 0.0)))
                hroutes_per_micron = float(getattr(plc, "hroutes_per_micron", getattr(benchmark, "hroutes_per_micron", 0.0)))
                vrouting_alloc = float(getattr(plc, "vrouting_alloc", 0.0))
                hrouting_alloc = float(getattr(plc, "hrouting_alloc", 0.0))
                smooth_range = int(getattr(plc, "congestion_smooth_range", 0))
                if grid_col > 0 and grid_row > 0 and vroutes_per_micron > 0 and hroutes_per_micron > 0:
                    congestion_pos = _sa_refine_congestion(
                        pos, nets, macro_to_nets, neighbors,
                        movable, sizes, half_w, half_h, sep_x, sep_y,
                        cw, ch, self.congestion_sa_iters, self.seed + 17,
                        grid_row, grid_col, vroutes_per_micron, hroutes_per_micron,
                        vrouting_alloc, hrouting_alloc, smooth_range,
                        plc,
                        congestion_weight=0.35,
                        t_start_factor=max(self.sa_t_start * 0.45, 0.03),
                        t_end_factor=max(self.sa_t_end * 0.5, 0.0005),
                        reheat_threshold=max(2_000, self.reheat_threshold // 2),
                        reheat_factor=max(1.5, self.reheat_factor - 0.5),
                    )
                    congestion_pos = _legalize(
                        congestion_pos, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y,
                    )
                    if _proxy_score_pos(benchmark, plc, congestion_pos) <= _proxy_score_pos(benchmark, plc, pos):
                        pos = congestion_pos
            except Exception:
                pass

        # ── Greedy pin flipping (proxy-aware when benchmark is available) ──
        if nets and plc is not None:
            _greedy_flip(pos, nets, macro_to_nets, movable, plc, benchmark=benchmark)

        # ── GPU post-refinement (congestion-aware differentiable fine-tune) ─
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

        # ── Compute per-macro connectivity weights (used in both greedy rounds) ─
        conn_weights = np.zeros(n_hard)
        for net in nets:
            w = net["weight"]
            for idx in net["hard_idx"]:
                conn_weights[idx] += w
        macro_pull_targets = _compute_macro_pull_targets(
            pos, nets, macro_to_nets, movable, half_w, half_h, cw, ch,
        ) if nets else None

        # ── Evaluator-guided greedy refinement ─────────────────────────────
        if plc is not None and self.greedy_passes > 0:
            pos = _greedy_refine_proxy(
                pos, benchmark, plc, movable, sizes,
                half_w, half_h, sep_x, sep_y, cw, ch,
                max_passes=self.greedy_passes,
                connectivity_weights=conn_weights,
                macro_pull_targets=macro_pull_targets,
            )

        # ── Post-greedy flip (second orientation pass after greedy repositioning) ─
        # Greedy shifts may unlock different optimal orientations.
        if nets and plc is not None:
            _greedy_flip(pos, nets, macro_to_nets, movable, plc, benchmark=benchmark)

        # ── Post-greedy GPU polish ──────────────────────────────────────────
        # Short differentiable pass to smooth any interaction effects from greedy
        if plc is not None and self.post_greedy_polish_steps > 0:
            pre_polish = pos.copy()
            polished_pos = _gpu_post_refine(
                benchmark,
                plc,
                pos,
                num_steps=self.post_greedy_polish_steps,
                lr=0.04,
                anchor_weight=0.05,
                seed=self.seed,
            )
            polished_pos = _legalize(polished_pos, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)
            try:
                if _proxy_score_pos(benchmark, plc, polished_pos) <= _proxy_score_pos(benchmark, plc, pre_polish):
                    pos = polished_pos
            except Exception:
                pass

        # ── Fine greedy (second cycle, small scales) ────────────────────────
        # After GPU polish, positions have shifted slightly; fine-scale greedy
        # captures sub-pixel improvements the coarser first pass missed.
        if plc is not None and self.second_greedy_passes > 0:
            pos = _greedy_refine_proxy(
                pos, benchmark, plc, movable, sizes,
                half_w, half_h, sep_x, sep_y, cw, ch,
                max_passes=self.second_greedy_passes,
                scales=(0.001, 0.003, 0.008, 0.02, 0.05),
                connectivity_weights=conn_weights,
                macro_pull_targets=macro_pull_targets,
            )

        # ── Micro GPU polish (tight anchor, after fine greedy) ──────────────
        # Very short differentiable pass to smooth fine-greedy interaction effects.
        if plc is not None and self.second_greedy_polish_steps > 0:
            pre_micro = pos.copy()
            micro_polished = _gpu_post_refine(
                benchmark,
                plc,
                pos,
                num_steps=self.second_greedy_polish_steps,
                lr=0.02,
                anchor_weight=0.1,
                seed=self.seed,
            )
            micro_polished = _legalize(micro_polished, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)
            try:
                if _proxy_score_pos(benchmark, plc, micro_polished) <= _proxy_score_pos(benchmark, plc, pre_micro):
                    pos = micro_polished
            except Exception:
                pass

        # ── Final proxy-aware macro shift cleanup ──────────────────────────
        if plc is not None:
            pos = _proxy_local_search(
                pos, benchmark, plc, movable, sizes,
                half_w, half_h, sep_x, sep_y, cw, ch,
                max_macros=20,
                shift_frac=0.02,
                time_budget=20.0,
            )

        # Build full placement tensor
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)

        # ── Fast soft macro recentering ────────────────────────────────────
        if plc is not None and benchmark.num_soft_macros > 0:
            # First update plc with final hard macro positions for pin lookups
            from macro_place.objective import _set_placement
            _set_placement(plc, full_pos, benchmark)
            soft_new = _recenter_soft_macros(pos, benchmark, plc)
            if len(soft_new) > 0:
                # Score with recentered soft macros vs original
                full_recentered = full_pos.clone()
                full_recentered[n_hard:] = torch.tensor(soft_new, dtype=torch.float32)
                try:
                    score_orig = float(compute_proxy_cost(full_pos, benchmark, plc)["proxy_cost"])
                    score_new = float(compute_proxy_cost(full_recentered, benchmark, plc)["proxy_cost"])
                    if score_new < score_orig:
                        full_pos = full_recentered
                except Exception:
                    pass

        if self.run_fd and plc is not None and benchmark.num_soft_macros > 0:
            soft_pos = _update_soft_macros(pos, benchmark, plc)
            full_pos[n_hard:] = torch.tensor(soft_pos, dtype=torch.float32)

        if self.capture_snapshots:
            if not self.debug_snapshots or not torch.equal(self.debug_snapshots[-1], full_pos):
                self.debug_snapshots.append(full_pos.clone())

        return full_pos
