"""
Analytical (gradient-based) placer — differentiable placement with Adam.

Approach:
  - Differentiable HPWL via log-sum-exp (LSE) approximation
  - Differentiable Gaussian density penalty (smooth spreading)
  - Differentiable overlap penalty (increasing weight schedule)
  - Differentiable congestion proxy (net bounding-box routing demand)
  - Quadratic placement initialization from net connectivity
  - Progressive density weight ramping (RePlAce-style)
  - Adam optimizer with cosine-annealed LR and projection to canvas bounds
  - Legalization post-processing (minimum displacement, reused from sa_placer)
  - SA polish after legalization for local refinement

Usage:
    PLACER_METHOD=analytical uv run evaluate submissions/placer.py --all
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from submissions.base import BasePlacer
from submissions.sa_placer import (
    _load_plc, _extract_nets, _legalize, _sa_refine,
    _compute_total_hpwl,
)


# ── Net tensor construction ────────────────────────────────────────────────

def _build_net_tensors(nets, device="cpu"):
    """
    Convert list-of-dict nets (from _extract_nets) into padded PyTorch tensors
    for vectorized LSE-HPWL computation.

    For each net, movable hard-macro pins are represented by (macro_idx, offset).
    Fixed pins (ports, soft macros) are collapsed to two virtual pins at the
    bounding-box extremes (fxmin/fxmax, fymin/fymax) — correct because HPWL
    decomposes into independent x and y terms.
    """
    num_nets = len(nets)
    if num_nets == 0:
        return None

    INF = float("inf")
    max_pins = max(
        len(n["hard_idx"]) + (2 if n["fxmin"] != INF else 0) for n in nets
    )

    pin_macro_idx = torch.zeros(num_nets, max_pins, dtype=torch.long, device=device)
    pin_offset_x = torch.zeros(num_nets, max_pins, device=device)
    pin_offset_y = torch.zeros(num_nets, max_pins, device=device)
    pin_fixed_x = torch.zeros(num_nets, max_pins, device=device)
    pin_fixed_y = torch.zeros(num_nets, max_pins, device=device)
    pin_is_fixed = torch.zeros(num_nets, max_pins, dtype=torch.bool, device=device)
    pin_mask = torch.zeros(num_nets, max_pins, dtype=torch.bool, device=device)
    net_weights = torch.zeros(num_nets, device=device)

    for i, net in enumerate(nets):
        idx = net["hard_idx"]
        n_hard_pins = len(idx)

        for j in range(n_hard_pins):
            pin_macro_idx[i, j] = idx[j]
            pin_offset_x[i, j] = net["hard_ox"][j]
            pin_offset_y[i, j] = net["hard_oy"][j]
            pin_mask[i, j] = True

        k = n_hard_pins
        if net["fxmin"] != INF:
            pin_is_fixed[i, k] = True
            pin_fixed_x[i, k] = net["fxmin"]
            pin_fixed_y[i, k] = net["fymin"]
            pin_mask[i, k] = True
            k += 1

            pin_is_fixed[i, k] = True
            pin_fixed_x[i, k] = net["fxmax"]
            pin_fixed_y[i, k] = net["fymax"]
            pin_mask[i, k] = True

        net_weights[i] = net["weight"]

    return {
        "pin_macro_idx": pin_macro_idx,
        "pin_offset_x": pin_offset_x,
        "pin_offset_y": pin_offset_y,
        "pin_fixed_x": pin_fixed_x,
        "pin_fixed_y": pin_fixed_y,
        "pin_is_fixed": pin_is_fixed,
        "pin_mask": pin_mask,
        "net_weights": net_weights,
    }


# ── Pin position helper ──────────────────────────────────────────────────

def _get_pin_positions(pos, net_data):
    """Compute pin x/y positions from macro positions + offsets."""
    macro_idx = net_data["pin_macro_idx"]
    safe_idx = macro_idx.clamp(0)

    px = pos[safe_idx, 0] + net_data["pin_offset_x"]
    py = pos[safe_idx, 1] + net_data["pin_offset_y"]

    is_fixed = net_data["pin_is_fixed"]
    px = torch.where(is_fixed, net_data["pin_fixed_x"], px)
    py = torch.where(is_fixed, net_data["pin_fixed_y"], py)
    return px, py


# ── Differentiable HPWL (log-sum-exp) ─────────────────────────────────────

def _lse_hpwl(pos, net_data, gamma=10.0):
    """
    Compute differentiable weighted HPWL using log-sum-exp approximation.

    HPWL_x = max(x) - min(x) ≈ γ·log(Σ exp(x/γ)) + γ·log(Σ exp(-x/γ))
    """
    px, py = _get_pin_positions(pos, net_data)
    mask = net_data["pin_mask"]
    LARGE = 1e10

    px_for_max = px.clone(); px_for_max[~mask] = -LARGE
    py_for_max = py.clone(); py_for_max[~mask] = -LARGE

    max_x = gamma * torch.logsumexp(px_for_max / gamma, dim=1)
    max_y = gamma * torch.logsumexp(py_for_max / gamma, dim=1)

    px_for_min = px.clone(); px_for_min[~mask] = LARGE
    py_for_min = py.clone(); py_for_min[~mask] = LARGE

    min_x = -gamma * torch.logsumexp(-px_for_min / gamma, dim=1)
    min_y = -gamma * torch.logsumexp(-py_for_min / gamma, dim=1)

    hpwl = (max_x - min_x) + (max_y - min_y)
    return (hpwl * net_data["net_weights"]).sum()


# ── Differentiable congestion proxy ───────────────────────────────────────

def _congestion_penalty(pos, net_data, cw, ch, grid_size=32):
    """
    Smooth congestion proxy: approximate routing demand per grid cell.

    For each net, the bounding box determines which cells need routing.
    We spread the net's demand smoothly using sigmoid-based soft indicators.
    Penalizes cells where demand exceeds a target capacity.
    """
    px, py = _get_pin_positions(pos, net_data)
    mask = net_data["pin_mask"]
    LARGE = 1e10

    # Net bounding boxes (using actual min/max for stability)
    px_max = px.clone(); px_max[~mask] = -LARGE
    py_max = py.clone(); py_max[~mask] = -LARGE
    px_min = px.clone(); px_min[~mask] = LARGE
    py_min = py.clone(); py_min[~mask] = LARGE

    net_xmin = px_min.min(dim=1).values  # [N_nets]
    net_xmax = px_max.max(dim=1).values
    net_ymin = py_min.min(dim=1).values
    net_ymax = py_max.max(dim=1).values

    device = pos.device
    cell_w = cw / grid_size
    cell_h = ch / grid_size

    # Grid cell centers
    cx = torch.linspace(cell_w / 2, cw - cell_w / 2, grid_size, device=device)
    cy = torch.linspace(cell_h / 2, ch - cell_h / 2, grid_size, device=device)

    # Soft indicator: is cell center inside net bounding box?
    # Using sigmoid for smooth differentiability
    sharpness = 4.0 / max(cell_w, cell_h)

    # cx: [G], net_xmin: [N] → indicator: [N, G]
    in_x = torch.sigmoid(sharpness * (cx.unsqueeze(0) - net_xmin.unsqueeze(1))) * \
           torch.sigmoid(sharpness * (net_xmax.unsqueeze(1) - cx.unsqueeze(0)))
    in_y = torch.sigmoid(sharpness * (cy.unsqueeze(0) - net_ymin.unsqueeze(1))) * \
           torch.sigmoid(sharpness * (net_ymax.unsqueeze(1) - cy.unsqueeze(0)))

    weights = net_data["net_weights"]  # [N]

    # Routing demand: [G, G] = sum over nets of weight * in_x * in_y
    demand = torch.einsum("ni,nj,n->ij", in_x, in_y, weights)

    # Target capacity (uniform distribution of total demand)
    total_demand = (weights * ((net_xmax - net_xmin) / cw) * ((net_ymax - net_ymin) / ch)).sum()
    target = total_demand / (grid_size * grid_size) * 2.0  # 2x headroom

    excess = torch.clamp(demand - target, min=0)
    return (excess ** 2).sum()


# ── Differentiable density penalty ────────────────────────────────────────

def _density_penalty(pos, sizes, movable_mask, cw, ch, grid_size=32, target_scale=1.35):
    """
    Smooth density penalty using Gaussian spreading.
    """
    device = pos.device
    cell_w = cw / grid_size
    cell_h = ch / grid_size

    cx = torch.linspace(cell_w / 2, cw - cell_w / 2, grid_size, device=device)
    cy = torch.linspace(cell_h / 2, ch - cell_h / 2, grid_size, device=device)

    mov_idx = torch.where(movable_mask)[0]
    if len(mov_idx) == 0:
        return torch.tensor(0.0, device=device)

    mov_pos = pos[mov_idx]
    mov_sizes = sizes[mov_idx]

    sigma_x = mov_sizes[:, 0] / 2 + 1e-6
    sigma_y = mov_sizes[:, 1] / 2 + 1e-6
    areas = mov_sizes[:, 0] * mov_sizes[:, 1]

    dx = (cx.unsqueeze(0) - mov_pos[:, 0:1]) / sigma_x.unsqueeze(1)
    dy = (cy.unsqueeze(0) - mov_pos[:, 1:2]) / sigma_y.unsqueeze(1)

    wx = torch.exp(-0.5 * dx ** 2)
    wy = torch.exp(-0.5 * dy ** 2)

    norm = areas / (2 * 3.14159 * sigma_x * sigma_y)
    wx_norm = wx * norm.unsqueeze(1)

    density = torch.einsum("mi,mj->ij", wx_norm, wy)

    total_area = areas.sum()
    target_density = total_area / (cw * ch)
    density_per_cell = density / (cell_w * cell_h)

    excess = torch.clamp(density_per_cell - target_density * target_scale, min=0)
    hotspot = density_per_cell.flatten()
    k = max(1, hotspot.numel() // 10)
    top_vals = hotspot.topk(k).values
    return ((excess ** 2).sum() + 0.25 * top_vals.pow(2).mean()) * cell_w * cell_h


# ── Overlap penalty (differentiable) ──────────────────────────────────────

def _overlap_penalty(pos, sizes, movable_mask):
    """
    Soft differentiable overlap penalty between all hard-macro pairs.
    """
    mov_idx = torch.where(movable_mask)[0]
    n = len(mov_idx)
    if n < 2:
        return torch.tensor(0.0, device=pos.device)

    mp = pos[mov_idx]
    ms = sizes[mov_idx]
    half = ms / 2

    dx = mp[:, 0].unsqueeze(1) - mp[:, 0].unsqueeze(0)
    dy = mp[:, 1].unsqueeze(1) - mp[:, 1].unsqueeze(0)

    sep_x = half[:, 0].unsqueeze(1) + half[:, 0].unsqueeze(0)
    sep_y = half[:, 1].unsqueeze(1) + half[:, 1].unsqueeze(0)

    ov_x = torch.clamp(sep_x - dx.abs(), min=0)
    ov_y = torch.clamp(sep_y - dy.abs(), min=0)

    overlap = ov_x * ov_y
    mask = torch.triu(torch.ones(n, n, device=pos.device, dtype=torch.bool), diagonal=1)
    return overlap[mask].sum()


def _halo_sizes(sizes, area_scale=0.10, base_scale=0.03):
    """
    Inflate macro sizes to reserve channels/whitespace during optimization.

    Inspired by macro halo / channel reservation ideas from congestion-aware
    macro placement work: larger macros get slightly larger effective halos.
    """
    area = sizes[:, 0] * sizes[:, 1]
    area_norm = area / area.max().clamp(min=1e-6)
    halo = base_scale + area_scale * area_norm.sqrt()
    scale = 1.0 + halo.unsqueeze(1)
    return sizes * scale


def _boundary_whitespace_penalty(pos, sizes, movable_mask, cw, ch):
    """
    Encourage large movable macros to stay somewhat closer to boundaries,
    opening central whitespace for routing channels.
    """
    mov_idx = torch.where(movable_mask)[0]
    if len(mov_idx) == 0:
        return torch.tensor(0.0, device=pos.device)

    mp = pos[mov_idx]
    ms = sizes[mov_idx]
    half_w = ms[:, 0] / 2
    half_h = ms[:, 1] / 2

    edge_dist = torch.stack(
        [
            mp[:, 0] - half_w,
            cw - (mp[:, 0] + half_w),
            mp[:, 1] - half_h,
            ch - (mp[:, 1] + half_h),
        ],
        dim=1,
    ).min(dim=1).values

    area = ms[:, 0] * ms[:, 1]
    area_norm = area / area.max().clamp(min=1e-6)
    # Smooth quadratic penalty on large edge distances for larger macros.
    return (area_norm * (edge_dist / max(cw, ch)).pow(2)).mean()


# ── Quadratic placement initialization ────────────────────────────────────

def _quadratic_init(nets, n_hard, movable_mask, fixed_pos_np, cw, ch):
    """
    Compute initial positions by solving the quadratic placement problem.

    Builds a weighted Laplacian from net connectivity (clique model) and
    solves Lx = bx, Ly = by where fixed nodes contribute to the RHS.
    Falls back to center-of-canvas if the system is singular.
    """
    movable_idx = np.where(movable_mask)[0]
    fixed_idx = np.where(~movable_mask)[0]
    n_mov = len(movable_idx)

    if n_mov == 0:
        return fixed_pos_np.copy()

    # Map original index -> movable index (-1 if fixed)
    mov_map = np.full(n_hard, -1, dtype=np.int64)
    for new_i, old_i in enumerate(movable_idx):
        mov_map[old_i] = new_i

    # Build Laplacian and RHS
    L = np.zeros((n_mov, n_mov))
    bx = np.zeros(n_mov)
    by = np.zeros(n_mov)

    INF = float("inf")

    for net in nets:
        idx = net["hard_idx"]
        w = net["weight"]
        # Clique model: weight / (num_pins - 1)
        has_fixed = net["fxmin"] != INF
        n_pins = len(idx) + (1 if has_fixed else 0)
        if n_pins < 2:
            continue
        clique_w = w / (n_pins - 1)

        mov_in_net = []
        for k in idx:
            mi = mov_map[k]
            if mi >= 0:
                mov_in_net.append((mi, k))

        # Movable-movable connections
        for a_i in range(len(mov_in_net)):
            ma, _ = mov_in_net[a_i]
            for b_i in range(a_i + 1, len(mov_in_net)):
                mb, _ = mov_in_net[b_i]
                L[ma, mb] -= clique_w
                L[mb, ma] -= clique_w
                L[ma, ma] += clique_w
                L[mb, mb] += clique_w

        # Movable-fixed connections
        for mi, orig_i in mov_in_net:
            for k in idx:
                if mov_map[k] < 0:  # fixed hard macro
                    L[mi, mi] += clique_w
                    bx[mi] += clique_w * fixed_pos_np[k, 0]
                    by[mi] += clique_w * fixed_pos_np[k, 1]

            if has_fixed:
                # Fixed pin bounding box center
                fx = (net["fxmin"] + net["fxmax"]) / 2
                fy = (net["fymin"] + net["fymax"]) / 2
                L[mi, mi] += clique_w
                bx[mi] += clique_w * fx
                by[mi] += clique_w * fy

    # Add small anchor to canvas center for numerical stability
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


# ── Placer class ───────────────────────────────────────────────────────────

class AnalyticalPlacer(BasePlacer):
    """
    Gradient-based macro placer.

    Uses differentiable HPWL + density + overlap + congestion penalties
    optimised with Adam + cosine LR schedule, followed by legalization.
    """

    def __init__(
        self,
        seed: int = 42,
        iters: int = 3_000,
        lr: float = 5.0,
        gamma_start: float = 50.0,
        gamma_end: float = 3.0,
        density_weight: float = 0.005,
        congestion_weight: float = 0.0005,
        boundary_weight: float = 0.02,
        overlap_weight_start: float = 0.01,
        overlap_weight_end: float = 20.0,
        num_candidates: int = 4,
        sa_polish_iters: int = 80_000,
    ):
        self.seed = seed
        self.iters = iters
        self.lr = lr
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.density_weight = density_weight
        self.congestion_weight = congestion_weight
        self.boundary_weight = boundary_weight
        self.overlap_weight_start = overlap_weight_start
        self.overlap_weight_end = overlap_weight_end
        self.num_candidates = num_candidates
        self.sa_polish_iters = sa_polish_iters

    def _optimize_candidate(
        self,
        benchmark: Benchmark,
        plc,
        net_data,
        sizes_t,
        movable_t,
        fixed_mask,
        sizes_np,
        movable_np,
        half_w,
        half_h,
        cw,
        ch,
        n_hard,
        init_pos,
        fixed_pos,
        density_weight,
        congestion_weight,
        boundary_weight,
        halo_area_scale,
        halo_base_scale,
        target_scale,
        seed_offset,
        iters,
    ):
        torch.manual_seed(self.seed + seed_offset)
        np.random.seed(self.seed + seed_offset)

        pos = init_pos.clone()
        movable_idx = torch.where(movable_t)[0]
        if len(movable_idx) > 0:
            noise_x = 0.02 * cw * (torch.rand(len(movable_idx)) - 0.5)
            noise_y = 0.02 * ch * (torch.rand(len(movable_idx)) - 0.5)
            pos[movable_idx, 0] += noise_x
            pos[movable_idx, 1] += noise_y

        inflated_sizes_t = _halo_sizes(
            sizes_t, area_scale=halo_area_scale, base_scale=halo_base_scale
        )
        pos_param = pos.clone().detach().requires_grad_(True)

        lo_x = torch.tensor(half_w, dtype=torch.float32)
        hi_x = torch.tensor([cw], dtype=torch.float32) - lo_x
        lo_y = torch.tensor(half_h, dtype=torch.float32)
        hi_y = torch.tensor([ch], dtype=torch.float32) - lo_y

        optimizer = torch.optim.Adam([pos_param], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iters, eta_min=self.lr * 0.01
        )

        best_snapshot = pos.clone()
        best_loss = float("inf")

        for step in range(iters):
            optimizer.zero_grad()
            frac = step / max(iters - 1, 1)
            gamma = self.gamma_start * (self.gamma_end / self.gamma_start) ** frac
            ov_weight = self.overlap_weight_start * (
                self.overlap_weight_end / max(self.overlap_weight_start, 1e-12)
            ) ** frac

            # Progressive density weight: start at 0.1x, ramp to full
            # (RePlAce-style penalty schedule for better convergence)
            density_ramp = 0.1 + 0.9 * min(1.0, frac * 2.5)
            eff_density_weight = density_weight * density_ramp

            loss = torch.tensor(0.0)
            if net_data is not None:
                loss = loss + _lse_hpwl(pos_param, net_data, gamma=gamma)

            loss = loss + eff_density_weight * _density_penalty(
                pos_param, inflated_sizes_t, movable_t, cw, ch, target_scale=target_scale
            )

            if net_data is not None and congestion_weight > 0:
                loss = loss + congestion_weight * _congestion_penalty(
                    pos_param, net_data, cw, ch
                )

            loss = loss + ov_weight * _overlap_penalty(
                pos_param, inflated_sizes_t, movable_t
            )

            if boundary_weight > 0:
                loss = loss + boundary_weight * _boundary_whitespace_penalty(
                    pos_param, inflated_sizes_t, movable_t, cw, ch
                )

            loss.backward()

            if pos_param.grad is not None:
                pos_param.grad[fixed_mask] = 0.0
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([pos_param], max_norm=max(cw, ch) * 2)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pos_param[:, 0].clamp_(lo_x, hi_x)
                pos_param[:, 1].clamp_(lo_y, hi_y)
                pos_param[fixed_mask] = fixed_pos
                loss_value = float(loss.item())
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_snapshot = pos_param.detach().clone()

        opt_pos = best_snapshot.detach().numpy().astype(np.float64)
        sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
        sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2
        legal_pos = _legalize(
            opt_pos, movable_np, sizes_np, half_w, half_h,
            cw, ch, n_hard, sep_x, sep_y,
        )

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(legal_pos, dtype=torch.float32)
        score = None
        if plc is not None:
            try:
                costs = compute_proxy_cost(full_pos, benchmark, plc)
                score = costs["proxy_cost"]
            except Exception:
                score = None

        return legal_pos, full_pos, score

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        movable_np = benchmark.get_movable_mask()[:n_hard].numpy()

        sizes_t = benchmark.macro_sizes[:n_hard].clone().float()
        movable_t = benchmark.get_movable_mask()[:n_hard]
        fixed_mask = ~movable_t

        # Load nets via PlacementCost
        plc = _load_plc(benchmark.name)
        if plc is not None:
            nets, macro_to_nets = _extract_nets(benchmark, plc)
        else:
            nets = []
            macro_to_nets = [[] for _ in range(n_hard)]

        net_data = _build_net_tensors(nets) if nets else None

        # Initialise: try quadratic placement from connectivity, fall back to initial.plc
        initial_pos = benchmark.macro_positions[:n_hard].clone().float()
        fixed_pos = initial_pos[fixed_mask].clone()

        if nets:
            quad_pos_np = _quadratic_init(
                nets, n_hard, movable_np,
                initial_pos.numpy().astype(np.float64), cw, ch,
            )
            quad_pos = torch.tensor(quad_pos_np, dtype=torch.float32)
        else:
            quad_pos = initial_pos.clone()

        # Per-candidate iteration budget: more iters than before
        per_candidate_iters = max(1500, self.iters // 2)

        candidate_settings = [
            # Quadratic init — wirelength-focused with moderate spreading
            dict(density=self.density_weight * 1.0, congestion=self.congestion_weight * 1.2,
                 boundary=self.boundary_weight * 0.3, halo_area=0.10, halo_base=0.03,
                 target_scale=1.35, iters=per_candidate_iters, init=quad_pos),
            # Quadratic init — strong density/congestion spreading
            dict(density=self.density_weight * 2.5, congestion=self.congestion_weight * 3.0,
                 boundary=self.boundary_weight * 0.5, halo_area=0.16, halo_base=0.05,
                 target_scale=1.15, iters=per_candidate_iters, init=quad_pos),
            # Initial.plc init — balanced
            dict(density=self.density_weight * 1.5, congestion=self.congestion_weight * 2.0,
                 boundary=self.boundary_weight * 0.5, halo_area=0.12, halo_base=0.04,
                 target_scale=1.25, iters=per_candidate_iters, init=initial_pos),
            # Quadratic init — aggressive spreading for congestion
            dict(density=self.density_weight * 4.0, congestion=self.congestion_weight * 4.0,
                 boundary=self.boundary_weight * 0.8, halo_area=0.18, halo_base=0.06,
                 target_scale=1.10, iters=per_candidate_iters, init=quad_pos),
        ][: max(1, self.num_candidates)]

        best_full = None
        best_score = float("inf")
        best_legal_pos = None
        fallback_full = None

        for idx, cfg in enumerate(candidate_settings):
            legal_pos, full_pos, score = self._optimize_candidate(
                benchmark, plc, net_data, sizes_t, movable_t, fixed_mask,
                sizes_np, movable_np, half_w, half_h, cw, ch, n_hard,
                cfg["init"], fixed_pos,
                density_weight=cfg["density"],
                congestion_weight=cfg["congestion"],
                boundary_weight=cfg["boundary"],
                halo_area_scale=cfg["halo_area"],
                halo_base_scale=cfg["halo_base"],
                target_scale=cfg["target_scale"],
                seed_offset=idx * 997,
                iters=cfg["iters"],
            )
            if fallback_full is None:
                fallback_full = full_pos
            if score is not None and score < best_score:
                best_score = score
                best_full = full_pos
                best_legal_pos = legal_pos

        if best_full is None:
            return fallback_full

        # SA polish: short SA refinement starting from the best analytical solution
        if nets and self.sa_polish_iters > 0 and best_legal_pos is not None:
            sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
            sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

            # Build neighbor adjacency for SA moves
            neighbors = [[] for _ in range(n_hard)]
            for net in nets:
                nidx = net["hard_idx"]
                for a in nidx:
                    for b in nidx:
                        if a != b:
                            neighbors[a].append(int(b))

            # Density weight calibration (same as SA placer)
            grid_col = grid_row = 0
            density_w = 0.0
            if plc is not None:
                try:
                    grid_col, grid_row = plc.grid_col, plc.grid_row
                    if grid_col > 0 and grid_row > 0:
                        from macro_place.objective import _set_placement
                        _set_placement(plc, best_full, benchmark)
                        wl_norm = plc.get_cost()
                        raw_hpwl = _compute_total_hpwl(best_legal_pos, nets)
                        if wl_norm > 1e-10 and raw_hpwl > 1e-10:
                            density_w = 0.5 * raw_hpwl / wl_norm
                except Exception:
                    grid_col = grid_row = 0
                    density_w = 0.0

            polished_pos = _sa_refine(
                best_legal_pos, nets, macro_to_nets, neighbors,
                movable_np, sizes_np, half_w, half_h, sep_x, sep_y,
                cw, ch,
                max_iters=self.sa_polish_iters,
                seed=self.seed + 7777,
                t_start_factor=0.06,
                t_end_factor=0.0005,
                density_weight=density_w,
                grid_col=grid_col,
                grid_row=grid_row,
                benchmark=benchmark,
            )

            polished_full = benchmark.macro_positions.clone()
            polished_full[:n_hard] = torch.tensor(polished_pos, dtype=torch.float32)

            # Keep polished result only if it actually improves proxy cost
            if plc is not None:
                try:
                    polished_costs = compute_proxy_cost(polished_full, benchmark, plc)
                    if polished_costs["proxy_cost"] < best_score:
                        best_full = polished_full
                except Exception:
                    pass

        return best_full
