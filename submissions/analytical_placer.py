"""
Analytical (gradient-based) placer — differentiable placement with Adam.

Approach:
  - Differentiable HPWL via log-sum-exp (LSE) approximation
  - Differentiable Gaussian density penalty (smooth spreading)
  - Differentiable overlap penalty (increasing weight schedule)
  - Differentiable congestion proxy (net bounding-box routing demand)
  - Adam optimizer with cosine-annealed LR and projection to canvas bounds
  - Legalization post-processing (minimum displacement, reused from sa_placer)

Usage:
    PLACER_METHOD=analytical uv run evaluate submissions/placer.py --all
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from submissions.base import BasePlacer
from submissions.sa_placer import _load_plc, _extract_nets, _legalize


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

def _density_penalty(pos, sizes, movable_mask, cw, ch, grid_size=32):
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

    excess = torch.clamp(density_per_cell - target_density * 1.5, min=0)
    return (excess ** 2).sum() * cell_w * cell_h


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
        overlap_weight_start: float = 0.01,
        overlap_weight_end: float = 20.0,
    ):
        self.seed = seed
        self.iters = iters
        self.lr = lr
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.density_weight = density_weight
        self.congestion_weight = congestion_weight
        self.overlap_weight_start = overlap_weight_start
        self.overlap_weight_end = overlap_weight_end

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
            nets, _ = _extract_nets(benchmark, plc)
        else:
            nets = []

        net_data = _build_net_tensors(nets) if nets else None

        # Initialise positions from initial.plc
        pos = benchmark.macro_positions[:n_hard].clone().float()
        fixed_pos = pos[fixed_mask].clone()

        pos_param = pos.clone().detach().requires_grad_(True)

        # Canvas bounds for projection
        lo_x = torch.tensor(half_w, dtype=torch.float32)
        hi_x = torch.tensor([cw], dtype=torch.float32) - lo_x
        lo_y = torch.tensor(half_h, dtype=torch.float32)
        hi_y = torch.tensor([ch], dtype=torch.float32) - lo_y

        optimizer = torch.optim.Adam([pos_param], lr=self.lr)
        # Cosine annealing LR schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=self.lr * 0.01
        )

        for step in range(self.iters):
            optimizer.zero_grad()

            frac = step / max(self.iters - 1, 1)

            # Anneal gamma (large → small = tighter HPWL approximation)
            gamma = self.gamma_start * (self.gamma_end / self.gamma_start) ** frac

            # Anneal overlap weight (increases over time to resolve overlaps)
            ov_weight = self.overlap_weight_start * (
                self.overlap_weight_end / max(self.overlap_weight_start, 1e-12)
            ) ** frac

            # Compute loss
            loss = torch.tensor(0.0)

            if net_data is not None:
                loss = loss + _lse_hpwl(pos_param, net_data, gamma=gamma)

            loss = loss + self.density_weight * _density_penalty(
                pos_param, sizes_t, movable_t, cw, ch
            )

            if net_data is not None and self.congestion_weight > 0:
                loss = loss + self.congestion_weight * _congestion_penalty(
                    pos_param, net_data, cw, ch
                )

            loss = loss + ov_weight * _overlap_penalty(
                pos_param, sizes_t, movable_t
            )

            loss.backward()

            # Zero out gradients for fixed macros
            if pos_param.grad is not None:
                pos_param.grad[fixed_mask] = 0.0

            optimizer.step()
            scheduler.step()

            # Project to canvas bounds
            with torch.no_grad():
                pos_param[:, 0].clamp_(lo_x, hi_x)
                pos_param[:, 1].clamp_(lo_y, hi_y)
                pos_param[fixed_mask] = fixed_pos

        # ── Legalization ───────────────────────────────────────────────────
        opt_pos = pos_param.detach().numpy().astype(np.float64)

        sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
        sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

        legal_pos = _legalize(
            opt_pos, movable_np, sizes_np, half_w, half_h,
            cw, ch, n_hard, sep_x, sep_y,
        )

        # Build full placement tensor
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(legal_pos, dtype=torch.float32)

        return full_pos
