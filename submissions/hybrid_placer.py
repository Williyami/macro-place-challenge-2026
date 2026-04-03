"""
Hybrid Placer — Analytical (gradient) initialisation → SA refinement.

Strategy:
  1. **Analytical phase**: Differentiable log-sum-exp HPWL + Gaussian density
     penalty optimised with Adam (~1000 steps).  This captures global
     connectivity structure quickly.
  2. **Legalization**: Minimum-displacement snap to resolve overlaps.
  3. **SA phase**: Full net-HPWL simulated annealing with density-aware moves
     (reuses the proven SA engine from sa_placer).

The intuition: analytical gets global structure right, SA handles local
optimisation and legalization details.

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

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from submissions.base import BasePlacer

# Reuse net extraction, legalization, and SA refinement from sa_placer
from submissions.sa_placer import (
    _load_plc,
    _extract_nets,
    _legalize,
    _sa_refine,
    _update_soft_macros,
)


# ── Differentiable HPWL (log-sum-exp) ──────────────────────────────────────

def _build_net_tensors(benchmark: Benchmark, plc):
    """
    Build padded tensor representation of nets for differentiable HPWL.

    Returns
    -------
    net_mask   : [N, max_pins] bool  — True for valid pin slots
    net_hard   : [N, max_pins] long  — hard macro index (0 if not hard pin)
    net_is_hard: [N, max_pins] bool  — True if pin belongs to a movable hard macro
    net_ox     : [N, max_pins] float — pin x offset (hard) or absolute x (fixed)
    net_oy     : [N, max_pins] float — pin y offset (hard) or absolute y (fixed)
    net_weight : [N] float
    """
    n_hard = benchmark.num_hard_macros

    hard_name_to_idx = {}
    for tensor_i, plc_i in enumerate(plc.hard_macro_indices):
        name = plc.modules_w_pins[plc_i].get_name()
        hard_name_to_idx[name] = tensor_i

    soft_name_to_pos = {}
    for plc_i in plc.soft_macro_indices:
        mod = plc.modules_w_pins[plc_i]
        soft_name_to_pos[mod.get_name()] = mod.get_pos()

    # Collect raw net data
    raw_nets = []
    for driver_name, sink_names in plc.nets.items():
        if driver_name not in plc.mod_name_to_indices:
            continue
        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        weight = plc.modules_w_pins[driver_plc_idx].get_weight()

        pins = []  # list of (is_hard, macro_idx_or_-1, ox, oy)
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

    net_mask = torch.zeros(N, max_pins, dtype=torch.bool)
    net_hard = torch.zeros(N, max_pins, dtype=torch.long)
    net_is_hard = torch.zeros(N, max_pins, dtype=torch.bool)
    net_ox = torch.zeros(N, max_pins, dtype=torch.float32)
    net_oy = torch.zeros(N, max_pins, dtype=torch.float32)
    net_weight = torch.zeros(N, dtype=torch.float32)

    for i, (w, pins) in enumerate(raw_nets):
        net_weight[i] = w
        for j, (is_hard, idx, ox, oy) in enumerate(pins):
            net_mask[i, j] = True
            net_ox[i, j] = ox
            net_oy[i, j] = oy
            if is_hard:
                net_is_hard[i, j] = True
                net_hard[i, j] = idx

    return net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight


def _lse_hpwl(pos, net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight, gamma=5.0):
    """
    Differentiable HPWL using log-sum-exp approximation.

    pos: [n_hard, 2] — current hard macro positions (requires_grad)
    Returns scalar HPWL estimate.
    """
    # Compute pin positions: for hard pins, pos[idx] + offset; for fixed, offset is absolute
    # pin_x[i,j] = is_hard * (pos[hard_idx, 0] + ox) + (1-is_hard) * ox
    hard_x = pos[net_hard, 0]  # [N, max_pins]
    hard_y = pos[net_hard, 1]
    is_h = net_is_hard.float()

    pin_x = is_h * (hard_x + net_ox) + (1 - is_h) * net_ox
    pin_y = is_h * (hard_y + net_oy) + (1 - is_h) * net_oy

    # Mask out invalid pins with large negative / positive values
    BIG = 1e6
    pin_x_max = pin_x.clone()
    pin_x_max[~net_mask] = -BIG
    pin_x_min = pin_x.clone()
    pin_x_min[~net_mask] = BIG

    pin_y_max = pin_y.clone()
    pin_y_max[~net_mask] = -BIG
    pin_y_min = pin_y.clone()
    pin_y_min[~net_mask] = BIG

    # LSE approximation: max(x) ≈ (1/γ) * log(Σ exp(γ*x))
    # min(x) ≈ -(1/γ) * log(Σ exp(-γ*x))
    inv_gamma = 1.0 / gamma

    x_max = inv_gamma * torch.logsumexp(gamma * pin_x_max, dim=1)
    x_min = -inv_gamma * torch.logsumexp(-gamma * pin_x_min, dim=1)
    y_max = inv_gamma * torch.logsumexp(gamma * pin_y_max, dim=1)
    y_min = -inv_gamma * torch.logsumexp(-gamma * pin_y_min, dim=1)

    hpwl = net_weight * ((x_max - x_min) + (y_max - y_min))
    return hpwl.sum()


def _density_penalty(pos, sizes, canvas_w, canvas_h, grid_n=8):
    """
    Gaussian-smoothed density penalty to spread macros apart.

    Penalises grid cells where total macro area exceeds target density.
    """
    n = pos.shape[0]
    # Grid cell centers
    cell_w = canvas_w / grid_n
    cell_h = canvas_h / grid_n
    cx = torch.linspace(cell_w / 2, canvas_w - cell_w / 2, grid_n)
    cy = torch.linspace(cell_h / 2, canvas_h - cell_h / 2, grid_n)
    grid_cx, grid_cy = torch.meshgrid(cx, cy, indexing='ij')  # [grid_n, grid_n]
    grid_cx = grid_cx.reshape(1, -1)  # [1, G]
    grid_cy = grid_cy.reshape(1, -1)

    # Macro contributions: Gaussian bell centered at macro position
    sigma_x = sizes[:, 0:1] * 0.5 + cell_w * 0.5  # [n, 1]
    sigma_y = sizes[:, 1:2] * 0.5 + cell_h * 0.5

    dx = pos[:, 0:1] - grid_cx  # [n, G]
    dy = pos[:, 1:2] - grid_cy

    weight_x = torch.exp(-0.5 * (dx / sigma_x) ** 2)
    weight_y = torch.exp(-0.5 * (dy / sigma_y) ** 2)
    area = sizes[:, 0:1] * sizes[:, 1:2]  # [n, 1]

    density = (area * weight_x * weight_y).sum(dim=0)  # [G]

    target = (sizes[:, 0] * sizes[:, 1]).sum() / (grid_n * grid_n)
    overflow = torch.relu(density - target * 2.0)
    return overflow.sum()


def _overlap_penalty(pos, sizes):
    """
    Differentiable overlap penalty between all pairs of macros.
    Smooth approximation of overlap area using softplus.
    """
    n = pos.shape[0]
    if n <= 1:
        return torch.tensor(0.0)

    half_w = sizes[:, 0] / 2  # [n]
    half_h = sizes[:, 1] / 2

    # Pairwise distances
    dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)  # [n, n]
    dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

    # Required separations
    sep_x = half_w.unsqueeze(1) + half_w.unsqueeze(0)
    sep_y = half_h.unsqueeze(1) + half_h.unsqueeze(0)

    # Overlap in each dimension (positive means overlap)
    overlap_x = torch.nn.functional.softplus(sep_x - torch.abs(dx), beta=5.0)
    overlap_y = torch.nn.functional.softplus(sep_y - torch.abs(dy), beta=5.0)

    # Overlap area (approximate) — ignore self-overlap
    mask = ~torch.eye(n, dtype=torch.bool)
    overlap_area = (overlap_x * overlap_y * mask.float()).sum() / 2
    return overlap_area


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
    Gradient-based placement using differentiable HPWL + density + overlap penalties.

    Returns hard macro positions as numpy array [n_hard, 2].
    """
    torch.manual_seed(seed)

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].clone().float()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard]
    fixed_mask = ~movable

    # Initialize from current positions
    init_pos = benchmark.macro_positions[:n_hard].clone().float()

    # Clamp to canvas
    init_pos[:, 0] = torch.clamp(init_pos[:, 0], half_w, cw - half_w)
    init_pos[:, 1] = torch.clamp(init_pos[:, 1], half_h, ch - half_h)

    # Build net tensors
    net_data = _build_net_tensors(benchmark, plc)
    if net_data is None:
        return init_pos.numpy().astype(np.float64)

    net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight = net_data

    # Optimizable positions
    pos = init_pos.clone().detach().requires_grad_(True)
    fixed_pos = init_pos.clone()

    optimizer = torch.optim.Adam([pos], lr=lr)

    best_pos = pos.data.clone()
    best_cost = float('inf')

    for step in range(num_steps):
        optimizer.zero_grad()

        # Project fixed macros back
        with torch.no_grad():
            pos.data[fixed_mask] = fixed_pos[fixed_mask]

        # HPWL loss
        hpwl = _lse_hpwl(pos, net_mask, net_hard, net_is_hard, net_ox, net_oy, net_weight, gamma=gamma)

        # Density penalty (ramp up over time)
        density_ramp = min(1.0, step / (num_steps * 0.3))
        density = _density_penalty(pos, sizes, cw, ch) * density_weight * density_ramp

        # Overlap penalty (ramp up aggressively)
        overlap_ramp = min(1.0, step / (num_steps * 0.2))
        overlap = _overlap_penalty(pos, sizes) * overlap_weight * overlap_ramp

        loss = hpwl + density + overlap
        loss.backward()
        optimizer.step()

        # Project to canvas bounds
        with torch.no_grad():
            pos.data[:, 0] = torch.clamp(pos.data[:, 0], half_w, cw - half_w)
            pos.data[:, 1] = torch.clamp(pos.data[:, 1], half_h, ch - half_h)
            pos.data[fixed_mask] = fixed_pos[fixed_mask]

        cost_val = loss.item()
        if cost_val < best_cost:
            best_cost = cost_val
            best_pos = pos.data.clone()

    return best_pos.numpy().astype(np.float64)


# ── HybridPlacer class ─────────────────────────────────────────────────────

class HybridPlacer(BasePlacer):
    """
    Hybrid placer: analytical gradient initialisation → SA refinement.

    Phase 1 (analytical): ~1000 Adam steps with LSE-HPWL + density + overlap.
    Phase 2 (legalize): Minimum-displacement overlap resolution.
    Phase 3 (SA): Full net-HPWL simulated annealing with 120K iterations.
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
        # SA phase
        sa_iters: int = 120_000,
        sa_t_start: float = 0.15,
        sa_t_end: float = 0.001,
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
        self.sa_iters = sa_iters
        self.sa_t_start = sa_t_start
        self.sa_t_end = sa_t_end
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

        # Load PlacementCost for net connectivity
        plc = _load_plc(benchmark.name)

        # ── Phase 1: Analytical placement ──────────────────────────────────
        if plc is not None:
            pos = _analytical_place(
                benchmark, plc,
                num_steps=self.analytical_steps,
                lr=self.analytical_lr,
                gamma=self.gamma,
                density_weight=self.density_weight,
                overlap_weight=self.overlap_weight,
                seed=self.seed,
            )
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

        # ── Phase 3: SA refinement ─────────────────────────────────────────
        if plc is not None:
            nets, macro_to_nets = _extract_nets(benchmark, plc)
        else:
            nets, macro_to_nets = [], [[] for _ in range(n_hard)]

        # Build neighbor adjacency
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
            )

        # Build full placement tensor
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)

        # Optionally update soft macros
        if self.run_fd and plc is not None and benchmark.num_soft_macros > 0:
            soft_pos = _update_soft_macros(pos, benchmark, plc)
            full_pos[n_hard:] = torch.tensor(soft_pos, dtype=torch.float32)

        if self.capture_snapshots:
            if not self.debug_snapshots or not torch.equal(self.debug_snapshots[-1], full_pos):
                self.debug_snapshots.append(full_pos.clone())

        return full_pos
