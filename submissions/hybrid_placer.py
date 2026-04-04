"""
Hybrid Placer v2 — Analytical (gradient) initialisation → SA refinement.

Improvements over v1 (avg 1.6972):
  - SA reheating: when stagnating for 10K iters, reheat temperature to escape
    local minima.
  - More SA iterations: 150K (up from 120K).
  - Higher density weight in analytical phase to better model congestion.
  - Higher-res density grid (16x16) in analytical phase.

Strategy:
  1. Analytical phase: Differentiable LSE-HPWL + density + overlap with Adam.
  2. Legalization: Minimum-displacement snap to resolve overlaps.
  3. SA phase: Full net-HPWL simulated annealing with reheating.

Usage:
    uv run evaluate submissions/hybrid_placer.py
    uv run evaluate submissions/hybrid_placer.py --all
"""

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

from submissions.sa_placer import (
    _load_plc,
    _extract_nets,
    _legalize,
    _sa_refine,
    _update_soft_macros,
)


# ── Differentiable HPWL (log-sum-exp) ──────────────────────────────────────

def _build_net_tensors(plc):
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
    cell_w = canvas_w / grid_n
    cell_h = canvas_h / grid_n
    cx = torch.linspace(cell_w / 2, canvas_w - cell_w / 2, grid_n)
    cy = torch.linspace(cell_h / 2, canvas_h - cell_h / 2, grid_n)
    grid_cx, grid_cy = torch.meshgrid(cx, cy, indexing='ij')
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
        return torch.tensor(0.0)

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
    dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

    sep_x = half_w.unsqueeze(1) + half_w.unsqueeze(0)
    sep_y = half_h.unsqueeze(1) + half_h.unsqueeze(0)

    overlap_x = torch.nn.functional.softplus(sep_x - torch.abs(dx), beta=5.0)
    overlap_y = torch.nn.functional.softplus(sep_y - torch.abs(dy), beta=5.0)

    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
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

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].clone().float()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard]
    fixed_mask = ~movable

    init_pos = benchmark.macro_positions[:n_hard].clone().float()
    init_pos[:, 0] = torch.clamp(init_pos[:, 0], half_w, cw - half_w)
    init_pos[:, 1] = torch.clamp(init_pos[:, 1], half_h, ch - half_h)

    net_data = _build_net_tensors(plc)
    if net_data is None:
        return init_pos.numpy().astype(np.float64)

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

    return best_pos.numpy().astype(np.float64)


# ── HybridPlacer class ─────────────────────────────────────────────────────

class HybridPlacer(BasePlacer):
    """
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
        # SA phase
        sa_iters: int = 150_000,
        sa_t_start: float = 0.15,
        sa_t_end: float = 0.001,
        reheat_threshold: int = 10_000,
        reheat_factor: float = 3.0,
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
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor
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
