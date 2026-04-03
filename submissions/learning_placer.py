"""
Learning-based placer — GNN-guided differentiable placement.

Approach:
  1. Extract netlist via PlacementCost (same as SA placer).
  2. Build graph adjacency from nets, compute macro features.
  3. GNN message passing → initial position prediction.
  4. Refine with differentiable LSE-HPWL + density + overlap penalties (Adam).
  5. Legalize to remove residual overlaps.

All training happens per-instance (no offline weights needed).
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from submissions.base import BasePlacer
from submissions.sa_placer import (
    _load_plc, _extract_nets, _legalize, _sa_refine, _compute_total_hpwl,
)


# ── Differentiable HPWL with pin offsets and fixed contributions ───────────

def _build_net_tensors(nets: list, device: torch.device):
    """
    Convert sa_placer net dicts to padded tensors for vectorized HPWL.

    Groups nets by pin count and pads into batched tensors for fast computation.
    Returns a list of (idx_batch, ox_batch, oy_batch, fixed_x, fixed_y, has_fixed, weights)
    tuples, one per pin-count group.
    """
    # Group nets by number of hard pins
    groups: dict = {}  # pin_count -> (idx_lists, ox_lists, oy_lists, fbboxes, weights)
    for net in nets:
        n_pins = len(net["hard_idx"])
        if n_pins == 0:
            continue
        if n_pins not in groups:
            groups[n_pins] = ([], [], [], [], [])
        g = groups[n_pins]
        g[0].append(net["hard_idx"])
        g[1].append(net["hard_ox"])
        g[2].append(net["hard_oy"])
        g[3].append([net["fxmin"], net["fxmax"], net["fymin"], net["fymax"]])
        g[4].append(net["weight"])

    batches = []
    for n_pins, (idx_l, ox_l, oy_l, fb_l, w_l) in groups.items():
        idx_batch = torch.tensor(np.array(idx_l), dtype=torch.long, device=device)  # [B, P]
        ox_batch = torch.tensor(np.array(ox_l), dtype=torch.float32, device=device)
        oy_batch = torch.tensor(np.array(oy_l), dtype=torch.float32, device=device)
        fb_batch = torch.tensor(fb_l, dtype=torch.float32, device=device)  # [B, 4]
        w_batch = torch.tensor(w_l, dtype=torch.float32, device=device)  # [B]
        batches.append((idx_batch, ox_batch, oy_batch, fb_batch, w_batch))

    return batches


def _lse_hpwl(pos: torch.Tensor, net_batches: list,
              gamma: float = 10.0) -> torch.Tensor:
    """
    Vectorized differentiable HPWL via log-sum-exp.

    Processes all nets of the same pin count in a single batched operation.
    """
    total = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

    for idx_batch, ox_batch, oy_batch, fb_batch, w_batch in net_batches:
        # idx_batch: [B, P], ox/oy: [B, P], fb: [B, 4], w: [B]
        B, P = idx_batch.shape

        # Gather pin positions: [B, P]
        px = pos[idx_batch, 0] + ox_batch
        py = pos[idx_batch, 1] + oy_batch

        # Check which nets have fixed pins (fxmin < 1e18)
        has_fixed = fb_batch[:, 0] < 1e18  # [B]

        if has_fixed.any():
            # For nets with fixed pins, append fixed bbox corners
            # fxmin, fxmax, fymin, fymax -> two extra "pins"
            fx = fb_batch[:, :2]  # [B, 2] - fxmin, fxmax
            fy = fb_batch[:, 2:]  # [B, 2] - fymin, fymax

            # Mask out fixed pins for nets without them (set to 0, won't affect LSE much)
            mask = has_fixed.unsqueeze(1).float()  # [B, 1]
            # Use large negative values for nets without fixed pins so they don't affect logsumexp
            big = torch.tensor(0.0, device=pos.device)
            fx_masked = torch.where(has_fixed.unsqueeze(1), fx, px[:, :1].detach().expand_as(fx))
            fy_masked = torch.where(has_fixed.unsqueeze(1), fy, py[:, :1].detach().expand_as(fy))

            px_full = torch.cat([px, fx_masked], dim=1)  # [B, P+2]
            py_full = torch.cat([py, fy_masked], dim=1)
        else:
            px_full = px
            py_full = py

        if px_full.shape[1] < 2:
            continue

        # LSE: [B]
        hpwl_x = gamma * (torch.logsumexp(px_full / gamma, 1) + torch.logsumexp(-px_full / gamma, 1))
        hpwl_y = gamma * (torch.logsumexp(py_full / gamma, 1) + torch.logsumexp(-py_full / gamma, 1))

        total = total + (w_batch * (hpwl_x + hpwl_y)).sum()

    return total


# ── Density penalty ─────────────────────────────────────────────────────────

def _smooth_density_penalty(pos: torch.Tensor, sizes: torch.Tensor,
                            cw: float, ch: float,
                            grid_n: int = 16,
                            target_util: float = 0.6) -> torch.Tensor:
    """
    Penalize grid cells above target utilization.
    Uses both area density and a congestion proxy (penalizes top cells harder).
    """
    cell_w = cw / grid_n
    cell_h = ch / grid_n
    cell_area = cell_w * cell_h

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    macro_l = pos[:, 0] - half_w
    macro_r = pos[:, 0] + half_w
    macro_b = pos[:, 1] - half_h
    macro_t = pos[:, 1] + half_h

    gx = torch.arange(grid_n, device=pos.device, dtype=pos.dtype) * cell_w + cell_w / 2
    gy = torch.arange(grid_n, device=pos.device, dtype=pos.dtype) * cell_h + cell_h / 2

    grid_l = gx - cell_w / 2
    grid_r = gx + cell_w / 2
    grid_b = gy - cell_h / 2
    grid_t = gy + cell_h / 2

    # Overlap in x: [N, Gx]
    ox = F.relu(torch.min(macro_r.unsqueeze(1), grid_r.unsqueeze(0))
                - torch.max(macro_l.unsqueeze(1), grid_l.unsqueeze(0)))
    oy = F.relu(torch.min(macro_t.unsqueeze(1), grid_t.unsqueeze(0))
                - torch.max(macro_b.unsqueeze(1), grid_b.unsqueeze(0)))

    overlap = ox.unsqueeze(2) * oy.unsqueeze(1)  # [N, Gx, Gy]
    density = overlap.sum(0) / cell_area  # [Gx, Gy]

    # Standard density penalty
    excess = F.relu(density - target_util)
    density_loss = excess.pow(2).mean()

    # Congestion proxy: penalize top 10% cells more aggressively
    flat = density.flatten()
    k = max(1, int(0.1 * flat.numel()))
    top_vals, _ = flat.topk(k)
    congestion_loss = top_vals.pow(2).mean()

    return density_loss + 0.5 * congestion_loss


# ── Overlap penalty ─────────────────────────────────────────────────────────

def _smooth_overlap_penalty(pos: torch.Tensor, sizes: torch.Tensor,
                            movable: torch.Tensor) -> torch.Tensor:
    """Smooth penalty for macro-macro overlaps."""
    n = pos.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=pos.device)

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    dx = (pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)).abs()
    dy = (pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)).abs()

    sep_x = half_w.unsqueeze(1) + half_w.unsqueeze(0)
    sep_y = half_h.unsqueeze(1) + half_h.unsqueeze(0)

    overlap_x = F.relu(sep_x - dx)
    overlap_y = F.relu(sep_y - dy)
    overlap_area = overlap_x * overlap_y

    mask = torch.ones(n, n, device=pos.device, dtype=pos.dtype)
    mask.fill_diagonal_(0)
    mov = movable.float()
    pair_mask = 1.0 - (1.0 - mov.unsqueeze(1)) * (1.0 - mov.unsqueeze(0))
    mask = mask * pair_mask

    return (overlap_area * mask).sum() / 2


# ── GNN ─────────────────────────────────────────────────────────────────────

class NetlistGNN(nn.Module):
    """GNN with residual message passing for netlist-aware macro embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 2,
                 num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.conv_layers, self.norms):
            msg = torch.mm(adj, h)
            h = norm(F.relu(conv(msg)) + h)
        return self.output_proj(h)


# ── Main placer ─────────────────────────────────────────────────────────────

class LearningPlacer(BasePlacer):
    """
    GNN-guided differentiable macro placer with SA polish.

    Phase 1: GNN predicts initial positions from netlist structure.
    Phase 2: Gradient-based refinement with LSE-HPWL + density + overlap.
    Phase 3: Legalization to remove residual overlaps.
    Phase 4: SA polish to optimize actual HPWL (not LSE approximation).
    Multi-start: runs multiple seeds and picks the best.
    """

    def __init__(self, seed: int = 42,
                 gnn_epochs: int = 60,
                 refine_epochs: int = 200,
                 gnn_lr: float = 3e-3,
                 refine_lr: float = 5.0,
                 gamma_start: float = 50.0,
                 gamma_end: float = 2.0,
                 sa_iters: int = 100_000,
                 num_starts: int = 3):
        self.seed = seed
        self.gnn_epochs = gnn_epochs
        self.refine_epochs = refine_epochs
        self.gnn_lr = gnn_lr
        self.refine_lr = refine_lr
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.sa_iters = sa_iters
        self.num_starts = num_starts

    def _run_one_seed(self, seed: int, benchmark: Benchmark,
                      nets_raw: list, macro_to_nets: list,
                      net_batches: list, adj_norm: torch.Tensor,
                      node_features: torch.Tensor,
                      sizes: torch.Tensor, half_w: torch.Tensor,
                      half_h: torch.Tensor, movable: torch.Tensor,
                      fixed_mask: torch.Tensor, orig_pos: torch.Tensor,
                      cw: float, ch: float, n_hard: int,
                      neighbors: list) -> tuple:
        """Run GNN + refine + legalize + SA for one seed. Returns (pos_np, hpwl)."""
        import random

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Phase 1: GNN ────────────────────────────────────────────────
        gnn = NetlistGNN(in_dim=7, hidden_dim=64, out_dim=2, num_layers=3)
        gnn_opt = torch.optim.Adam(gnn.parameters(), lr=self.gnn_lr)

        for epoch in range(self.gnn_epochs):
            frac = epoch / max(self.gnn_epochs, 1)
            gamma = self.gamma_start * (self.gamma_end / self.gamma_start) ** frac

            gnn_opt.zero_grad()
            raw_pos = gnn(node_features, adj_norm)

            pos = torch.zeros_like(raw_pos)
            pos[:, 0] = raw_pos[:, 0] * (cw - sizes[:, 0]) + half_w
            pos[:, 1] = raw_pos[:, 1] * (ch - sizes[:, 1]) + half_h

            if fixed_mask.any():
                pos = torch.where(fixed_mask.unsqueeze(1), orig_pos, pos)

            loss_wl = _lse_hpwl(pos, net_batches, gamma)
            loss_den = _smooth_density_penalty(pos, sizes, cw, ch) * (cw * ch) * 0.015
            loss_ov = _smooth_overlap_penalty(pos, sizes, movable) * 0.1

            loss = loss_wl + loss_den + loss_ov
            loss.backward()
            gnn_opt.step()

        with torch.no_grad():
            raw_pos = gnn(node_features, adj_norm)
            init_pos = torch.zeros(n_hard, 2)
            init_pos[:, 0] = raw_pos[:, 0] * (cw - sizes[:, 0]) + half_w
            init_pos[:, 1] = raw_pos[:, 1] * (ch - sizes[:, 1]) + half_h
            if fixed_mask.any():
                init_pos = torch.where(fixed_mask.unsqueeze(1), orig_pos, init_pos)

        # ── Phase 2: Gradient refinement ────────────────────────────────
        pos_param = init_pos.clone().detach().requires_grad_(True)
        refine_opt = torch.optim.Adam([pos_param], lr=self.refine_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            refine_opt, T_max=self.refine_epochs, eta_min=0.1
        )

        best_pos = init_pos.clone()
        best_cost = float('inf')

        for epoch in range(self.refine_epochs):
            frac = epoch / max(self.refine_epochs, 1)
            gamma = self.gamma_start * (self.gamma_end / self.gamma_start) ** frac
            ov_weight = 0.05 + 3.0 * frac

            refine_opt.zero_grad()

            pos = torch.stack([
                pos_param[:, 0].clamp(half_w, cw - half_w),
                pos_param[:, 1].clamp(half_h, ch - half_h),
            ], dim=1)

            if fixed_mask.any():
                pos = torch.where(fixed_mask.unsqueeze(1), orig_pos, pos)

            loss_wl = _lse_hpwl(pos, net_batches, gamma)
            den_weight = 0.03 + 0.05 * frac  # ramp up density pressure
            loss_den = _smooth_density_penalty(pos, sizes, cw, ch) * (cw * ch) * den_weight
            loss_ov = _smooth_overlap_penalty(pos, sizes, movable) * ov_weight

            loss = loss_wl + loss_den + loss_ov

            with torch.no_grad():
                ov_area = _smooth_overlap_penalty(pos, sizes, movable).item()
                cost = loss_wl.item()
                if ov_area < 1.0 and cost < best_cost:
                    best_cost = cost
                    best_pos = pos.detach().clone()

            loss.backward()

            if fixed_mask.any():
                pos_param.grad[fixed_mask] = 0

            refine_opt.step()
            scheduler.step()

        # ── Phase 3: Legalization ───────────────────────────────────────
        pos_np = best_pos.detach().numpy().astype(np.float64)
        sizes_np = sizes.numpy().astype(np.float64)
        half_w_np = sizes_np[:, 0] / 2
        half_h_np = sizes_np[:, 1] / 2
        movable_np = movable.numpy()

        sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
        sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

        legal_pos = _legalize(
            pos_np, movable_np, sizes_np, half_w_np, half_h_np,
            cw, ch, n_hard, sep_x, sep_y,
        )

        # ── Phase 4: SA polish ──────────────────────────────────────────
        if nets_raw and self.sa_iters > 0:
            legal_pos = _sa_refine(
                legal_pos, nets_raw, macro_to_nets, neighbors,
                movable_np, sizes_np, half_w_np, half_h_np,
                sep_x, sep_y, cw, ch,
                self.sa_iters, seed,
                t_start_factor=0.12,  # full SA exploration from GNN init
                t_end_factor=0.0008,
            )

        hpwl = _compute_total_hpwl(legal_pos, nets_raw) if nets_raw else float('inf')
        return legal_pos, hpwl

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        sizes = benchmark.macro_sizes[:n_hard].float()
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard]
        fixed_mask = benchmark.macro_fixed[:n_hard]
        orig_pos = benchmark.macro_positions[:n_hard].float().clone()

        # ── Extract nets via PlacementCost ────────────────────────────────
        plc = _load_plc(benchmark.name)
        if plc is not None:
            nets_raw, macro_to_nets = _extract_nets(benchmark, plc)
        else:
            nets_raw, macro_to_nets = [], [[] for _ in range(n_hard)]

        # Build batched net tensors for vectorized HPWL
        device = torch.device("cpu")
        net_batches = _build_net_tensors(nets_raw, device)

        # ── Build adjacency and neighbors ────────────────────────────────
        adj = torch.zeros(n_hard, n_hard)
        neighbors = [[] for _ in range(n_hard)]
        for net in nets_raw:
            idx = net["hard_idx"]
            w = 1.0 / max(len(idx) - 1, 1)
            for a in idx:
                for b in idx:
                    if a != b:
                        adj[a, b] += w
                        neighbors[int(a)].append(int(b))

        deg = adj.sum(1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        # ── Node features ────────────────────────────────────────────────
        feat_w = sizes[:, 0] / cw
        feat_h = sizes[:, 1] / ch
        feat_area = (sizes[:, 0] * sizes[:, 1]) / (cw * ch)
        feat_deg = adj.sum(1)
        feat_deg = feat_deg / feat_deg.max().clamp(min=1)
        feat_x = orig_pos[:, 0] / cw
        feat_y = orig_pos[:, 1] / ch
        feat_fixed = fixed_mask.float()

        node_features = torch.stack([
            feat_w, feat_h, feat_area, feat_deg, feat_x, feat_y, feat_fixed
        ], dim=1)

        # ── Multi-start (adapt to benchmark size) ────────────────────────
        # Only 1 start — budget goes to longer SA polish
        num_starts = 1

        best_pos = None
        best_hpwl = float('inf')

        for s in range(num_starts):
            seed = self.seed + s * 1000
            pos_np, hpwl = self._run_one_seed(
                seed, benchmark, nets_raw, macro_to_nets, net_batches,
                adj_norm, node_features, sizes, half_w, half_h,
                movable, fixed_mask, orig_pos, cw, ch, n_hard, neighbors,
            )
            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_pos = pos_np

        # ── Build output ─────────────────────────────────────────────────
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)

        return full_pos
