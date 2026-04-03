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
from submissions.sa_placer import _load_plc, _extract_nets, _legalize


# ── Differentiable HPWL with pin offsets and fixed contributions ───────────

def _build_net_tensors(nets: list, device: torch.device):
    """
    Convert sa_placer net dicts to PyTorch tensors for differentiable HPWL.

    Returns lists of tensors per net:
      hard_idx[k]: LongTensor of macro indices
      hard_ox[k], hard_oy[k]: float offsets
      fixed_bbox[k]: [fxmin, fxmax, fymin, fymax] (inf/-inf if no fixed pins)
      weights[k]: float net weight
    """
    net_hard_idx = []
    net_hard_ox = []
    net_hard_oy = []
    net_fixed_bbox = []
    net_w = []

    for net in nets:
        net_hard_idx.append(torch.tensor(net["hard_idx"], dtype=torch.long, device=device))
        net_hard_ox.append(torch.tensor(net["hard_ox"], dtype=torch.float32, device=device))
        net_hard_oy.append(torch.tensor(net["hard_oy"], dtype=torch.float32, device=device))
        net_fixed_bbox.append(torch.tensor(
            [net["fxmin"], net["fxmax"], net["fymin"], net["fymax"]],
            dtype=torch.float32, device=device
        ))
        net_w.append(net["weight"])

    return net_hard_idx, net_hard_ox, net_hard_oy, net_fixed_bbox, torch.tensor(net_w, device=device)


def _lse_hpwl(pos: torch.Tensor, net_hard_idx: list, net_hard_ox: list,
              net_hard_oy: list, net_fixed_bbox: list, net_weights: torch.Tensor,
              gamma: float = 10.0) -> torch.Tensor:
    """
    Differentiable HPWL via log-sum-exp, including pin offsets and fixed pins.

    For each net: pin positions = pos[hard_idx] + offset, plus fixed bbox bounds.
    """
    total = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
    INF = float("inf")

    for k in range(len(net_hard_idx)):
        idx = net_hard_idx[k]
        ox = net_hard_ox[k]
        oy = net_hard_oy[k]
        fb = net_fixed_bbox[k]

        # Hard macro pin positions
        px = pos[idx, 0] + ox  # [num_pins]
        py = pos[idx, 1] + oy

        # Include fixed pin bbox if present
        fxmin, fxmax, fymin, fymax = fb[0], fb[1], fb[2], fb[3]
        has_fixed = fxmin < 1e18

        if has_fixed:
            # Add fixed pin positions as additional "pins"
            # Use the bbox corners as representatives
            fx = torch.tensor([fxmin, fxmax], device=pos.device, dtype=pos.dtype)
            fy = torch.tensor([fymin, fymax], device=pos.device, dtype=pos.dtype)
            px = torch.cat([px, fx])
            py = torch.cat([py, fy])

        if len(px) < 2:
            continue

        # LSE approximation: max ≈ γ·log(Σexp(x/γ)), min ≈ -γ·log(Σexp(-x/γ))
        hpwl_x = gamma * (torch.logsumexp(px / gamma, 0) + torch.logsumexp(-px / gamma, 0))
        hpwl_y = gamma * (torch.logsumexp(py / gamma, 0) + torch.logsumexp(-py / gamma, 0))

        total = total + net_weights[k] * (hpwl_x + hpwl_y)

    return total


# ── Density penalty ─────────────────────────────────────────────────────────

def _smooth_density_penalty(pos: torch.Tensor, sizes: torch.Tensor,
                            cw: float, ch: float,
                            grid_n: int = 16,
                            target_util: float = 0.7) -> torch.Tensor:
    """Penalize grid cells above target utilization."""
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

    excess = F.relu(density - target_util)
    return excess.pow(2).mean()


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
    GNN-guided differentiable macro placer.

    Phase 1: GNN predicts initial positions from netlist structure.
    Phase 2: Gradient-based refinement with LSE-HPWL + density + overlap.
    Phase 3: Legalization to remove residual overlaps.
    """

    def __init__(self, seed: int = 42,
                 gnn_epochs: int = 150,
                 refine_epochs: int = 600,
                 gnn_lr: float = 3e-3,
                 refine_lr: float = 5.0,
                 gamma_start: float = 50.0,
                 gamma_end: float = 5.0):
        self.seed = seed
        self.gnn_epochs = gnn_epochs
        self.refine_epochs = refine_epochs
        self.gnn_lr = gnn_lr
        self.refine_lr = refine_lr
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

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

        # Build net tensors for differentiable HPWL
        device = torch.device("cpu")
        net_hard_idx, net_hard_ox, net_hard_oy, net_fixed_bbox, net_weights = \
            _build_net_tensors(nets_raw, device)

        # ── Build adjacency from nets ────────────────────────────────────
        adj = torch.zeros(n_hard, n_hard)
        for net in nets_raw:
            idx = net["hard_idx"]
            w = 1.0 / max(len(idx) - 1, 1)
            for a in idx:
                for b in idx:
                    if a != b:
                        adj[a, b] += w

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

        # ── Phase 1: GNN training ────────────────────────────────────────
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

            loss_wl = _lse_hpwl(pos, net_hard_idx, net_hard_ox, net_hard_oy,
                                net_fixed_bbox, net_weights, gamma)
            loss_den = _smooth_density_penalty(pos, sizes, cw, ch) * (cw * ch) * 0.01
            loss_ov = _smooth_overlap_penalty(pos, sizes, movable) * 0.1

            loss = loss_wl + loss_den + loss_ov
            loss.backward()
            gnn_opt.step()

        # Get GNN output as initialization
        with torch.no_grad():
            raw_pos = gnn(node_features, adj_norm)
            init_pos = torch.zeros(n_hard, 2)
            init_pos[:, 0] = raw_pos[:, 0] * (cw - sizes[:, 0]) + half_w
            init_pos[:, 1] = raw_pos[:, 1] * (ch - sizes[:, 1]) + half_h
            if fixed_mask.any():
                init_pos = torch.where(fixed_mask.unsqueeze(1), orig_pos, init_pos)

        # ── Phase 2: Direct position refinement ──────────────────────────
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
            ov_weight = 0.01 + 2.0 * frac

            refine_opt.zero_grad()

            pos = torch.stack([
                pos_param[:, 0].clamp(half_w, cw - half_w),
                pos_param[:, 1].clamp(half_h, ch - half_h),
            ], dim=1)

            if fixed_mask.any():
                pos = torch.where(fixed_mask.unsqueeze(1), orig_pos, pos)

            loss_wl = _lse_hpwl(pos, net_hard_idx, net_hard_ox, net_hard_oy,
                                net_fixed_bbox, net_weights, gamma)
            loss_den = _smooth_density_penalty(pos, sizes, cw, ch) * (cw * ch) * 0.02
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

        # ── Phase 3: Legalization ────────────────────────────────────────
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

        # ── Build output ─────────────────────────────────────────────────
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(legal_pos, dtype=torch.float32)

        return full_pos
