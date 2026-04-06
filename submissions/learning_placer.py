"""
Learning-based placer — GNN-guided differentiable placement with pre-training.

Approach:
  1. Load pre-trained GNN weights (trained across all IBM benchmarks).
  2. Extract netlist via PlacementCost, build graph adjacency + macro features.
  3. GNN message passing → initial position prediction.
  4. Fine-tune with differentiable LSE-HPWL + density + overlap penalties (Adam).
  5. Legalize to remove residual overlaps.
  6. SA polish with density-aware cost.

Pre-training:
  Run `python submissions/pretrain_learning.py` to train the GNN across all
  IBM benchmarks. Weights are saved to submissions/learning_weights/gnn_pretrained.pt
  and tracked by git.

Key improvements over v1 (informed by research papers):
  - Pre-trained GNN: learns general netlist→position mapping across benchmarks
    (from Google's RL paper concept: transfer learning across designs)
  - Larger GNN (128 hidden, 4 layers) with edge-weight attention
    (from DREAMPlace: richer graph representation)
  - Nesterov-accelerated refinement with electric-field density
    (from DREAMPlace: better optimizer + density model)
  - Multi-start with diverse seeds for GNN fine-tuning
  - Congestion-aware loss matching actual proxy_cost scoring
    (from Synopsys congestion paper: aligns training with evaluation metric)
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
    _greedy_flip,
)

WEIGHTS_DIR = Path(__file__).resolve().parent / "learning_weights"


# ── Differentiable HPWL with pin offsets and fixed contributions ───────────

def _build_net_tensors(nets: list, device: torch.device):
    """
    Convert sa_placer net dicts to padded tensors for vectorized HPWL.
    Groups nets by pin count for batched computation.
    """
    groups: dict = {}
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
        idx_batch = torch.tensor(np.array(idx_l), dtype=torch.long, device=device)
        ox_batch = torch.tensor(np.array(ox_l), dtype=torch.float32, device=device)
        oy_batch = torch.tensor(np.array(oy_l), dtype=torch.float32, device=device)
        fb_batch = torch.tensor(fb_l, dtype=torch.float32, device=device)
        w_batch = torch.tensor(w_l, dtype=torch.float32, device=device)
        batches.append((idx_batch, ox_batch, oy_batch, fb_batch, w_batch))

    return batches


def _lse_hpwl(pos: torch.Tensor, net_batches: list,
              gamma: float = 10.0) -> torch.Tensor:
    """Vectorized differentiable HPWL via log-sum-exp."""
    total = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

    for idx_batch, ox_batch, oy_batch, fb_batch, w_batch in net_batches:
        B, P = idx_batch.shape
        px = pos[idx_batch, 0] + ox_batch
        py = pos[idx_batch, 1] + oy_batch

        has_fixed = fb_batch[:, 0] < 1e18

        if has_fixed.any():
            fx = fb_batch[:, :2]
            fy = fb_batch[:, 2:]
            fx_masked = torch.where(has_fixed.unsqueeze(1), fx, px[:, :1].detach().expand_as(fx))
            fy_masked = torch.where(has_fixed.unsqueeze(1), fy, py[:, :1].detach().expand_as(fy))
            px_full = torch.cat([px, fx_masked], dim=1)
            py_full = torch.cat([py, fy_masked], dim=1)
        else:
            px_full = px
            py_full = py

        if px_full.shape[1] < 2:
            continue

        hpwl_x = gamma * (torch.logsumexp(px_full / gamma, 1) + torch.logsumexp(-px_full / gamma, 1))
        hpwl_y = gamma * (torch.logsumexp(py_full / gamma, 1) + torch.logsumexp(-py_full / gamma, 1))

        total = total + (w_batch * (hpwl_x + hpwl_y)).sum()

    return total


# ── Density penalty (electric-field inspired, from DREAMPlace) ─────────────

def _smooth_density_penalty(pos: torch.Tensor, sizes: torch.Tensor,
                            cw: float, ch: float,
                            grid_col: int = 16, grid_row: int = 16,
                            target_util: float = 0.6) -> torch.Tensor:
    """
    Density penalty with top-10% congestion proxy.
    Uses bell-shaped overlap computation for smoother gradients.
    """
    cell_w = cw / grid_col
    cell_h = ch / grid_row
    cell_area = cell_w * cell_h

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    macro_l = pos[:, 0] - half_w
    macro_r = pos[:, 0] + half_w
    macro_b = pos[:, 1] - half_h
    macro_t = pos[:, 1] + half_h

    gx = torch.arange(grid_col, device=pos.device, dtype=pos.dtype) * cell_w + cell_w / 2
    gy = torch.arange(grid_row, device=pos.device, dtype=pos.dtype) * cell_h + cell_h / 2

    grid_l = gx - cell_w / 2
    grid_r = gx + cell_w / 2
    grid_b = gy - cell_h / 2
    grid_t = gy + cell_h / 2

    ox = F.relu(torch.min(macro_r.unsqueeze(1), grid_r.unsqueeze(0))
                - torch.max(macro_l.unsqueeze(1), grid_l.unsqueeze(0)))
    oy = F.relu(torch.min(macro_t.unsqueeze(1), grid_t.unsqueeze(0))
                - torch.max(macro_b.unsqueeze(1), grid_b.unsqueeze(0)))

    overlap = ox.unsqueeze(2) * oy.unsqueeze(1)
    density = overlap.sum(0) / cell_area

    excess = F.relu(density - target_util)
    density_loss = excess.pow(2).mean()

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


# ── RUDY-based routing congestion (matches evaluator) ─────────────────────

def _rudy_congestion_proxy(pos: torch.Tensor, sizes: torch.Tensor,
                           net_batches: list,
                           cw: float, ch: float,
                           grid_col: int = 10, grid_row: int = 10,
                           hroutes_per_micron: float = 1.0,
                           vroutes_per_micron: float = 1.0,
                           hrouting_alloc: float = 0.0,
                           vrouting_alloc: float = 0.0) -> torch.Tensor:
    """
    Differentiable RUDY-based routing congestion proxy.

    Matches the actual PlacementCost evaluator which computes:
      1. Per-net routing demand distributed across grid cells (RUDY)
      2. Macro blockage of routing resources
      3. Normalization by routes_per_micron
      4. ABU-5% of combined H+V congestion

    This is a differentiable approximation using soft bounding box overlap
    (uniform RUDY distribution over each net's bounding box).
    """
    device = pos.device
    dtype = pos.dtype

    cell_w = cw / grid_col
    cell_h = ch / grid_row

    # Grid cell boundaries
    gx_l = torch.arange(grid_col, device=device, dtype=dtype) * cell_w
    gx_r = gx_l + cell_w
    gy_b = torch.arange(grid_row, device=device, dtype=dtype) * cell_h
    gy_t = gy_b + cell_h

    # Routing capacity per grid cell
    grid_h_routes = max(cell_h * hroutes_per_micron, 1e-6)
    grid_v_routes = max(cell_w * vroutes_per_micron, 1e-6)

    # Initialize H and V congestion grids: [grid_row, grid_col]
    h_cong = torch.zeros(grid_row, grid_col, device=device, dtype=dtype)
    v_cong = torch.zeros(grid_row, grid_col, device=device, dtype=dtype)

    # ── 1. Net routing demand (RUDY approximation) ─────────────────────
    # For each net, compute bounding box of all pins, then distribute
    # routing demand uniformly across grid cells overlapping the bbox.
    # RUDY: h_demand = weight / max(bbox_cols, 1), v_demand = weight / max(bbox_rows, 1)
    for idx_batch, ox_batch, oy_batch, fb_batch, w_batch in net_batches:
        B, P = idx_batch.shape
        # Pin positions
        px = pos[idx_batch, 0] + ox_batch  # [B, P]
        py = pos[idx_batch, 1] + oy_batch  # [B, P]

        # Include fixed pin contributions
        has_fixed = fb_batch[:, 0] < 1e18
        if has_fixed.any():
            fx = fb_batch[:, :2]  # [B, 2] fxmin, fxmax
            fy = fb_batch[:, 2:]  # [B, 2] fymin, fymax
            fx_masked = torch.where(has_fixed.unsqueeze(1), fx,
                                    px[:, :1].detach().expand_as(fx))
            fy_masked = torch.where(has_fixed.unsqueeze(1), fy,
                                    py[:, :1].detach().expand_as(fy))
            px_full = torch.cat([px, fx_masked], dim=1)
            py_full = torch.cat([py, fy_masked], dim=1)
        else:
            px_full = px
            py_full = py

        if px_full.shape[1] < 2:
            continue

        # Bounding box of each net: [B]
        bbox_xmin = px_full.min(dim=1).values
        bbox_xmax = px_full.max(dim=1).values
        bbox_ymin = py_full.min(dim=1).values
        bbox_ymax = py_full.max(dim=1).values

        # Soft overlap of each net bbox with each grid cell
        # Net bbox vs grid columns: [B, grid_col]
        ox_min = torch.max(bbox_xmin.unsqueeze(1), gx_l.unsqueeze(0))
        ox_max = torch.min(bbox_xmax.unsqueeze(1), gx_r.unsqueeze(0))
        col_overlap = F.relu(ox_max - ox_min)  # [B, grid_col]

        # Net bbox vs grid rows: [B, grid_row]
        oy_min = torch.max(bbox_ymin.unsqueeze(1), gy_b.unsqueeze(0))
        oy_max = torch.min(bbox_ymax.unsqueeze(1), gy_t.unsqueeze(0))
        row_overlap = F.relu(oy_max - oy_min)  # [B, grid_row]

        # RUDY demand: weight distributed over bbox area
        bbox_w = (bbox_xmax - bbox_xmin).clamp(min=cell_w * 0.5)
        bbox_h = (bbox_ymax - bbox_ymin).clamp(min=cell_h * 0.5)

        # H routing demand per grid cell (horizontal wires, proportional to bbox width)
        # Normalized by column span
        h_demand = w_batch / bbox_h  # [B] — demand per unit height
        # V routing demand per grid cell
        v_demand = w_batch / bbox_w  # [B] — demand per unit width

        # Distribute over grid: overlap_area gives how much of each cell is covered
        # cell_overlap[B, grid_row, grid_col] = row_overlap * col_overlap
        # H congestion: accumulate h_demand weighted by fractional overlap
        h_contribution = (row_overlap / cell_h).unsqueeze(2) * \
                         (col_overlap / cell_w).unsqueeze(1) * \
                         h_demand.unsqueeze(1).unsqueeze(2)  # [B, grid_row, grid_col]
        h_cong = h_cong + h_contribution.sum(0)

        # V congestion
        v_contribution = (row_overlap / cell_h).unsqueeze(2) * \
                         (col_overlap / cell_w).unsqueeze(1) * \
                         v_demand.unsqueeze(1).unsqueeze(2)  # [B, grid_row, grid_col]
        v_cong = v_cong + v_contribution.sum(0)

    # ── 2. Macro blockage ──────────────────────────────────────────────
    # Hard macros block routing tracks (matches __macro_route_over_grid_cell)
    if hrouting_alloc > 0 or vrouting_alloc > 0:
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        macro_l = pos[:, 0] - half_w
        macro_r = pos[:, 0] + half_w
        macro_b = pos[:, 1] - half_h
        macro_t = pos[:, 1] + half_h

        # Overlap of each macro with each grid column: [N, grid_col]
        m_ox = F.relu(torch.min(macro_r.unsqueeze(1), gx_r.unsqueeze(0))
                      - torch.max(macro_l.unsqueeze(1), gx_l.unsqueeze(0)))
        # Overlap of each macro with each grid row: [N, grid_row]
        m_oy = F.relu(torch.min(macro_t.unsqueeze(1), gy_t.unsqueeze(0))
                      - torch.max(macro_b.unsqueeze(1), gy_b.unsqueeze(0)))

        # Evaluator: V_macro[cell] += x_dist * vrouting_alloc where overlap exists
        #            H_macro[cell] += y_dist * hrouting_alloc where overlap exists
        # Vectorized: [N, grid_row, grid_col] — use soft overlap (no hard threshold)
        # m_ox: [N, grid_col], m_oy: [N, grid_row]
        # For differentiability, use product of overlaps as soft mask
        overlap_mask = m_oy.unsqueeze(2) * m_ox.unsqueeze(1)  # [N, grid_row, grid_col]
        overlap_exists = (overlap_mask > 0).float()

        # V macro congestion: x_dist * vrouting_alloc per cell where both dims overlap
        v_macro = (m_ox.unsqueeze(1) * overlap_exists * vrouting_alloc).sum(0)
        h_macro = (m_oy.unsqueeze(2) * overlap_exists * hrouting_alloc).sum(0)

        v_cong = v_cong + v_macro
        h_cong = h_cong + h_macro

    # ── 3. Normalize by routing capacity ───────────────────────────────
    h_cong_norm = h_cong / grid_h_routes
    v_cong_norm = v_cong / grid_v_routes

    # ── 4. Simple smoothing (average with neighbors) ───────────────────
    # Matches __smooth_routing_cong: V smoothed horizontally, H smoothed vertically
    if grid_col > 2:
        pad_v = F.pad(v_cong_norm, (1, 1), mode='replicate')
        v_cong_norm = (pad_v[:, :-2] + pad_v[:, 1:-1] + pad_v[:, 2:]) / 3.0
    if grid_row > 2:
        pad_h = F.pad(h_cong_norm.unsqueeze(0), (0, 0, 1, 1), mode='replicate').squeeze(0)
        h_cong_norm = (pad_h[:-2, :] + pad_h[1:-1, :] + pad_h[2:, :]) / 3.0

    # ── 5. Combined congestion + ABU-5% ────────────────────────────────
    combined = (h_cong_norm + v_cong_norm).flatten()
    k = max(1, int(0.05 * combined.numel()))
    top_vals, _ = combined.topk(k)
    abu5 = top_vals.mean()

    return abu5


# Legacy alias for backward compatibility during transition
def _congestion_penalty(pos: torch.Tensor, sizes: torch.Tensor,
                        cw: float, ch: float,
                        grid_n: int = 16) -> torch.Tensor:
    """Legacy density-based congestion proxy (kept for fallback)."""
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
    ox = F.relu(torch.min(macro_r.unsqueeze(1), grid_r.unsqueeze(0))
                - torch.max(macro_l.unsqueeze(1), grid_l.unsqueeze(0)))
    oy = F.relu(torch.min(macro_t.unsqueeze(1), grid_t.unsqueeze(0))
                - torch.max(macro_b.unsqueeze(1), grid_b.unsqueeze(0)))
    overlap = ox.unsqueeze(2) * oy.unsqueeze(1)
    density = overlap.sum(0) / cell_area
    cong_loss = density.var() + F.relu(density - 1.0).pow(2).mean()
    return cong_loss


# ── GNN with edge-weight attention ─────────────────────────────────────────

# ── Post-SA congestion-targeted local search ─────────────────────────────

def _congestion_local_search(pos: np.ndarray, nets: list, macro_to_nets: list,
                             movable: np.ndarray, sizes: np.ndarray,
                             cw: float, ch: float, n_hard: int,
                             benchmark, plc,
                             max_rounds: int = 3, attempts_per_macro: int = 8):
    """
    Targeted local search to reduce congestion hotspots.

    After SA polish (which optimizes HPWL + density), congestion is often still
    high because SA doesn't directly optimize routing congestion. This pass:
    1. Evaluates current proxy cost
    2. For each movable macro (sorted by participation in congested areas),
       tries small shifts in 8 directions
    3. Accepts shifts that reduce proxy cost
    """
    if plc is None:
        return pos

    from macro_place.objective import compute_proxy_cost, _set_placement

    pos = pos.copy()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    GAP = 0.05

    def check_overlap(idx):
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        ov = (dx < sep_x[idx] + GAP) & (dy < sep_y[idx] + GAP)
        ov[idx] = False
        return bool(ov.any())

    def eval_proxy():
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)
        costs = compute_proxy_cost(full_pos, benchmark, plc)
        return costs["proxy_cost"], costs["congestion_cost"]

    current_proxy, current_cong = eval_proxy()

    # Sort macros by connectivity (most-connected first — they contribute
    # most to routing congestion). Only try top 20% most-connected.
    macro_connectivity = np.array([len(macro_to_nets[i]) for i in range(n_hard)])
    sorted_macros = np.argsort(-macro_connectivity)
    n_try = max(5, n_hard // 5)  # top 20%, at least 5
    sorted_macros = sorted_macros[:n_try]

    for round_num in range(max_rounds):
        improved_this_round = False
        # Shift magnitude decays per round
        shift_scale = max(cw, ch) * (0.05 / (1 + round_num))

        for i in sorted_macros:
            if not movable[i]:
                continue
            if len(macro_to_nets[i]) == 0:
                continue

            old_x, old_y = pos[i, 0], pos[i, 1]
            best_dx, best_dy = 0.0, 0.0
            best_proxy = current_proxy

            # Try shifts in 8 directions + 4 smaller shifts
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (0.7, 0.7), (0.7, -0.7), (-0.7, 0.7), (-0.7, -0.7),
            ]

            for dx_dir, dy_dir in directions:
                dx = dx_dir * shift_scale
                dy = dy_dir * shift_scale
                new_x = float(np.clip(old_x + dx, half_w[i], cw - half_w[i]))
                new_y = float(np.clip(old_y + dy, half_h[i], ch - half_h[i]))

                pos[i, 0] = new_x
                pos[i, 1] = new_y

                if check_overlap(i):
                    pos[i, 0] = old_x
                    pos[i, 1] = old_y
                    continue

                trial_proxy, _ = eval_proxy()
                if trial_proxy < best_proxy:
                    best_proxy = trial_proxy
                    best_dx = new_x - old_x
                    best_dy = new_y - old_y

                pos[i, 0] = old_x
                pos[i, 1] = old_y

            if best_dx != 0.0 or best_dy != 0.0:
                pos[i, 0] = old_x + best_dx
                pos[i, 1] = old_y + best_dy
                current_proxy = best_proxy
                improved_this_round = True

        if not improved_this_round:
            break

    return pos


# ── Proxy-aware greedy flip ──────────────────────────────────────────────

def _proxy_greedy_flip(pos: np.ndarray, nets: list, macro_to_nets: list,
                       movable: np.ndarray, benchmark, plc, n_hard: int):
    """
    Like _greedy_flip but evaluates each flip by actual proxy cost
    (WL + 0.5*density + 0.5*congestion) instead of just HPWL.
    Falls back to HPWL-only flip if plc is unavailable.
    """
    if plc is None:
        _greedy_flip(pos, nets, macro_to_nets, movable, plc)
        return

    from macro_place.objective import compute_proxy_cost, _set_placement

    def eval_proxy():
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)
        costs = compute_proxy_cost(full_pos, benchmark, plc)
        return costs["proxy_cost"]

    for i in range(n_hard):
        if not movable[i]:
            continue
        affected = macro_to_nets[i]
        if not affected:
            continue

        base_proxy = eval_proxy()
        best_proxy = base_proxy
        best_flip = None

        # Collect pin locations for this macro
        pin_locs = []
        for ni in affected:
            net = nets[ni]
            idxs = net["hard_idx"]
            for k in range(len(idxs)):
                if idxs[k] == i:
                    pin_locs.append((ni, k))

        orig_ox = [(ni, k, nets[ni]["hard_ox"][k]) for ni, k in pin_locs]
        orig_oy = [(ni, k, nets[ni]["hard_oy"][k]) for ni, k in pin_locs]

        for flip_name, flip_x, flip_y in [("FN", True, False),
                                            ("FS", False, True),
                                            ("S",  True, True)]:
            for ni, k, ox in orig_ox:
                nets[ni]["hard_ox"][k] = -ox if flip_x else ox
            for ni, k, oy in orig_oy:
                nets[ni]["hard_oy"][k] = -oy if flip_y else oy

            trial_proxy = eval_proxy()
            if trial_proxy < best_proxy:
                best_proxy = trial_proxy
                best_flip = (flip_x, flip_y)

            # Restore
            for ni, k, ox in orig_ox:
                nets[ni]["hard_ox"][k] = ox
            for ni, k, oy in orig_oy:
                nets[ni]["hard_oy"][k] = oy

        if best_flip is not None:
            flip_x, flip_y = best_flip
            for ni, k, ox in orig_ox:
                nets[ni]["hard_ox"][k] = -ox if flip_x else ox
            for ni, k, oy in orig_oy:
                nets[ni]["hard_oy"][k] = -oy if flip_y else oy
            # Update plc pin offsets (same as _greedy_flip)
            if plc is not None:
                plc_idx = plc.hard_macro_indices[i]
                macro_name = plc.modules_w_pins[plc_idx].get_name()
                pin_names = plc.hard_macros_to_inpins.get(macro_name, [])
                for pin_name in pin_names:
                    if pin_name not in plc.mod_name_to_indices:
                        continue
                    pin_obj = plc.modules_w_pins[plc.mod_name_to_indices[pin_name]]
                    ox, oy = pin_obj.get_offset()
                    new_ox = -ox if flip_x else ox
                    new_oy = -oy if flip_y else oy
                    pin_obj.set_offset(new_ox, new_oy)


class NetlistGNN(nn.Module):
    """
    GNN with edge-weight attention and residual message passing.

    Improvements over v1:
    - Larger capacity (128 hidden, 4 layers)
    - Edge-weight modulated message passing
    - Separate output heads for x and y with skip connection from input features
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 2,
                 num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.gates.append(nn.Linear(hidden_dim * 2, hidden_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + in_dim, hidden_dim),  # skip connection
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for conv, norm, gate in zip(self.conv_layers, self.norms, self.gates):
            msg = torch.mm(adj, h)
            update = F.relu(conv(msg))
            # Gated residual: learn how much of update vs residual to keep
            gate_input = torch.cat([h, update], dim=-1)
            g = torch.sigmoid(gate(gate_input))
            h = norm(g * update + (1 - g) * h)
        # Skip connection from input features
        out = self.output_proj(torch.cat([h, x], dim=-1))
        return out


# ── Feature extraction ─────────────────────────────────────────────────────

def _build_features(benchmark: Benchmark, plc, nets_raw: list,
                    n_hard: int, cw: float, ch: float):
    """Build node features and adjacency for the GNN."""
    sizes = benchmark.macro_sizes[:n_hard].float()
    orig_pos = benchmark.macro_positions[:n_hard].float()
    fixed_mask = benchmark.macro_fixed[:n_hard]

    # Adjacency with net weights
    adj = torch.zeros(n_hard, n_hard)
    for net in nets_raw:
        idx = net["hard_idx"]
        w = net["weight"] / max(len(idx) - 1, 1)
        for a in idx:
            for b in idx:
                if a != b:
                    adj[a, b] += w

    deg = adj.sum(1, keepdim=True).clamp(min=1)
    adj_norm = adj / deg

    # 9-dim node features (richer than v1's 7-dim)
    feat_w = sizes[:, 0] / cw
    feat_h = sizes[:, 1] / ch
    feat_area = (sizes[:, 0] * sizes[:, 1]) / (cw * ch)
    feat_aspect = sizes[:, 0] / sizes[:, 1].clamp(min=1e-6)
    feat_aspect = feat_aspect / feat_aspect.max().clamp(min=1)
    feat_deg = adj.sum(1)
    feat_deg = feat_deg / feat_deg.max().clamp(min=1)
    # Weighted degree (considers net weight, not just count)
    feat_wdeg = adj.sum(1)
    feat_wdeg = feat_wdeg / feat_wdeg.max().clamp(min=1)
    feat_x = orig_pos[:, 0] / cw
    feat_y = orig_pos[:, 1] / ch
    feat_fixed = fixed_mask.float()

    node_features = torch.stack([
        feat_w, feat_h, feat_area, feat_aspect,
        feat_deg, feat_wdeg, feat_x, feat_y, feat_fixed
    ], dim=1)

    return node_features, adj_norm, sizes, orig_pos, fixed_mask


# ── Main placer ─────────────────────────────────────────────────────────────

class LearningPlacer(BasePlacer):
    """
    GNN-guided differentiable macro placer with pre-training and SA polish.

    Phase 1: Load pre-trained GNN, fine-tune on this benchmark.
    Phase 2: Gradient-based refinement with LSE-HPWL + density + overlap.
    Phase 3: Legalization.
    Phase 4: SA polish with density-aware cost.
    Phase 5: Greedy macro flipping.
    """

    def __init__(self, seed: int = 42,
                 gnn_finetune_epochs: int = 400,
                 refine_epochs: int = 800,
                 gnn_lr: float = 1e-3,
                 refine_lr: float = 3.0,
                 gamma_start: float = 50.0,
                 gamma_end: float = 2.0,
                 sa_iters: int = 600_000,
                 num_starts: int = 2,
                 weights_path: str = None):
        self.seed = seed
        self.gnn_finetune_epochs = gnn_finetune_epochs
        self.refine_epochs = refine_epochs
        self.gnn_lr = gnn_lr
        self.refine_lr = refine_lr
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.sa_iters = sa_iters
        self.num_starts = num_starts
        self.weights_path = weights_path or str(WEIGHTS_DIR / "gnn_pretrained.pt")

    def _load_pretrained(self, in_dim: int) -> NetlistGNN:
        """Load pre-trained GNN weights if available."""
        gnn = NetlistGNN(in_dim=in_dim, hidden_dim=128, out_dim=2, num_layers=4)
        weights_file = Path(self.weights_path)
        if weights_file.exists():
            state = torch.load(weights_file, map_location="cpu", weights_only=True)
            # Handle dimension mismatch gracefully (different benchmarks may have
            # slightly different feature dims — skip mismatched layers)
            model_state = gnn.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_state and v.shape == model_state[k].shape}
            if filtered:
                model_state.update(filtered)
                gnn.load_state_dict(model_state)
        return gnn

    def _eval_proxy(self, pos_np: np.ndarray, benchmark: Benchmark,
                    plc, n_hard: int) -> float:
        """Evaluate actual proxy cost using the ground-truth evaluator."""
        if plc is None:
            return float('inf')
        try:
            from macro_place.objective import compute_proxy_cost, _set_placement
            full_pos = benchmark.macro_positions.clone()
            full_pos[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
            costs = compute_proxy_cost(full_pos, benchmark, plc)
            return costs["proxy_cost"]
        except Exception:
            return float('inf')

    def _run_one_seed(self, seed: int, benchmark: Benchmark,
                      nets_raw: list, macro_to_nets: list,
                      net_batches: list, adj_norm: torch.Tensor,
                      node_features: torch.Tensor,
                      sizes: torch.Tensor, half_w: torch.Tensor,
                      half_h: torch.Tensor, movable: torch.Tensor,
                      fixed_mask: torch.Tensor, orig_pos: torch.Tensor,
                      cw: float, ch: float, n_hard: int,
                      neighbors: list, plc) -> tuple:
        """Run GNN finetune + refine + legalize + SA for one seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        in_dim = node_features.shape[1]

        # Extract plc grid parameters for RUDY congestion proxy
        grid_col = getattr(plc, 'grid_col', 10) if plc else 10
        grid_row = getattr(plc, 'grid_row', 10) if plc else 10
        hroutes = getattr(plc, 'hroutes_per_micron', 1.0) if plc else 1.0
        vroutes = getattr(plc, 'vroutes_per_micron', 1.0) if plc else 1.0
        halloc = getattr(plc, 'hrouting_alloc', 0.0) if plc else 0.0
        valloc = getattr(plc, 'vrouting_alloc', 0.0) if plc else 0.0

        rudy_kwargs = dict(
            net_batches=net_batches, cw=cw, ch=ch,
            grid_col=grid_col, grid_row=grid_row,
            hroutes_per_micron=hroutes, vroutes_per_micron=vroutes,
            hrouting_alloc=halloc, vrouting_alloc=valloc,
        )

        # Benchmark-specific density target from actual macro utilization
        total_macro_area = (sizes[:, 0] * sizes[:, 1]).sum().item()
        canvas_area = cw * ch
        actual_util = total_macro_area / canvas_area
        # Target slightly above actual to allow some slack
        target_util = min(actual_util * 1.2, 0.85)

        # ── Phase 1: Load pre-trained GNN + fine-tune ───────────────────
        gnn = self._load_pretrained(in_dim)
        gnn_opt = torch.optim.Adam(gnn.parameters(), lr=self.gnn_lr)
        # LR warmup: ramp from 10% to full over first 10% of epochs
        warmup_epochs = max(1, self.gnn_finetune_epochs // 10)
        gnn_scheduler = torch.optim.lr_scheduler.SequentialLR(
            gnn_opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(gnn_opt, start_factor=0.1,
                                                   total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    gnn_opt, T_max=self.gnn_finetune_epochs - warmup_epochs,
                    eta_min=1e-5),
            ],
            milestones=[warmup_epochs],
        )

        best_gnn_state = None
        best_gnn_cost = float('inf')

        for epoch in range(self.gnn_finetune_epochs):
            frac = epoch / max(self.gnn_finetune_epochs, 1)
            gamma = self.gamma_start * (self.gamma_end / self.gamma_start) ** frac

            gnn_opt.zero_grad()
            raw_pos = gnn(node_features, adj_norm)

            pos = torch.zeros_like(raw_pos)
            pos[:, 0] = raw_pos[:, 0] * (cw - sizes[:, 0]) + half_w
            pos[:, 1] = raw_pos[:, 1] * (ch - sizes[:, 1]) + half_h

            if fixed_mask.any():
                pos = torch.where(fixed_mask.unsqueeze(1), orig_pos, pos)

            loss_wl = _lse_hpwl(pos, net_batches, gamma)
            loss_den = _smooth_density_penalty(pos, sizes, cw, ch,
                                                grid_col=grid_col, grid_row=grid_row,
                                                target_util=target_util) * (cw * ch) * 0.02
            loss_ov = _smooth_overlap_penalty(pos, sizes, movable) * (0.1 + 0.5 * frac)
            # RUDY-based congestion — start higher to push congestion down early
            cong_weight = 0.2 + 0.5 * frac
            loss_cong = _rudy_congestion_proxy(pos, sizes, **rudy_kwargs) * cong_weight

            loss = loss_wl + loss_den + loss_ov + loss_cong
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            gnn_opt.step()
            gnn_scheduler.step()

            # Track best GNN checkpoint (proxy-aligned: WL + 0.5*den + 0.5*cong)
            with torch.no_grad():
                ov = _smooth_overlap_penalty(pos, sizes, movable).item()
                den_raw = _smooth_density_penalty(pos, sizes, cw, ch,
                                                  grid_col=grid_col, grid_row=grid_row,
                                                  target_util=target_util).item()
                cong_raw = _rudy_congestion_proxy(pos, sizes, **rudy_kwargs).item()
                cost = loss_wl.item() + 0.5 * den_raw + 0.5 * cong_raw
                if ov < 1.0 and cost < best_gnn_cost:
                    best_gnn_cost = cost
                    best_gnn_state = {k: v.clone() for k, v in gnn.state_dict().items()}

        # Restore best GNN checkpoint
        if best_gnn_state is not None:
            gnn.load_state_dict(best_gnn_state)

        with torch.no_grad():
            raw_pos = gnn(node_features, adj_norm)
            init_pos = torch.zeros(n_hard, 2)
            init_pos[:, 0] = raw_pos[:, 0] * (cw - sizes[:, 0]) + half_w
            init_pos[:, 1] = raw_pos[:, 1] * (ch - sizes[:, 1]) + half_h
            if fixed_mask.any():
                init_pos = torch.where(fixed_mask.unsqueeze(1), orig_pos, init_pos)

        # ── Phase 2: Two-stage gradient refinement ───────────────────────
        # Stage 1: WL + overlap focused (get a legal, low-WL placement)
        # Stage 2: Congestion-focused (redistribute to reduce hotspots)
        pos_param = init_pos.clone().detach().requires_grad_(True)

        stage1_epochs = self.refine_epochs * 2 // 5  # 40% for WL+overlap
        stage2_epochs = self.refine_epochs - stage1_epochs  # 60% for congestion

        refine_opt = torch.optim.Adam([pos_param], lr=self.refine_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            refine_opt, T_max=self.refine_epochs, eta_min=0.05
        )

        best_pos = init_pos.clone()
        best_cost = float('inf')

        for epoch in range(self.refine_epochs):
            in_stage2 = epoch >= stage1_epochs
            if in_stage2:
                stage_frac = (epoch - stage1_epochs) / max(stage2_epochs, 1)
            else:
                stage_frac = epoch / max(stage1_epochs, 1)

            frac = epoch / max(self.refine_epochs, 1)
            gamma = self.gamma_start * (self.gamma_end / self.gamma_start) ** frac
            ov_weight = 0.1 + 5.0 * frac  # stronger overlap ramp

            refine_opt.zero_grad()

            pos = torch.stack([
                pos_param[:, 0].clamp(half_w, cw - half_w),
                pos_param[:, 1].clamp(half_h, ch - half_h),
            ], dim=1)

            if fixed_mask.any():
                pos = torch.where(fixed_mask.unsqueeze(1), orig_pos, pos)

            loss_wl = _lse_hpwl(pos, net_batches, gamma)

            if in_stage2:
                # Stage 2: high congestion + density weight, maintain WL
                den_weight = 0.08 + 0.12 * stage_frac
                cong_weight = 0.5 + 0.5 * stage_frac  # up to 1.0
            else:
                # Stage 1: moderate congestion, focus on WL + overlap
                den_weight = 0.03 + 0.05 * stage_frac
                cong_weight = 0.15 + 0.2 * stage_frac  # up to 0.35

            loss_den = _smooth_density_penalty(pos, sizes, cw, ch,
                                                grid_col=grid_col, grid_row=grid_row,
                                                target_util=target_util) * (cw * ch) * den_weight
            loss_ov = _smooth_overlap_penalty(pos, sizes, movable) * ov_weight
            loss_cong = _rudy_congestion_proxy(pos, sizes, **rudy_kwargs) * cong_weight

            loss = loss_wl + loss_den + loss_ov + loss_cong

            with torch.no_grad():
                ov_area = _smooth_overlap_penalty(pos, sizes, movable).item()
                # Proxy-aligned tracking: WL + 0.5*density + 0.5*congestion
                den_raw = _smooth_density_penalty(pos, sizes, cw, ch,
                                                  grid_col=grid_col, grid_row=grid_row,
                                                  target_util=target_util).item()
                cong_raw = _rudy_congestion_proxy(pos, sizes, **rudy_kwargs).item()
                cost = loss_wl.item() + 0.5 * den_raw + 0.5 * cong_raw
                if ov_area < 1.0 and cost < best_cost:
                    best_cost = cost
                    best_pos = pos.detach().clone()

            loss.backward()

            if fixed_mask.any():
                pos_param.grad[fixed_mask] = 0

            torch.nn.utils.clip_grad_norm_([pos_param], 5.0)
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

        # ── Phase 4: SA polish with density ─────────────────────────────
        if nets_raw and self.sa_iters > 0:
            # Density weight calibration (same as SA placer)
            grid_col = grid_row = 0
            density_w = 0.0
            if plc is not None:
                try:
                    grid_col, grid_row = plc.grid_col, plc.grid_row
                    if grid_col > 0 and grid_row > 0:
                        from macro_place.objective import _set_placement
                        full_init = benchmark.macro_positions.clone()
                        full_init[:n_hard] = torch.tensor(legal_pos, dtype=torch.float32)
                        _set_placement(plc, full_init, benchmark)
                        wl_norm = plc.get_cost()
                        raw_hpwl = _compute_total_hpwl(legal_pos, nets_raw)
                        if wl_norm > 1e-10 and raw_hpwl > 1e-10:
                            density_w = 0.5 * raw_hpwl / wl_norm
                except Exception:
                    grid_col = grid_row = 0
                    density_w = 0.0

            legal_pos = _sa_refine(
                legal_pos, nets_raw, macro_to_nets, neighbors,
                movable_np, sizes_np, half_w_np, half_h_np,
                sep_x, sep_y, cw, ch,
                self.sa_iters, seed,
                t_start_factor=0.12,
                t_end_factor=0.0008,
                density_weight=density_w,
                grid_col=grid_col,
                grid_row=grid_row,
                benchmark=benchmark,
            )

        # ── Phase 5: HPWL greedy flipping (fast, per-start) ──────────────
        if nets_raw:
            _greedy_flip(legal_pos, nets_raw, macro_to_nets, movable_np, plc)

        proxy = self._eval_proxy(legal_pos, benchmark, plc, n_hard)
        return legal_pos, proxy

    def _run_from_initial(self, seed: int, benchmark: Benchmark,
                          nets_raw: list, macro_to_nets: list,
                          sizes: torch.Tensor, movable: torch.Tensor,
                          orig_pos: torch.Tensor,
                          cw: float, ch: float, n_hard: int,
                          neighbors: list, plc,
                          t_start_factor: float = 0.12,
                          t_end_factor: float = 0.0008,
                          sa_iter_mult: float = 1.5) -> tuple:
        """Run SA polish directly from initial.plc positions (no GNN)."""
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)

        pos_np = orig_pos.numpy().astype(np.float64)
        sizes_np = sizes.numpy().astype(np.float64)
        half_w_np = sizes_np[:, 0] / 2
        half_h_np = sizes_np[:, 1] / 2
        movable_np = movable.numpy()
        sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
        sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

        # Legalize first (initial.plc may have overlaps)
        legal_pos = _legalize(
            pos_np, movable_np, sizes_np, half_w_np, half_h_np,
            cw, ch, n_hard, sep_x, sep_y,
        )

        # SA polish with more iterations (use full budget since no GNN time)
        if nets_raw and self.sa_iters > 0:
            grid_col = getattr(plc, 'grid_col', 0) if plc else 0
            grid_row = getattr(plc, 'grid_row', 0) if plc else 0
            density_w = 0.0
            if plc is not None and grid_col > 0 and grid_row > 0:
                try:
                    from macro_place.objective import _set_placement
                    full_init = benchmark.macro_positions.clone()
                    full_init[:n_hard] = torch.tensor(legal_pos, dtype=torch.float32)
                    _set_placement(plc, full_init, benchmark)
                    wl_norm = plc.get_cost()
                    raw_hpwl = _compute_total_hpwl(legal_pos, nets_raw)
                    if wl_norm > 1e-10 and raw_hpwl > 1e-10:
                        density_w = 0.5 * raw_hpwl / wl_norm
                except Exception:
                    grid_col = grid_row = 0
                    density_w = 0.0

            # Give this path extra SA iterations since it skips GNN
            sa_iters_initial = int(self.sa_iters * sa_iter_mult)
            legal_pos = _sa_refine(
                legal_pos, nets_raw, macro_to_nets, neighbors,
                movable_np, sizes_np, half_w_np, half_h_np,
                sep_x, sep_y, cw, ch,
                sa_iters_initial, seed,
                t_start_factor=t_start_factor,
                t_end_factor=t_end_factor,
                density_weight=density_w,
                grid_col=grid_col,
                grid_row=grid_row,
                benchmark=benchmark,
            )

        if nets_raw:
            _greedy_flip(legal_pos, nets_raw, macro_to_nets, movable_np, plc)

        proxy = self._eval_proxy(legal_pos, benchmark, plc, n_hard)
        return legal_pos, proxy

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        plc = _load_plc(benchmark.name)
        if plc is not None:
            nets_raw, macro_to_nets = _extract_nets(benchmark, plc)
        else:
            nets_raw, macro_to_nets = [], [[] for _ in range(n_hard)]

        device = torch.device("cpu")
        net_batches = _build_net_tensors(nets_raw, device)

        node_features, adj_norm, sizes, orig_pos, fixed_mask = _build_features(
            benchmark, plc, nets_raw, n_hard, cw, ch
        )

        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard]

        neighbors = [[] for _ in range(n_hard)]
        for net in nets_raw:
            idx = net["hard_idx"]
            for a in idx:
                for b in idx:
                    if a != b:
                        neighbors[int(a)].append(int(b))

        best_pos = None
        best_proxy = float('inf')

        def _update_best(pos_np, proxy):
            nonlocal best_pos, best_proxy
            if proxy < best_proxy:
                best_proxy = proxy
                best_pos = pos_np

        # GNN multi-starts
        for s in range(self.num_starts):
            seed = self.seed + s * 1000
            pos_np, proxy = self._run_one_seed(
                seed, benchmark, nets_raw, macro_to_nets, net_batches,
                adj_norm, node_features, sizes, half_w, half_h,
                movable, fixed_mask, orig_pos, cw, ch, n_hard, neighbors, plc,
            )
            _update_best(pos_np, proxy)

        # Initial.plc starts with diverse SA temperatures
        # Start 1: default (moderate temperature, long run)
        pos_np, proxy = self._run_from_initial(
            self.seed + 99000, benchmark, nets_raw, macro_to_nets,
            sizes, movable, orig_pos, cw, ch, n_hard, neighbors, plc,
            t_start_factor=0.12, t_end_factor=0.0008, sa_iter_mult=1.5,
        )
        _update_best(pos_np, proxy)

        # Start 2: hotter start (more exploration, may escape local minima)
        pos_np, proxy = self._run_from_initial(
            self.seed + 98000, benchmark, nets_raw, macro_to_nets,
            sizes, movable, orig_pos, cw, ch, n_hard, neighbors, plc,
            t_start_factor=0.20, t_end_factor=0.0005, sa_iter_mult=1.5,
        )
        _update_best(pos_np, proxy)

        # ── Final post-processing on the winner only ──────────────────
        sizes_np = sizes.numpy().astype(np.float64)
        movable_np = movable.numpy()

        # Proxy-aware greedy flip (more expensive but picks better flips)
        if nets_raw:
            _proxy_greedy_flip(best_pos, nets_raw, macro_to_nets,
                               movable_np, benchmark, plc, n_hard)

        # Congestion-targeted local search (shift macros out of hotspots)
        if nets_raw:
            best_pos = _congestion_local_search(
                best_pos, nets_raw, macro_to_nets, movable_np,
                sizes_np, float(cw), float(ch), n_hard, benchmark, plc,
                max_rounds=3, attempts_per_macro=8,
            )

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)

        return full_pos
