#!/usr/bin/env python3
"""
Pre-train the learning placer GNN across all available benchmarks.

Trains the GNN to predict good macro positions by optimizing differentiable
LSE-HPWL + density + overlap + congestion across IBM and non-IBM benchmarks.
Weights are saved to submissions/learning_weights/gnn_pretrained.pt.

Improvements over v1 (informed by research papers):
  - Non-IBM benchmarks (ariane, nvdla, mempool) for better generalization
  - Data augmentation via flipping/rotation (8x effective dataset, from AutoDMP)
  - Congestion-aware loss matching actual proxy_cost scoring
  - Curriculum learning: train on smaller designs first (from HRLP)
  - Per-macro local HPWL dense loss for better gradient signal (from HRLP)
  - Prioritized training: more epochs on harder benchmarks (from MCTS+RL paper)

Usage:
    cd macro-place-challenge-2026
    uv run python submissions/pretrain_learning.py [--epochs 200] [--rounds 5]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir, load_benchmark
from submissions.learning_placer import (
    NetlistGNN, _build_net_tensors, _build_features,
    _lse_hpwl, _smooth_density_penalty, _smooth_overlap_penalty,
)
from submissions.sa_placer import _load_plc, _extract_nets

WEIGHTS_DIR = Path(__file__).resolve().parent / "learning_weights"

IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
    "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
    "ibm16", "ibm17", "ibm18",
]

# Non-IBM benchmarks available via Flows/ directories
NON_IBM_BENCHMARKS = [
    # (name, netlist_path_relative, plc_path_relative)
    ("ariane133_ng45",
     "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping/netlist.pb.txt",
     "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping/initial.plc"),
    ("ariane136_ng45",
     "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping/netlist.pb.txt",
     "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping/initial.plc"),
    ("nvdla_ng45",
     "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping/netlist.pb.txt",
     "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping/initial.plc"),
    ("mempool_tile_ng45",
     "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping/netlist.pb.txt",
     "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping/initial.plc"),
    ("ariane136_asap7",
     "external/MacroPlacement/Flows/ASAP7/ariane136/netlist/output_CT_Grouping/netlist.pb.txt",
     "external/MacroPlacement/Flows/ASAP7/ariane136/netlist/output_CT_Grouping/initial.plc"),
    ("nvdla_asap7",
     "external/MacroPlacement/Flows/ASAP7/nvdla/netlist/output_CT_Grouping/netlist.pb.txt",
     "external/MacroPlacement/Flows/ASAP7/nvdla/netlist/output_CT_Grouping/initial.plc"),
    ("mempool_tile_asap7",
     "external/MacroPlacement/Flows/ASAP7/mempool_tile/netlist/output_CT_Grouping/netlist.pb.txt",
     "external/MacroPlacement/Flows/ASAP7/mempool_tile/netlist/output_CT_Grouping/initial.plc"),
]


def load_ibm_benchmark_data(name: str):
    """Load an IBM benchmark and extract all data needed for training."""
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    benchmark, _ = load_benchmark_from_dir(str(root))
    return _extract_training_data(name, benchmark)


def load_non_ibm_benchmark_data(name: str, netlist_path: str, plc_path: str):
    """Load a non-IBM benchmark and extract training data."""
    if not Path(netlist_path).exists():
        return None
    plc_file = plc_path if Path(plc_path).exists() else None
    benchmark, _ = load_benchmark(netlist_path, plc_file)
    benchmark.name = name
    return _extract_training_data(name, benchmark)


def _extract_training_data(name: str, benchmark: Benchmark):
    """Extract features, adjacency, and net tensors for training."""
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    plc = _load_plc(name)
    if plc is None:
        return None

    nets_raw, macro_to_nets = _extract_nets(benchmark, plc)
    net_batches = _build_net_tensors(nets_raw, torch.device("cpu"))
    node_features, adj_norm, sizes, orig_pos, fixed_mask = _build_features(
        benchmark, plc, nets_raw, n_hard, cw, ch
    )

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard]

    # Build per-macro net index lists for local HPWL
    macro_net_indices = [[] for _ in range(n_hard)]
    for net_idx, net in enumerate(nets_raw):
        for mi in net["hard_idx"]:
            macro_net_indices[int(mi)].append(net_idx)

    return {
        "name": name,
        "n_hard": n_hard,
        "cw": cw,
        "ch": ch,
        "net_batches": net_batches,
        "nets_raw": nets_raw,
        "macro_net_indices": macro_net_indices,
        "node_features": node_features,
        "adj_norm": adj_norm,
        "sizes": sizes,
        "half_w": half_w,
        "half_h": half_h,
        "movable": movable,
        "fixed_mask": fixed_mask,
        "orig_pos": orig_pos,
        "plc": plc,
    }


# ── Data augmentation ──────────────────────────────────────────────────────

def augment_data(data: dict, transform: int) -> dict:
    """
    Apply geometric transform to training data for augmentation.

    Transforms (applied to positions and features):
      0: identity
      1: flip X
      2: flip Y
      3: flip X+Y
      4: rotate 90°
      5: rotate 90° + flip X
      6: rotate 90° + flip Y
      7: rotate 90° + flip X+Y

    From AutoDMP: placement problem has rotational/translational symmetry,
    so augmenting with flips and rotations multiplies effective dataset.
    """
    if transform == 0:
        return data

    cw, ch = data["cw"], data["ch"]
    node_features = data["node_features"].clone()
    orig_pos = data["orig_pos"].clone()
    sizes = data["sizes"].clone()

    # Feature indices: feat_x=6, feat_y=7
    feat_x = node_features[:, 6].clone()
    feat_y = node_features[:, 7].clone()
    pos_x = orig_pos[:, 0].clone()
    pos_y = orig_pos[:, 1].clone()
    size_w = sizes[:, 0].clone()
    size_h = sizes[:, 1].clone()

    rotate = transform >= 4
    flip_x = (transform % 4) in (1, 3)
    flip_y = (transform % 4) in (2, 3)

    if rotate:
        # Swap X/Y axes
        feat_x, feat_y = feat_y.clone(), feat_x.clone()
        pos_x, pos_y = pos_y.clone(), pos_x.clone()
        size_w, size_h = size_h.clone(), size_w.clone()
        # Also update width/height features (indices 0, 1)
        node_features[:, 0], node_features[:, 1] = node_features[:, 1].clone(), node_features[:, 0].clone()
        # Aspect ratio inverts
        node_features[:, 3] = 1.0 / node_features[:, 3].clamp(min=1e-6)
        node_features[:, 3] = node_features[:, 3] / node_features[:, 3].max().clamp(min=1)

    if flip_x:
        feat_x = 1.0 - feat_x
        pos_x = (cw if not rotate else ch) - pos_x

    if flip_y:
        feat_y = 1.0 - feat_y
        pos_y = (ch if not rotate else cw) - pos_y

    node_features[:, 6] = feat_x
    node_features[:, 7] = feat_y
    orig_pos_new = torch.stack([pos_x, pos_y], dim=1)
    sizes_new = torch.stack([size_w, size_h], dim=1)

    new_cw = ch if rotate else cw
    new_ch = cw if rotate else ch

    aug = dict(data)
    aug["node_features"] = node_features
    aug["orig_pos"] = orig_pos_new
    aug["sizes"] = sizes_new
    aug["half_w"] = sizes_new[:, 0] / 2
    aug["half_h"] = sizes_new[:, 1] / 2
    aug["cw"] = new_cw
    aug["ch"] = new_ch
    # Net batches and adjacency don't change with spatial transforms
    return aug


# ── Congestion loss ────────────────────────────────────────────────────────

def _congestion_loss(pos: torch.Tensor, sizes: torch.Tensor,
                     cw: float, ch: float, grid_n: int = 16) -> torch.Tensor:
    """
    Differentiable congestion proxy inspired by routing demand estimation.

    From the Synopsys paper and DREAMPlace: estimates horizontal and vertical
    routing congestion by computing macro pin density in grid cells, weighted
    by adjacency (net connections cross grid boundaries).

    Approximated as the variance of macro coverage across grid cells —
    high variance means some cells are very congested while others are empty.
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

    ox = F.relu(torch.min(macro_r.unsqueeze(1), grid_r.unsqueeze(0))
                - torch.max(macro_l.unsqueeze(1), grid_l.unsqueeze(0)))
    oy = F.relu(torch.min(macro_t.unsqueeze(1), grid_t.unsqueeze(0))
                - torch.max(macro_b.unsqueeze(1), grid_b.unsqueeze(0)))

    overlap = ox.unsqueeze(2) * oy.unsqueeze(1)
    density = overlap.sum(0) / cell_area

    # Congestion = variance of density (spread should be even)
    cong_loss = density.var() + F.relu(density - 1.0).pow(2).mean()
    return cong_loss


# ── Per-macro local HPWL ───────────────────────────────────────────────────

def _local_hpwl_loss(pos: torch.Tensor, nets_raw: list,
                     macro_net_indices: list, n_hard: int,
                     movable: torch.Tensor) -> torch.Tensor:
    """
    Per-macro local HPWL loss for denser gradient signal.

    From HRLP paper: instead of only global HPWL, compute the HPWL
    contribution of each macro's local nets. This gives each node
    better gradient information about its own position quality.
    """
    local_costs = torch.zeros(n_hard, device=pos.device, dtype=pos.dtype)

    for mi in range(n_hard):
        if not movable[mi]:
            continue
        net_indices = macro_net_indices[mi]
        if not net_indices:
            continue

        cost = torch.tensor(0.0, device=pos.device)
        for ni in net_indices:
            net = nets_raw[ni]
            idx = net["hard_idx"]
            if len(idx) < 2:
                continue
            px = pos[idx, 0]
            py = pos[idx, 1]
            cost = cost + (px.max() - px.min() + py.max() - py.min()) * net["weight"]

        local_costs[mi] = cost

    return local_costs.mean()


# ── Training step ──────────────────────────────────────────────────────────

def train_step(gnn, optimizer, data, gamma, frac, use_congestion=True,
               use_local_hpwl=True, local_hpwl_weight=0.0):
    """One training step on a single benchmark (or augmented version)."""
    optimizer.zero_grad()

    raw_pos = gnn(data["node_features"], data["adj_norm"])

    cw, ch = data["cw"], data["ch"]
    sizes = data["sizes"]
    half_w = data["half_w"]
    half_h = data["half_h"]

    pos = torch.zeros_like(raw_pos)
    pos[:, 0] = raw_pos[:, 0] * (cw - sizes[:, 0]) + half_w
    pos[:, 1] = raw_pos[:, 1] * (ch - sizes[:, 1]) + half_h

    if data["fixed_mask"].any():
        pos = torch.where(data["fixed_mask"].unsqueeze(1), data["orig_pos"], pos)

    # Global HPWL loss
    loss_wl = _lse_hpwl(pos, data["net_batches"], gamma)

    # Density loss
    loss_den = _smooth_density_penalty(
        pos, sizes, cw, ch, grid_n=16, target_util=0.5
    ) * (cw * ch) * (0.01 + 0.04 * frac)

    # Overlap loss
    loss_ov = _smooth_overlap_penalty(
        pos, sizes, data["movable"]
    ) * (0.1 + 1.0 * frac)

    loss = loss_wl + loss_den + loss_ov

    # Congestion loss (matches actual scoring: proxy = WL + 0.5*density + 0.5*congestion)
    if use_congestion:
        loss_cong = _congestion_loss(pos, sizes, cw, ch, grid_n=16) * (cw * ch) * (0.005 + 0.02 * frac)
        loss = loss + loss_cong

    # Per-macro local HPWL loss (dense gradient signal from HRLP paper)
    if use_local_hpwl and local_hpwl_weight > 0 and data.get("nets_raw"):
        loss_local = _local_hpwl_loss(
            pos, data["nets_raw"], data["macro_net_indices"],
            data["n_hard"], data["movable"]
        ) * local_hpwl_weight
        loss = loss + loss_local

    loss.backward()
    torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
    optimizer.step()

    return loss.item(), loss_wl.item()


def main():
    parser = argparse.ArgumentParser(description="Pre-train learning placer GNN (v2)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs per benchmark per round")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of round-robin passes over all benchmarks")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-start", type=float, default=50.0)
    parser.add_argument("--gamma-end", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--no-congestion", action="store_true",
                        help="Disable congestion loss")
    parser.add_argument("--no-local-hpwl", action="store_true",
                        help="Disable per-macro local HPWL loss")
    parser.add_argument("--no-non-ibm", action="store_true",
                        help="Only train on IBM benchmarks")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning (sort by difficulty)")
    parser.add_argument("--no-prioritized", action="store_true",
                        help="Disable prioritized training")
    parser.add_argument("--augment-transforms", type=int, default=4,
                        help="Number of augmentation transforms (1-8, default 4: identity+3 flips)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load all benchmarks ────────────────────────────────────────────
    print("Loading benchmarks...")
    all_data = []

    # IBM benchmarks
    for name in IBM_BENCHMARKS:
        print(f"  {name}...", end=" ", flush=True)
        try:
            data = load_ibm_benchmark_data(name)
            if data is not None:
                all_data.append(data)
                print(f"OK ({data['n_hard']} macros, {len(data['net_batches'])} net groups)")
            else:
                print("SKIP (no plc)")
        except Exception as e:
            print(f"SKIP ({e})")

    # Non-IBM benchmarks
    if not args.no_non_ibm:
        for name, netlist_path, plc_path in NON_IBM_BENCHMARKS:
            print(f"  {name}...", end=" ", flush=True)
            try:
                data = load_non_ibm_benchmark_data(name, netlist_path, plc_path)
                if data is not None:
                    all_data.append(data)
                    print(f"OK ({data['n_hard']} macros, {len(data['net_batches'])} net groups)")
                else:
                    print("SKIP (no plc)")
            except Exception as e:
                print(f"SKIP ({e})")

    if not all_data:
        print("ERROR: No benchmarks loaded!")
        return

    # ── Curriculum learning: sort by number of macros (easy → hard) ────
    if not args.no_curriculum:
        all_data.sort(key=lambda d: d["n_hard"])
        print(f"\nCurriculum order (easy→hard): "
              + ", ".join(f"{d['name']}({d['n_hard']})" for d in all_data))

    # All benchmarks should have in_dim=9
    in_dim = all_data[0]["node_features"].shape[1]
    print(f"\nFeature dim: {in_dim}, Benchmarks: {len(all_data)}")

    gnn = NetlistGNN(in_dim=in_dim, hidden_dim=128, out_dim=2, num_layers=4)

    # Determine augmentation transforms to use
    n_transforms = 1 if args.no_augment else min(args.augment_transforms, 8)
    print(f"Augmentation transforms: {n_transforms}")

    # Estimate total steps for scheduler
    effective_benchmarks = len(all_data) * n_transforms
    total_steps = args.rounds * args.epochs * effective_benchmarks

    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    # ── Prioritized training: track per-benchmark loss for adaptive epochs ─
    bench_losses = {d["name"]: [] for d in all_data}

    step = 0
    t0 = time.time()

    use_congestion = not args.no_congestion
    use_local_hpwl = not args.no_local_hpwl

    print(f"\nTraining: {args.rounds} rounds × {args.epochs} epochs × "
          f"{effective_benchmarks} effective benchmarks")
    print(f"  Congestion loss: {'ON' if use_congestion else 'OFF'}")
    print(f"  Local HPWL loss: {'ON' if use_local_hpwl else 'OFF'}")
    print(f"  Curriculum: {'ON' if not args.no_curriculum else 'OFF'}")
    print(f"  Prioritized: {'ON' if not args.no_prioritized else 'OFF'}")
    print(f"Total steps: ~{total_steps}\n")

    for round_idx in range(args.rounds):
        round_frac = round_idx / args.rounds

        # ── Determine benchmark order ──────────────────────────────────
        if args.no_curriculum:
            # Random shuffle each round
            rng = np.random.default_rng(args.seed + round_idx)
            order = rng.permutation(len(all_data)).tolist()
        else:
            # Curriculum: in early rounds follow easy→hard order,
            # in later rounds shuffle more
            order = list(range(len(all_data)))
            if round_idx > 0:
                # Partial shuffle: swap some elements based on round progress
                rng = np.random.default_rng(args.seed + round_idx)
                n_swaps = int(len(order) * min(round_frac, 0.8))
                for _ in range(n_swaps):
                    i, j = rng.integers(0, len(order), size=2)
                    order[i], order[j] = order[j], order[i]

        round_loss = 0.0
        round_wl = 0.0
        round_steps = 0

        for bench_idx in order:
            data = all_data[bench_idx]

            # ── Prioritized training: compute epochs for this benchmark ─
            if not args.no_prioritized and round_idx > 0 and bench_losses[data["name"]]:
                # More epochs for benchmarks with higher recent loss
                recent_loss = np.mean(bench_losses[data["name"]][-3:])
                all_recent = [np.mean(bench_losses[d["name"]][-3:])
                              for d in all_data if bench_losses[d["name"]]]
                if all_recent:
                    mean_loss = np.mean(all_recent)
                    # Scale epochs: 0.5x-2x based on relative loss
                    ratio = recent_loss / max(mean_loss, 1e-10)
                    ratio = np.clip(ratio, 0.5, 2.0)
                    bench_epochs = int(args.epochs * ratio)
                else:
                    bench_epochs = args.epochs
            else:
                bench_epochs = args.epochs

            # ── Local HPWL weight: ramp up over training ───────────────
            local_hpwl_weight = 0.1 * (1.0 - round_frac)  # stronger early, fade out

            bench_loss_sum = 0.0

            for t in range(n_transforms):
                aug_data = augment_data(data, t)
                for epoch in range(bench_epochs):
                    frac = step / max(total_steps, 1)
                    gamma = args.gamma_start * (args.gamma_end / args.gamma_start) ** frac

                    loss, wl = train_step(
                        gnn, optimizer, aug_data, gamma, frac,
                        use_congestion=use_congestion,
                        use_local_hpwl=use_local_hpwl,
                        local_hpwl_weight=local_hpwl_weight,
                    )
                    scheduler.step()

                    round_loss += loss
                    round_wl += wl
                    round_steps += 1
                    step += 1
                    bench_loss_sum += loss

            bench_losses[data["name"]].append(
                bench_loss_sum / max(bench_epochs * n_transforms, 1)
            )

        avg_loss = round_loss / max(round_steps, 1)
        avg_wl = round_wl / max(round_steps, 1)
        elapsed = time.time() - t0
        print(f"Round {round_idx + 1}/{args.rounds}: "
              f"avg_loss={avg_loss:.2f}, avg_wl={avg_wl:.2f}, "
              f"steps={round_steps}, time={elapsed:.1f}s")

    # Save weights
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WEIGHTS_DIR / "gnn_pretrained.pt"
    torch.save(gnn.state_dict(), out_path)
    print(f"\nWeights saved to {out_path}")
    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
