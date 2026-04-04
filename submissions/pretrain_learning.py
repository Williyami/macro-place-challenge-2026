#!/usr/bin/env python3
"""
Pre-train the learning placer GNN across all IBM benchmarks.

Trains the GNN to predict good macro positions by optimizing differentiable
LSE-HPWL + density + overlap across all 17 IBM benchmarks in a round-robin
fashion. Weights are saved to submissions/learning_weights/gnn_pretrained.pt.

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
from macro_place.loader import load_benchmark_from_dir
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


def load_benchmark_data(name: str):
    """Load benchmark and extract all data needed for training."""
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    benchmark, _ = load_benchmark_from_dir(str(root))

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

    return {
        "name": name,
        "n_hard": n_hard,
        "cw": cw,
        "ch": ch,
        "net_batches": net_batches,
        "node_features": node_features,
        "adj_norm": adj_norm,
        "sizes": sizes,
        "half_w": half_w,
        "half_h": half_h,
        "movable": movable,
        "fixed_mask": fixed_mask,
        "orig_pos": orig_pos,
    }


def train_step(gnn, optimizer, data, gamma, frac):
    """One training step on a single benchmark."""
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

    loss_wl = _lse_hpwl(pos, data["net_batches"], gamma)
    loss_den = _smooth_density_penalty(
        pos, sizes, cw, ch, grid_n=16, target_util=0.5
    ) * (cw * ch) * (0.01 + 0.04 * frac)
    loss_ov = _smooth_overlap_penalty(
        pos, sizes, data["movable"]
    ) * (0.1 + 1.0 * frac)

    loss = loss_wl + loss_den + loss_ov
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
    optimizer.step()

    return loss.item(), loss_wl.item()


def main():
    parser = argparse.ArgumentParser(description="Pre-train learning placer GNN")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs per round-robin pass")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of round-robin passes over all benchmarks")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-start", type=float, default=50.0)
    parser.add_argument("--gamma-end", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading benchmarks...")
    all_data = []
    for name in IBM_BENCHMARKS:
        print(f"  {name}...", end=" ", flush=True)
        data = load_benchmark_data(name)
        if data is not None:
            all_data.append(data)
            print(f"OK ({data['n_hard']} macros, {len(data['net_batches'])} net groups)")
        else:
            print("SKIP (no plc)")

    if not all_data:
        print("ERROR: No benchmarks loaded!")
        return

    # All benchmarks should have in_dim=9 (our feature set)
    in_dim = all_data[0]["node_features"].shape[1]
    print(f"\nFeature dim: {in_dim}, Benchmarks: {len(all_data)}")

    gnn = NetlistGNN(in_dim=in_dim, hidden_dim=128, out_dim=2, num_layers=4)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.rounds * args.epochs, eta_min=args.lr * 0.01
    )

    total_steps = args.rounds * args.epochs * len(all_data)
    step = 0
    t0 = time.time()

    print(f"\nTraining: {args.rounds} rounds × {args.epochs} epochs × {len(all_data)} benchmarks")
    print(f"Total steps: {total_steps}\n")

    for round_idx in range(args.rounds):
        # Shuffle benchmark order each round for better generalization
        rng = np.random.default_rng(args.seed + round_idx)
        order = rng.permutation(len(all_data))

        round_loss = 0.0
        round_wl = 0.0
        round_steps = 0

        for bench_idx in order:
            data = all_data[bench_idx]
            for epoch in range(args.epochs):
                frac = step / total_steps
                gamma = args.gamma_start * (args.gamma_end / args.gamma_start) ** frac

                loss, wl = train_step(gnn, optimizer, data, gamma, frac)
                scheduler.step()

                round_loss += loss
                round_wl += wl
                round_steps += 1
                step += 1

        avg_loss = round_loss / round_steps
        avg_wl = round_wl / round_steps
        elapsed = time.time() - t0
        print(f"Round {round_idx + 1}/{args.rounds}: "
              f"avg_loss={avg_loss:.2f}, avg_wl={avg_wl:.2f}, "
              f"time={elapsed:.1f}s")

    # Save weights
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WEIGHTS_DIR / "gnn_pretrained.pt"
    torch.save(gnn.state_dict(), out_path)
    print(f"\nWeights saved to {out_path}")
    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
