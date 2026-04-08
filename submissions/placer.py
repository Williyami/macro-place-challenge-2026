"""
Competition entry point.

Wraps the current best placer so the evaluation harness can load it via:
    uv run evaluate submissions/placer.py

Switch methods via env var:
    PLACER_METHOD=sa          uv run evaluate submissions/placer.py --all
    PLACER_METHOD=analytical  uv run evaluate submissions/placer.py --all
    PLACER_METHOD=hybrid      uv run evaluate submissions/placer.py --all
    PLACER_METHOD=learning    uv run evaluate submissions/placer.py --all
    PLACER_METHOD=will_seed   uv run evaluate submissions/placer.py --all
    PLACER_METHOD=sa_v2       uv run evaluate submissions/placer.py --all
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from macro_place.benchmark import Benchmark

# ── Strategy selection ──────────────────────────────────────────────────────
# Set PLACER_METHOD env var to switch between approaches.
# Each team member works on their own module under submissions/.
METHOD = os.environ.get("PLACER_METHOD", "sa")


class MacroPlacer:
    """
    Entry point for the evaluation harness.
    Delegates to the selected placer implementation.
    """

    def __init__(self, seed: int = 42):
        if METHOD == "sa":
            from submissions.sa_placer import SAPlacer
            self._inner = SAPlacer(
                seed=seed,
                max_iters=120_000,
                run_fd=False,
                snapshot_interval=2_000,
                trace_interval=500,
                t_start_factor=0.12,
                t_end_factor=0.0008,
            )
        elif METHOD == "analytical":
            from submissions.analytical_placer import AnalyticalPlacer
            self._inner = AnalyticalPlacer(seed=seed)
        elif METHOD == "hybrid":
            from submissions.hybrid_placer import HybridPlacer
            self._inner = HybridPlacer(seed=seed)
        elif METHOD == "learning":
            from submissions.learning_placer import LearningPlacer
            self._inner = LearningPlacer(seed=seed)
        elif METHOD == "sa_v2":
            from submissions.sa_v2_placer import SAV2Placer
            self._inner = SAV2Placer(
                seed=seed,
                max_iters=120_000,
                run_fd=False,
                snapshot_interval=2_000,
                trace_interval=500,
                t_start_factor=0.15,
                t_end_factor=0.001,
                reheat_threshold=5_000,
                lahc_length=0,
                greedy_tail_frac=0.05,
                greedy_local_passes=3,
                adaptive_moves=True,
                gpu_refine_steps=200,
                congestion_weight_factor=1.0,
            )
        elif METHOD == "will_seed":
            from submissions.will_seed.placer import WillSeedPlacer
            self._inner = WillSeedPlacer(seed=seed)
        else:
            raise ValueError(
                f"Unknown PLACER_METHOD={METHOD!r}. "
                "Options: sa, analytical, hybrid, learning, will_seed, sa_v2"
            )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._inner.place(benchmark)
