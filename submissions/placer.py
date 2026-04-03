"""
Competition entry point.

Wraps the current best placer so the evaluation harness can load it via:
    uv run evaluate submissions/placer.py

To switch approaches, change METHOD below.
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
# Set METHOD env var or change this default to switch:
#   "sa"        — full net-HPWL simulated annealing (current best)
#   "will_seed" — edge-based fast SA
METHOD = os.environ.get("PLACER_METHOD", "sa")


class MacroPlacer:
    """
    Entry point for the evaluation harness.
    Delegates to the selected placer implementation.
    """

    def __init__(self, seed: int = 42):
        if METHOD == "will_seed":
            from submissions.will_seed.placer import WillSeedPlacer
            self._inner = WillSeedPlacer(seed=seed)
        else:
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

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._inner.place(benchmark)
