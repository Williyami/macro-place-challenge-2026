"""
Competition entry point.

Wraps the current best placer so the evaluation harness can load it via:
    uv run evaluate submissions/placer.py

To switch approaches, change the import/class alias below.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from macro_place.benchmark import Benchmark
from submissions.sa_placer import SAPlacer


class MacroPlacer:
    """
    Entry point for the evaluation harness.

    The harness discovers the first class in this file with a ``place`` method
    and calls ``place(benchmark)``.  This class delegates to the best available
    placer implementation.

    To try a different approach:
        from submissions.analytical_placer import AnalyticalPlacer as _Inner
    """

    def __init__(self, seed: int = 42):
        self._inner = SAPlacer(
            seed=seed,
            max_iters=100_000,
            run_fd=False,
            snapshot_interval=2_000,
            trace_interval=500,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._inner.place(benchmark)
