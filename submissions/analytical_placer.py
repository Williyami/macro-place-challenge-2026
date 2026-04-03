"""
Analytical (gradient-based) placer — Phase 2 stub.

Planned approach:
  - Differentiable HPWL via log-sum-exp (LSE) approximation
  - Differentiable density penalty (Gaussian spreading)
  - Adam optimizer with projection to feasible region
  - Legalization post-processing (minimum displacement)

Status: NOT YET IMPLEMENTED.

Usage (when ready):
    uv run evaluate submissions/analytical_placer.py
"""

import torch
from macro_place.benchmark import Benchmark
from submissions.base import BasePlacer


class AnalyticalPlacer(BasePlacer):
    """
    Gradient-based macro placer.

    Will use differentiable HPWL + density penalty optimised with Adam,
    followed by legalization to resolve overlaps.
    """

    def __init__(self, seed: int = 42, iters: int = 5_000, lr: float = 0.01):
        self.seed = seed
        self.iters = iters
        self.lr = lr

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        raise NotImplementedError(
            "AnalyticalPlacer is not yet implemented (Phase 2). "
            "Run submissions/sa_placer.py for the current best placer."
        )
