"""
Base class for all macro placers.

Usage:
    class MyPlacer(BasePlacer):
        def place(self, benchmark: Benchmark) -> torch.Tensor:
            ...
"""

from abc import ABC, abstractmethod
import torch
from macro_place.benchmark import Benchmark


class BasePlacer(ABC):
    @abstractmethod
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Place macros and return a [num_macros, 2] float32 tensor of (x, y) centers.

        Requirements:
            - Fixed macros (benchmark.macro_fixed) must stay at original positions.
            - Hard macros must not overlap each other.
            - All macros must remain within canvas bounds.
            - Returned tensor shape: [num_macros, 2], float32.
        """
        ...
