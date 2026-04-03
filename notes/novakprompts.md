You are working on the macro-place-challenge-2026 repo at the project root.
This is a chip macro placement competition — minimize proxy_cost = 1.0×WL + 0.5×density + 0.5×congestion across 17 IBM benchmarks.

YOUR FOCUS: Hybrid placement methods that combine multiple techniques.

## Current state
- `submissions/sa_placer.py` — working SA placer (avg ~1.57, our current best). Uses net-HPWL objective, 120K iterations, exponential cooling.
- `submissions/will_seed/placer.py` — simpler edge-based SA (avg ~1.53, slightly better).
- `submissions/analytical_placer.py` — gradient-based approach (being built by teammate).
- Baselines: SA baseline avg ~2.13, RePlAce avg ~1.46, will_seed avg ~1.53.
- Gap to close: ~8% behind RePlAce.

## Your task
Create `submissions/hybrid_placer.py` with a HybridPlacer class that extends BasePlacer (from submissions/base.py).

Promising hybrid strategies to try:
1. **Analytical → SA pipeline**: Use a quick gradient-based pass (log-sum-exp HPWL + density penalty, ~1000 Adam steps) to get a good initial placement, then refine with SA. The intuition: analytical gets global structure right, SA handles local optimization and legalization.
2. **Force-directed + SA**: Compute net-based attractive forces and density-based repulsive forces to get initial positions, then SA to polish. Simpler than full analytical but captures connectivity structure.
3. **Multi-start SA with clustering**: Partition macros into groups by connectivity (spectral clustering or simple BFS), place groups roughly, then SA within groups, then global SA.
4. **SA with density-aware moves**: Modify the SA cost to include a density term (not just HPWL), so moves that spread macros apart are preferred when density is high.

The class signature must be:
```python
from submissions.base import BasePlacer

class HybridPlacer(BasePlacer):
    def __init__(self, seed: int = 42):
        ...
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Return [num_macros, 2] float32 tensor
        ...
