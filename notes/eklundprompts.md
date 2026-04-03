# Person 3: RL / GNN / Learning-Based Methods

## Team setup
There are 3 people working in parallel via LiveShare, each building a different
placement strategy. ALL methods live in the same repo under submissions/ and share
a single entry point (submissions/placer.py). You can benchmark ANY method at any
time by setting the PLACER_METHOD env var:

    export PATH="$HOME/.local/bin:$PATH"
    PLACER_METHOD=sa          uv run evaluate submissions/placer.py --all
    PLACER_METHOD=analytical  uv run evaluate submissions/placer.py --all
    PLACER_METHOD=hybrid      uv run evaluate submissions/placer.py --all
    PLACER_METHOD=learning    uv run evaluate submissions/placer.py --all
    PLACER_METHOD=will_seed   uv run evaluate submissions/placer.py --all

Compare your method against others frequently to see where you stand. The goal is
to find the best approach across all methods — we'll submit whichever wins.
Your file is YOUR file, but run the others to compare.

## Context
You are working on the macro-place-challenge-2026 repo at the project root.
This is a chip macro placement competition — minimize proxy_cost = 1.0×WL + 0.5×density + 0.5×congestion across 17 IBM benchmarks.

YOUR FOCUS: Learning-based placement methods (RL, GNN, neural combinatorial optimization).

## Current state
- `submissions/sa_placer.py` — working SA placer (avg ~1.57). Good but struggles on some benchmarks.
- Baselines: SA baseline avg ~2.13, RePlAce avg ~1.46 (target to beat).
- The benchmark data is already in PyTorch tensors — net connectivity, macro sizes, positions, pin offsets are all available.

## Your task
Create `submissions/learning_placer.py` with a LearningPlacer class that extends BasePlacer (from submissions/base.py).

Promising approaches (pick one or combine):
1. **GNN-guided placement**: Build a graph from the netlist (macros = nodes, nets = hyperedges). Use a GNN (e.g., GCN/GAT with PyTorch Geometric or manual message passing) to compute macro embeddings, then decode to (x,y) positions. Train on the proxy cost signal directly (REINFORCE or straight-through estimator). Can train per-benchmark since each is a separate optimization problem.
2. **RL sequential placement** (Google's approach from Nature 2021): Place macros one at a time onto a grid. State = current partial placement + macro features. Action = grid cell. Policy = small CNN/MLP. Train with REINFORCE using proxy_cost as reward. Key: order macros by size (largest first) to reduce branching.
3. **Neural combinatorial optimization**: Treat placement as continuous optimization. Use a neural network to predict good initial positions, then refine with gradient descent on differentiable proxy cost. The network takes netlist features as input and outputs [N, 2] positions.
4. **Learning-enhanced SA**: Use a small neural net to predict good move proposals for SA (instead of random Gaussian shifts). Train the proposal network to maximize acceptance rate or minimize cost.

The class signature must be:
```python
from submissions.base import BasePlacer

class LearningPlacer(BasePlacer):
    def __init__(self, seed: int = 42):
        ...
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Return [num_macros, 2] float32 tensor
        ...
```

## How to benchmark
    export PATH="$HOME/.local/bin:$PATH"
    PLACER_METHOD=learning uv run evaluate submissions/placer.py -b ibm01   # single
    PLACER_METHOD=learning uv run evaluate submissions/placer.py --all      # all 17

## Key APIs & data
- Benchmark.macro_positions: [N, 2] initial positions (x, y centers)
- Benchmark.macro_sizes: [N, 2] (width, height)
- Benchmark.macro_fixed: [N] bool — fixed macros must stay put
- Benchmark.canvas_width/canvas_height: float
- Benchmark.num_hard_macros: int (hard macros are indices [0, num_hard))
- Benchmark.num_soft_macros: int (soft macros are indices [num_hard, num_macros))
- Benchmark.get_movable_mask(): ~macro_fixed
- Net connectivity: use _extract_nets(benchmark, plc) from sa_placer.py or build from benchmark data
- _load_plc(benchmark.name) from sa_placer.py loads the PlacementCost evaluator
- compute_proxy_cost(placement, benchmark, plc) from macro_place.objective — the actual scoring function
- For overlap-free placements, you can use _legalize() from sa_placer.py as a post-processing step
- PyTorch is already a dependency. Add torch-geometric or other deps to pyproject.toml if needed (uv add <package>).

## Important constraints
- place() must run in <60 seconds per benchmark (competition time limit)
- Training can happen inside place() (per-instance optimization) or offline
- If training offline, store model weights in submissions/ and load them in __init__
- The placement MUST be valid: no overlaps, within bounds, fixed macros unchanged

## Rules
- Only create/edit `submissions/learning_placer.py` (and any model weight files in submissions/)
- Do NOT touch placer.py, sa_placer.py, analytical_placer.py, or hybrid_placer.py
- The router in submissions/placer.py already handles PLACER_METHOD=learning and imports LearningPlacer from your file
- You CAN import helper functions from sa_placer.py (net extraction, legalization, HPWL computation)
