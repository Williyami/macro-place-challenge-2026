You are working on the macro-place-challenge-2026 repo at the project root.
This is a chip macro placement competition — minimize proxy_cost = 1.0×WL + 0.5×density + 0.5×congestion across 17 IBM benchmarks.

YOUR FOCUS: SA and Analytical placement methods.

## Current state
- `submissions/sa_placer.py` — working SA placer (our current best, avg ~1.57). Uses full net-HPWL, 120K iters, exponential cooling. Temperature params are exposed as t_start_factor/t_end_factor.
- `submissions/analytical_placer.py` — stub, NOT YET IMPLEMENTED.
- Baselines: SA baseline avg ~2.13, RePlAce avg ~1.46, will_seed avg ~1.53.
- ibm04 is our worst benchmark (1.57 vs SA baseline 1.50) — we're WORSE than baseline there.

## Your tasks (priority order)
1. **Fix ibm04 regression** — investigate why ibm04 is worse than SA baseline. Try per-benchmark temperature tuning, more iterations, or different move ratios. The SA code already accepts t_start_factor and t_end_factor params.
2. **Implement AnalyticalPlacer** in `submissions/analytical_placer.py` — differentiable placement using:
   - Log-sum-exp HPWL approximation (differentiable)
   - Gaussian density penalty (smooth, differentiable)
   - Adam optimizer with projection to keep macros in canvas bounds
   - Legalization post-processing (can reuse _legalize from sa_placer.py)
   - The class must extend BasePlacer from submissions/base.py and implement place(benchmark) -> Tensor[num_macros, 2]
3. **SA improvements** — consider: adaptive cooling schedule, reheating on stagnation, better move selection (e.g., window-based moves instead of global random), multi-start with different seeds.

## How to benchmark
    export PATH="$HOME/.local/bin:$PATH"
    PLACER_METHOD=sa uv run evaluate submissions/placer.py -b ibm04       # single
    PLACER_METHOD=sa uv run evaluate submissions/placer.py --all          # all 17
    PLACER_METHOD=analytical uv run evaluate submissions/placer.py --all  # analytical

## Key APIs
- Benchmark dataclass: macro_positions[N,2], macro_sizes[N,2], macro_fixed[N], canvas_width/height, num_hard_macros, get_movable_mask()
- compute_proxy_cost(placement, benchmark, plc) -> dict with proxy_cost, wirelength_cost, density_cost, congestion_cost
- Your placer's place() receives a Benchmark and must return a [num_macros, 2] float32 tensor

## Rules
- Only edit files in submissions/sa_placer.py and submissions/analytical_placer.py
- The router in submissions/placer.py already handles PLACER_METHOD=sa and PLACER_METHOD=analytical
- Do NOT touch submissions/placer.py, hybrid_placer.py, or learning_placer.py — other team members own those
