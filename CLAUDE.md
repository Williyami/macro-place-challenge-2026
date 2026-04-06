# Claude Code Instructions — Macro Placement Challenge

## Project Structure

- `submissions/` — placer implementations (one per team member)
- `submissions/placer.py` — router that dispatches to the right placer via `PLACER_METHOD` env var
- `notes/` — per-person benchmark logs and research material
- `notes/benchmark history/` — auto-generated history graph and raw data

## Running Benchmarks

```bash
# Run a specific method on all IBM benchmarks
PLACER_METHOD=learning uv run evaluate submissions/placer.py --all --no-media

# Available methods: sa, analytical, hybrid, learning, will_seed
```

## After Running Benchmarks

Every time you run benchmarks, you MUST do all three of these steps:

### 1. Log results to the correct notes file

Each team member has their own notes file:
- Learning/RL placer (Person 3): `notes/(RL method)eklundnotes.md`
- SA + Analytical placer (Person 1): `notes/(SA + Analytical method) Omnellnotes.md`
- Hybrid placer (GPU-accelerated analytical phase, Person 2): `notes/(Hybrid method)novaknotes.md`

Add a new `## Date — Description` section with a markdown table matching the existing format in that file. Include: Benchmark, Proxy, WL, Density, Congestion, Time, and vs SA baseline comparison.

### 2. Regenerate the benchmark history graph

```bash
uv run python "notes/benchmark history/generate_benchmark_history.py"
```

This parses all notes files and regenerates:
- `notes/benchmark history/benchmark_history_raw.md`
- `notes/benchmark history/benchmark_history_summary.md`
- `notes/benchmark history/benchmark_history_full_suite.png`

The PNG is embedded in the README via `![Benchmark history summary](notes/benchmark%20history/benchmark_history_full_suite.png)`.

### 3. Update the README leaderboard

Update the table in `README.md` under `## Leaderboard` with the new AVG proxy cost, best/worst scores, and total runtime.

## Pre-training the Learning Placer

```bash
uv run python submissions/pretrain_learning.py --epochs 50 --rounds 5 --augment-transforms 2
```

Weights save to `submissions/learning_weights/gnn_pretrained.pt`. Always re-run benchmarks after retraining.

## Key Files for Learning Placer

- `submissions/learning_placer.py` — GNN + differentiable placement + SA polish
- `submissions/pretrain_learning.py` — pre-training script across all benchmarks
- `submissions/learning_weights/gnn_pretrained.pt` — pre-trained weights (git-tracked)
