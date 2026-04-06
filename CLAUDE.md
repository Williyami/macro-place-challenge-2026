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

# Available methods: sa, analytical, hybrid, learning, will_seed, sa_v2
```

## After Running Benchmarks

Every time you run the full IBM benchmark suite (`--all`), you MUST do all three of these steps.
Do not do these logging/update steps for quick single-benchmark test runs.

### 1. Log results to the correct notes file

Each team member has their own notes file:
- Learning/RL placer (Person 3): `notes/(RL method)eklundnotes.md`
- SA + Analytical placer (Person 1): `notes/(SA + Analytical method) Omnellnotes.md`
- Hybrid placer (GPU-accelerated analytical phase, Person 2): `notes/(Hybrid method)novaknotes.md`

Add a new `## Date — Description` section with a markdown table matching the existing format in that file. Include: Benchmark, Proxy, WL, Density, Congestion, Time, and vs SA baseline comparison.

**SA V2 (Eklund)**: Log to `notes/(RL method)eklundnotes.md` with "SA V2" in the heading (e.g. `## 2026-04-06 — SA V2: description`). The benchmark history generator identifies SA V2 sections by looking for "SA V2" or "sa_v2" in the heading.

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

## Key Files for SA V2 Placer (Eklund)

- `submissions/sa_v2_placer.py` — SA V2 with HPWL caching, adaptive moves, LAHC, greedy local search
- `submissions/sa_placer.py` — Original SA V1 (kept as baseline)

## Key Files for Learning Placer

- `submissions/learning_placer.py` — GNN + differentiable placement + SA polish
- `submissions/pretrain_learning.py` — pre-training script across all benchmarks
- `submissions/learning_weights/gnn_pretrained.pt` — pre-trained weights (git-tracked)
