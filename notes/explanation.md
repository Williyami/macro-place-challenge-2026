## Team setup
There are 3 people working in parallel, each building a different
placement strategy. ALL methods live in the same repo under submissions/ and share
a single entry point (submissions/placer.py). You can benchmark ANY method at any
time by setting the PLACER_METHOD env var:

    PLACER_METHOD=sa          uv run evaluate submissions/placer.py --all
    PLACER_METHOD=analytical  uv run evaluate submissions/placer.py --all
    PLACER_METHOD=hybrid      uv run evaluate submissions/placer.py --all
    PLACER_METHOD=learning    uv run evaluate submissions/placer.py --all
    PLACER_METHOD=will_seed   uv run evaluate submissions/placer.py --all

Compare your method against others frequently to see where you stand. The goal is
to find the best approach across all methods — we'll submit whichever wins.
Your file is YOUR file, but run the others to compare.


---

## Summary

| Person | Method | File | Env var | Prompt above |
|--------|--------|------|---------|------|
| 1 | SA + Analytical | `sa_placer.py`, `analytical_placer.py` | `PLACER_METHOD=sa` or `analytical` | SA/Analytical |
| 2 | Hybrid | `hybrid_placer.py` (new) | `PLACER_METHOD=hybrid` | Hybrid |
| 3 | RL/GNN/Learning | `learning_placer.py` (new) | `PLACER_METHOD=learning` | Learning |

The router in [placer.py](submissions/placer.py) already handles all five methods. Each person's Claude works in their own file(s) — no merge conflicts. Benchmark any method with `PLACER_METHOD=<name> uv run evaluate submissions/placer.py --all`.

---

## Pre-training the Learning Placer

The learning placer uses a GNN that can be pre-trained across all IBM benchmarks.
Pre-trained weights are saved to `submissions/learning_weights/gnn_pretrained.pt`
and tracked by git so teammates don't need to retrain.

**To retrain (if you change the GNN architecture or training):**

```bash
uv run python submissions/pretrain_learning.py --epochs 150 --rounds 5
```

Options:
- `--epochs N` — training epochs per benchmark per round (default: 150)
- `--rounds N` — round-robin passes over all benchmarks (default: 5)
- `--lr F` — learning rate (default: 1e-3)
- `--seed N` — random seed (default: 42)

This takes ~5-10 min and produces `submissions/learning_weights/gnn_pretrained.pt`.
The learning placer automatically loads these weights at evaluation time.

**To evaluate after pre-training:**

```bash
PLACER_METHOD=learning uv run evaluate submissions/placer.py --all --no-media
```
