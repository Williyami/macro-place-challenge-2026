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

## Pre-training the Learning Placer (v2)

The learning placer uses a GNN pre-trained across all available benchmarks
(17 IBM + 4 NanGate45 non-IBM). Pre-trained weights are saved to
`submissions/learning_weights/gnn_pretrained.pt` and tracked by git.

**v2 improvements (informed by research papers):**
- Non-IBM benchmarks (ariane, nvdla, mempool) for better generalization
- Data augmentation via flipping (from AutoDMP paper)
- Congestion-aware loss matching actual proxy_cost scoring (from Synopsys paper)
- Curriculum learning: small designs first (from HRLP paper)
- Per-net local HPWL loss for denser gradient signal (from HRLP paper)
- Prioritized training: more epochs on harder benchmarks (from MCTS+RL paper)

**To retrain (if you change the GNN architecture or training):**

```bash
uv run python submissions/pretrain_learning.py --epochs 50 --rounds 5 --augment-transforms 2
```

Options:
- `--epochs N` — training epochs per benchmark per round (default: 150)
- `--rounds N` — round-robin passes over all benchmarks (default: 5)
- `--augment-transforms N` — number of augmentation transforms 1-8 (default: 4)
- `--lr F` — learning rate (default: 1e-3)
- `--seed N` — random seed (default: 42)
- `--no-augment` — disable data augmentation
- `--no-congestion` — disable congestion loss
- `--no-local-hpwl` — disable per-net local HPWL loss
- `--no-non-ibm` — only train on IBM benchmarks
- `--no-curriculum` — disable curriculum learning
- `--no-prioritized` — disable prioritized training

Recommended: `--epochs 50 --rounds 5 --augment-transforms 2` (~30 min on CPU).
Full training: `--epochs 150 --rounds 5 --augment-transforms 4` (~2-3 hours).

**To evaluate after pre-training:**

```bash
PLACER_METHOD=learning uv run evaluate submissions/placer.py --all --no-media
```
