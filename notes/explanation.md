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
