# Learning Placer — Benchmark Results Log

## 2026-04-03 — Stronger density + congestion proxy + target_util=0.6
Method: `PLACER_METHOD=learning` — added top-10% congestion penalty, adaptive multi-start

### Full suite run (this config):
| Benchmark | Proxy | WL | Density | Congestion | Time | vs SA baseline |
|-----------|-------|-----|---------|------------|------|----------------|
| ibm01 | 1.2429 | 0.069 | 1.029 | 1.318 | 5.55s | BETTER (1.3166) |
| ibm02 | 1.7320 | 0.080 | 0.960 | 2.344 | 8.87s | BETTER (1.9072) |
| ibm03 | 1.7874 | 0.080 | 1.152 | 2.264 | 5.47s | WORSE (1.7401) |
| ibm04 | 1.6619 | 0.069 | 1.108 | 2.077 | 7.05s | WORSE (1.5037) |
| ibm06 | 2.0635 | 0.063 | 1.075 | 2.925 | 6.28s | BETTER (2.5057) |
| ibm07 | 1.6977 | 0.065 | 1.095 | 2.171 | 4.70s | BETTER (2.0229) |
| ibm08 | 1.7969 | 0.068 | 1.130 | 2.329 | 6.03s | BETTER (1.9239) |
| ibm09 | 1.2970 | 0.058 | 1.043 | 1.436 | 4.55s | BETTER (1.3875) |
| ibm10 | 1.7425 | 0.064 | 0.991 | 2.367 | 39.29s | BETTER (2.1108) |
| ibm11 | 1.4576 | 0.054 | 1.130 | 1.676 | 5.10s | BETTER (1.7111) |
| ibm12 | 2.0335 | 0.064 | 1.097 | 2.842 | 20.98s | BETTER (2.8261) |
| ibm13 | 1.6555 | 0.053 | 1.139 | 2.066 | 6.02s | BETTER (1.9141) |
| ibm14 | 1.7262 | 0.051 | 1.102 | 2.248 | 9.94s | BETTER (2.2750) |
| ibm15 | 1.8130 | 0.058 | 1.134 | 2.377 | 10.29s | BETTER (2.3000) |
| ibm16 | 1.7684 | 0.048 | 1.070 | 2.370 | 9.52s | BETTER (2.2337) |
| ibm17 | 1.7908 | 0.053 | 1.011 | 2.465 | 14.32s | BETTER (3.6726) |
| ibm18 | 1.8329 | 0.053 | 1.109 | 2.451 | 6.87s | BETTER (2.7755) |
| **AVG** | **1.7117** | | | | | **SA: 2.1251** |

Beat SA baseline on 15/17. Beat RePlAce on 1/17.
**Main weakness: congestion (2.0-2.9 on most benchmarks). WL is excellent.**

## 2026-04-04 — Full suite rerun (`--no-media`)
Method: `PLACER_METHOD=learning` — current code, benchmark sweep without visualization output

### Full suite run (this config):
| Benchmark | Proxy | WL | Density | Congestion | Time | vs SA baseline |
|-----------|-------|-----|---------|------------|------|----------------|
| ibm01 | 1.2707 | 0.067 | 1.046 | 1.360 | 21.37s | BETTER (1.3166) |
| ibm02 | 1.7309 | 0.079 | 0.945 | 2.358 | 21.11s | BETTER (1.9072) |
| ibm03 | 1.7636 | 0.079 | 1.149 | 2.220 | 15.17s | WORSE (1.7401) |
| ibm04 | 1.6595 | 0.068 | 1.105 | 2.077 | 18.84s | WORSE (1.5037) |
| ibm06 | 2.0338 | 0.063 | 1.055 | 2.886 | 20.31s | BETTER (2.5057) |
| ibm07 | 1.7092 | 0.065 | 1.109 | 2.180 | 17.22s | BETTER (2.0229) |
| ibm08 | 1.8257 | 0.067 | 1.140 | 2.377 | 19.19s | BETTER (1.9239) |
| ibm09 | 1.2826 | 0.057 | 1.025 | 1.426 | 13.84s | BETTER (1.3875) |
| ibm10 | 1.7108 | 0.063 | 0.987 | 2.309 | 104.54s | BETTER (2.1108) |
| ibm11 | 1.4296 | 0.054 | 1.112 | 1.640 | 19.33s | BETTER (1.7111) |
| ibm12 | 2.0185 | 0.063 | 1.092 | 2.819 | 49.63s | BETTER (2.8261) |
| ibm13 | 1.6382 | 0.053 | 1.124 | 2.046 | 14.55s | BETTER (1.9141) |
| ibm14 | 1.7314 | 0.051 | 1.107 | 2.254 | 29.41s | BETTER (2.2750) |
| ibm15 | 1.8209 | 0.058 | 1.145 | 2.381 | 27.85s | BETTER (2.3000) |
| ibm16 | 1.7527 | 0.048 | 1.066 | 2.343 | 20.48s | BETTER (2.2337) |
| ibm17 | 1.7912 | 0.053 | 1.010 | 2.468 | 30.82s | BETTER (3.6726) |
| ibm18 | 1.8331 | 0.053 | 1.112 | 2.448 | 14.08s | BETTER (2.7755) |
| **AVG** | **1.7060** | | | | 457.74s | **SA: 2.1251** |

Beat SA baseline on 15/17. Beat RePlAce on 1/17.
**This rerun is slightly better than the 2026-04-03 full-suite average (1.7117 -> 1.7060).**

## 2026-04-04 — v2: Pre-trained GNN + Density-aware SA + Greedy Flipping

Changes:
- Pre-trained GNN across all 17 IBM benchmarks (128 hidden, 4 layers, gated residual, 5 rounds)
- 9-dim node features (added aspect ratio, weighted degree)
- Fine-tune GNN per instance (100 epochs) instead of training from scratch (60 epochs)
- Density-aware SA polish (same as SA placer improvements)
- Greedy macro flipping post-processing
- 3 multi-starts with best selection

### Full suite run (this config):
| Benchmark | Proxy | WL | Density | Congestion | Time | vs SA baseline |
|-----------|-------|-----|---------|------------|------|----------------|
| ibm01 | 1.1795 | 0.069 | 0.890 | 1.330 | 84.55s | BETTER (1.3166) |
| ibm02 | 1.6912 | — | — | — | — | BETTER (1.9072) |
| ibm03 | 1.7467 | — | — | — | — | WORSE (1.7401) |
| ibm04 | 1.4569 | — | — | — | — | BETTER (1.5037) |
| ibm06 | 1.8393 | — | — | — | — | BETTER (2.5057) |
| ibm07 | 1.6561 | — | — | — | — | BETTER (2.0229) |
| ibm08 | 1.6987 | — | — | — | — | BETTER (1.9239) |
| ibm09 | 1.1960 | — | — | — | — | BETTER (1.3875) |
| ibm10 | 1.6763 | — | — | — | — | BETTER (2.1108) |
| ibm11 | 1.3852 | — | — | — | — | BETTER (1.7111) |
| ibm12 | 1.9648 | — | — | — | — | BETTER (2.8261) |
| ibm13 | 1.5334 | — | — | — | — | BETTER (1.9141) |
| ibm14 | 1.6263 | — | — | — | — | BETTER (2.2750) |
| ibm15 | 1.7841 | — | — | — | — | BETTER (2.3000) |
| ibm16 | 1.6803 | — | — | — | — | BETTER (2.2337) |
| ibm17 | 1.7660 | — | — | — | — | BETTER (3.6726) |
| ibm18 | 1.7985 | — | — | — | — | BETTER (2.7755) |
| **AVG** | **1.6282** | | | | 2000.80s | **SA: 2.1251** |

Beat SA baseline on 15/17. Beat RePlAce on 0/17.
**Major improvement: 1.7060 -> 1.6282 (-4.6%).**
Pre-trained GNN + density SA polish + flipping all contributed.
Still behind SA placer (1.4803) — the GNN initial placement helps but SA from initial.plc converges better.

## 2026-04-05 — v3: Research-paper-informed pretraining improvements

Changes (informed by research papers in notes/researchmaterial/):
- **Non-IBM benchmarks**: Pre-trained across 21 benchmarks (17 IBM + 4 NanGate45: ariane133, ariane136, nvdla, mempool)
- **Data augmentation**: Flip transforms (2x effective dataset) — from AutoDMP paper
- **Congestion-aware loss**: Added differentiable congestion penalty to both pretraining and fine-tuning, aligning with actual proxy_cost scoring (WL + 0.5×density + 0.5×congestion) — from Synopsys congestion paper
- **Curriculum learning**: Benchmarks sorted easy→hard by macro count — from HRLP paper
- **Per-net local HPWL loss**: Denser gradient signal per net — from HRLP paper
- **Prioritized training**: More epochs on harder benchmarks (0.5x-2x scaling) — from MCTS+RL paper

Pretrained with: `--epochs 50 --rounds 5 --augment-transforms 2` (~30 min on CPU)

### Full suite run (this config):
| Benchmark | Proxy | WL | Density | Congestion | Time | vs SA baseline |
|-----------|-------|-----|---------|------------|------|----------------|
| ibm01 | 1.1544 | 0.069 | 0.850 | 1.321 | 62.04s | BETTER (1.3166) |
| ibm02 | 1.6974 | 0.080 | 0.902 | 2.334 | 66.76s | BETTER (1.9072) |
| ibm03 | 1.7582 | 0.079 | 1.129 | 2.230 | 91.24s | WORSE (1.7401) |
| ibm04 | 1.4279 | 0.070 | 0.801 | 1.914 | 107.16s | BETTER (1.5037) |
| ibm06 | 2.0281 | 0.063 | 1.041 | 2.889 | 69.52s | BETTER (2.5057) |
| ibm07 | 1.6603 | 0.065 | 0.975 | 2.216 | 112.88s | BETTER (2.0229) |
| ibm08 | 1.6859 | 0.068 | 0.943 | 2.292 | 140.93s | BETTER (1.9239) |
| ibm09 | 1.1986 | 0.058 | 0.893 | 1.388 | 109.42s | BETTER (1.3875) |
| ibm10 | 1.6453 | 0.058 | 0.917 | 2.258 | 331.76s | BETTER (2.1108) |
| ibm11 | 1.3709 | 0.055 | 0.991 | 1.642 | 138.93s | BETTER (1.7111) |
| ibm12 | 1.9361 | 0.063 | 0.976 | 2.769 | 248.14s | BETTER (2.8261) |
| ibm13 | 1.5649 | 0.054 | 1.018 | 2.004 | 65.76s | BETTER (1.9141) |
| ibm14 | 1.6528 | 0.051 | 0.975 | 2.228 | 144.81s | BETTER (2.2750) |
| ibm15 | 1.7820 | 0.058 | 1.074 | 2.375 | 111.23s | BETTER (2.3000) |
| ibm16 | 1.6646 | 0.049 | 0.934 | 2.297 | 153.31s | BETTER (2.2337) |
| ibm17 | 1.7720 | 0.053 | 0.954 | 2.485 | 210.96s | BETTER (3.6726) |
| ibm18 | 1.8005 | 0.053 | 1.042 | 2.453 | 110.45s | BETTER (2.7755) |
| **AVG** | **1.6353** | | | | 2275.30s | **SA: 2.1251** |

Beat SA baseline on 16/17. Beat RePlAce on 1/17 (ibm02).
**Improvement over v2: 1.6282 -> 1.6353 (+0.4%) — slight regression.**
Density scores improved across the board (many now below 1.0), but congestion remains the bottleneck.
The research-paper improvements helped density but didn't yet translate to overall proxy gain.
Notable improvements: ibm04 (1.4569→1.4279), ibm11 (1.3852→1.3709).
Notable regressions: ibm12 (1.9648→1.9361 better), ibm06 still worst (2.0281).
Now beats SA on 16/17 (was 15/17) — ibm03 is the only loss.

## 2026-04-06 — v4: Congestion-focused overhaul + compute increase

Changes:
- **Congestion weight 10-20x increase**: GNN fine-tune max 0.025→0.5, refinement stage 2 up to 1.0
- **Two-stage refinement**: stage 1 (40%) WL+overlap focus, stage 2 (60%) congestion-heavy
- **Evaluator grid dims**: density penalty uses actual `grid_col`/`grid_row` instead of hardcoded 16
- **Benchmark-specific density target**: computed from actual macro utilization (×1.2 slack)
- **Best GNN checkpoint tracking**: restores best epoch by WL+congestion, not last
- **LR warmup + cosine decay**: for GNN fine-tuning stability
- **Initial.plc multi-start**: extra SA-only run from original positions with 1.5× SA iters
- **Compute budget increase**: finetune 100→400, refine 300→800, SA 100K→500K, starts 3→5+1
- **Retrained GNN** with matching congestion weight fix

### Full suite run (this config):
| Benchmark | Proxy | WL | Density | Congestion | Time | vs SA baseline |
|-----------|-------|-----|---------|------------|------|----------------|
| ibm01 | 1.1477 | 0.067 | 0.859 | 1.304 | 671.70s | BETTER (1.3166) |
| ibm02 | 1.5618 | 0.076 | 0.690 | 2.282 | 388.45s | BETTER (1.9072) |
| ibm03 | 1.7196 | 0.079 | 1.065 | 2.217 | 536.91s | BETTER (1.7401) |
| ibm04 | 1.3948 | 0.069 | 0.780 | 1.871 | 710.03s | BETTER (1.5037) |
| ibm06 | 1.6858 | 0.063 | 0.704 | 2.543 | 429.12s | BETTER (2.5057) |
| ibm07 | 1.4806 | 0.065 | 0.806 | 2.025 | 612.93s | BETTER (2.0229) |
| ibm08 | 1.7002 | 0.068 | 0.940 | 2.324 | 793.02s | BETTER (1.9239) |
| ibm09 | 1.1825 | 0.057 | 0.895 | 1.355 | 770.13s | BETTER (1.3875) |
| ibm10 | 1.6436 | 0.058 | 0.958 | 2.214 | 1476.55s | BETTER (2.1108) |
| ibm11 | 1.3557 | 0.054 | 0.979 | 1.624 | 634.07s | BETTER (1.7111) |
| ibm12 | 1.6348 | 0.060 | 0.750 | 2.400 | 971.40s | BETTER (2.8261) |
| ibm13 | 1.5596 | 0.054 | 1.024 | 1.988 | 592.36s | BETTER (1.9141) |
| ibm14 | 1.5885 | 0.051 | 0.946 | 2.129 | 1378.06s | BETTER (2.2750) |
| ibm15 | 1.7827 | 0.057 | 1.069 | 2.381 | 2703.37s | BETTER (2.3000) |
| ibm16 | 1.5033 | 0.049 | 0.811 | 2.098 | 1321.16s | BETTER (2.2337) |
| ibm17 | 1.7580 | 0.052 | 0.945 | 2.467 | 1438.32s | BETTER (3.6726) |
| ibm18 | 1.7984 | 0.053 | 1.043 | 2.447 | 674.22s | BETTER (2.7755) |
| **AVG** | **1.5587** | | | | 16101.83s | **SA: 2.1251** |

Beat SA baseline on 17/17. Beat RePlAce on 2/17 (ibm02, ibm12).
**Major improvement: 1.6353 → 1.5587 (-4.7%).**
Congestion-focused changes drove big gains on ibm06 (-16.9%), ibm07 (-10.8%), ibm16 (-9.7%), ibm12 (-15.6%).
Density scores excellent (mostly 0.7-1.0). Congestion still the main gap vs RePlAce.
Close to RePlAce on ibm07 (-1.2%), ibm18 (-1.5%), ibm16 (-1.7%).

## 2026-04-06 — SA V2: GPU-assisted post-refinement with CPU fallback

Method: `PLACER_METHOD=sa_v2` — SA V2 with HPWL caching, adaptive moves, LAHC, greedy local search, and GPU-assisted post-refinement

### Full suite run (this config):
| Benchmark | Proxy | WL | Density | Congestion | Time | vs SA baseline |
|-----------|-------|-----|---------|------------|------|----------------|
| ibm01 | 1.2564 | 0.088 | 0.927 | 1.410 | 80.75s | BETTER (1.3166) |
| ibm02 | 1.6908 | 0.085 | 0.783 | 2.429 | 77.08s | BETTER (1.9072) |
| ibm03 | 1.4712 | 0.086 | 0.829 | 1.940 | 62.24s | BETTER (1.7401) |
| ibm04 | 1.5237 | 0.078 | 0.886 | 2.005 | 262.14s | WORSE (1.5037) |
| ibm06 | 1.7958 | 0.067 | 0.788 | 2.669 | 58.71s | BETTER (2.5057) |
| ibm07 | 1.5767 | 0.068 | 0.901 | 2.118 | 78.22s | BETTER (2.0229) |
| ibm08 | 1.6220 | 0.074 | 0.935 | 2.161 | 95.94s | BETTER (1.9239) |
| ibm09 | 1.2071 | 0.060 | 0.898 | 1.396 | 74.49s | BETTER (1.3875) |
| ibm10 | 1.5070 | 0.076 | 0.746 | 2.117 | 324.50s | BETTER (2.1108) |
| ibm11 | 1.3136 | 0.057 | 0.917 | 1.597 | 98.48s | BETTER (1.7111) |
| ibm12 | 1.7353 | 0.066 | 0.799 | 2.540 | 285.93s | BETTER (2.8261) |
| ibm13 | 1.4951 | 0.057 | 0.950 | 1.927 | 111.21s | BETTER (1.9141) |
| ibm14 | 1.6527 | 0.054 | 0.998 | 2.201 | 371.52s | BETTER (2.2750) |
| ibm15 | 1.6473 | 0.059 | 0.976 | 2.201 | 196.69s | BETTER (2.3000) |
| ibm16 | 1.6145 | 0.050 | 0.910 | 2.218 | 407.15s | BETTER (2.2337) |
| ibm17 | 1.7702 | 0.054 | 0.967 | 2.465 | 712.17s | BETTER (3.6726) |
| ibm18 | 1.8050 | 0.053 | 1.058 | 2.446 | 181.13s | BETTER (2.7755) |
| **AVG** | **1.5697** | | | | 3478.35s | **SA: 2.1251** |

Beat SA baseline on 16/17. Beat RePlAce on 1/17 (ibm02).
**Result:** Much better than the SA baseline, but still behind our best logged SA V1 run (1.4803) and the current learning placer (1.5587).
The GPU-assisted fallback path is integrated and validated, but this first SA V2 full-suite run is still congestion-limited on ibm04/ibm10/ibm17.

## 2026-04-07 — SA V2: fast full-suite rerun (`--no-media`)

Method: `PLACER_METHOD=sa_v2` — SA V2 with HPWL caching, adaptive moves, LAHC, greedy local search, and GPU-assisted post-refinement

### Full suite run (this config):
| Benchmark | Proxy | WL | Density | Congestion | Time | vs SA baseline |
|-----------|-------|-----|---------|------------|------|----------------|
| ibm01 | 1.2515 | 0.088 | 0.934 | 1.393 | 23.43s | BETTER (1.3166) |
| ibm02 | 1.6672 | 0.086 | 0.768 | 2.395 | 14.96s | BETTER (1.9072) |
| ibm03 | 1.5042 | 0.087 | 0.834 | 2.000 | 15.90s | BETTER (1.7401) |
| ibm04 | 1.5122 | 0.078 | 0.882 | 1.987 | 97.32s | WORSE (1.5037) |
| ibm06 | 1.7702 | 0.066 | 0.803 | 2.607 | 15.51s | BETTER (2.5057) |
| ibm07 | 1.5731 | 0.068 | 0.892 | 2.119 | 21.37s | BETTER (2.0229) |
| ibm08 | 1.6782 | 0.076 | 0.922 | 2.282 | 31.19s | BETTER (1.9239) |
| ibm09 | 1.2071 | 0.061 | 0.902 | 1.390 | 22.93s | BETTER (1.3875) |
| ibm10 | 1.5059 | 0.076 | 0.740 | 2.120 | 48.62s | BETTER (2.1108) |
| ibm11 | 1.3292 | 0.058 | 0.916 | 1.627 | 28.50s | BETTER (1.7111) |
| ibm12 | 1.7301 | 0.066 | 0.804 | 2.525 | 45.90s | BETTER (2.8261) |
| ibm13 | 1.5038 | 0.057 | 0.946 | 1.948 | 28.28s | BETTER (1.9141) |
| ibm14 | 1.6763 | 0.054 | 1.004 | 2.240 | 57.27s | BETTER (2.2750) |
| ibm15 | 1.6607 | 0.059 | 0.998 | 2.205 | 46.20s | BETTER (2.3000) |
| ibm16 | 1.6149 | 0.051 | 0.897 | 2.231 | 58.27s | BETTER (2.2337) |
| ibm17 | 1.7691 | 0.054 | 0.969 | 2.461 | 74.23s | BETTER (3.6726) |
| ibm18 | 1.8011 | 0.053 | 1.055 | 2.440 | 39.07s | BETTER (2.7755) |
| **AVG** | **1.5738** | | | | 668.95s | **SA: 2.1251** |

Beat SA baseline on 16/17. Beat RePlAce on 1/17 (ibm02).
**Result:** Quality is essentially tied with the previous SA V2 full-suite run (1.5697 -> 1.5738, +0.3%), but runtime improved dramatically (3478.35s -> 668.95s, about 5.2x faster).
This keeps SA V2 clearly ahead of the SA baseline while making it much more practical under the competition runtime limit.
