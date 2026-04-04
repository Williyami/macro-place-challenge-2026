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
