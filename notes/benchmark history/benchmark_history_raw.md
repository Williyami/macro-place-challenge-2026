# Benchmark History Raw Data

Generated from the note files in this directory.

## Run-Level Records

| Run ID | Method | Date | Title | Scope | Benchmarks Logged | Plotted Proxy | Full-Suite Avg Proxy | Total Runtime (s) | Source |
|--------|--------|------|-------|-------|-------------------|---------------|----------------------|-------------------|--------|
| L1 | Learning Placer | 2026-04-03 | Stronger density + congestion proxy + target_util=0.6 | full_suite | 17 | 1.7117 | 1.7117 | 170.83 | (RL method)eklundnotes.md |
| L2 | Learning Placer | 2026-04-04 | Full suite rerun (`--no-media`) | full_suite | 17 | 1.7060 | 1.7060 | 457.74 | (RL method)eklundnotes.md |
| L3 | Learning Placer | 2026-04-04 | v2: Pre-trained GNN + Density-aware SA + Greedy Flipping | full_suite | 17 | 1.6282 | 1.6282 | 84.55 | (RL method)eklundnotes.md |
| L4 | Learning Placer | 2026-04-05 | v3: Research-paper-informed pretraining improvements | full_suite | 17 | 1.6353 | 1.6353 | 2275.30 | (RL method)eklundnotes.md |
| L5 | Learning Placer | 2026-04-06 | v4: Congestion-focused overhaul + compute increase | full_suite | 17 | 1.5587 | 1.5587 | 16101.80 | (RL method)eklundnotes.md |
| L6 | Learning Placer | 2026-04-07 | v6: fast full-suite rerun (`--no-media`) | full_suite | 17 | 1.4828 | 1.4828 | - | (RL method)eklundnotes.md |
| S1 | SA Placer | 2026-04-03 | SA Placer (PLACER_METHOD=sa) | full_suite | 17 | 1.5765 | 1.5765 | 215.00 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | 2026-04-04 | SA Placer (PLACER_METHOD=sa) | full_suite | 17 | 1.4850 | 1.4850 | 999.38 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | 2026-04-04, v2 | SA Placer (PLACER_METHOD=sa) | full_suite | 17 | 1.4803 | 1.4803 | 676.83 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | 2026-04-05 | SA Placer (PLACER_METHOD=sa) | full_suite | 17 | 1.4844 | 1.4844 | 503.00 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | 2026-04-05 | SA Placer (full_suite) — S5 | full_suite | 17 | 1.4836 | 1.4836 | - | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | 2026-04-05 | SA Placer (full_suite) — S6 | full_suite | 17 | 1.4814 | 1.4814 | - | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | 2026-04-03 | Analytical Placer (PLACER_METHOD=analytical) | full_suite | 17 | 1.7310 | 1.7310 | 406.00 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | 2026-04-04 | Analytical Placer (PLACER_METHOD=analytical) | full_suite | 17 | 1.7394 | 1.7394 | 598.68 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | 2026-04-05 | Analytical Placer v3 (PLACER_METHOD=analytical) | full_suite | 17 | 1.7232 | 1.7232 | 1362.00 | (SA + Analytical method) Omnellnotes.md |
| H1 | HybridPlacer | 2026-04-03 | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.6972 | 1.6972 | 1492.67 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | 2026-04-04, --no-media | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.6977 | 1.6977 | 324.72 | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | 2026-04-06 | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.6966 | 1.6966 | 3998.65 | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | 2026-04-07, v4 --no-media | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.5954 | 1.5954 | 12369.19 | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | 2026-04-07, v5 | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.5723 | 1.5723 | 24641.71 | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | 2026-04-08, v6 | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.4828 | 1.4828 | 797.87 | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | 2026-04-08, v7 | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.4828 | 1.4828 | 797.56 | (Hybrid method)novaknotes.md |
| V1 | SA V2 (Eklund) | 2026-04-06 | SA V2: GPU-assisted post-refinement with CPU fallback | full_suite | 17 | 1.5697 | 1.5697 | 3478.35 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | 2026-04-07 | SA V2: fast full-suite rerun (`--no-media`) | full_suite | 17 | 1.5738 | 1.5738 | 668.95 | (RL method)eklundnotes.md |

## Benchmark-Level Records

| Run ID | Method | Benchmark | Proxy | WL | Density | Congestion | Time (s) | Source |
|--------|--------|-----------|-------|----|---------|------------|----------|--------|
| L1 | Learning Placer | ibm01 | 1.2429 | 0.069 | 1.029 | 1.318 | 5.55 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm02 | 1.7320 | 0.080 | 0.960 | 2.344 | 8.87 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm03 | 1.7874 | 0.080 | 1.152 | 2.264 | 5.47 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm04 | 1.6619 | 0.069 | 1.108 | 2.077 | 7.05 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm06 | 2.0635 | 0.063 | 1.075 | 2.925 | 6.28 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm07 | 1.6977 | 0.065 | 1.095 | 2.171 | 4.70 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm08 | 1.7969 | 0.068 | 1.130 | 2.329 | 6.03 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm09 | 1.2970 | 0.058 | 1.043 | 1.436 | 4.55 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm10 | 1.7425 | 0.064 | 0.991 | 2.367 | 39.29 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm11 | 1.4576 | 0.054 | 1.130 | 1.676 | 5.10 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm12 | 2.0335 | 0.064 | 1.097 | 2.842 | 20.98 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm13 | 1.6555 | 0.053 | 1.139 | 2.066 | 6.02 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm14 | 1.7262 | 0.051 | 1.102 | 2.248 | 9.94 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm15 | 1.8130 | 0.058 | 1.134 | 2.377 | 10.29 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm16 | 1.7684 | 0.048 | 1.070 | 2.370 | 9.52 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm17 | 1.7908 | 0.053 | 1.011 | 2.465 | 14.32 | (RL method)eklundnotes.md |
| L1 | Learning Placer | ibm18 | 1.8329 | 0.053 | 1.109 | 2.451 | 6.87 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm01 | 1.2707 | 0.067 | 1.046 | 1.360 | 21.37 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm02 | 1.7309 | 0.079 | 0.945 | 2.358 | 21.11 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm03 | 1.7636 | 0.079 | 1.149 | 2.220 | 15.17 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm04 | 1.6595 | 0.068 | 1.105 | 2.077 | 18.84 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm06 | 2.0338 | 0.063 | 1.055 | 2.886 | 20.31 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm07 | 1.7092 | 0.065 | 1.109 | 2.180 | 17.22 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm08 | 1.8257 | 0.067 | 1.140 | 2.377 | 19.19 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm09 | 1.2826 | 0.057 | 1.025 | 1.426 | 13.84 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm10 | 1.7108 | 0.063 | 0.987 | 2.309 | 104.54 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm11 | 1.4296 | 0.054 | 1.112 | 1.640 | 19.33 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm12 | 2.0185 | 0.063 | 1.092 | 2.819 | 49.63 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm13 | 1.6382 | 0.053 | 1.124 | 2.046 | 14.55 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm14 | 1.7314 | 0.051 | 1.107 | 2.254 | 29.41 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm15 | 1.8209 | 0.058 | 1.145 | 2.381 | 27.85 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm16 | 1.7527 | 0.048 | 1.066 | 2.343 | 20.48 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm17 | 1.7912 | 0.053 | 1.010 | 2.468 | 30.82 | (RL method)eklundnotes.md |
| L2 | Learning Placer | ibm18 | 1.8331 | 0.053 | 1.112 | 2.448 | 14.08 | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm01 | 1.1795 | 0.069 | 0.890 | 1.330 | 84.55 | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm02 | 1.6912 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm03 | 1.7467 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm04 | 1.4569 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm06 | 1.8393 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm07 | 1.6561 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm08 | 1.6987 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm09 | 1.1960 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm10 | 1.6763 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm11 | 1.3852 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm12 | 1.9648 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm13 | 1.5334 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm14 | 1.6263 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm15 | 1.7841 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm16 | 1.6803 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm17 | 1.7660 | - | - | - | - | (RL method)eklundnotes.md |
| L3 | Learning Placer | ibm18 | 1.7985 | - | - | - | - | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm01 | 1.1544 | 0.069 | 0.850 | 1.321 | 62.04 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm02 | 1.6974 | 0.080 | 0.902 | 2.334 | 66.76 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm03 | 1.7582 | 0.079 | 1.129 | 2.230 | 91.24 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm04 | 1.4279 | 0.070 | 0.801 | 1.914 | 107.16 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm06 | 2.0281 | 0.063 | 1.041 | 2.889 | 69.52 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm07 | 1.6603 | 0.065 | 0.975 | 2.216 | 112.88 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm08 | 1.6859 | 0.068 | 0.943 | 2.292 | 140.93 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm09 | 1.1986 | 0.058 | 0.893 | 1.388 | 109.42 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm10 | 1.6453 | 0.058 | 0.917 | 2.258 | 331.76 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm11 | 1.3709 | 0.055 | 0.991 | 1.642 | 138.93 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm12 | 1.9361 | 0.063 | 0.976 | 2.769 | 248.14 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm13 | 1.5649 | 0.054 | 1.018 | 2.004 | 65.76 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm14 | 1.6528 | 0.051 | 0.975 | 2.228 | 144.81 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm15 | 1.7820 | 0.058 | 1.074 | 2.375 | 111.23 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm16 | 1.6646 | 0.049 | 0.934 | 2.297 | 153.31 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm17 | 1.7720 | 0.053 | 0.954 | 2.485 | 210.96 | (RL method)eklundnotes.md |
| L4 | Learning Placer | ibm18 | 1.8005 | 0.053 | 1.042 | 2.453 | 110.45 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm01 | 1.1477 | 0.067 | 0.859 | 1.304 | 671.70 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm02 | 1.5618 | 0.076 | 0.690 | 2.282 | 388.45 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm03 | 1.7196 | 0.079 | 1.065 | 2.217 | 536.91 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm04 | 1.3948 | 0.069 | 0.780 | 1.871 | 710.03 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm06 | 1.6858 | 0.063 | 0.704 | 2.543 | 429.12 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm07 | 1.4806 | 0.065 | 0.806 | 2.025 | 612.93 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm08 | 1.7002 | 0.068 | 0.940 | 2.324 | 793.02 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm09 | 1.1825 | 0.057 | 0.895 | 1.355 | 770.13 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm10 | 1.6436 | 0.058 | 0.958 | 2.214 | 1476.55 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm11 | 1.3557 | 0.054 | 0.979 | 1.624 | 634.07 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm12 | 1.6348 | 0.060 | 0.750 | 2.400 | 971.40 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm13 | 1.5596 | 0.054 | 1.024 | 1.988 | 592.36 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm14 | 1.5885 | 0.051 | 0.946 | 2.129 | 1378.06 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm15 | 1.7827 | 0.057 | 1.069 | 2.381 | 2703.37 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm16 | 1.5033 | 0.049 | 0.811 | 2.098 | 1321.16 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm17 | 1.7580 | 0.052 | 0.945 | 2.467 | 1438.32 | (RL method)eklundnotes.md |
| L5 | Learning Placer | ibm18 | 1.7984 | 0.053 | 1.043 | 2.447 | 674.22 | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm01 | 1.1167 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm02 | 1.5970 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm03 | 1.3886 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm04 | 1.3923 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm06 | 1.6923 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm07 | 1.4864 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm08 | 1.5223 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm09 | 1.1035 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm10 | 1.3697 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm11 | 1.2315 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm12 | 1.6441 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm13 | 1.3902 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm14 | 1.6145 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm15 | 1.5939 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm16 | 1.5277 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm17 | 1.7493 | - | - | - | - | (RL method)eklundnotes.md |
| L6 | Learning Placer | ibm18 | 1.7871 | - | - | - | - | (RL method)eklundnotes.md |
| S1 | SA Placer | ibm01 | 1.2923 | 0.067 | 1.070 | 1.381 | 12.50 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm02 | 1.7310 | 0.075 | 0.901 | 2.411 | 7.32 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm03 | 1.4484 | 0.078 | 0.895 | 1.846 | 7.54 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm04 | 1.5738 | 0.069 | 1.041 | 1.968 | 37.37 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm06 | 1.7894 | 0.062 | 0.879 | 2.575 | 5.71 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm07 | 1.6716 | 0.065 | 1.057 | 2.157 | 8.68 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm08 | 1.6239 | 0.068 | 0.993 | 2.118 | 10.27 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm09 | 1.2075 | 0.057 | 0.954 | 1.347 | 8.92 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm10 | 1.4190 | 0.062 | 0.785 | 1.928 | 16.94 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm11 | 1.2881 | 0.054 | 0.965 | 1.504 | 8.39 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm12 | 1.6791 | 0.060 | 0.861 | 2.378 | 10.85 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm13 | 1.5025 | 0.053 | 1.014 | 1.884 | 9.35 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm14 | 1.6838 | 0.051 | 1.077 | 2.190 | 15.60 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm15 | 1.6916 | 0.058 | 1.045 | 2.223 | 13.67 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm16 | 1.6194 | 0.048 | 0.960 | 2.183 | 15.99 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm17 | 1.7679 | 0.053 | 0.983 | 2.447 | 16.00 | (SA + Analytical method) Omnellnotes.md |
| S1 | SA Placer | ibm18 | 1.8108 | 0.053 | 1.078 | 2.438 | 9.95 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm01 | 1.1362 | 0.069 | 0.827 | 1.307 | 43.79 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm02 | 1.5992 | 0.076 | 0.728 | 2.319 | 24.71 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm03 | 1.3975 | 0.079 | 0.771 | 1.865 | 35.23 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm04 | 1.3680 | 0.072 | 0.779 | 1.814 | 132.07 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm06 | 1.7328 | 0.063 | 0.764 | 2.576 | 35.86 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm07 | 1.4800 | 0.065 | 0.808 | 2.022 | 49.05 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm08 | 1.5224 | 0.070 | 0.840 | 2.066 | 45.89 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm09 | 1.1079 | 0.058 | 0.820 | 1.281 | 39.97 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm10 | 1.3624 | 0.065 | 0.707 | 1.889 | 81.26 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm11 | 1.2394 | 0.055 | 0.847 | 1.521 | 45.97 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm12 | 1.6420 | 0.061 | 0.750 | 2.412 | 77.21 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm13 | 1.3908 | 0.054 | 0.875 | 1.799 | 37.38 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm14 | 1.5988 | 0.051 | 0.948 | 2.146 | 91.98 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm15 | 1.5994 | 0.058 | 0.917 | 2.165 | 64.54 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm16 | 1.5308 | 0.048 | 0.848 | 2.116 | 70.92 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm17 | 1.7504 | 0.053 | 0.941 | 2.454 | 84.36 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | ibm18 | 1.7875 | 0.053 | 1.034 | 2.435 | 39.19 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm01 | 1.1281 | 0.069 | 0.822 | 1.296 | 22.34 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm02 | 1.6074 | 0.076 | 0.727 | 2.335 | 14.76 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm03 | 1.3752 | 0.079 | 0.771 | 1.865 | 35.23 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm04 | 1.3680 | 0.072 | 0.779 | 1.814 | 132.07 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm06 | 1.6848 | 0.063 | 0.764 | 2.576 | 35.86 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm07 | 1.4792 | 0.065 | 0.808 | 2.022 | 49.05 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm08 | 1.5132 | 0.070 | 0.840 | 2.066 | 45.89 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm09 | 1.1103 | 0.058 | 0.819 | 1.286 | 21.16 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm10 | 1.3703 | 0.065 | 0.707 | 1.889 | 81.26 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm11 | 1.2305 | 0.055 | 0.847 | 1.521 | 45.97 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm12 | 1.6458 | 0.061 | 0.750 | 2.412 | 77.21 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm13 | 1.3953 | 0.054 | 0.875 | 1.799 | 37.38 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm14 | 1.6007 | 0.051 | 0.948 | 2.146 | 91.98 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm15 | 1.5964 | 0.058 | 0.917 | 2.165 | 64.54 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm16 | 1.5238 | 0.048 | 0.848 | 2.116 | 70.92 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm17 | 1.7495 | 0.053 | 0.941 | 2.452 | 97.50 | (SA + Analytical method) Omnellnotes.md |
| S3 | SA Placer | ibm18 | 1.7872 | 0.053 | 1.034 | 2.434 | 49.45 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm01 | 1.1362 | 0.069 | 0.827 | 1.307 | 20.97 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm02 | 1.5992 | 0.076 | 0.728 | 2.319 | 12.81 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm03 | 1.3935 | 0.079 | 0.769 | 1.860 | 12.87 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm04 | 1.3954 | 0.072 | 0.812 | 1.834 | 15.81 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm06 | 1.6926 | 0.063 | 0.709 | 2.550 | 12.78 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm07 | 1.4801 | 0.065 | 0.808 | 2.022 | 18.14 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm08 | 1.5224 | 0.070 | 0.840 | 2.066 | 21.20 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm09 | 1.1088 | 0.058 | 0.820 | 1.283 | 17.91 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm10 | 1.3624 | 0.065 | 0.707 | 1.889 | 43.40 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm11 | 1.2390 | 0.055 | 0.847 | 1.520 | 22.32 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm12 | 1.6440 | 0.061 | 0.755 | 2.412 | 37.08 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm13 | 1.3908 | 0.054 | 0.875 | 1.799 | 23.51 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm14 | 1.5988 | 0.051 | 0.948 | 2.146 | 47.46 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm15 | 1.6034 | 0.058 | 0.917 | 2.173 | 34.65 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm16 | 1.5308 | 0.048 | 0.848 | 2.116 | 51.71 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm17 | 1.7504 | 0.053 | 0.941 | 2.454 | 79.25 | (SA + Analytical method) Omnellnotes.md |
| S4 | SA Placer | ibm18 | 1.7877 | 0.053 | 1.034 | 2.435 | 30.88 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm01 | 1.1362 | 0.069 | 0.827 | 1.307 | 20.73 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm02 | 1.5992 | 0.076 | 0.728 | 2.319 | 12.61 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm03 | 1.3935 | 0.079 | 0.769 | 1.860 | 12.65 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm04 | 1.3658 | 0.072 | 0.779 | 1.809 | 63.56 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm06 | 1.7103 | 0.063 | 0.751 | 2.543 | 16.31 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm07 | 1.4801 | 0.065 | 0.808 | 2.022 | 17.52 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm08 | 1.5224 | 0.070 | 0.840 | 2.066 | 20.45 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm09 | 1.1088 | 0.058 | 0.820 | 1.283 | 17.55 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm10 | 1.3624 | 0.065 | 0.707 | 1.889 | 41.57 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm11 | 1.2390 | 0.055 | 0.847 | 1.520 | 21.36 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm12 | 1.6440 | 0.061 | 0.755 | 2.412 | 33.85 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm13 | 1.3908 | 0.054 | 0.875 | 1.799 | 24.48 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm14 | 1.5988 | 0.051 | 0.948 | 2.146 | 45.08 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm15 | 1.6034 | 0.058 | 0.917 | 2.173 | 34.14 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm16 | 1.5308 | 0.048 | 0.848 | 2.116 | 46.92 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm17 | 1.7494 | 0.053 | 0.941 | 2.452 | 68.17 | (SA + Analytical method) Omnellnotes.md |
| S5 | SA Placer | ibm18 | 1.7865 | 0.053 | 1.034 | 2.433 | 38.66 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm01 | 1.1138 | 0.069 | 0.822 | 1.286 | 35.47 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm02 | 1.5983 | 0.076 | 0.727 | 2.315 | 46.75 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm03 | 1.3893 | 0.079 | 0.769 | 1.849 | 38.91 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm04 | 1.3811 | 0.072 | 0.779 | 1.839 | 103.13 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm06 | 1.6887 | 0.063 | 0.764 | 2.481 | 48.28 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm07 | 1.4894 | 0.065 | 0.808 | 2.040 | 76.26 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm08 | 1.5220 | 0.070 | 0.840 | 2.066 | 57.35 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm09 | 1.1022 | 0.058 | 0.819 | 1.269 | 37.50 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm10 | 1.3719 | 0.065 | 0.707 | 1.904 | 97.83 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm11 | 1.2274 | 0.055 | 0.847 | 1.498 | 68.25 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm12 | 1.6432 | 0.060 | 0.758 | 2.407 | 83.68 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm13 | 1.3905 | 0.054 | 0.873 | 1.799 | 66.67 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm14 | 1.6101 | 0.052 | 0.947 | 2.171 | 77.99 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm15 | 1.5952 | 0.058 | 0.916 | 2.158 | 65.49 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm16 | 1.5277 | 0.049 | 0.847 | 2.111 | 88.86 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm17 | 1.7478 | 0.053 | 0.941 | 2.448 | 128.40 | (SA + Analytical method) Omnellnotes.md |
| S6 | SA Placer | ibm18 | 1.7858 | 0.053 | 1.033 | 2.432 | 74.31 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm01 | 1.3845 | 0.103 | 1.017 | 1.546 | 10.07 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm02 | 1.7839 | 0.089 | 0.920 | 2.470 | 7.40 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm03 | 1.6980 | 0.095 | 1.054 | 2.152 | 7.75 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm04 | 1.6478 | 0.082 | 1.058 | 2.074 | 17.59 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm06 | 1.9669 | 0.071 | 0.989 | 2.803 | 5.16 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm07 | 1.7728 | 0.074 | 1.060 | 2.338 | 7.24 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm08 | 1.8069 | 0.076 | 1.085 | 2.377 | 16.97 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm09 | 1.3438 | 0.063 | 1.052 | 1.509 | 9.06 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm10 | 1.7661 | 0.066 | 1.003 | 2.398 | 55.92 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm11 | 1.4751 | 0.061 | 1.079 | 1.748 | 15.39 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm12 | 2.0331 | 0.068 | 1.027 | 2.904 | 40.36 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm13 | 1.6363 | 0.060 | 1.090 | 2.063 | 20.43 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm14 | 1.7747 | 0.055 | 1.078 | 2.362 | 99.19 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm15 | 1.8680 | 0.060 | 1.130 | 2.485 | 15.87 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm16 | 1.8090 | 0.051 | 1.056 | 2.461 | 34.04 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm17 | 1.8310 | 0.056 | 1.025 | 2.525 | 35.99 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | ibm18 | 1.8287 | 0.055 | 1.097 | 2.451 | 7.45 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm01 | 1.3845 | 0.103 | 1.017 | 1.546 | 16.87 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm02 | 1.7839 | 0.089 | 0.920 | 2.470 | 14.58 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm03 | 1.6980 | 0.095 | 1.054 | 2.152 | 15.07 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm04 | 1.6156 | 0.084 | 1.015 | 2.048 | 25.28 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm06 | 1.9669 | 0.071 | 0.989 | 2.803 | 18.40 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm07 | 1.7728 | 0.074 | 1.060 | 2.338 | 19.57 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm08 | 1.8769 | 0.076 | 1.121 | 2.482 | 28.42 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm09 | 1.3438 | 0.063 | 1.052 | 1.509 | 12.22 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm10 | 1.7747 | 0.066 | 1.025 | 2.393 | 108.72 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm11 | 1.5111 | 0.061 | 1.133 | 1.768 | 26.81 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm12 | 2.0684 | 0.068 | 1.076 | 2.926 | 49.58 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm13 | 1.7159 | 0.060 | 1.130 | 2.182 | 24.60 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm14 | 1.7998 | 0.055 | 1.103 | 2.387 | 113.30 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm15 | 1.8600 | 0.061 | 1.133 | 2.465 | 22.30 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm16 | 1.7438 | 0.051 | 1.044 | 2.342 | 47.24 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm17 | 1.8249 | 0.056 | 1.014 | 2.524 | 47.09 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | ibm18 | 1.8287 | 0.055 | 1.097 | 2.451 | 8.65 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm01 | 1.3280 | 0.073 | 1.034 | 1.476 | 34.21 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm02 | 1.8140 | 0.079 | 0.988 | 2.482 | 31.83 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm03 | 1.7517 | 0.081 | 1.130 | 2.210 | 32.76 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm04 | 1.6324 | 0.070 | 1.049 | 2.075 | 62.14 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm06 | 2.0549 | 0.064 | 1.073 | 2.909 | 23.25 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm07 | 1.7149 | 0.065 | 1.105 | 2.194 | 37.87 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm08 | 1.8806 | 0.069 | 1.160 | 2.464 | 69.87 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm09 | 1.3377 | 0.058 | 1.059 | 1.501 | 32.39 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm10 | 1.6909 | 0.057 | 0.993 | 2.275 | 153.98 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm11 | 1.4683 | 0.054 | 1.125 | 1.703 | 51.43 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm12 | 1.9980 | 0.062 | 1.072 | 2.799 | 125.00 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm13 | 1.6787 | 0.053 | 1.132 | 2.119 | 63.68 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm14 | 1.7218 | 0.051 | 1.103 | 2.240 | 280.44 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm15 | 1.8235 | 0.058 | 1.135 | 2.397 | 67.47 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm16 | 1.7702 | 0.048 | 1.067 | 2.377 | 141.99 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm17 | 1.7936 | 0.053 | 1.009 | 2.473 | 119.90 | (SA + Analytical method) Omnellnotes.md |
| A3 | Analytical Placer | ibm18 | 1.8347 | 0.053 | 1.109 | 2.454 | 33.59 | (SA + Analytical method) Omnellnotes.md |
| H1 | HybridPlacer | ibm01 | 1.2458 | 0.066 | 1.038 | 1.322 | 61.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm02 | 1.7708 | 0.080 | 0.989 | 2.393 | 75.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm03 | 1.7771 | 0.079 | 1.140 | 2.257 | 57.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm04 | 1.6132 | 0.069 | 1.061 | 2.028 | 57.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm06 | 2.0241 | 0.063 | 1.040 | 2.882 | 33.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm07 | 1.6769 | 0.064 | 1.075 | 2.150 | 53.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm08 | 1.8352 | 0.067 | 1.137 | 2.398 | 84.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm09 | 1.2520 | 0.057 | 0.999 | 1.391 | 66.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm10 | 1.6931 | 0.059 | 0.994 | 2.273 | 173.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm11 | 1.4224 | 0.054 | 1.091 | 1.647 | 63.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm12 | 1.9868 | 0.062 | 1.066 | 2.783 | 112.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm13 | 1.6506 | 0.053 | 1.117 | 2.079 | 64.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm14 | 1.7277 | 0.050 | 1.108 | 2.246 | 213.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm15 | 1.8122 | 0.058 | 1.139 | 2.371 | 86.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm16 | 1.7517 | 0.047 | 1.053 | 2.355 | 109.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm17 | 1.7813 | 0.052 | 1.004 | 2.454 | 123.00 | (Hybrid method)novaknotes.md |
| H1 | HybridPlacer | ibm18 | 1.8320 | 0.053 | 1.111 | 2.447 | 62.00 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm01 | 1.2477 | 0.066 | 1.034 | 1.330 | 11.28 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm02 | 1.7787 | 0.080 | 0.991 | 2.407 | 17.21 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm03 | 1.7756 | 0.079 | 1.140 | 2.254 | 12.08 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm04 | 1.6133 | 0.069 | 1.057 | 2.032 | 14.20 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm06 | 2.0305 | 0.063 | 1.053 | 2.882 | 8.20 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm07 | 1.6873 | 0.064 | 1.083 | 2.162 | 9.58 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm08 | 1.8384 | 0.067 | 1.135 | 2.407 | 14.96 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm09 | 1.2531 | 0.057 | 1.006 | 1.386 | 10.07 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm10 | 1.6922 | 0.059 | 0.996 | 2.269 | 47.64 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm11 | 1.4209 | 0.054 | 1.089 | 1.646 | 12.23 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm12 | 1.9854 | 0.062 | 1.067 | 2.780 | 29.96 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm13 | 1.6461 | 0.053 | 1.114 | 2.073 | 13.71 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm14 | 1.7276 | 0.050 | 1.113 | 2.241 | 42.95 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm15 | 1.8139 | 0.058 | 1.143 | 2.370 | 21.77 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm16 | 1.7381 | 0.047 | 1.050 | 2.331 | 23.69 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm17 | 1.7802 | 0.052 | 1.003 | 2.453 | 24.28 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | ibm18 | 1.8326 | 0.053 | 1.106 | 2.454 | 10.89 | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm01 | 1.2362 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm02 | 1.7591 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm03 | 1.7848 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm04 | 1.6224 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm06 | 2.0324 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm07 | 1.6864 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm08 | 1.8275 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm09 | 1.2505 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm10 | 1.6908 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm11 | 1.4337 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm12 | 1.9814 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm13 | 1.6423 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm14 | 1.7259 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm15 | 1.8156 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm16 | 1.7436 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm17 | 1.7761 | - | - | - | - | (Hybrid method)novaknotes.md |
| H3 | HybridPlacer | ibm18 | 1.8340 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm01 | 1.1782 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm02 | 1.6815 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm03 | 1.7161 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm04 | 1.4399 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm06 | 1.9132 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm07 | 1.5315 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm08 | 1.6530 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm09 | 1.1387 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm10 | 1.6386 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm11 | 1.3913 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm12 | 1.9018 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm13 | 1.5037 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm14 | 1.6309 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm15 | 1.6724 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm16 | 1.5795 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm17 | 1.7543 | - | - | - | - | (Hybrid method)novaknotes.md |
| H4 | HybridPlacer | ibm18 | 1.7968 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm01 | 1.1204 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm02 | 1.6241 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm03 | 1.6933 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm04 | 1.4383 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm06 | 1.8583 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm07 | 1.5253 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm08 | 1.6217 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm09 | 1.1247 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm10 | 1.6137 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm11 | 1.3907 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm12 | 1.7448 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm13 | 1.5358 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm14 | 1.6439 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm15 | 1.6654 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm16 | 1.5827 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm17 | 1.7531 | - | - | - | - | (Hybrid method)novaknotes.md |
| H5 | HybridPlacer | ibm18 | 1.7929 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm01 | 1.1167 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm02 | 1.5970 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm03 | 1.3886 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm04 | 1.3923 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm06 | 1.6923 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm07 | 1.4864 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm08 | 1.5223 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm09 | 1.1035 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm10 | 1.3697 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm11 | 1.2315 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm12 | 1.6441 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm13 | 1.3902 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm14 | 1.6145 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm15 | 1.5939 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm16 | 1.5277 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm17 | 1.7493 | - | - | - | - | (Hybrid method)novaknotes.md |
| H6 | HybridPlacer | ibm18 | 1.7871 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm01 | 1.1167 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm02 | 1.5970 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm03 | 1.3886 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm04 | 1.3923 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm06 | 1.6923 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm07 | 1.4864 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm08 | 1.5223 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm09 | 1.1035 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm10 | 1.3697 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm11 | 1.2315 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm12 | 1.6441 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm13 | 1.3902 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm14 | 1.6145 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm15 | 1.5939 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm16 | 1.5277 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm17 | 1.7493 | - | - | - | - | (Hybrid method)novaknotes.md |
| H7 | HybridPlacer | ibm18 | 1.7871 | - | - | - | - | (Hybrid method)novaknotes.md |
| V1 | SA V2 (Eklund) | ibm01 | 1.2564 | 0.088 | 0.927 | 1.410 | 80.75 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm02 | 1.6908 | 0.085 | 0.783 | 2.429 | 77.08 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm03 | 1.4712 | 0.086 | 0.829 | 1.940 | 62.24 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm04 | 1.5237 | 0.078 | 0.886 | 2.005 | 262.14 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm06 | 1.7958 | 0.067 | 0.788 | 2.669 | 58.71 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm07 | 1.5767 | 0.068 | 0.901 | 2.118 | 78.22 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm08 | 1.6220 | 0.074 | 0.935 | 2.161 | 95.94 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm09 | 1.2071 | 0.060 | 0.898 | 1.396 | 74.49 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm10 | 1.5070 | 0.076 | 0.746 | 2.117 | 324.50 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm11 | 1.3136 | 0.057 | 0.917 | 1.597 | 98.48 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm12 | 1.7353 | 0.066 | 0.799 | 2.540 | 285.93 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm13 | 1.4951 | 0.057 | 0.950 | 1.927 | 111.21 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm14 | 1.6527 | 0.054 | 0.998 | 2.201 | 371.52 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm15 | 1.6473 | 0.059 | 0.976 | 2.201 | 196.69 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm16 | 1.6145 | 0.050 | 0.910 | 2.218 | 407.15 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm17 | 1.7702 | 0.054 | 0.967 | 2.465 | 712.17 | (RL method)eklundnotes.md |
| V1 | SA V2 (Eklund) | ibm18 | 1.8050 | 0.053 | 1.058 | 2.446 | 181.13 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm01 | 1.2515 | 0.088 | 0.934 | 1.393 | 23.43 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm02 | 1.6672 | 0.086 | 0.768 | 2.395 | 14.96 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm03 | 1.5042 | 0.087 | 0.834 | 2.000 | 15.90 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm04 | 1.5122 | 0.078 | 0.882 | 1.987 | 97.32 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm06 | 1.7702 | 0.066 | 0.803 | 2.607 | 15.51 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm07 | 1.5731 | 0.068 | 0.892 | 2.119 | 21.37 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm08 | 1.6782 | 0.076 | 0.922 | 2.282 | 31.19 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm09 | 1.2071 | 0.061 | 0.902 | 1.390 | 22.93 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm10 | 1.5059 | 0.076 | 0.740 | 2.120 | 48.62 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm11 | 1.3292 | 0.058 | 0.916 | 1.627 | 28.50 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm12 | 1.7301 | 0.066 | 0.804 | 2.525 | 45.90 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm13 | 1.5038 | 0.057 | 0.946 | 1.948 | 28.28 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm14 | 1.6763 | 0.054 | 1.004 | 2.240 | 57.27 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm15 | 1.6607 | 0.059 | 0.998 | 2.205 | 46.20 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm16 | 1.6149 | 0.051 | 0.897 | 2.231 | 58.27 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm17 | 1.7691 | 0.054 | 0.969 | 2.461 | 74.23 | (RL method)eklundnotes.md |
| V2 | SA V2 (Eklund) | ibm18 | 1.8011 | 0.053 | 1.055 | 2.440 | 39.07 | (RL method)eklundnotes.md |
