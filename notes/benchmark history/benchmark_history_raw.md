# Benchmark History Raw Data

Generated from the note files in this directory.

## Run-Level Records

| Run ID | Method | Date | Title | Scope | Benchmarks Logged | Plotted Proxy | Full-Suite Avg Proxy | Total Runtime (s) | Source |
|--------|--------|------|-------|-------|-------------------|---------------|----------------------|-------------------|--------|
| L1 | Learning Placer | 2026-04-03 | Stronger density + congestion proxy + target_util=0.6 | full_suite | 17 | 1.7117 | 1.7117 | 170.83 | (RL method)eklundnotes.md |
| L2 | Learning Placer | 2026-04-04 | Full suite rerun (`--no-media`) | full_suite | 17 | 1.7060 | 1.7060 | 457.74 | (RL method)eklundnotes.md |
| S1 | SA Placer | 2026-04-03 | SA Placer (PLACER_METHOD=sa) | full_suite | 17 | 1.5765 | 1.5765 | 215.00 | (SA + Analytical method) Omnellnotes.md |
| S2 | SA Placer | 2026-04-04 | SA Placer (PLACER_METHOD=sa) | full_suite | 17 | 1.4850 | 1.4850 | 999.38 | (SA + Analytical method) Omnellnotes.md |
| A1 | Analytical Placer | 2026-04-03 | Analytical Placer (PLACER_METHOD=analytical) | full_suite | 17 | 1.7310 | 1.7310 | 406.00 | (SA + Analytical method) Omnellnotes.md |
| A2 | Analytical Placer | 2026-04-04 | Analytical Placer (PLACER_METHOD=analytical) | full_suite | 17 | 1.7394 | 1.7394 | 598.68 | (SA + Analytical method) Omnellnotes.md |
| H1 | HybridPlacer | 2026-04-03 | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.6972 | 1.6972 | 1492.67 | (Hybrid method)novaknotes.md |
| H2 | HybridPlacer | 2026-04-04 | Analytical -> SA pipeline benchmark run | full_suite | 17 | 1.6977 | 1.6977 | 324.72 | (Hybrid method)novaknotes.md |

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
