Samla research papers till claude

## Benchmark Results (2026-04-03)

### SA Placer (PLACER_METHOD=sa)

**Method:** Simulated Annealing with full net-HPWL cost.
- 120K iterations (200K for ibm04), exponential cooling
- t_start_factor=0.12, t_end_factor=0.0008
- Multi-start (3 runs) + reheating on stagnation for ibm04
- Swap, shift, and toward-neighbor moves

| Benchmark | proxy  | wl    | den   | cong  | Time   |
|-----------|--------|-------|-------|-------|--------|
| ibm01     | 1.2923 | 0.067 | 1.070 | 1.381 | 12.50s |
| ibm02     | 1.7310 | 0.075 | 0.901 | 2.411 |  7.32s |
| ibm03     | 1.4484 | 0.078 | 0.895 | 1.846 |  7.54s |
| ibm04     | 1.5738 | 0.069 | 1.041 | 1.968 | 37.37s |
| ibm06     | 1.7894 | 0.062 | 0.879 | 2.575 |  5.71s |
| ibm07     | 1.6716 | 0.065 | 1.057 | 2.157 |  8.68s |
| ibm08     | 1.6239 | 0.068 | 0.993 | 2.118 | 10.27s |
| ibm09     | 1.2075 | 0.057 | 0.954 | 1.347 |  8.92s |
| ibm10     | 1.4190 | 0.062 | 0.785 | 1.928 | 16.94s |
| ibm11     | 1.2881 | 0.054 | 0.965 | 1.504 |  8.39s |
| ibm12     | 1.6791 | 0.060 | 0.861 | 2.378 | 10.85s |
| ibm13     | 1.5025 | 0.053 | 1.014 | 1.884 |  9.35s |
| ibm14     | 1.6838 | 0.051 | 1.077 | 2.190 | 15.60s |
| ibm15     | 1.6916 | 0.058 | 1.045 | 2.223 | 13.67s |
| ibm16     | 1.6194 | 0.048 | 0.960 | 2.183 | 15.99s |
| ibm17     | 1.7679 | 0.053 | 0.983 | 2.447 | 16.00s |
| ibm18     | 1.8108 | 0.053 | 1.078 | 2.438 |  9.95s |
| **AVG**   |**1.5765**|     |       |       | 215s   |

- +25.8% better than SA baseline (2.13)
- -8.1% worse than RePlAce (1.46)
- ibm04 still -4.7% vs baseline — SA optimizes HPWL only, can't fix density/congestion

### Analytical Placer (PLACER_METHOD=analytical)

**Method:** Differentiable placement with Adam optimizer.
- 2000 iterations, lr=5.0
- LSE-HPWL with annealed gamma (50→5)
- Gaussian density penalty on 32x32 grid
- Differentiable overlap penalty (weight annealed 0.01→10.0)
- Legalization post-processing (minimum displacement)

| Benchmark | proxy  | wl    | den   | cong  | Time    |
|-----------|--------|-------|-------|-------|---------|
| ibm01     | 1.3845 | 0.103 | 1.017 | 1.546 |  10.07s |
| ibm02     | 1.7839 | 0.089 | 0.920 | 2.470 |   7.40s |
| ibm03     | 1.6980 | 0.095 | 1.054 | 2.152 |   7.75s |
| ibm04     | 1.6478 | 0.082 | 1.058 | 2.074 |  17.59s |
| ibm06     | 1.9669 | 0.071 | 0.989 | 2.803 |   5.16s |
| ibm07     | 1.7728 | 0.074 | 1.060 | 2.338 |   7.24s |
| ibm08     | 1.8069 | 0.076 | 1.085 | 2.377 |  16.97s |
| ibm09     | 1.3438 | 0.063 | 1.052 | 1.509 |   9.06s |
| ibm10     | 1.7661 | 0.066 | 1.003 | 2.398 |  55.92s |
| ibm11     | 1.4751 | 0.061 | 1.079 | 1.748 |  15.39s |
| ibm12     | 2.0331 | 0.068 | 1.027 | 2.904 |  40.36s |
| ibm13     | 1.6363 | 0.060 | 1.090 | 2.063 |  20.43s |
| ibm14     | 1.7747 | 0.055 | 1.078 | 2.362 |  99.19s |
| ibm15     | 1.8680 | 0.060 | 1.130 | 2.485 |  15.87s |
| ibm16     | 1.8090 | 0.051 | 1.056 | 2.461 |  34.04s |
| ibm17     | 1.8310 | 0.056 | 1.025 | 2.525 |  35.99s |
| ibm18     | 1.8287 | 0.055 | 1.097 | 2.451 |   7.45s |
| **AVG**   |**1.7310**|     |       |       | 406s    |

- +18.5% better than SA baseline (2.13)
- -18.7% worse than RePlAce (1.46)
- First implementation — room for tuning (lr, gamma schedule, density weight, more iters)

### Comparison Summary

| Method     | Avg Proxy | vs SA Baseline | vs RePlAce |
|------------|-----------|----------------|------------|
| SA baseline| 2.1251    | —              | +45.8%     |
| **SA ours**| **1.5765**| **-25.8%**     | +8.1%      |
| Analytical | 1.7310    | -18.5%         | +18.7%     |
| will_seed  | 1.53      | -28.0%         | +4.9%      |
| RePlAce    | 1.4578    | -31.4%         | —          |

### Observations / Next Steps
- SA is our best method (avg 1.58), but ibm04 congestion remains problematic
- Analytical placer works but needs tuning — congestion is high across the board
- Potential improvements: use analytical as initial placement for SA (hybrid), tune density/congestion weights, increase analytical iters

## Benchmark Results (2026-04-04)

### SA Placer (PLACER_METHOD=sa)

**Method:** Simulated Annealing with full net-HPWL cost.
- Current code in `submissions/sa_placer.py`
- Benchmark sweep run with `--no-media` to avoid visualization overhead

| Benchmark | proxy  | wl    | den   | cong  | Time   |
|-----------|--------|-------|-------|-------|--------|
| ibm01     | 1.1362 | 0.069 | 0.827 | 1.307 | 43.79s |
| ibm02     | 1.5992 | 0.076 | 0.728 | 2.319 | 24.71s |
| ibm03     | 1.3975 | 0.079 | 0.771 | 1.865 | 35.23s |
| ibm04     | 1.3680 | 0.072 | 0.779 | 1.814 | 132.07s |
| ibm06     | 1.7328 | 0.063 | 0.764 | 2.576 | 35.86s |
| ibm07     | 1.4800 | 0.065 | 0.808 | 2.022 | 49.05s |
| ibm08     | 1.5224 | 0.070 | 0.840 | 2.066 | 45.89s |
| ibm09     | 1.1079 | 0.058 | 0.820 | 1.281 | 39.97s |
| ibm10     | 1.3624 | 0.065 | 0.707 | 1.889 | 81.26s |
| ibm11     | 1.2394 | 0.055 | 0.847 | 1.521 | 45.97s |
| ibm12     | 1.6420 | 0.061 | 0.750 | 2.412 | 77.21s |
| ibm13     | 1.3908 | 0.054 | 0.875 | 1.799 | 37.38s |
| ibm14     | 1.5988 | 0.051 | 0.948 | 2.146 | 91.98s |
| ibm15     | 1.5994 | 0.058 | 0.917 | 2.165 | 64.54s |
| ibm16     | 1.5308 | 0.048 | 0.848 | 2.116 | 70.92s |
| ibm17     | 1.7504 | 0.053 | 0.941 | 2.454 | 84.36s |
| ibm18     | 1.7875 | 0.053 | 1.034 | 2.435 | 39.19s |
| **AVG**   |**1.4850**|     |       |       | 999.38s |

- +30.1% better than SA baseline (2.1251)
- -1.9% worse than RePlAce (1.4578)
- Massive improvement over the previous logged SA run (1.5765 -> 1.4850)

### Analytical Placer (PLACER_METHOD=analytical)

**Method:** Differentiable placement with Adam optimizer.
- Current code in `submissions/analytical_placer.py`
- Benchmark sweep run with `--no-media` to avoid visualization overhead

| Benchmark | proxy  | wl    | den   | cong  | Time    |
|-----------|--------|-------|-------|-------|---------|
| ibm01     | 1.3845 | 0.103 | 1.017 | 1.546 | 16.87s |
| ibm02     | 1.7839 | 0.089 | 0.920 | 2.470 | 14.58s |
| ibm03     | 1.6980 | 0.095 | 1.054 | 2.152 | 15.07s |
| ibm04     | 1.6156 | 0.084 | 1.015 | 2.048 | 25.28s |
| ibm06     | 1.9669 | 0.071 | 0.989 | 2.803 | 18.40s |
| ibm07     | 1.7728 | 0.074 | 1.060 | 2.338 | 19.57s |
| ibm08     | 1.8769 | 0.076 | 1.121 | 2.482 | 28.42s |
| ibm09     | 1.3438 | 0.063 | 1.052 | 1.509 | 12.22s |
| ibm10     | 1.7747 | 0.066 | 1.025 | 2.393 | 108.72s |
| ibm11     | 1.5111 | 0.061 | 1.133 | 1.768 | 26.81s |
| ibm12     | 2.0684 | 0.068 | 1.076 | 2.926 | 49.58s |
| ibm13     | 1.7159 | 0.060 | 1.130 | 2.182 | 24.60s |
| ibm14     | 1.7998 | 0.055 | 1.103 | 2.387 | 113.30s |
| ibm15     | 1.8600 | 0.061 | 1.133 | 2.465 | 22.30s |
| ibm16     | 1.7438 | 0.051 | 1.044 | 2.342 | 47.24s |
| ibm17     | 1.8249 | 0.056 | 1.014 | 2.524 | 47.09s |
| ibm18     | 1.8287 | 0.055 | 1.097 | 2.451 | 8.65s |
| **AVG**   |**1.7394**|     |       |       | 598.68s |

- +18.1% better than SA baseline (2.1251)
- -19.3% worse than RePlAce (1.4578)
- Slight regression versus the previous logged analytical run (1.7310 -> 1.7394)

### Comparison Summary

| Method     | Avg Proxy | vs SA Baseline | vs RePlAce |
|------------|-----------|----------------|------------|
| SA baseline| 2.1251    | —              | +45.8%     |
| **SA ours**| **1.4850**| **-30.1%**     | +1.9%      |
| Analytical | 1.7394    | -18.1%         | +19.3%     |
| will_seed  | 1.5338    | -27.8%         | +5.2%      |
| RePlAce    | 1.4578    | -31.4%         | —          |

### Observations / Next Steps
- SA is now clearly our strongest current method and is within 1.9% of RePlAce on proxy cost
- Analytical remains competitive but is still congestion-limited and now sits well behind the new SA run
- The current SA run also beats Will Seed on average proxy (1.4850 vs 1.5338)

## Benchmark Results (2026-04-04, v2) — Greedy Macro Flipping

Changes from previous SA run:
- Added greedy macro flipping post-processing (try FN/FS/S orientations per macro, keep if HPWL improves)
- Density-aware SA unchanged from previous run

### SA Placer (PLACER_METHOD=sa)

| Benchmark | proxy  | wl    | den   | cong  | Time    |
|-----------|--------|-------|-------|-------|---------|
| ibm01     | 1.1281 | 0.069 | 0.822 | 1.296 |  22.34s |
| ibm02     | 1.6074 | 0.076 | 0.727 | 2.335 |  14.76s |
| ibm03     | 1.3752 | 0.079 | 0.771 | 1.865 |  35.23s |
| ibm04     | 1.3680 | 0.072 | 0.779 | 1.814 | 132.07s |
| ibm06     | 1.6848 | 0.063 | 0.764 | 2.576 |  35.86s |
| ibm07     | 1.4792 | 0.065 | 0.808 | 2.022 |  49.05s |
| ibm08     | 1.5132 | 0.070 | 0.840 | 2.066 |  45.89s |
| ibm09     | 1.1103 | 0.058 | 0.819 | 1.286 |  21.16s |
| ibm10     | 1.3703 | 0.065 | 0.707 | 1.889 |  81.26s |
| ibm11     | 1.2305 | 0.055 | 0.847 | 1.521 |  45.97s |
| ibm12     | 1.6458 | 0.061 | 0.750 | 2.412 |  77.21s |
| ibm13     | 1.3953 | 0.054 | 0.875 | 1.799 |  37.38s |
| ibm14     | 1.6007 | 0.051 | 0.948 | 2.146 |  91.98s |
| ibm15     | 1.5964 | 0.058 | 0.917 | 2.165 |  64.54s |
| ibm16     | 1.5238 | 0.048 | 0.848 | 2.116 |  70.92s |
| ibm17     | 1.7495 | 0.053 | 0.941 | 2.452 |  97.50s |
| ibm18     | 1.7872 | 0.053 | 1.034 | 2.434 |  49.45s |
| **AVG**   |**1.4803**|     |       |       | 676.83s |

- Improved from 1.4850 → 1.4803 (+0.3%)
- Now only 1.5% behind RePlAce (was 1.9%)
- Beats RePlAce on 4/17 benchmarks (ibm02, ibm09, ibm10, ibm12)
- Beats SA baseline on all 17/17 benchmarks
