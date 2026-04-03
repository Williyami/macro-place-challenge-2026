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
