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

*Results pending...*
