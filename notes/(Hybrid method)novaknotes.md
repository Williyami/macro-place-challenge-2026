# Novak's Notes — HybridPlacer Results

## Method: Analytical -> SA Pipeline (`submissions/hybrid_placer.py`)

Three-phase hybrid approach:

### Phase 1: Analytical Placement (~1000 Adam steps, ~2s)
- Differentiable **log-sum-exp (LSE) HPWL** approximation for global wirelength
- **Gaussian density penalty** on a grid to spread macros and reduce congestion
- **Differentiable overlap penalty** using softplus to push macros apart
- Both density and overlap ramp up gradually (density at 30% progress, overlap at 20%)
- Adam optimizer, lr=0.5, gamma=5.0
- Purpose: captures global connectivity structure quickly, gives SA a much better starting point

### Phase 2: Legalization (~0.5s)
- Minimum-displacement overlap resolution (reuses `_legalize` from sa_placer)
- Places largest macros first, spirals outward to find non-overlapping positions

### Phase 3: SA Refinement (120K iterations, ~55-200s depending on benchmark size)
- Full net-HPWL simulated annealing (reuses `_sa_refine` from sa_placer)
- Three move types: SHIFT (50%), SWAP (30%), MOVE TOWARD NEIGHBOR (20%)
- Exponential cooling schedule: T_start=0.15*canvas, T_end=0.001*canvas
- Purpose: local optimization and fine-tuning from the analytical starting point

### Key Insight
The analytical phase gives SA a much better starting point than the hand-crafted initial.plc positions. SA then converges to a better local minimum because it's already near the basin of a good solution.

---

## Benchmark Results (2026-04-03)

All benchmarks: **VALID (zero overlaps)**
Total runtime: **1492.67s** (~25 min for all 17 benchmarks)

| Benchmark |  Proxy |    WL | Density | Congestion | SA Baseline | RePlAce |  vs SA | vs RePlAce |  Time |
|-----------|--------|-------|---------|------------|-------------|---------|--------|------------|-------|
| ibm01     | 1.2458 | 0.066 |   1.038 |      1.322 |      1.3166 |  0.9976 |  +5.4% |     -24.9% |   61s |
| ibm02     | 1.7708 | 0.080 |   0.989 |      2.393 |      1.9072 |  1.8370 |  +7.2% |      +3.6% |   75s |
| ibm03     | 1.7771 | 0.079 |   1.140 |      2.257 |      1.7401 |  1.3222 |  -2.1% |     -34.4% |   57s |
| ibm04     | 1.6132 | 0.069 |   1.061 |      2.028 |      1.5037 |  1.3024 |  -7.3% |     -23.9% |   57s |
| ibm06     | 2.0241 | 0.063 |   1.040 |      2.882 |      2.5057 |  1.6187 | +19.2% |     -25.0% |   33s |
| ibm07     | 1.6769 | 0.064 |   1.075 |      2.150 |      2.0229 |  1.4633 | +17.1% |     -14.6% |   53s |
| ibm08     | 1.8352 | 0.067 |   1.137 |      2.398 |      1.9239 |  1.4285 |  +4.6% |     -28.5% |   84s |
| ibm09     | 1.2520 | 0.057 |   0.999 |      1.391 |      1.3875 |  1.1194 |  +9.8% |     -11.8% |   66s |
| ibm10     | 1.6931 | 0.059 |   0.994 |      2.273 |      2.1108 |  1.5009 | +19.8% |     -12.8% |  173s |
| ibm11     | 1.4224 | 0.054 |   1.091 |      1.647 |      1.7111 |  1.1774 | +16.9% |     -20.8% |   63s |
| ibm12     | 1.9868 | 0.062 |   1.066 |      2.783 |      2.8261 |  1.7261 | +29.7% |     -15.1% |  112s |
| ibm13     | 1.6506 | 0.053 |   1.117 |      2.079 |      1.9141 |  1.3355 | +13.8% |     -23.6% |   64s |
| ibm14     | 1.7277 | 0.050 |   1.108 |      2.246 |      2.2750 |  1.5436 | +24.1% |     -11.9% |  213s |
| ibm15     | 1.8122 | 0.058 |   1.139 |      2.371 |      2.3000 |  1.5159 | +21.2% |     -19.5% |   86s |
| ibm16     | 1.7517 | 0.047 |   1.053 |      2.355 |      2.2337 |  1.4780 | +21.6% |     -18.5% |  109s |
| ibm17     | 1.7813 | 0.052 |   1.004 |      2.454 |      3.6726 |  1.6446 | +51.5% |      -8.3% |  123s |
| ibm18     | 1.8320 | 0.053 |   1.111 |      2.447 |      2.7755 |  1.7722 | +34.0% |      -3.4% |   62s |
| **AVG**   | **1.6972** | —  |       — |          — | **2.1251**  | **1.4578** | **+20.1%** | **-16.4%** | — |

## Summary

- **Average proxy cost: 1.6972** (all 17 IBM benchmarks)
- **Beats SA baseline** (2.1251) by **20.1%** on average — beats SA on **15 of 17** benchmarks
- **Behind RePlAce** (1.4578) by **16.4%** — but beats RePlAce on **1 benchmark** (ibm02: +3.6%)
- Wirelength is very low across all benchmarks (good HPWL optimization)
- Main gap vs RePlAce is in **congestion** — analytical spreading helps but not enough
- Zero overlaps on all benchmarks

## Leaderboard Position

| Rank | Method | Avg Proxy |
|------|--------|-----------|
| 1 | RePlAce (baseline) | 1.4578 |
| 2 | **HybridPlacer (ours)** | **1.6972** |
| 3 | Will (Partcl) | 1.5338 |
| 4 | SA (baseline) | 2.1251 |

Currently 3rd place — behind RePlAce and Will's seed placer (1.5338).

## Next Steps / Ideas to Improve
- Increase SA iterations (try 200K-500K) — more iterations should help close the gap
- Tune analytical phase: higher density_weight to reduce congestion penalty
- Add congestion-aware term to the analytical cost function
- Try force-directed soft macro update (run_fd=True) to improve soft macro positions
- Multi-start: run analytical phase with different random seeds, pick best for SA
- Density-aware SA moves: penalize moves that increase local density during SA
