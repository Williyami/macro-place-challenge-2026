# Novak's Notes — HybridPlacer Results

## Method: Analytical -> SA Pipeline (GPU-accelerated analytical phase) (`submissions/hybrid_placer.py`)

Three-phase hybrid approach:

### Phase 1: Analytical Placement (GPU-accelerated when CUDA is available, ~1000 Adam steps, ~2s)
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

## Summary (v1)

- **Average proxy cost: 1.6972** (all 17 IBM benchmarks)
- **Beats SA baseline** (2.1251) by **20.1%** on average — beats SA on **15 of 17** benchmarks
- **Behind RePlAce** (1.4578) by **16.4%** — but beats RePlAce on **1 benchmark** (ibm02: +3.6%)
- Wirelength is very low across all benchmarks (good HPWL optimization)
- Main gap vs RePlAce is in **congestion** — analytical spreading helps but not enough
- Zero overlaps on all benchmarks

---

## v2 Results (2026-04-04) — SA Reheating + 150K iters + 16x16 density grid

Changes from v1:
- SA reheating (reheat after 10K stagnant iters, 3x temperature boost)
- 150K SA iterations (up from 120K)
- 16x16 density grid in analytical phase (up from 8x8)
- Analytical phase unchanged (gamma=5, 1000 steps, lr=0.5)

| Benchmark |  v1 Proxy |  v2 Proxy | Change |
|-----------|-----------|-----------|--------|
| ibm01     |    1.2458 |    1.2686 |  +1.8% |
| ibm02     |    1.7708 |    1.7673 |  -0.2% |
| ibm03     |    1.7771 |    1.7794 |  +0.1% |
| ibm04     |    1.6132 |    1.6172 |  +0.2% |
| ibm06     |    2.0241 |    2.0451 |  +1.0% |
| ibm07     |    1.6769 |    1.6875 |  +0.6% |
| ibm08     |    1.8352 |    1.8351 |   0.0% |
| ibm09     |    1.2520 |    1.2523 |   0.0% |
| ibm10     |    1.6931 |    1.6915 |  -0.1% |
| ibm11     |    1.4224 |    1.4263 |  +0.3% |
| ibm12     |    1.9868 |    2.0034 |  +0.8% |
| ibm13     |    1.6506 |    1.6454 |  -0.3% |
| ibm14     |    1.7277 |    1.7243 |  -0.2% |
| ibm15     |    1.8122 |    1.8156 |  +0.2% |
| ibm16     |    1.7517 |    1.7554 |  +0.2% |
| ibm17     |    1.7813 |    1.7799 |  -0.1% |
| ibm18     |    1.8320 |    1.8325 |   0.0% |
| **AVG**   | **1.6972**| **1.7016**|  +0.3% |

**Conclusion**: v2 is essentially identical to v1 (+0.3%). SA reheating, more iterations,
and finer density grid did not meaningfully improve results. The bottleneck is **congestion**,
not HPWL optimization or local minima trapping.

Also tested and rejected:
- Gamma annealing (50→5): worse on ibm01 (1.30 vs 1.25), no improvement elsewhere
- Higher LR (5.0): overshoots, worse convergence
- Multi-start SA (2 runs): 2x slower, negligible improvement
- run_fd=True (soft macro FD): 24 min per benchmark, worse congestion
- analytical_placer.py's _build_net_tensors: different fixed-pin treatment, marginally worse

## Leaderboard Position

| Rank | Method | Avg Proxy |
|------|--------|-----------|
| 1 | RePlAce (baseline) | 1.4578 |
| 2 | Will (Partcl) | 1.5338 |
| 3 | **HybridPlacer (ours)** | **1.6972** |
| 4 | SA (baseline) | 2.1251 |

## Key Insight

The gap to RePlAce (~16%) is almost entirely in **congestion** (avg 2.25 vs ~1.5).
Our wirelength is already very competitive. To close the gap, we need:
- A congestion-aware objective (not just HPWL + density)
- Better cell-aware spreading that models routing tracks
- Possibly a completely different approach for congestion (e.g., routing-driven placement)

## Benchmark Results (2026-04-04, --no-media)

All benchmarks: **VALID (zero overlaps)**
Total runtime: **324.72s** (~5.4 min for all 17 benchmarks, run with `--no-media`)

| Benchmark |  Proxy |    WL | Density | Congestion | SA Baseline | RePlAce |  vs SA | vs RePlAce |    Time |
|-----------|--------|-------|---------|------------|-------------|---------|--------|------------|---------|
| ibm01     | 1.2477 | 0.066 |   1.034 |      1.330 |      1.3166 |  0.9976 |  +5.2% |     -25.1% |  11.28s |
| ibm02     | 1.7787 | 0.080 |   0.991 |      2.407 |      1.9072 |  1.8370 |  +6.7% |      +3.2% |  17.21s |
| ibm03     | 1.7756 | 0.079 |   1.140 |      2.254 |      1.7401 |  1.3222 |  -2.0% |     -34.3% |  12.08s |
| ibm04     | 1.6133 | 0.069 |   1.057 |      2.032 |      1.5037 |  1.3024 |  -7.3% |     -23.9% |  14.20s |
| ibm06     | 2.0305 | 0.063 |   1.053 |      2.882 |      2.5057 |  1.6187 | +19.0% |     -25.4% |   8.20s |
| ibm07     | 1.6873 | 0.064 |   1.083 |      2.162 |      2.0229 |  1.4633 | +16.6% |     -15.3% |   9.58s |
| ibm08     | 1.8384 | 0.067 |   1.135 |      2.407 |      1.9239 |  1.4285 |  +4.4% |     -28.7% |  14.96s |
| ibm09     | 1.2531 | 0.057 |   1.006 |      1.386 |      1.3875 |  1.1194 |  +9.7% |     -11.9% |  10.07s |
| ibm10     | 1.6922 | 0.059 |   0.996 |      2.269 |      2.1108 |  1.5009 | +19.8% |     -12.7% |  47.64s |
| ibm11     | 1.4209 | 0.054 |   1.089 |      1.646 |      1.7111 |  1.1774 | +17.0% |     -20.7% |  12.23s |
| ibm12     | 1.9854 | 0.062 |   1.067 |      2.780 |      2.8261 |  1.7261 | +29.7% |     -15.0% |  29.96s |
| ibm13     | 1.6461 | 0.053 |   1.114 |      2.073 |      1.9141 |  1.3355 | +14.0% |     -23.3% |  13.71s |
| ibm14     | 1.7276 | 0.050 |   1.113 |      2.241 |      2.2750 |  1.5436 | +24.1% |     -11.9% |  42.95s |
| ibm15     | 1.8139 | 0.058 |   1.143 |      2.370 |      2.3000 |  1.5159 | +21.1% |     -19.7% |  21.77s |
| ibm16     | 1.7381 | 0.047 |   1.050 |      2.331 |      2.2337 |  1.4780 | +22.2% |     -17.6% |  23.69s |
| ibm17     | 1.7802 | 0.052 |   1.003 |      2.453 |      3.6726 |  1.6446 | +51.5% |      -8.2% |  24.28s |
| ibm18     | 1.8326 | 0.053 |   1.106 |      2.454 |      2.7755 |  1.7722 | +34.0% |      -3.4% |  10.89s |
| **AVG**   | **1.6977** | —  |       — |          — | **2.1251**  | **1.4578** | **+20.1%** | **-16.5%** | — |

- Essentially unchanged from the previous hybrid run (1.6972 -> 1.6977)
- Same overall ranking profile, but much faster wall-clock runtime because media generation was disabled

---

## v3 Results (2026-04-06) — 300K SA iters + RUDY congestion helpers

Changes from v1:
- SA iterations increased to 300K (up from 150K)
- Added RUDY congestion estimation helpers (grid-based L-shaped routing + macro blockage)
- Congestion-aware SA function implemented but not used in final run (too slow per-move)
- Tested congestion-aware SA, density-tracking SA — both hurt more than helped
- The pure HPWL-only SA with more iterations was the best approach

| Benchmark | v3 Proxy |    WL | Density | Congestion | v1 Proxy | Change | SA Baseline | vs SA |  Time |
|-----------|----------|-------|---------|------------|----------|--------|-------------|-------|-------|
| ibm01     |   1.2362 | 0.065 |   1.035 |      1.308 |   1.2477 |  -0.9% |      1.3166 | +6.1% |   98s |
| ibm02     |   1.7591 | 0.079 |   0.982 |      2.378 |   1.7787 |  -1.1% |      1.9072 | +7.8% |  109s |
| ibm03     |   1.7819 | 0.078 |   1.147 |      2.261 |   1.7756 |  +0.4% |      1.7401 | -2.4% |  103s |
| ibm04     |   1.6181 | 0.068 |   1.060 |      2.040 |   1.6133 |  +0.3% |      1.5037 | -7.6% |  101s |
| ibm06     |   2.0324 | 0.063 |   1.050 |      2.889 |   2.0305 |  +0.1% |      2.5057 |+18.9% |   73s |
| ibm07     |   1.6864 | 0.064 |   1.084 |      2.160 |   1.6873 |  -0.1% |      2.0229 |+16.6% |  111s |
| ibm08     |   1.8342 | 0.067 |   1.141 |      2.394 |   1.8384 |  -0.2% |      1.9239 | +4.7% |  121s |
| ibm09     |   1.2505 | 0.057 |   1.000 |      1.387 |   1.2531 |  -0.2% |      1.3875 | +9.9% |  121s |
| ibm10     |   1.6951 | 0.059 |   0.994 |      2.279 |   1.6922 |  +0.2% |      2.1108 |+19.7% |  263s |
| ibm11     |   1.4258 | 0.054 |   1.086 |      1.658 |   1.4209 |  +0.3% |      1.7111 |+16.7% |  109s |
| ibm12     |   1.9837 | 0.062 |   1.066 |      2.778 |   1.9854 |  -0.1% |      2.8261 |+29.8% |  175s |
| ibm13     |   1.6342 | 0.053 |   1.116 |      2.048 |   1.6461 |  -0.7% |      1.9141 |+14.6% |  111s |
| ibm14     |   1.7251 | 0.050 |   1.107 |      2.242 |   1.7276 |  -0.1% |      2.2750 |+24.2% |  283s |
| ibm15     |   1.8139 | 0.057 |   1.142 |      2.371 |   1.8139 |   0.0% |      2.3000 |+21.1% |  137s |
| ibm16     |   1.7451 | 0.047 |   1.044 |      2.352 |   1.7381 |  +0.4% |      2.2337 |+21.9% |  158s |
| ibm17     |   1.7833 | 0.052 |   1.006 |      2.457 |   1.7802 |  +0.2% |      3.6726 |+51.4% |    — |
| ibm18     |   1.8340 | 0.053 |   1.109 |      2.453 |   1.8326 |  +0.1% |      2.7755 |+33.9% |   61s |
| **AVG**   | **1.6964** |    — |       — |          — | **1.6977**| **-0.1%** | **2.1251** |**+20.2%** | — |

### Conclusion

v3 (300K SA iters) is essentially tied with v1 (1.6964 vs 1.6977, -0.1%). 
More SA iterations help marginally on some benchmarks (ibm01: -0.9%, ibm02: -1.1%, ibm13: -0.7%) 
but hurt slightly on others, netting out to roughly zero improvement at ~2x runtime cost.

**Congestion-aware SA approaches tested and rejected:**
- RUDY incremental congestion in SA: too slow (~10x overhead per move), marginal congestion improvement
- Density-tracking SA: slow and didn't improve proxy cost
- Split approach (80% HPWL SA + 20% congestion SA): slight congestion improvement but offset by worse HPWL

**Key takeaway**: The bottleneck remains congestion, but SA-level congestion optimization
is too slow and imprecise to help. The evaluator's L-shaped routing + smoothing + abu(5%)
is hard to approximate incrementally. A fundamentally different approach (e.g., DREAMPlace-style
differentiable placement with congestion-aware density) would be needed to close the gap to RePlAce.

---

## Benchmark Results (2026-04-06)

All benchmarks: **VALID (zero overlaps)**
Total runtime: **3998.65s** (~66.6 min for all 17 benchmarks)

| Benchmark |  Proxy | SA Baseline | RePlAce |  vs SA | vs RePlAce | Overlaps |
|-----------|--------|-------------|---------|--------|------------|----------|
| ibm01     | 1.2362 |      1.3166 |  0.9976 |  +6.1% |     -23.9% |        0 |
| ibm02     | 1.7591 |      1.9072 |  1.8370 |  +7.8% |      +4.2% |        0 |
| ibm03     | 1.7848 |      1.7401 |  1.3222 |  -2.6% |     -35.0% |        0 |
| ibm04     | 1.6224 |      1.5037 |  1.3024 |  -7.9% |     -24.6% |        0 |
| ibm06     | 2.0324 |      2.5057 |  1.6187 | +18.9% |     -25.6% |        0 |
| ibm07     | 1.6864 |      2.0229 |  1.4633 | +16.6% |     -15.2% |        0 |
| ibm08     | 1.8275 |      1.9239 |  1.4285 |  +5.0% |     -27.9% |        0 |
| ibm09     | 1.2505 |      1.3875 |  1.1194 |  +9.9% |     -11.7% |        0 |
| ibm10     | 1.6908 |      2.1108 |  1.5009 | +19.9% |     -12.7% |        0 |
| ibm11     | 1.4337 |      1.7111 |  1.1774 | +16.2% |     -21.8% |        0 |
| ibm12     | 1.9814 |      2.8261 |  1.7261 | +29.9% |     -14.8% |        0 |
| ibm13     | 1.6423 |      1.9141 |  1.3355 | +14.2% |     -23.0% |        0 |
| ibm14     | 1.7259 |      2.2750 |  1.5436 | +24.1% |     -11.8% |        0 |
| ibm15     | 1.8156 |      2.3000 |  1.5159 | +21.1% |     -19.8% |        0 |
| ibm16     | 1.7436 |      2.2337 |  1.4780 | +21.9% |     -18.0% |        0 |
| ibm17     | 1.7761 |      3.6726 |  1.6446 | +51.6% |      -8.0% |        0 |
| ibm18     | 1.8340 |      2.7755 |  1.7722 | +33.9% |      -3.5% |        0 |
| **AVG**   | **1.6966** | **2.1251** | **1.4578** | **+20.2%** | **-16.4%** | **0** |

### Summary

- Average proxy cost improved slightly to **1.6966**
- Beats the SA baseline by **20.2%** on average with **zero overlaps**
- Still trails RePlAce by **16.4%** on average
- Runtime rose substantially to **3998.65s**, so the gain over prior runs is very small relative to the extra cost
