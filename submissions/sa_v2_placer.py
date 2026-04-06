"""
SA V2 Placer (Eklund) — Improved Simulated Annealing.

Key improvements over SA V1:
  1. Net HPWL caching: cache per-net HPWL, only recompute affected nets on moves.
  2. Adaptive move probabilities: track accept rates per move type and bias toward
     successful types.
  3. Late-acceptance hill climbing (LAHC) combined with SA for better escapes.
  4. Size-aware move radius: shift distance scales with macro size relative to canvas.
  5. Congestion-aware cost: lightweight congestion penalty in SA acceptance.
  6. Improved initial placement via force-directed warm-start option.
  7. Greedy local search post-processing (steepest descent after SA cools).
  8. Multi-start with diverse seeds and perturbation strategies.

Usage:
    PLACER_METHOD=sa_v2 uv run evaluate submissions/placer.py --all
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _sav2_device() -> torch.device:
    """Prefer CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from submissions.base import BasePlacer


# ── benchmark loading (shared with sa_placer) ──────────────────────────────

def _load_plc(name: str):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / d / "netlist" / "output_CT_Grouping"
        )
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"), str(base / "initial.plc")
            )
            return plc
    return None


# ── net extraction ─────────────────────────────────────────────────────────

def _extract_nets(benchmark: Benchmark, plc):
    """Build net data structures for fast delta-HPWL computation."""
    n_hard = benchmark.num_hard_macros

    hard_name_to_idx = {}
    for tensor_i, plc_i in enumerate(plc.hard_macro_indices):
        name = plc.modules_w_pins[plc_i].get_name()
        hard_name_to_idx[name] = tensor_i

    soft_name_to_pos = {}
    for plc_i in plc.soft_macro_indices:
        mod = plc.modules_w_pins[plc_i]
        soft_name_to_pos[mod.get_name()] = mod.get_pos()

    nets = []
    macro_to_nets = [[] for _ in range(n_hard)]

    INF = float("inf")

    for driver_name, sink_names in plc.nets.items():
        if driver_name not in plc.mod_name_to_indices:
            continue

        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        driver_obj = plc.modules_w_pins[driver_plc_idx]
        weight = driver_obj.get_weight()

        hard_idx_list = []
        hard_ox_list = []
        hard_oy_list = []
        fxmin, fxmax = INF, -INF
        fymin, fymax = INF, -INF

        for pin_name in [driver_name] + sink_names:
            if pin_name not in plc.mod_name_to_indices:
                continue
            pin_plc_idx = plc.mod_name_to_indices[pin_name]
            pin_obj = plc.modules_w_pins[pin_plc_idx]
            pin_type = pin_obj.get_type()

            if pin_type == "PORT":
                x, y = pin_obj.get_pos()
                fxmin = min(fxmin, x); fxmax = max(fxmax, x)
                fymin = min(fymin, y); fymax = max(fymax, y)

            elif pin_type == "MACRO_PIN":
                parent_name = pin_obj.get_macro_name()

                if parent_name in hard_name_to_idx:
                    macro_idx = hard_name_to_idx[parent_name]
                    ox, oy = pin_obj.get_offset()
                    hard_idx_list.append(macro_idx)
                    hard_ox_list.append(ox)
                    hard_oy_list.append(oy)

                elif parent_name in soft_name_to_pos:
                    sx, sy = soft_name_to_pos[parent_name]
                    ox, oy = pin_obj.get_offset()
                    px, py = sx + ox, sy + oy
                    fxmin = min(fxmin, px); fxmax = max(fxmax, px)
                    fymin = min(fymin, py); fymax = max(fymax, py)

        if not hard_idx_list:
            continue
        total_pins = len(hard_idx_list) + (0 if fxmin == INF else 1)
        if total_pins < 2 and fxmin == INF:
            continue

        net_idx = len(nets)
        nets.append({
            "weight": weight,
            "hard_idx": np.array(hard_idx_list, dtype=np.int32),
            "hard_ox":  np.array(hard_ox_list,  dtype=np.float64),
            "hard_oy":  np.array(hard_oy_list,  dtype=np.float64),
            "fxmin": fxmin,
            "fxmax": fxmax,
            "fymin": fymin,
            "fymax": fymax,
        })
        for mid in hard_idx_list:
            macro_to_nets[mid].append(net_idx)

    return nets, macro_to_nets


# ── per-net HPWL helpers ───────────────────────────────────────────────────

def _net_hpwl(net: dict, pos: np.ndarray) -> float:
    hx = pos[net["hard_idx"], 0] + net["hard_ox"]
    hy = pos[net["hard_idx"], 1] + net["hard_oy"]
    xmin = min(float(hx.min()), net["fxmin"])
    xmax = max(float(hx.max()), net["fxmax"])
    ymin = min(float(hy.min()), net["fymin"])
    ymax = max(float(hy.max()), net["fymax"])
    return net["weight"] * ((xmax - xmin) + (ymax - ymin))


def _net_hpwl_override(net: dict, pos: np.ndarray,
                       m: int, new_x: float, new_y: float) -> float:
    idx = net["hard_idx"]
    ox  = net["hard_ox"]
    oy  = net["hard_oy"]

    hx = pos[idx, 0] + ox
    hy = pos[idx, 1] + oy

    mask = (idx == m)
    if mask.any():
        hx = hx.copy()
        hy = hy.copy()
        hx[mask] = new_x + ox[mask]
        hy[mask] = new_y + oy[mask]

    xmin = min(float(hx.min()), net["fxmin"])
    xmax = max(float(hx.max()), net["fxmax"])
    ymin = min(float(hy.min()), net["fymin"])
    ymax = max(float(hy.max()), net["fymax"])
    return net["weight"] * ((xmax - xmin) + (ymax - ymin))


# ── cached HPWL management ────────────────────────────────────────────────

def _build_net_hpwl_cache(pos: np.ndarray, nets: list) -> np.ndarray:
    """Compute HPWL for every net and return as array."""
    cache = np.empty(len(nets))
    for i, net in enumerate(nets):
        cache[i] = _net_hpwl(net, pos)
    return cache


def _delta_hpwl_cached(m: int, new_x: float, new_y: float,
                       pos: np.ndarray, nets: list, macro_to_nets: list,
                       net_cache: np.ndarray) -> tuple[float, list]:
    """Compute delta HPWL using cached values. Returns (delta, list of (net_idx, new_hpwl))."""
    delta = 0.0
    updates = []
    for net_idx in macro_to_nets[m]:
        net = nets[net_idx]
        new_val = _net_hpwl_override(net, pos, m, new_x, new_y)
        delta += new_val - net_cache[net_idx]
        updates.append((net_idx, new_val))
    return delta, updates


def _compute_total_hpwl_from_cache(net_cache: np.ndarray) -> float:
    return float(net_cache.sum())


# ── density grid helpers ──────────────────────────────────────────────────

def _macro_cell_overlaps(mx, my, mw, mh, grid_col, grid_row, gw, gh):
    x_lo, x_hi = mx - mw / 2, mx + mw / 2
    y_lo, y_hi = my - mh / 2, my + mh / 2
    c_min = max(0, int(x_lo / gw))
    c_max = min(grid_col - 1, int(x_hi / gw))
    r_min = max(0, int(y_lo / gh))
    r_max = min(grid_row - 1, int(y_hi / gh))
    result = []
    for r in range(r_min, r_max + 1):
        cy_lo, cy_hi = r * gh, (r + 1) * gh
        ov_y = min(y_hi, cy_hi) - max(y_lo, cy_lo)
        if ov_y <= 0:
            continue
        for c in range(c_min, c_max + 1):
            cx_lo, cx_hi = c * gw, (c + 1) * gw
            ov_x = min(x_hi, cx_hi) - max(x_lo, cx_lo)
            if ov_x > 0:
                result.append((r * grid_col + c, ov_x * ov_y))
    return result


def _build_density_grid(pos, sizes, n_hard, grid_col, grid_row, gw, gh,
                        benchmark=None):
    grid = np.zeros(grid_col * grid_row)
    for i in range(n_hard):
        for ci, area in _macro_cell_overlaps(
            pos[i, 0], pos[i, 1], sizes[i, 0], sizes[i, 1],
            grid_col, grid_row, gw, gh,
        ):
            grid[ci] += area
    if benchmark is not None:
        all_sizes = benchmark.macro_sizes.numpy()
        all_pos = benchmark.macro_positions.numpy()
        for i in range(n_hard, len(all_sizes)):
            for ci, area in _macro_cell_overlaps(
                all_pos[i, 0], all_pos[i, 1],
                all_sizes[i, 0], all_sizes[i, 1],
                grid_col, grid_row, gw, gh,
            ):
                grid[ci] += area
    return grid


def _density_cost_from_grid(grid_occupied, grid_area, n_cells):
    densities = grid_occupied / grid_area
    nonzero = sorted([float(d) for d in densities if d > 0], reverse=True)
    if not nonzero:
        return 0.0
    cnt = max(1, n_cells // 10)
    return 0.5 * sum(nonzero[:cnt]) / cnt


# ── legalization ───────────────────────────────────────────────────────────

def _legalize(pos, movable, sizes, half_w, half_h, cw, ch, n, sep_x, sep_y):
    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    gap = 0.05

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            conflict = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
            conflict[idx] = False
            if not conflict.any():
                placed[idx] = True
                continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best_p = legal[idx].copy()
        best_d = float("inf")

        for r in range(1, 150):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])
                    if placed.any():
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        conflict = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
                        conflict[idx] = False
                        if conflict.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        found = True
            if found:
                break

        legal[idx] = best_p
        placed[idx] = True

    return legal


# ── SA V2 refinement ─────────────────────────────────────────────────────

def _sa_v2_refine(
    pos: np.ndarray,
    nets: list,
    macro_to_nets: list,
    neighbors: list,
    movable: np.ndarray,
    sizes: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    cw: float,
    ch: float,
    max_iters: int,
    seed: int,
    snapshot_interval: int = 0,
    snapshot_callback=None,
    trace_interval: int = 0,
    trace_callback=None,
    t_start_factor: float = 0.15,
    t_end_factor: float = 0.001,
    reheat_threshold: int = 0,
    reheat_factor: float = 3.0,
    density_weight: float = 0.0,
    grid_col: int = 0,
    grid_row: int = 0,
    benchmark=None,
    lahc_length: int = 200,
    greedy_tail_frac: float = 0.05,
    adaptive_moves: bool = True,
) -> np.ndarray:
    """SA V2 loop with cached HPWL, adaptive moves, and LAHC.

    New parameters
    --------------
    lahc_length : int
        Late Acceptance Hill Climbing history length. During the LAHC phase
        a move is accepted if cost < history[step % length] even if SA would
        reject it. Set 0 to disable.
    greedy_tail_frac : float
        Fraction of iterations at the end devoted to greedy-only (T=0) search.
    adaptive_moves : bool
        If True, adapt move-type probabilities based on acceptance rates.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos

    pos = pos.copy()
    n_hard = len(movable)
    GAP = 0.05

    # Pre-compute macro areas for weighted random selection (prefer smaller macros
    # early since large macros constrain the space)
    macro_areas = sizes[movable_idx, 0] * sizes[movable_idx, 1]
    # Inverse-area weights for selection (smaller macros more likely)
    inv_area_weights = 1.0 / (macro_areas + 1e-10)
    inv_area_weights /= inv_area_weights.sum()

    def check_overlap(idx: int) -> bool:
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        ov = (dx < sep_x[idx] + GAP) & (dy < sep_y[idx] + GAP)
        ov[idx] = False
        return bool(ov.any())

    # Build HPWL cache
    net_cache = _build_net_hpwl_cache(pos, nets)
    current_hpwl = float(net_cache.sum())
    best_pos = pos.copy()
    best_hpwl = current_hpwl

    T_start = max(cw, ch) * t_start_factor
    T_end   = max(cw, ch) * t_end_factor

    # Density tracking
    use_density = density_weight > 0 and grid_col > 0 and grid_row > 0
    if use_density:
        gw = cw / grid_col
        gh = ch / grid_row
        grid_area = gw * gh
        n_grid = grid_col * grid_row
        grid_occupied = _build_density_grid(
            pos, sizes, n_hard, grid_col, grid_row, gw, gh, benchmark,
        )
        current_density = _density_cost_from_grid(grid_occupied, grid_area, n_grid)
        current_cost = current_hpwl + density_weight * current_density
        best_cost = current_cost
    else:
        current_density = 0.0
        current_cost = current_hpwl
        best_cost = current_hpwl

    # LAHC history
    use_lahc = lahc_length > 0
    if use_lahc:
        lahc_history = np.full(lahc_length, current_cost)

    # Adaptive move probabilities
    # Move types: 0=SHIFT, 1=SWAP, 2=TOWARD_NEIGHBOR, 3=MIRROR_SHIFT
    n_move_types = 4
    move_accepts = np.ones(n_move_types)  # Laplace smoothing
    move_attempts = np.ones(n_move_types) * 2
    move_probs = np.array([0.45, 0.25, 0.20, 0.10])  # initial

    # Reheating state
    steps_since_improvement = 0

    # Greedy tail boundary
    greedy_start = int(max_iters * (1.0 - greedy_tail_frac))

    if trace_callback is not None:
        trace_callback({
            "step": 0,
            "temperature": T_start,
            "current_hpwl": current_hpwl,
            "best_hpwl": best_hpwl,
        })

    for step in range(max_iters):
        frac = step / max_iters

        # Temperature schedule
        if step >= greedy_start:
            T = 0.0  # greedy tail
        else:
            sa_frac = step / greedy_start
            T = T_start * (T_end / T_start) ** sa_frac

        # Reheat if stagnating
        if reheat_threshold > 0 and steps_since_improvement >= reheat_threshold:
            T = min(T * reheat_factor, T_start * 0.5)
            steps_since_improvement = 0

        # Select move type
        if adaptive_moves and step > 500 and step % 200 == 0:
            rates = move_accepts / move_attempts
            move_probs = rates / rates.sum()
            # Blend with uniform to maintain exploration
            move_probs = 0.8 * move_probs + 0.2 * np.ones(n_move_types) / n_move_types

        r = rng.random()
        cum = 0.0
        move_type = 0
        for mt in range(n_move_types):
            cum += move_probs[mt]
            if r < cum:
                move_type = mt
                break

        # Choose macro — mix uniform and inverse-area weighted
        if rng.random() < 0.3:
            i_loc = np_rng.choice(len(movable_idx), p=inv_area_weights)
        else:
            i_loc = rng.randrange(len(movable_idx))
        i = movable_idx[i_loc]
        old_x, old_y = pos[i, 0], pos[i, 1]

        move_attempts[move_type] += 1

        if move_type == 0:
            # ── SHIFT ────────────────────────────────────────────────────
            # Size-aware shift: smaller macros get relatively larger moves
            macro_diag = math.sqrt(sizes[i, 0]**2 + sizes[i, 1]**2)
            canvas_diag = math.sqrt(cw**2 + ch**2)
            base_shift = T * (0.3 + 0.7 * (1 - frac))
            # Scale inversely with macro size (small macros explore more)
            size_factor = 1.0 + 0.5 * (1.0 - macro_diag / canvas_diag)
            shift = base_shift * size_factor

            new_x = float(np.clip(old_x + rng.gauss(0, shift), half_w[i], cw - half_w[i]))
            new_y = float(np.clip(old_y + rng.gauss(0, shift), half_h[i], ch - half_h[i]))

            pos[i, 0] = new_x; pos[i, 1] = new_y
            if check_overlap(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue
            pos[i, 0] = old_x; pos[i, 1] = old_y

            delta_hpwl, hpwl_updates = _delta_hpwl_cached(
                i, new_x, new_y, pos, nets, macro_to_nets, net_cache)
            delta = delta_hpwl

            old_cells = new_cells = None
            if use_density:
                old_cells = _macro_cell_overlaps(old_x, old_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                new_cells = _macro_cell_overlaps(new_x, new_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                for ci, a in old_cells: grid_occupied[ci] -= a
                for ci, a in new_cells: grid_occupied[ci] += a
                new_dens = _density_cost_from_grid(grid_occupied, grid_area, n_grid)
                delta += density_weight * (new_dens - current_density)

            new_cost = current_cost + delta

            # Acceptance: SA + LAHC
            accept = False
            if delta < 0:
                accept = True
            elif T > 0 and rng.random() < math.exp(-delta / max(T, 1e-12)):
                accept = True
            elif use_lahc and new_cost < lahc_history[step % lahc_length]:
                accept = True

            if accept:
                pos[i, 0] = new_x; pos[i, 1] = new_y
                current_hpwl += delta_hpwl
                for ni, val in hpwl_updates:
                    net_cache[ni] = val
                if use_density:
                    current_density = new_dens
                current_cost = new_cost
                move_accepts[move_type] += 1

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                if use_density:
                    for ci, a in new_cells: grid_occupied[ci] -= a
                    for ci, a in old_cells: grid_occupied[ci] += a
                steps_since_improvement += 1

            if use_lahc:
                lahc_history[step % lahc_length] = current_cost

        elif move_type == 1:
            # ── SWAP ─────────────────────────────────────────────────────
            if neighbors[i] and rng.random() < 0.7:
                cands = [j for j in neighbors[i] if movable[j]]
                j = rng.choice(cands) if cands else rng.choice(movable_idx)
            else:
                j = rng.choice(movable_idx)
            if i == j:
                continue

            old_jx, old_jy = pos[j, 0], pos[j, 1]
            new_ix = float(np.clip(old_jx, half_w[i], cw - half_w[i]))
            new_iy = float(np.clip(old_jy, half_h[i], ch - half_h[i]))
            new_jx = float(np.clip(old_x,  half_w[j], cw - half_w[j]))
            new_jy = float(np.clip(old_y,  half_h[j], ch - half_h[j]))

            pos[i, 0] = new_ix; pos[i, 1] = new_iy
            pos[j, 0] = new_jx; pos[j, 1] = new_jy

            if check_overlap(i) or check_overlap(j):
                pos[i, 0] = old_x;  pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                continue

            # Delta HPWL for swap using cache
            affected = list(set(macro_to_nets[i]) | set(macro_to_nets[j]))
            new_hpwl_aff = sum(_net_hpwl(nets[k], pos) for k in affected)
            old_hpwl_aff = sum(net_cache[k] for k in affected)
            delta_hpwl = new_hpwl_aff - old_hpwl_aff

            # Revert positions for density check
            pos[i, 0] = old_x;  pos[i, 1] = old_y
            pos[j, 0] = old_jx; pos[j, 1] = old_jy
            delta = delta_hpwl

            old_cells_i = old_cells_j = new_cells_i = new_cells_j = None
            if use_density:
                old_cells_i = _macro_cell_overlaps(old_x, old_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                old_cells_j = _macro_cell_overlaps(old_jx, old_jy, sizes[j, 0], sizes[j, 1], grid_col, grid_row, gw, gh)
                new_cells_i = _macro_cell_overlaps(new_ix, new_iy, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                new_cells_j = _macro_cell_overlaps(new_jx, new_jy, sizes[j, 0], sizes[j, 1], grid_col, grid_row, gw, gh)
                for ci, a in old_cells_i: grid_occupied[ci] -= a
                for ci, a in old_cells_j: grid_occupied[ci] -= a
                for ci, a in new_cells_i: grid_occupied[ci] += a
                for ci, a in new_cells_j: grid_occupied[ci] += a
                new_dens = _density_cost_from_grid(grid_occupied, grid_area, n_grid)
                delta += density_weight * (new_dens - current_density)

            new_cost = current_cost + delta

            accept = False
            if delta < 0:
                accept = True
            elif T > 0 and rng.random() < math.exp(-delta / max(T, 1e-12)):
                accept = True
            elif use_lahc and new_cost < lahc_history[step % lahc_length]:
                accept = True

            if accept:
                pos[i, 0] = new_ix; pos[i, 1] = new_iy
                pos[j, 0] = new_jx; pos[j, 1] = new_jy
                current_hpwl += delta_hpwl
                # Update cache for all affected nets (need to recompute in swapped state)
                for k in affected:
                    net_cache[k] = _net_hpwl(nets[k], pos)
                if use_density:
                    current_density = new_dens
                current_cost = new_cost
                move_accepts[move_type] += 1

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                if use_density:
                    for ci, a in new_cells_i: grid_occupied[ci] -= a
                    for ci, a in new_cells_j: grid_occupied[ci] -= a
                    for ci, a in old_cells_i: grid_occupied[ci] += a
                    for ci, a in old_cells_j: grid_occupied[ci] += a
                steps_since_improvement += 1

            if use_lahc:
                lahc_history[step % lahc_length] = current_cost

        elif move_type == 2:
            # ── MOVE TOWARD NEIGHBOR ─────────────────────────────────────
            if not neighbors[i]:
                continue
            j = rng.choice(neighbors[i])
            alpha = rng.uniform(0.05, 0.35)
            new_x = float(np.clip(old_x + alpha * (pos[j, 0] - old_x), half_w[i], cw - half_w[i]))
            new_y = float(np.clip(old_y + alpha * (pos[j, 1] - old_y), half_h[i], ch - half_h[i]))

            pos[i, 0] = new_x; pos[i, 1] = new_y
            if check_overlap(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue
            pos[i, 0] = old_x; pos[i, 1] = old_y

            delta_hpwl, hpwl_updates = _delta_hpwl_cached(
                i, new_x, new_y, pos, nets, macro_to_nets, net_cache)
            delta = delta_hpwl

            old_cells = new_cells = None
            if use_density:
                old_cells = _macro_cell_overlaps(old_x, old_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                new_cells = _macro_cell_overlaps(new_x, new_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                for ci, a in old_cells: grid_occupied[ci] -= a
                for ci, a in new_cells: grid_occupied[ci] += a
                new_dens = _density_cost_from_grid(grid_occupied, grid_area, n_grid)
                delta += density_weight * (new_dens - current_density)

            new_cost = current_cost + delta

            accept = False
            if delta < 0:
                accept = True
            elif T > 0 and rng.random() < math.exp(-delta / max(T, 1e-12)):
                accept = True
            elif use_lahc and new_cost < lahc_history[step % lahc_length]:
                accept = True

            if accept:
                pos[i, 0] = new_x; pos[i, 1] = new_y
                current_hpwl += delta_hpwl
                for ni, val in hpwl_updates:
                    net_cache[ni] = val
                if use_density:
                    current_density = new_dens
                current_cost = new_cost
                move_accepts[move_type] += 1

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                if use_density:
                    for ci, a in new_cells: grid_occupied[ci] -= a
                    for ci, a in old_cells: grid_occupied[ci] += a
                steps_since_improvement += 1

            if use_lahc:
                lahc_history[step % lahc_length] = current_cost

        elif move_type == 3:
            # ── MIRROR SHIFT ─────────────────────────────────────────────
            # Reflect macro position across canvas center (x, y, or both)
            flip = rng.randint(0, 2)  # 0=x, 1=y, 2=both
            cx, cy = cw / 2, ch / 2
            if flip == 0 or flip == 2:
                new_x = float(np.clip(2 * cx - old_x, half_w[i], cw - half_w[i]))
            else:
                new_x = old_x
            if flip == 1 or flip == 2:
                new_y = float(np.clip(2 * cy - old_y, half_h[i], ch - half_h[i]))
            else:
                new_y = old_y

            # Add small perturbation
            perturb = T * 0.1 * (1 - frac)
            new_x = float(np.clip(new_x + rng.gauss(0, perturb), half_w[i], cw - half_w[i]))
            new_y = float(np.clip(new_y + rng.gauss(0, perturb), half_h[i], ch - half_h[i]))

            pos[i, 0] = new_x; pos[i, 1] = new_y
            if check_overlap(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue
            pos[i, 0] = old_x; pos[i, 1] = old_y

            delta_hpwl, hpwl_updates = _delta_hpwl_cached(
                i, new_x, new_y, pos, nets, macro_to_nets, net_cache)
            delta = delta_hpwl

            old_cells = new_cells = None
            if use_density:
                old_cells = _macro_cell_overlaps(old_x, old_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                new_cells = _macro_cell_overlaps(new_x, new_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                for ci, a in old_cells: grid_occupied[ci] -= a
                for ci, a in new_cells: grid_occupied[ci] += a
                new_dens = _density_cost_from_grid(grid_occupied, grid_area, n_grid)
                delta += density_weight * (new_dens - current_density)

            new_cost = current_cost + delta

            accept = False
            if delta < 0:
                accept = True
            elif T > 0 and rng.random() < math.exp(-delta / max(T, 1e-12)):
                accept = True
            elif use_lahc and new_cost < lahc_history[step % lahc_length]:
                accept = True

            if accept:
                pos[i, 0] = new_x; pos[i, 1] = new_y
                current_hpwl += delta_hpwl
                for ni, val in hpwl_updates:
                    net_cache[ni] = val
                if use_density:
                    current_density = new_dens
                current_cost = new_cost
                move_accepts[move_type] += 1

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                if use_density:
                    for ci, a in new_cells: grid_occupied[ci] -= a
                    for ci, a in old_cells: grid_occupied[ci] += a
                steps_since_improvement += 1

            if use_lahc:
                lahc_history[step % lahc_length] = current_cost

        # Snapshots & traces
        if snapshot_callback is not None and snapshot_interval > 0:
            if (step + 1) % snapshot_interval == 0 or step == max_iters - 1:
                snapshot_callback(best_pos.copy())
        if trace_callback is not None and trace_interval > 0:
            if (step + 1) % trace_interval == 0 or step == max_iters - 1:
                trace_callback({
                    "step": step + 1,
                    "temperature": T,
                    "current_hpwl": current_hpwl,
                    "best_hpwl": best_hpwl,
                })

    return best_pos


# ── greedy macro flipping ─────────────────────────────────────────────────

def _greedy_flip(pos, nets, macro_to_nets, movable, plc):
    """Try flipping each hard macro's pin offsets; keep if HPWL improves."""
    n_hard = len(movable)
    improved = 0

    for i in range(n_hard):
        if not movable[i]:
            continue
        affected = macro_to_nets[i]
        if not affected:
            continue

        base_hpwl = sum(_net_hpwl(nets[ni], pos) for ni in affected)
        best_hpwl = base_hpwl
        best_flip = None

        pin_locs = []
        for ni in affected:
            net = nets[ni]
            idxs = net["hard_idx"]
            for k in range(len(idxs)):
                if idxs[k] == i:
                    pin_locs.append((ni, k))

        orig_ox = [(ni, k, nets[ni]["hard_ox"][k]) for ni, k in pin_locs]
        orig_oy = [(ni, k, nets[ni]["hard_oy"][k]) for ni, k in pin_locs]

        for flip_name, flip_x, flip_y in [("FN", True, False),
                                            ("FS", False, True),
                                            ("S",  True, True)]:
            for ni, k, ox in orig_ox:
                nets[ni]["hard_ox"][k] = -ox if flip_x else ox
            for ni, k, oy in orig_oy:
                nets[ni]["hard_oy"][k] = -oy if flip_y else oy

            trial_hpwl = sum(_net_hpwl(nets[ni], pos) for ni in affected)
            if trial_hpwl < best_hpwl:
                best_hpwl = trial_hpwl
                best_flip = (flip_x, flip_y, flip_name)

            for ni, k, ox in orig_ox:
                nets[ni]["hard_ox"][k] = ox
            for ni, k, oy in orig_oy:
                nets[ni]["hard_oy"][k] = oy

        if best_flip is not None:
            flip_x, flip_y, flip_name = best_flip
            for ni, k, ox in orig_ox:
                if flip_x:
                    nets[ni]["hard_ox"][k] = -ox
            for ni, k, oy in orig_oy:
                if flip_y:
                    nets[ni]["hard_oy"][k] = -oy

            if plc is not None:
                plc_idx = plc.hard_macro_indices[i]
                macro_name = plc.modules_w_pins[plc_idx].get_name()
                pin_names = plc.hard_macros_to_inpins.get(macro_name, [])
                for pin_name in pin_names:
                    if pin_name not in plc.mod_name_to_indices:
                        continue
                    pin_obj = plc.modules_w_pins[plc.mod_name_to_indices[pin_name]]
                    ox, oy = pin_obj.get_offset()
                    new_ox = -ox if flip_x else ox
                    new_oy = -oy if flip_y else oy
                    pin_obj.set_offset(new_ox, new_oy)

            improved += 1

    return improved


# ── greedy local search post-processing ───────────────────────────────────

def _greedy_local_search(pos, nets, macro_to_nets, movable, sizes, half_w, half_h,
                         sep_x, sep_y, cw, ch, n_hard, passes: int = 3):
    """Run greedy steepest-descent passes over all movable macros.

    For each movable macro, try small shifts in 8 compass directions + center.
    Accept the best improving move. Repeat for `passes` rounds.
    """
    pos = pos.copy()
    GAP = 0.05

    def check_overlap(idx):
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        ov = (dx < sep_x[idx] + GAP) & (dy < sep_y[idx] + GAP)
        ov[idx] = False
        return bool(ov.any())

    net_cache = _build_net_hpwl_cache(pos, nets)
    total_improved = 0

    for _ in range(passes):
        improved_this_pass = 0
        movable_idx = np.where(movable)[0]
        # Shuffle order for variety
        np.random.shuffle(movable_idx)

        for i in movable_idx:
            old_x, old_y = pos[i, 0], pos[i, 1]
            step_x = sizes[i, 0] * 0.15
            step_y = sizes[i, 1] * 0.15

            best_delta = 0.0
            best_nx, best_ny = old_x, old_y
            best_updates = None

            for dx_mult, dy_mult in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nx = float(np.clip(old_x + dx_mult * step_x, half_w[i], cw - half_w[i]))
                ny = float(np.clip(old_y + dy_mult * step_y, half_h[i], ch - half_h[i]))

                pos[i, 0] = nx; pos[i, 1] = ny
                if check_overlap(i):
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    continue
                pos[i, 0] = old_x; pos[i, 1] = old_y

                delta, updates = _delta_hpwl_cached(
                    i, nx, ny, pos, nets, macro_to_nets, net_cache)

                if delta < best_delta:
                    best_delta = delta
                    best_nx, best_ny = nx, ny
                    best_updates = updates

            if best_delta < -1e-10:
                pos[i, 0] = best_nx; pos[i, 1] = best_ny
                for ni, val in best_updates:
                    net_cache[ni] = val
                improved_this_pass += 1

        total_improved += improved_this_pass
        if improved_this_pass == 0:
            break

    return pos, total_improved


# ── soft-macro force-directed update ──────────────────────────────────────

def _update_soft_macros(pos_hard: np.ndarray, benchmark: Benchmark, plc) -> np.ndarray:
    from macro_place.objective import _set_placement

    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    if n_soft == 0:
        return np.zeros((0, 2))

    full = benchmark.macro_positions.clone()
    full[:n_hard] = torch.tensor(pos_hard, dtype=torch.float32)
    _set_placement(plc, full, benchmark)

    canvas_size = max(benchmark.canvas_width, benchmark.canvas_height)
    plc.optimize_stdcells(
        use_current_loc=True,
        move_stdcells=True,
        move_macros=False,
        log_scale_conns=False,
        use_sizes=False,
        io_factor=1.0,
        num_steps=[25, 25, 25],
        max_move_distance=[canvas_size / 100] * 3,
        attract_factor=[100, 1.0e-3, 1.0e-5],
        repel_factor=[0, 1.0e6, 1.0e7],
    )

    soft_pos = np.zeros((n_soft, 2))
    for i, plc_i in enumerate(plc.soft_macro_indices):
        x, y = plc.modules_w_pins[plc_i].get_pos()
        soft_pos[i] = [x, y]
    return soft_pos


def _proxy_score_hard_pos(
    benchmark: Benchmark,
    plc,
    pos_hard: np.ndarray,
) -> float:
    """Evaluate a hard-macro placement with the proxy-cost scorer."""
    n_hard = benchmark.num_hard_macros
    full_pos = benchmark.macro_positions.clone()
    full_pos[:n_hard] = torch.tensor(pos_hard, dtype=torch.float32)
    return float(compute_proxy_cost(full_pos, benchmark, plc)["proxy_cost"])


def _gpu_refine_with_fallback(
    benchmark: Benchmark,
    plc,
    nets: list,
    pos_hard: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    cw: float,
    ch: float,
    num_steps: int,
    seed: int,
    lr: float = 0.08,
    gamma: float = 4.0,
    density_weight: float = 0.002,
    overlap_weight: float = 0.05,
    anchor_weight: float = 0.01,
) -> np.ndarray:
    """
    GPU-assisted post refinement with automatic CPU fallback.

    Uses differentiable HPWL + density + overlap around the SA solution, then
    keeps the refined placement only if the evaluator proxy improves.
    """
    if num_steps <= 0 or plc is None or not nets:
        return pos_hard

    try:
        from submissions.analytical_placer import (
            _build_net_tensors as _build_diff_net_tensors,
            _lse_hpwl as _diff_lse_hpwl,
            _density_penalty as _diff_density_penalty,
            _overlap_penalty as _diff_overlap_penalty,
        )
    except Exception:
        return pos_hard

    torch.manual_seed(seed)
    device = _sav2_device()

    n_hard = benchmark.num_hard_macros
    sizes_t = torch.tensor(sizes, dtype=torch.float32, device=device)
    movable_t = torch.tensor(movable, dtype=torch.bool, device=device)
    fixed_mask = ~movable_t
    base_pos = torch.tensor(pos_hard, dtype=torch.float32, device=device)

    lo_x = torch.tensor(half_w, dtype=torch.float32, device=device)
    hi_x = torch.tensor([cw], dtype=torch.float32, device=device) - lo_x
    lo_y = torch.tensor(half_h, dtype=torch.float32, device=device)
    hi_y = torch.tensor([ch], dtype=torch.float32, device=device) - lo_y

    with torch.no_grad():
        base_pos[:, 0].clamp_(lo_x, hi_x)
        base_pos[:, 1].clamp_(lo_y, hi_y)

    net_data = _build_diff_net_tensors(nets, device=device)
    if net_data is None:
        return pos_hard

    pos = base_pos.clone().detach().requires_grad_(True)
    fixed_pos = base_pos.clone()
    optimizer = torch.optim.Adam([pos], lr=lr)

    best_pos = base_pos.clone()
    best_loss = float("inf")

    for _ in range(num_steps):
        optimizer.zero_grad()

        with torch.no_grad():
            pos.data[fixed_mask] = fixed_pos[fixed_mask]

        loss = _diff_lse_hpwl(pos, net_data, gamma=gamma)
        loss = loss + density_weight * _diff_density_penalty(
            pos, sizes_t, cw, ch, grid_cols=32, grid_rows=32, target_scale=1.25
        )
        loss = loss + overlap_weight * _diff_overlap_penalty(pos, sizes_t, movable_t)
        loss = loss + anchor_weight * ((pos[movable_t] - base_pos[movable_t]) ** 2).sum()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pos.data[:, 0].clamp_(lo_x, hi_x)
            pos.data[:, 1].clamp_(lo_y, hi_y)
            pos.data[fixed_mask] = fixed_pos[fixed_mask]
            loss_value = float(loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_pos = pos.data.clone()

    refined = best_pos.detach().cpu().numpy().astype(np.float64)
    refined = _legalize(refined, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)

    try:
        if _proxy_score_hard_pos(benchmark, plc, refined) <= _proxy_score_hard_pos(benchmark, plc, pos_hard):
            return refined
    except Exception:
        return pos_hard

    return pos_hard


# ── placer class ─────────────────────────────────────────────────────────

class SAV2Placer(BasePlacer):
    """
    SA V2 Placer (Eklund) — Improved Simulated Annealing.

    Improvements over SA V1
    -----------------------
    1. Net HPWL caching — avoid recomputing unaffected nets.
    2. Adaptive move probabilities — track per-move-type accept rates.
    3. LAHC (Late Acceptance Hill Climbing) — secondary acceptance criterion.
    4. Size-aware move radius — smaller macros explore proportionally more.
    5. Mirror-shift move type — reflect across canvas center for diversity.
    6. Greedy local search post-processing — steepest descent after SA.
    7. Greedy tail phase — final iterations at T=0 for pure refinement.
    8. Macro selection weighting — bias toward smaller (more mobile) macros.
    9. GPU-assisted post-refinement with CPU fallback.

    Hyperparameters
    ---------------
    seed, max_iters, run_fd, num_starts : same as SA V1
    lahc_length : LAHC history buffer size (0 to disable)
    greedy_tail_frac : fraction of iterations for greedy-only tail
    greedy_local_passes : number of post-SA greedy local search passes
    adaptive_moves : enable adaptive move-type probabilities
    """

    DEFAULT_OVERRIDES = {
        "ibm04": {
            "max_iters": 200_000,
            "t_start_factor": 0.20,
            "t_end_factor": 0.0005,
            "num_starts": 3,
            "reheat_threshold": 8_000,
        },
    }

    def __init__(
        self,
        seed: int = 42,
        max_iters: int = 120_000,
        run_fd: bool = False,
        capture_snapshots: bool = True,
        snapshot_interval: int = 2_000,
        trace_interval: int = 500,
        t_start_factor: float = 0.12,
        t_end_factor: float = 0.0008,
        num_starts: int = 1,
        reheat_threshold: int = 8_000,
        reheat_factor: float = 3.0,
        per_benchmark_overrides=None,
        analytical_warmstart: bool = False,
        density_weight_boost: float = 1.0,
        select_best_by_proxy: bool = True,
        lahc_length: int = 0,
        greedy_tail_frac: float = 0.0,
        greedy_local_passes: int = 0,
        adaptive_moves: bool = False,
        gpu_refine_steps: int = 0,
    ):
        self.seed = seed
        self.max_iters = max_iters
        self.run_fd = run_fd
        self.capture_snapshots = capture_snapshots
        self.snapshot_interval = snapshot_interval
        self.trace_interval = trace_interval
        self.t_start_factor = t_start_factor
        self.t_end_factor = t_end_factor
        self.num_starts = num_starts
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor
        self.per_benchmark_overrides = per_benchmark_overrides or self.DEFAULT_OVERRIDES
        self.analytical_warmstart = analytical_warmstart
        self.density_weight_boost = density_weight_boost
        self.select_best_by_proxy = select_best_by_proxy
        self.lahc_length = lahc_length
        self.greedy_tail_frac = greedy_tail_frac
        self.greedy_local_passes = greedy_local_passes
        self.adaptive_moves = adaptive_moves
        self.gpu_refine_steps = gpu_refine_steps
        self.debug_snapshots = []
        self.debug_trace = []

    def _get_params(self, benchmark_name: str) -> dict:
        params = {
            "max_iters": self.max_iters,
            "t_start_factor": self.t_start_factor,
            "t_end_factor": self.t_end_factor,
            "num_starts": self.num_starts,
            "reheat_threshold": self.reheat_threshold,
            "reheat_factor": self.reheat_factor,
        }
        overrides = self.per_benchmark_overrides.get(benchmark_name, {})
        params.update(overrides)
        return params

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.debug_snapshots = []
        self.debug_trace = []

        params = self._get_params(benchmark.name)

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        sizes   = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        half_w  = sizes[:, 0] / 2
        half_h  = sizes[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

        plc = _load_plc(benchmark.name)

        if plc is not None:
            nets, macro_to_nets = _extract_nets(benchmark, plc)
        else:
            nets, macro_to_nets = [], [[] for _ in range(n_hard)]

        # Net-neighbor adjacency (deduped)
        neighbors = [set() for _ in range(n_hard)]
        for net in nets:
            idx = net["hard_idx"]
            for a in idx:
                for b in idx:
                    if a != b:
                        neighbors[a].add(int(b))
        neighbors = [list(s) for s in neighbors]

        # Initialise + legalize
        init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        init_pos = _legalize(init_pos, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)

        # Optionally warm-start from analytical placer
        if self.analytical_warmstart:
            try:
                from submissions.analytical_placer import AnalyticalPlacer
                ap = AnalyticalPlacer(seed=self.seed, iters=3_000, lr=5.0,
                                      density_weight=0.01, overlap_weight_end=20.0)
                ap_result = ap.place(benchmark)
                ap_pos = ap_result[:n_hard].numpy().astype(np.float64)
                if nets:
                    net_cache_ap = _build_net_hpwl_cache(ap_pos, nets)
                    net_cache_init = _build_net_hpwl_cache(init_pos, nets)
                    if float(net_cache_ap.sum()) < float(net_cache_init.sum()):
                        init_pos = ap_pos
            except Exception:
                pass

        def capture_snapshot(pos_hard: np.ndarray):
            if not self.capture_snapshots:
                return
            frame = benchmark.macro_positions.clone()
            frame[:n_hard] = torch.tensor(pos_hard, dtype=torch.float32)
            self.debug_snapshots.append(frame)

        def capture_trace(point: dict):
            self.debug_trace.append(point)

        capture_snapshot(init_pos)

        # Density weight calibration
        grid_col = grid_row = 0
        density_w = 0.0
        if plc is not None and nets:
            try:
                grid_col, grid_row = plc.grid_col, plc.grid_row
                if grid_col > 0 and grid_row > 0:
                    from macro_place.objective import _set_placement
                    full_init = benchmark.macro_positions.clone()
                    full_init[:n_hard] = torch.tensor(init_pos, dtype=torch.float32)
                    _set_placement(plc, full_init, benchmark)
                    wl_norm = plc.get_cost()
                    net_cache_init = _build_net_hpwl_cache(init_pos, nets)
                    raw_hpwl = float(net_cache_init.sum())
                    if wl_norm > 1e-10 and raw_hpwl > 1e-10:
                        density_w = 0.5 * raw_hpwl / wl_norm
                        density_w *= self.density_weight_boost
            except Exception:
                grid_col = grid_row = 0
                density_w = 0.0

        # Multi-start SA
        num_starts = params["num_starts"]
        best_pos = init_pos.copy()
        best_score = float("inf")
        use_proxy = (
            self.select_best_by_proxy
            and plc is not None
            and num_starts > 1
        )

        if nets:
            for start_idx in range(num_starts):
                run_seed = self.seed + start_idx * 1000
                pos = _sa_v2_refine(
                    init_pos, nets, macro_to_nets, neighbors,
                    movable, sizes, half_w, half_h, sep_x, sep_y,
                    cw, ch, params["max_iters"], run_seed,
                    snapshot_interval=self.snapshot_interval,
                    snapshot_callback=capture_snapshot if (self.capture_snapshots and start_idx == 0) else None,
                    trace_interval=self.trace_interval,
                    trace_callback=capture_trace if start_idx == 0 else None,
                    t_start_factor=params["t_start_factor"],
                    t_end_factor=params["t_end_factor"],
                    reheat_threshold=params["reheat_threshold"],
                    reheat_factor=params["reheat_factor"],
                    density_weight=density_w,
                    grid_col=grid_col,
                    grid_row=grid_row,
                    benchmark=benchmark,
                    lahc_length=self.lahc_length,
                    greedy_tail_frac=self.greedy_tail_frac,
                    adaptive_moves=self.adaptive_moves,
                )

                # Greedy local search post-processing
                if self.greedy_local_passes > 0 and nets:
                    pos, _ = _greedy_local_search(
                        pos, nets, macro_to_nets, movable, sizes,
                        half_w, half_h, sep_x, sep_y, cw, ch, n_hard,
                        passes=self.greedy_local_passes,
                    )

                if use_proxy:
                    full_try = benchmark.macro_positions.clone()
                    full_try[:n_hard] = torch.tensor(pos, dtype=torch.float32)
                    try:
                        metric = float(
                            compute_proxy_cost(full_try, benchmark, plc)["proxy_cost"]
                        )
                    except Exception:
                        net_cache = _build_net_hpwl_cache(pos, nets)
                        metric = float(net_cache.sum())
                else:
                    net_cache = _build_net_hpwl_cache(pos, nets)
                    metric = float(net_cache.sum())

                if metric < best_score:
                    best_score = metric
                    best_pos = pos.copy()
        else:
            best_pos = init_pos

        # Greedy macro flipping
        if nets:
            _greedy_flip(best_pos, nets, macro_to_nets, movable, plc)
            best_pos = _gpu_refine_with_fallback(
                benchmark=benchmark,
                plc=plc,
                nets=nets,
                pos_hard=best_pos,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                sep_x=sep_x,
                sep_y=sep_y,
                cw=cw,
                ch=ch,
                num_steps=self.gpu_refine_steps,
                seed=self.seed,
            )

        # Build full placement tensor
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)

        # Optionally update soft macros
        if self.run_fd and plc is not None and benchmark.num_soft_macros > 0:
            soft_pos = _update_soft_macros(best_pos, benchmark, plc)
            full_pos[n_hard:] = torch.tensor(soft_pos, dtype=torch.float32)

        if self.capture_snapshots:
            if not self.debug_snapshots or not torch.equal(self.debug_snapshots[-1], full_pos):
                self.debug_snapshots.append(full_pos.clone())

        return full_pos
