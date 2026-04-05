"""
SA Placer — Simulated Annealing with full net-HPWL cost.

Key improvements over will_seed:
  - Full HPWL (net bounding box) instead of pairwise-edge approximation.
    Includes fixed soft-macro and port positions in each net's bounding box.
  - Many more iterations (default 100 K vs 3 K).
  - Optional soft-macro force-directed refinement after SA.

Usage:
    uv run evaluate submissions/sa_placer.py
    uv run evaluate submissions/sa_placer.py --all
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path so sibling packages resolve correctly
# regardless of how this file is loaded (direct exec vs package import).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark
from submissions.base import BasePlacer


# ── benchmark loading ────────────────────────────────────────────────────────

def _load_plc(name: str):
    """Load PlacementCost for a benchmark by name.  Returns None on failure."""
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


# ── net extraction ───────────────────────────────────────────────────────────

def _extract_nets(benchmark: Benchmark, plc):
    """
    Build net data structures for fast delta-HPWL computation.

    Hard macros (indices 0..n_hard-1 in pos array) are movable.
    Soft macros and ports are treated as fixed contributors to bounding boxes.

    Returns
    -------
    nets : list of dicts
        Each dict has:
          'weight'   : float
          'hard_idx' : np.int32 [k]   — tensor indices of hard macro pins
          'hard_ox'  : np.float64 [k] — x offsets (0 for soft, actual for hard)
          'hard_oy'  : np.float64 [k]
          'fxmin/fxmax/fymin/fymax' : float — bounding box of fixed pins
                                       (inf/-inf when no fixed pins)
    macro_to_nets : list of lists
        macro_to_nets[i] = list of net indices macro i participates in.
    """
    n_hard = benchmark.num_hard_macros

    # hard macro name -> tensor index (0-based)
    hard_name_to_idx = {}
    for tensor_i, plc_i in enumerate(plc.hard_macro_indices):
        name = plc.modules_w_pins[plc_i].get_name()
        hard_name_to_idx[name] = tensor_i

    # soft macro name -> current (x, y) position (fixed during SA)
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
                    # Movable hard macro pin
                    macro_idx = hard_name_to_idx[parent_name]
                    ox, oy = pin_obj.get_offset()
                    hard_idx_list.append(macro_idx)
                    hard_ox_list.append(ox)
                    hard_oy_list.append(oy)

                elif parent_name in soft_name_to_pos:
                    # Fixed soft macro pin (offset is 0,0 for SoftMacroPin)
                    sx, sy = soft_name_to_pos[parent_name]
                    ox, oy = pin_obj.get_offset()   # always (0,0) for soft
                    px, py = sx + ox, sy + oy
                    fxmin = min(fxmin, px); fxmax = max(fxmax, px)
                    fymin = min(fymin, py); fymax = max(fymax, py)

        # Skip nets with no hard macro pins or fewer than 2 total pins
        if not hard_idx_list:
            continue
        total_pins = len(hard_idx_list) + (0 if fxmin == INF else 1)
        if total_pins < 2 and fxmin == INF:
            # Only one hard macro and no fixed pins → skip (zero HPWL contribution)
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


# ── per-net HPWL helpers ─────────────────────────────────────────────────────

def _net_hpwl(net: dict, pos: np.ndarray) -> float:
    """HPWL contribution of a single net given current hard-macro positions."""
    hx = pos[net["hard_idx"], 0] + net["hard_ox"]
    hy = pos[net["hard_idx"], 1] + net["hard_oy"]
    xmin = min(float(hx.min()), net["fxmin"])
    xmax = max(float(hx.max()), net["fxmax"])
    ymin = min(float(hy.min()), net["fymin"])
    ymax = max(float(hy.max()), net["fymax"])
    return net["weight"] * ((xmax - xmin) + (ymax - ymin))


def _net_hpwl_override(net: dict, pos: np.ndarray,
                       m: int, new_x: float, new_y: float) -> float:
    """HPWL for a net with macro m temporarily at (new_x, new_y)."""
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


def _delta_hpwl(m: int, new_x: float, new_y: float,
                pos: np.ndarray, nets: list, macro_to_nets: list) -> float:
    """Change in total HPWL when macro m moves to (new_x, new_y)."""
    delta = 0.0
    for net_idx in macro_to_nets[m]:
        net = nets[net_idx]
        delta += _net_hpwl_override(net, pos, m, new_x, new_y) - _net_hpwl(net, pos)
    return delta


def _compute_total_hpwl(pos: np.ndarray, nets: list) -> float:
    return sum(_net_hpwl(net, pos) for net in nets)


# ── density grid helpers ──────────────────────────────────────────────────────

def _macro_cell_overlaps(mx, my, mw, mh, grid_col, grid_row, gw, gh):
    """Return list of (cell_idx, overlap_area) for a macro centered at (mx,my)."""
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
    """Build initial density grid (total macro area per cell)."""
    grid = np.zeros(grid_col * grid_row)
    for i in range(n_hard):
        for ci, area in _macro_cell_overlaps(
            pos[i, 0], pos[i, 1], sizes[i, 0], sizes[i, 1],
            grid_col, grid_row, gw, gh,
        ):
            grid[ci] += area
    # Include soft macros (fixed during SA)
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
    """Compute density cost matching evaluator: 0.5 * avg of top 10%."""
    densities = grid_occupied / grid_area
    nonzero = sorted([float(d) for d in densities if d > 0], reverse=True)
    if not nonzero:
        return 0.0
    cnt = max(1, n_cells // 10)
    return 0.5 * sum(nonzero[:cnt]) / cnt


# ── legalization (minimum displacement) ─────────────────────────────────────

def _legalize(pos, movable, sizes, half_w, half_h, cw, ch, n, sep_x, sep_y):
    """Resolve hard-macro overlaps with minimum displacement (same as will_seed)."""
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


# ── SA refinement ─────────────────────────────────────────────────────────────

def _sa_refine(
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
) -> np.ndarray:
    """SA loop minimising proxy cost (HPWL + density + periphery).

    Parameters
    ----------
    reheat_threshold : int
        If > 0, reheat the temperature when no improvement is found for this
        many consecutive iterations.  0 disables reheating.
    reheat_factor : float
        On reheat, multiply current temperature by this factor (capped at T_start).
    density_weight : float
        Weight for density penalty in HPWL units.  0 disables density tracking.
    grid_col, grid_row : int
        Grid dimensions for density tracking.
    benchmark : Benchmark or None
        Needed for soft-macro density contributions.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos

    pos = pos.copy()
    n_hard = len(movable)

    GAP = 0.05  # overlap safety gap (μm)

    def check_overlap(idx: int) -> bool:
        """True if macro idx overlaps any other hard macro."""
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        ov = (dx < sep_x[idx] + GAP) & (dy < sep_y[idx] + GAP)
        ov[idx] = False
        return bool(ov.any())

    current_hpwl = _compute_total_hpwl(pos, nets)
    best_pos = pos.copy()
    best_hpwl = current_hpwl

    T_start = max(cw, ch) * t_start_factor
    T_end   = max(cw, ch) * t_end_factor

    # ── Density tracking ──────────────────────────────────────────────────
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
        best_cost = current_hpwl + density_weight * current_density
    else:
        current_density = 0.0
        best_cost = current_hpwl

    # Reheating state
    steps_since_improvement = 0

    if trace_callback is not None:
        trace_callback(
            {
                "step": 0,
                "temperature": T_start,
                "current_hpwl": current_hpwl,
                "best_hpwl": best_hpwl,
            }
        )

    for step in range(max_iters):
        frac = step / max_iters
        T = T_start * (T_end / T_start) ** frac

        # Reheat if stagnating
        if reheat_threshold > 0 and steps_since_improvement >= reheat_threshold:
            T = min(T * reheat_factor, T_start * 0.5)
            steps_since_improvement = 0

        move = rng.random()
        i = rng.choice(movable_idx)
        old_x, old_y = pos[i, 0], pos[i, 1]

        if move < 0.5:
            # ── SHIFT ──────────────────────────────────────────────────────
            shift = T * (0.3 + 0.7 * (1 - frac))
            new_x = float(np.clip(old_x + rng.gauss(0, shift), half_w[i], cw - half_w[i]))
            new_y = float(np.clip(old_y + rng.gauss(0, shift), half_h[i], ch - half_h[i]))

            pos[i, 0] = new_x; pos[i, 1] = new_y
            if check_overlap(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue

            pos[i, 0] = old_x; pos[i, 1] = old_y
            delta_hpwl = _delta_hpwl(i, new_x, new_y, pos, nets, macro_to_nets)
            delta = delta_hpwl

            # Density penalty
            old_cells = new_cells = None
            if use_density:
                old_cells = _macro_cell_overlaps(old_x, old_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                new_cells = _macro_cell_overlaps(new_x, new_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                for ci, a in old_cells: grid_occupied[ci] -= a
                for ci, a in new_cells: grid_occupied[ci] += a
                new_dens = _density_cost_from_grid(grid_occupied, grid_area, n_grid)
                delta += density_weight * (new_dens - current_density)

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                pos[i, 0] = new_x; pos[i, 1] = new_y
                current_hpwl += delta_hpwl
                if use_density:
                    current_density = new_dens
                cost = current_hpwl + density_weight * current_density if use_density else current_hpwl
                if cost < best_cost:
                    best_cost = cost
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

        elif move < 0.80:
            # ── SWAP ───────────────────────────────────────────────────────
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

            # Delta HPWL for swap: union of nets of both macros
            # pos is in swapped state after overlap check — compute new HPWL
            affected = list(set(macro_to_nets[i]) | set(macro_to_nets[j]))
            new_hpwl_aff = sum(_net_hpwl(nets[k], pos) for k in affected)
            # Revert to old state, compute old HPWL
            pos[i, 0] = old_x;  pos[i, 1] = old_y
            pos[j, 0] = old_jx; pos[j, 1] = old_jy
            old_hpwl_aff = sum(_net_hpwl(nets[k], pos) for k in affected)
            delta_hpwl = new_hpwl_aff - old_hpwl_aff
            delta = delta_hpwl

            # Density penalty for swap (two macros move)
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

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                pos[i, 0] = new_ix; pos[i, 1] = new_iy
                pos[j, 0] = new_jx; pos[j, 1] = new_jy
                current_hpwl += delta_hpwl
                if use_density:
                    current_density = new_dens
                cost = current_hpwl + density_weight * current_density if use_density else current_hpwl
                if cost < best_cost:
                    best_cost = cost
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

        else:
            # ── MOVE TOWARD NEIGHBOR ───────────────────────────────────────
            if not neighbors[i]:
                continue
            j = rng.choice(neighbors[i])
            alpha = rng.uniform(0.05, 0.3)
            new_x = float(np.clip(old_x + alpha * (pos[j, 0] - old_x), half_w[i], cw - half_w[i]))
            new_y = float(np.clip(old_y + alpha * (pos[j, 1] - old_y), half_h[i], ch - half_h[i]))

            pos[i, 0] = new_x; pos[i, 1] = new_y
            if check_overlap(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue

            pos[i, 0] = old_x; pos[i, 1] = old_y
            delta_hpwl = _delta_hpwl(i, new_x, new_y, pos, nets, macro_to_nets)
            delta = delta_hpwl

            # Density penalty
            old_cells = new_cells = None
            if use_density:
                old_cells = _macro_cell_overlaps(old_x, old_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                new_cells = _macro_cell_overlaps(new_x, new_y, sizes[i, 0], sizes[i, 1], grid_col, grid_row, gw, gh)
                for ci, a in old_cells: grid_occupied[ci] -= a
                for ci, a in new_cells: grid_occupied[ci] += a
                new_dens = _density_cost_from_grid(grid_occupied, grid_area, n_grid)
                delta += density_weight * (new_dens - current_density)

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                pos[i, 0] = new_x; pos[i, 1] = new_y
                current_hpwl += delta_hpwl
                if use_density:
                    current_density = new_dens
                cost = current_hpwl + density_weight * current_density if use_density else current_hpwl
                if cost < best_cost:
                    best_cost = cost
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

        if snapshot_callback is not None and snapshot_interval > 0:
            if (step + 1) % snapshot_interval == 0 or step == max_iters - 1:
                snapshot_callback(best_pos.copy())
        if trace_callback is not None and trace_interval > 0:
            if (step + 1) % trace_interval == 0 or step == max_iters - 1:
                trace_callback(
                    {
                        "step": step + 1,
                        "temperature": T,
                        "current_hpwl": current_hpwl,
                        "best_hpwl": best_hpwl,
                    }
                )

    return best_pos


# ── greedy macro flipping ───────────────────────────────────────────────────

def _greedy_flip(pos, nets, macro_to_nets, movable, plc):
    """Try flipping each hard macro's pin offsets; keep if HPWL improves.

    Orientations tried (relative to current N):
      FN  → negate x_offset  (mirror across Y axis)
      FS  → negate y_offset  (mirror across X axis)
      S   → negate both      (180° rotation)

    Updates *nets* pin offsets in-place and sets plc pin offsets to match.
    """
    n_hard = len(movable)
    improved = 0

    for i in range(n_hard):
        if not movable[i]:
            continue
        affected = macro_to_nets[i]
        if not affected:
            continue

        # Current HPWL for affected nets
        base_hpwl = sum(_net_hpwl(nets[ni], pos) for ni in affected)
        best_hpwl = base_hpwl
        best_flip = None  # None = keep current orientation

        # Collect (net_idx, pin_position_in_array) for all pins of macro i
        pin_locs = []  # (net_idx, array_index_within_net)
        for ni in affected:
            net = nets[ni]
            idxs = net["hard_idx"]
            for k in range(len(idxs)):
                if idxs[k] == i:
                    pin_locs.append((ni, k))

        # Save original offsets
        orig_ox = [(ni, k, nets[ni]["hard_ox"][k]) for ni, k in pin_locs]
        orig_oy = [(ni, k, nets[ni]["hard_oy"][k]) for ni, k in pin_locs]

        for flip_name, flip_x, flip_y in [("FN", True, False),
                                            ("FS", False, True),
                                            ("S",  True, True)]:
            # Apply flip
            for ni, k, ox in orig_ox:
                nets[ni]["hard_ox"][k] = -ox if flip_x else ox
            for ni, k, oy in orig_oy:
                nets[ni]["hard_oy"][k] = -oy if flip_y else oy

            trial_hpwl = sum(_net_hpwl(nets[ni], pos) for ni in affected)
            if trial_hpwl < best_hpwl:
                best_hpwl = trial_hpwl
                best_flip = (flip_x, flip_y, flip_name)

            # Restore originals
            for ni, k, ox in orig_ox:
                nets[ni]["hard_ox"][k] = ox
            for ni, k, oy in orig_oy:
                nets[ni]["hard_oy"][k] = oy

        if best_flip is not None:
            flip_x, flip_y, flip_name = best_flip
            # Apply the winning flip permanently to net data
            for ni, k, ox in orig_ox:
                if flip_x:
                    nets[ni]["hard_ox"][k] = -ox
            for ni, k, oy in orig_oy:
                if flip_y:
                    nets[ni]["hard_oy"][k] = -oy

            # Update plc pin offsets so evaluation reflects the flip
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


# ── soft-macro force-directed update ────────────────────────────────────────

def _update_soft_macros(pos_hard: np.ndarray, benchmark: Benchmark, plc) -> np.ndarray:
    """
    Run PlacementCost.optimize_stdcells to reposition soft macros given
    the current hard macro placement.  Returns soft macro positions
    as [num_soft, 2] numpy array.
    """
    from macro_place.objective import _set_placement

    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    if n_soft == 0:
        return np.zeros((0, 2))

    # Build a full placement tensor to push hard positions into plc
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


# ── placer class ─────────────────────────────────────────────────────────────

class SAPlacer(BasePlacer):
    """
    Simulated Annealing placer with full net-HPWL cost.

    Hyperparameters
    ---------------
    seed         : RNG seed for reproducibility
    max_iters    : SA iterations per benchmark
    run_fd       : whether to run soft-macro FD at the end
    num_starts   : number of independent SA runs (best result kept)
    reheat_threshold : iterations without improvement before reheating (0=off)
    per_benchmark_overrides : dict mapping benchmark name to param overrides
    """

    # Default per-benchmark overrides for known difficult benchmarks.
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
        max_iters: int = 100_000,
        run_fd: bool = False,
        capture_snapshots: bool = True,
        snapshot_interval: int = 2_000,
        trace_interval: int = 500,
        t_start_factor: float = 0.15,
        t_end_factor: float = 0.001,
        num_starts: int = 1,
        reheat_threshold: int = 0,
        reheat_factor: float = 3.0,
        per_benchmark_overrides=None,
        analytical_warmstart: bool = False,
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
        self.debug_snapshots = []
        self.debug_trace = []

    def _get_params(self, benchmark_name: str) -> dict:
        """Return effective parameters, applying per-benchmark overrides."""
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

        # Precompute pairwise separation matrices (used for O(N) overlap check)
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

        # Load PlacementCost for net connectivity
        plc = _load_plc(benchmark.name)

        if plc is not None:
            nets, macro_to_nets = _extract_nets(benchmark, plc)
        else:
            nets, macro_to_nets = [], [[] for _ in range(n_hard)]

        # Build net-neighbor adjacency (for TOWARD_NEIGHBOR moves)
        neighbors = [[] for _ in range(n_hard)]
        for net in nets:
            idx = net["hard_idx"]
            for a in idx:
                for b in idx:
                    if a != b:
                        neighbors[a].append(int(b))

        # Initialise from hand-crafted initial.plc positions + legalize
        init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        init_pos = _legalize(init_pos, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)

        # Optionally warm-start from analytical placer output
        if self.analytical_warmstart:
            try:
                from submissions.analytical_placer import AnalyticalPlacer
                ap = AnalyticalPlacer(seed=self.seed, iters=3_000, lr=5.0,
                                      density_weight=0.01, overlap_weight_end=20.0)
                ap_result = ap.place(benchmark)
                ap_pos = ap_result[:n_hard].numpy().astype(np.float64)
                # Compare HPWL: use analytical only if it's better
                if nets:
                    ap_hpwl = _compute_total_hpwl(ap_pos, nets)
                    init_hpwl = _compute_total_hpwl(init_pos, nets)
                    if ap_hpwl < init_hpwl:
                        init_pos = ap_pos
            except Exception:
                pass  # Fall back to default init

        def capture_snapshot(pos_hard: np.ndarray):
            if not self.capture_snapshots:
                return
            frame = benchmark.macro_positions.clone()
            frame[:n_hard] = torch.tensor(pos_hard, dtype=torch.float32)
            self.debug_snapshots.append(frame)

        def capture_trace(point: dict):
            self.debug_trace.append(point)

        capture_snapshot(init_pos)

        # ── Density weight calibration ────────────────────────────────────
        # proxy = 1.0*WL_norm + 0.5*density + 0.5*congestion
        # Convert density_penalty to HPWL-equivalent units so SA can
        # co-optimise both in a single acceptance criterion.
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
                    raw_hpwl = _compute_total_hpwl(init_pos, nets)
                    if wl_norm > 1e-10 and raw_hpwl > 1e-10:
                        # density_weight converts: delta_density → HPWL-equivalent
                        # proxy uses weight 0.5 for density, so:
                        density_w = 0.5 * raw_hpwl / wl_norm
            except Exception:
                grid_col = grid_row = 0
                density_w = 0.0

        # Multi-start SA: run multiple times with different seeds, keep best
        num_starts = params["num_starts"]
        best_pos = init_pos
        best_hpwl = float("inf")

        if nets:
            for start_idx in range(num_starts):
                run_seed = self.seed + start_idx * 1000
                pos = _sa_refine(
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
                )
                hpwl = _compute_total_hpwl(pos, nets)
                if hpwl < best_hpwl:
                    best_hpwl = hpwl
                    best_pos = pos
        else:
            best_pos = init_pos

        # Greedy macro flipping post-processing
        if nets:
            _greedy_flip(best_pos, nets, macro_to_nets, movable, plc)

        # Build full placement tensor
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)

        # Optionally update soft macros via force-directed placement
        if self.run_fd and plc is not None and benchmark.num_soft_macros > 0:
            soft_pos = _update_soft_macros(best_pos, benchmark, plc)
            full_pos[n_hard:] = torch.tensor(soft_pos, dtype=torch.float32)

        if self.capture_snapshots:
            if not self.debug_snapshots or not torch.equal(self.debug_snapshots[-1], full_pos):
                self.debug_snapshots.append(full_pos.clone())

        return full_pos
