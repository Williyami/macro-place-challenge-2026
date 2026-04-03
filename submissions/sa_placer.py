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
) -> np.ndarray:
    """SA loop minimising full net-HPWL while enforcing hard-macro legality."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos

    pos = pos.copy()

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
            delta = _delta_hpwl(i, new_x, new_y, pos, nets, macro_to_nets)

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                pos[i, 0] = new_x; pos[i, 1] = new_y
                current_hpwl += delta
                if current_hpwl < best_hpwl:
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()

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
            delta = new_hpwl_aff - old_hpwl_aff

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                pos[i, 0] = new_ix; pos[i, 1] = new_iy
                pos[j, 0] = new_jx; pos[j, 1] = new_jy
                current_hpwl += delta
                if current_hpwl < best_hpwl:
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()

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
            delta = _delta_hpwl(i, new_x, new_y, pos, nets, macro_to_nets)

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                pos[i, 0] = new_x; pos[i, 1] = new_y
                current_hpwl += delta
                if current_hpwl < best_hpwl:
                    best_hpwl = current_hpwl
                    best_pos = pos.copy()

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
    """

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
    ):
        self.seed = seed
        self.max_iters = max_iters
        self.run_fd = run_fd
        self.capture_snapshots = capture_snapshots
        self.snapshot_interval = snapshot_interval
        self.trace_interval = trace_interval
        self.t_start_factor = t_start_factor
        self.t_end_factor = t_end_factor
        self.debug_snapshots = []
        self.debug_trace = []

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.debug_snapshots = []
        self.debug_trace = []

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
        pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        pos = _legalize(pos, movable, sizes, half_w, half_h, cw, ch, n_hard, sep_x, sep_y)

        def capture_snapshot(pos_hard: np.ndarray):
            if not self.capture_snapshots:
                return
            frame = benchmark.macro_positions.clone()
            frame[:n_hard] = torch.tensor(pos_hard, dtype=torch.float32)
            self.debug_snapshots.append(frame)

        def capture_trace(point: dict):
            self.debug_trace.append(point)

        capture_snapshot(pos)

        # Run SA
        if nets:
            pos = _sa_refine(
                pos, nets, macro_to_nets, neighbors,
                movable, sizes, half_w, half_h, sep_x, sep_y,
                cw, ch, self.max_iters, self.seed,
                snapshot_interval=self.snapshot_interval,
                snapshot_callback=capture_snapshot if self.capture_snapshots else None,
                trace_interval=self.trace_interval,
                trace_callback=capture_trace,
                t_start_factor=self.t_start_factor,
                t_end_factor=self.t_end_factor,
            )

        # Build full placement tensor
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos, dtype=torch.float32)

        # Optionally update soft macros via force-directed placement
        if self.run_fd and plc is not None and benchmark.num_soft_macros > 0:
            soft_pos = _update_soft_macros(pos, benchmark, plc)
            full_pos[n_hard:] = torch.tensor(soft_pos, dtype=torch.float32)

        if self.capture_snapshots:
            if not self.debug_snapshots or not torch.equal(self.debug_snapshots[-1], full_pos):
                self.debug_snapshots.append(full_pos.clone())

        return full_pos
