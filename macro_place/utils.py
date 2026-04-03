"""
Utility functions for placement validation and visualization.
"""

import sys
from pathlib import Path

import torch
from typing import Tuple, List, Optional

from macro_place.benchmark import Benchmark


def validate_placement(
    placement: torch.Tensor, benchmark: Benchmark, check_overlaps: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate placement legality.

    Checks:
    - All macros within canvas bounds
    - No NaN/Inf values
    - Correct shape
    - Fixed macros at original positions
    - No macro overlaps (optional, can be slow for large designs)

    Args:
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object
        check_overlaps: If True, check for macro-to-macro overlaps (default: True)

    Returns:
        (is_valid, violations)
    """
    violations = []

    # Check shape
    if placement.shape != (benchmark.num_macros, 2):
        violations.append(
            f"Shape mismatch: expected {(benchmark.num_macros, 2)}, got {placement.shape}"
        )
        return False, violations

    # Check for NaN/Inf
    if torch.isnan(placement).any():
        violations.append("Placement contains NaN values")
    if torch.isinf(placement).any():
        violations.append("Placement contains Inf values")

    # Check bounds
    x_coords = placement[:, 0]
    y_coords = placement[:, 1]
    widths = benchmark.macro_sizes[:, 0]
    heights = benchmark.macro_sizes[:, 1]

    x_min = x_coords - widths / 2
    x_max = x_coords + widths / 2
    y_min = y_coords - heights / 2
    y_max = y_coords + heights / 2

    if (x_min < 0).any() or (x_max > benchmark.canvas_width).any():
        violations.append("Macros outside horizontal canvas bounds")
    if (y_min < 0).any() or (y_max > benchmark.canvas_height).any():
        violations.append("Macros outside vertical canvas bounds")

    # Check fixed macros
    fixed_mask = benchmark.macro_fixed
    if fixed_mask.any():
        original_pos = benchmark.macro_positions[fixed_mask]
        new_pos = placement[fixed_mask]
        if not torch.allclose(original_pos, new_pos, atol=1e-3):
            violations.append("Fixed macros have been moved")

    # Check overlaps among hard macros only (soft macros naturally overlap)
    if check_overlaps:
        overlap_count = 0
        num_hard = benchmark.num_hard_macros
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                # Get bounding boxes
                lx_i, ux_i = x_min[i].item(), x_max[i].item()
                ly_i, uy_i = y_min[i].item(), y_max[i].item()
                lx_j, ux_j = x_min[j].item(), x_max[j].item()
                ly_j, uy_j = y_min[j].item(), y_max[j].item()

                # Check if boxes overlap (NOT just touching)
                # No overlap if: one box is completely to the left, right, above, or below the other
                if not (lx_i >= ux_j or ux_i <= lx_j or ly_i >= uy_j or uy_i <= ly_j):
                    overlap_count += 1
                    if overlap_count <= 5:  # Only report first 5 to avoid spam
                        violations.append(f"Macros {i} and {j} overlap")

        if overlap_count > 5:
            violations.append(f"... and {overlap_count - 5} more overlaps")

    return len(violations) == 0, violations


def _draw_canvas(ax, benchmark):
    """Draw the canvas border on an axis."""
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (0, 0),
            benchmark.canvas_width,
            benchmark.canvas_height,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
    )
    ax.set_xlim(0, benchmark.canvas_width)
    ax.set_ylim(0, benchmark.canvas_height)
    ax.set_aspect("equal")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")


def _draw_hard_macros(ax, placement, benchmark):
    """Draw only hard macro outlines (no fill, no soft macros)."""
    from matplotlib.patches import Rectangle

    num_hard = benchmark.num_hard_macros
    for i in range(num_hard):
        x, y = placement[i].tolist()
        w, h = benchmark.macro_sizes[i].tolist()
        color = "red" if benchmark.macro_fixed[i] else "black"
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                fill=False,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
        )


def visualize_placement(
    placement: torch.Tensor,
    benchmark: Benchmark,
    save_path: Optional[str] = None,
    plc=None,
):
    """
    Visualize placement as 3 side-by-side panels:
      1. Placement (hard + soft macros, pins, nets)
      2. Density heatmap (hard macros only)
      3. Congestion heatmap (hard macros only)

    Args:
        placement: [num_macros, 2] positions
        benchmark: Benchmark data
        save_path: Optional path to save figure
        plc: Optional PlacementCost object (enables nets + heatmaps)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle, Patch
        from matplotlib.lines import Line2D
        from matplotlib.collections import LineCollection
    except ImportError:
        print(
            "Error: matplotlib not installed. Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # ── Panel 1: Full placement ──────────────────────────────────────────
    ax = axes[0]
    _draw_canvas(ax, benchmark)

    num_hard = benchmark.num_hard_macros
    for i in range(benchmark.num_macros):
        x, y = placement[i].tolist()
        w, h = benchmark.macro_sizes[i].tolist()
        is_soft = i >= num_hard
        color = (
            "red"
            if benchmark.macro_fixed[i]
            else "lightsteelblue"
            if is_soft
            else "blue"
        )
        alpha = 0.25 if is_soft else 0.5
        linestyle = "dashed" if is_soft else "solid"
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                fill=True,
                facecolor=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
                linestyle=linestyle,
            )
        )

    # Macro pins
    if benchmark.macro_pin_offsets:
        all_pin_x, all_pin_y = [], []
        for i, offsets in enumerate(benchmark.macro_pin_offsets):
            if offsets.shape[0] == 0:
                continue
            cx, cy = placement[i].tolist()
            all_pin_x.extend((cx + offsets[:, 0]).tolist())
            all_pin_y.extend((cy + offsets[:, 1]).tolist())
        if all_pin_x:
            ax.scatter(all_pin_x, all_pin_y, s=3, c="darkslateblue", zorder=6)

    # I/O pins
    if benchmark.port_positions.shape[0] > 0:
        ax.scatter(
            benchmark.port_positions[:, 0].tolist(),
            benchmark.port_positions[:, 1].tolist(),
            s=8, c="green", zorder=5, edgecolors="darkgreen", linewidths=0.3,
        )

    # Net connections
    if plc is not None:
        lines = []
        for driver_name, sink_names in plc.nets.items():
            if driver_name not in plc.mod_name_to_indices:
                continue
            coords = []
            driver_idx = plc.mod_name_to_indices[driver_name]
            dx, dy = plc.modules_w_pins[driver_idx].get_pos()
            coords.append((dx, dy))
            for sink_name in sink_names:
                if sink_name not in plc.mod_name_to_indices:
                    continue
                sink_idx = plc.mod_name_to_indices[sink_name]
                sx, sy = plc.modules_w_pins[sink_idx].get_pos()
                coords.append((sx, sy))
            if len(coords) < 2:
                continue
            avg_x = sum(c[0] for c in coords) / len(coords)
            avg_y = sum(c[1] for c in coords) / len(coords)
            for cx, cy in coords:
                lines.append([(avg_x, avg_y), (cx, cy)])
        if lines:
            ax.add_collection(
                LineCollection(lines, colors="gray", alpha=0.05, linewidths=0.5, zorder=1)
            )

    ax.set_title(f"{benchmark.name} — Placement")
    legend_elements = [
        Patch(facecolor="blue", alpha=0.5, edgecolor="black", label="Hard macros"),
        Patch(facecolor="lightsteelblue", alpha=0.1, edgecolor="black",
              linestyle="dashed", label="Soft macros"),
        Patch(facecolor="red", alpha=0.3, edgecolor="black", label="Fixed macros"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkslateblue",
               markeredgecolor="darkslateblue", markersize=5, label="Macro pins"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
               markeredgecolor="darkgreen", markersize=6, label="I/O pins"),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    legend.set_zorder(10)
    # Invisible colorbar so panel 1 has the same width as panels 2 & 3
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_visible(False)

    # ── Panel 2: Density heatmap ─────────────────────────────────────────
    ax = axes[1]
    _draw_canvas(ax, benchmark)
    extent = (0, benchmark.canvas_width, 0, benchmark.canvas_height)

    if plc is not None:
        plc.get_density_cost()
        nrow, ncol = benchmark.grid_rows, benchmark.grid_cols
        dens = np.asarray(plc.grid_cells, dtype=float).reshape(nrow, ncol)
        vmax = max(float(np.max(dens)), 1e-9)
        im_dens = ax.imshow(
            dens, origin="lower", extent=extent, aspect="equal",
            cmap="Blues", alpha=0.6, vmin=0.0, vmax=vmax, zorder=0,
            interpolation="nearest",
        )
        fig.colorbar(im_dens, ax=ax, fraction=0.046, pad=0.04, label="Density")

    _draw_hard_macros(ax, placement, benchmark)
    ax.set_title(f"{benchmark.name} — Density")

    # ── Panel 3: Congestion heatmap ──────────────────────────────────────
    ax = axes[2]
    _draw_canvas(ax, benchmark)

    if plc is not None:
        plc.get_congestion_cost()
        nrow, ncol = benchmark.grid_rows, benchmark.grid_cols
        h_cong = np.asarray(plc.H_routing_cong, dtype=float).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=float).reshape(nrow, ncol)
        cong = np.maximum(h_cong, v_cong)
        pos = cong[cong > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        vmax = max(vmax, 1e-9)
        im_cong = ax.imshow(
            cong, origin="lower", extent=extent, aspect="equal",
            cmap="hot", alpha=0.6, vmin=0.0, vmax=vmax, zorder=0,
            interpolation="nearest",
        )
        fig.colorbar(im_cong, ax=ax, fraction=0.046, pad=0.04, label="Congestion (max H/V)")

    _draw_hard_macros(ax, placement, benchmark)
    ax.set_title(f"{benchmark.name} — Congestion")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def save_visualization_bundle(
    benchmark: Benchmark,
    initial_placement: torch.Tensor,
    final_placement: torch.Tensor,
    output_dir: str,
    plc=None,
    animation_frames: Optional[List[torch.Tensor]] = None,
    cost_history: Optional[List[dict]] = None,
    gif_name: str = "placement.gif",
):
    """
    Save a README-style media bundle for a benchmark run.

    Output layout:
      output_dir/
        initial.png
        final.png
        frames/frame_0000.png ...
        placement.gif   (if Pillow is available and multiple frames exist)
        cost_trace.gif  (if cost history is available)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualize_placement(initial_placement, benchmark, save_path=str(output_path / "initial.png"), plc=plc)
    visualize_placement(final_placement, benchmark, save_path=str(output_path / "final.png"), plc=plc)

    if animation_frames and len(animation_frames) >= 2:
        frames_dir = output_path / "frames"
        frames_dir.mkdir(exist_ok=True)

        frame_paths = []
        for idx, frame in enumerate(animation_frames):
            frame_path = frames_dir / f"frame_{idx:04d}.png"
            visualize_placement(frame, benchmark, save_path=str(frame_path), plc=plc)
            frame_paths.append(frame_path)

        try:
            from PIL import Image
        except ImportError:
            print(
                "Warning: Pillow not installed; saved PNG frames but skipped GIF generation.",
                file=sys.stderr,
            )
        else:
            images = []
            for frame_path in frame_paths:
                with Image.open(frame_path) as image:
                    images.append(image.convert("P", palette=Image.ADAPTIVE))

            if images:
                images[0].save(
                    output_path / gif_name,
                    save_all=True,
                    append_images=images[1:],
                    duration=120,
                    loop=0,
                    optimize=False,
                    disposal=2,
                )

    if cost_history and len(cost_history) >= 2:
        _save_cost_trace_gif(cost_history, output_path / "cost_trace.gif")


def _save_cost_trace_gif(cost_history: List[dict], gif_path: Path):
    """Save an animated cost trace GIF showing optimization progress over time."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
    except ImportError:
        print(
            "Warning: matplotlib or Pillow not installed; skipped cost trace GIF generation.",
            file=sys.stderr,
        )
        return

    steps = [point["step"] for point in cost_history]
    current = [point["current_hpwl"] for point in cost_history]
    best = [point["best_hpwl"] for point in cost_history]
    temps = [point["temperature"] for point in cost_history]

    y_min = min(min(current), min(best))
    y_max = max(max(current), max(best))
    if y_min == y_max:
        y_max = y_min + 1.0

    temp_min = min(temps)
    temp_max = max(temps)
    if temp_min == temp_max:
        temp_max = temp_min + 1.0

    frame_indices = list(range(1, len(cost_history) + 1))
    if len(frame_indices) > 80:
        stride = max(1, len(frame_indices) // 80)
        frame_indices = list(range(1, len(cost_history) + 1, stride))
        if frame_indices[-1] != len(cost_history):
            frame_indices.append(len(cost_history))

    images = []
    full_x_max = max(steps[-1], 1)
    for end in frame_indices:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(10, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.patch.set_facecolor("white")
        ax1.set_facecolor("white")
        ax2.set_facecolor("white")

        current_steps = steps[:end]
        current_vals = current[:end]
        best_vals = best[:end]
        temp_vals = temps[:end]

        ax1.plot(
            steps,
            current,
            color="lightsteelblue",
            linewidth=1.5,
            alpha=0.45,
        )
        ax1.plot(
            steps,
            best,
            color="navajowhite",
            linewidth=1.5,
            alpha=0.45,
        )
        ax2.plot(
            steps,
            temps,
            color="mediumseagreen",
            linewidth=1.25,
            alpha=0.35,
        )

        ax1.plot(
            current_steps,
            current_vals,
            color="steelblue",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Current HPWL",
        )
        ax1.plot(
            current_steps,
            best_vals,
            color="darkorange",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Best HPWL",
        )
        ax2.plot(
            current_steps,
            temp_vals,
            color="seagreen",
            linewidth=1.75,
            marker="o",
            markersize=2.5,
            label="Temperature",
        )

        current_step = current_steps[-1]
        ax1.axvline(current_step, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax2.axvline(current_step, color="gray", linestyle=":", linewidth=1, alpha=0.7)

        ax1.set_xlim(0, full_x_max)
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(temp_min, temp_max)
        ax1.set_ylabel("HPWL")
        ax2.set_ylabel("Temp")
        ax2.set_xlabel("Iteration")
        ax1.set_title("Optimization Trace")
        ax1.grid(True, alpha=0.25)
        ax2.grid(True, alpha=0.25)
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        fig.subplots_adjust(left=0.11, right=0.90, top=0.92, bottom=0.10, hspace=0.12)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        images.append(image)
        plt.close(fig)

    if not images:
        return

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
        disposal=2,
    )
