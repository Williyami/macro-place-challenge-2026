from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
NOTES_ROOT = ROOT.parent


@dataclass
class BenchmarkRow:
    benchmark: str
    proxy: float
    wl: float | None
    density: float | None
    congestion: float | None
    time_s: float | None


@dataclass
class RunRecord:
    run_id: str
    method: str
    date: str
    title: str
    scope: str
    benchmarks_logged: int
    plotted_proxy: float
    full_suite_avg_proxy: float | None
    total_runtime_s: float | None
    source_file: str
    source_heading: str


def parse_time_seconds(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)s", text)
    return float(match.group(1)) if match else None


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_learning() -> tuple[list[RunRecord], list[dict[str, object]]]:
    path = NOTES_ROOT / "(RL method)eklundnotes.md"
    text = path.read_text()
    blocks = re.split(r"(?=^## )", text, flags=re.M)

    runs: list[RunRecord] = []
    rows: list[dict[str, object]] = []
    run_index = 0

    bullet_re = re.compile(
        r"^- (ibm\d+): proxy=([0-9.]+) \(wl=([0-9.]+) den=([0-9.]+) cong=([0-9.]+)\) ([0-9.]+)s",
        re.M,
    )

    for block in blocks:
        if not block.startswith("## "):
            continue
        heading = block.splitlines()[0][3:].strip()
        if "—" in heading:
            date, title = [part.strip() for part in heading.split("—", 1)]
        else:
            date, title = "", heading

        method = "Learning Placer"
        benchmarks: list[BenchmarkRow] = []

        for match in bullet_re.finditer(block):
            benchmarks.append(
                BenchmarkRow(
                    benchmark=match.group(1),
                    proxy=float(match.group(2)),
                    wl=float(match.group(3)),
                    density=float(match.group(4)),
                    congestion=float(match.group(5)),
                    time_s=float(match.group(6)),
                )
            )

        full_suite_avg = None
        table_match = re.search(
            r"^\| Benchmark \|.*?^\|\s*\*\*AVG\*\*\s*\|\s*\*\*([0-9.]+)\*\*.*$",
            block,
            flags=re.M | re.S,
        )
        if table_match:
            full_suite_avg = float(table_match.group(1))
            table_lines = [line for line in block.splitlines() if line.startswith("|")]
            for line in table_lines:
                parts = [part.strip() for part in line.strip().strip("|").split("|")]
                if len(parts) < 6:
                    continue
                if parts[0] in {"Benchmark", "-----------", "**AVG**"}:
                    continue
                if not re.fullmatch(r"ibm\d+", parts[0]):
                    continue
                benchmarks.append(
                    BenchmarkRow(
                        benchmark=parts[0],
                        proxy=float(parts[1]),
                        wl=_safe_float(parts[2]),
                        density=_safe_float(parts[3]),
                        congestion=_safe_float(parts[4]),
                        time_s=parse_time_seconds(parts[5]),
                    )
                )

        if not benchmarks:
            continue

        run_index += 1
        run_id = f"L{run_index}"
        plotted_proxy = full_suite_avg if full_suite_avg is not None else mean(b.proxy for b in benchmarks)
        total_runtime_s = sum(b.time_s for b in benchmarks if b.time_s is not None) or None
        scope = "full_suite" if full_suite_avg is not None else "partial"
        runs.append(
            RunRecord(
                run_id=run_id,
                method=method,
                date=date,
                title=title,
                scope=scope,
                benchmarks_logged=len(benchmarks),
                plotted_proxy=plotted_proxy,
                full_suite_avg_proxy=full_suite_avg,
                total_runtime_s=total_runtime_s,
                source_file=path.name,
                source_heading=heading,
            )
        )
        for benchmark in benchmarks:
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "benchmark": benchmark.benchmark,
                    "proxy": benchmark.proxy,
                    "wl": benchmark.wl,
                    "density": benchmark.density,
                    "congestion": benchmark.congestion,
                    "time_s": benchmark.time_s,
                    "source_file": path.name,
                }
            )

    return runs, rows


def parse_hybrid() -> tuple[list[RunRecord], list[dict[str, object]]]:
    path = NOTES_ROOT / "(Hybrid method)novaknotes.md"
    text = path.read_text()
    runs: list[RunRecord] = []
    rows: list[dict[str, object]] = []
    sections = re.split(r"(?=^## Benchmark Results \()", text, flags=re.M)
    hybrid_count = 0

    for section in sections:
        if not section.startswith("## Benchmark Results ("):
            continue
        hybrid_count += 1
        run_id = f"H{hybrid_count}"
        date_match = re.match(r"## Benchmark Results \(([^)]+)\)", section)
        date = date_match.group(1) if date_match else ""
        runtime_match = re.search(r"Total runtime:\s+\*\*([0-9.]+)s", section)
        total_runtime_s = float(runtime_match.group(1)) if runtime_match else None

        benchmarks: list[BenchmarkRow] = []
        # Only parse the FIRST table in the section (stop after AVG row)
        in_table = False
        for line in section.splitlines():
            if not line.startswith("|"):
                if in_table and benchmarks:
                    break  # left the first table, stop
                continue
            in_table = True
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if parts[0] in {"Benchmark", "-----------"}:
                continue
            if parts[0] == "**AVG**":
                break  # end of first table
            if not re.fullmatch(r"ibm\d+", parts[0]):
                continue
            # Support both the original detailed hybrid table:
            # Benchmark | Proxy | WL | Density | Congestion | ... | Time
            # and later summary-only tables:
            # Benchmark | Proxy | SA Baseline | RePlAce | vs SA | vs RePlAce | Overlaps
            wl = density = congestion = time_s = None
            if len(parts) >= 10:
                wl = _safe_float(parts[2])
                density = _safe_float(parts[3])
                congestion = _safe_float(parts[4])
                time_s = parse_time_seconds(parts[9])
            elif len(parts) >= 7:
                wl = density = congestion = None
                time_s = None
            else:
                continue
            benchmarks.append(
                BenchmarkRow(
                    benchmark=parts[0],
                    proxy=float(parts[1]),
                    wl=wl,
                    density=density,
                    congestion=congestion,
                    time_s=time_s,
                )
            )

        avg_match = re.search(r"^\|\s*\*\*AVG\*\*\s*\|\s*\*\*([0-9.]+)\*\*", section, flags=re.M)
        full_suite_avg = float(avg_match.group(1)) if avg_match else None
        runs.append(
            RunRecord(
                run_id=run_id,
                method="HybridPlacer",
                date=date,
                title="Analytical -> SA pipeline benchmark run",
                scope="full_suite",
                benchmarks_logged=len(benchmarks),
                plotted_proxy=full_suite_avg if full_suite_avg is not None else mean(b.proxy for b in benchmarks),
                full_suite_avg_proxy=full_suite_avg,
                total_runtime_s=total_runtime_s,
                source_file=path.name,
                source_heading="Benchmark Results",
            )
        )
        for benchmark in benchmarks:
            rows.append(
                {
                    "run_id": run_id,
                    "method": "HybridPlacer",
                    "benchmark": benchmark.benchmark,
                    "proxy": benchmark.proxy,
                    "wl": benchmark.wl,
                    "density": benchmark.density,
                    "congestion": benchmark.congestion,
                    "time_s": benchmark.time_s,
                    "source_file": path.name,
                }
            )

    return runs, rows


def parse_sa_analytical() -> tuple[list[RunRecord], list[dict[str, object]]]:
    path = NOTES_ROOT / "(SA + Analytical method) Omnellnotes.md"
    text = path.read_text()
    runs: list[RunRecord] = []
    rows: list[dict[str, object]] = []
    sections = re.split(r"(?=^## Benchmark Results \()", text, flags=re.M)
    sa_count = 0
    analytical_count = 0

    for section in sections:
        if not section.startswith("## Benchmark Results ("):
            continue
        date_match = re.match(r"## Benchmark Results \(([^)]+)\)", section)
        date = date_match.group(1) if date_match else ""
        blocks = re.split(r"(?=^### )", section, flags=re.M)

        for block in blocks:
            if not block.startswith("### "):
                continue
            heading = block.splitlines()[0][4:].strip()
            if heading.startswith("SA Placer"):
                sa_count += 1
                run_id, method = f"S{sa_count}", "SA Placer"
            elif heading.startswith("Analytical Placer"):
                analytical_count += 1
                run_id, method = f"A{analytical_count}", "Analytical Placer"
            else:
                continue

            benchmarks: list[BenchmarkRow] = []
            for line in block.splitlines():
                if not line.startswith("|"):
                    continue
                parts = [part.strip() for part in line.strip().strip("|").split("|")]
                if len(parts) < 6:
                    continue
                if parts[0] in {"Benchmark", "-----------", "**AVG**"}:
                    continue
                if not re.fullmatch(r"ibm\d+", parts[0]):
                    continue
                benchmarks.append(
                    BenchmarkRow(
                        benchmark=parts[0],
                        proxy=float(parts[1]),
                        wl=_safe_float(parts[2]),
                        density=_safe_float(parts[3]),
                        congestion=_safe_float(parts[4]),
                        time_s=parse_time_seconds(parts[5]),
                    )
                )

            avg_match = re.search(
                r"^\|\s*\*\*AVG\*\*\s*\|\s*\*?\*?([0-9.]+)\*?\*?.*?([0-9.]+)s\s*\|?$",
                block,
                flags=re.M,
            )
            avg_proxy = float(avg_match.group(1)) if avg_match else mean(b.proxy for b in benchmarks)
            total_runtime_s = float(avg_match.group(2)) if avg_match else None
            runs.append(
                RunRecord(
                    run_id=run_id,
                    method=method,
                    date=date,
                    title=heading,
                    scope="full_suite",
                    benchmarks_logged=len(benchmarks),
                    plotted_proxy=avg_proxy,
                    full_suite_avg_proxy=avg_proxy,
                    total_runtime_s=total_runtime_s,
                    source_file=path.name,
                    source_heading=heading,
                )
            )
            for benchmark in benchmarks:
                rows.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "benchmark": benchmark.benchmark,
                        "proxy": benchmark.proxy,
                        "wl": benchmark.wl,
                        "density": benchmark.density,
                        "congestion": benchmark.congestion,
                        "time_s": benchmark.time_s,
                        "source_file": path.name,
                    }
                )

    return runs, rows


def build_markdown(runs: list[RunRecord], rows: list[dict[str, object]]) -> str:
    out: list[str] = []
    out.append("# Benchmark History Raw Data")
    out.append("")
    out.append("Generated from the note files in this directory.")
    out.append("")
    out.append("## Run-Level Records")
    out.append("")
    out.append("| Run ID | Method | Date | Title | Scope | Benchmarks Logged | Plotted Proxy | Full-Suite Avg Proxy | Total Runtime (s) | Source |")
    out.append("|--------|--------|------|-------|-------|-------------------|---------------|----------------------|-------------------|--------|")
    for run in runs:
        full_avg = f"{run.full_suite_avg_proxy:.4f}" if run.full_suite_avg_proxy is not None else "-"
        total_runtime = f"{run.total_runtime_s:.2f}" if run.total_runtime_s is not None else "-"
        out.append(
            f"| {run.run_id} | {run.method} | {run.date or '-'} | {run.title} | {run.scope} | "
            f"{run.benchmarks_logged} | {run.plotted_proxy:.4f} | {full_avg} | {total_runtime} | "
            f"{run.source_file} |"
        )

    out.append("")
    out.append("## Benchmark-Level Records")
    out.append("")
    out.append("| Run ID | Method | Benchmark | Proxy | WL | Density | Congestion | Time (s) | Source |")
    out.append("|--------|--------|-----------|-------|----|---------|------------|----------|--------|")
    for row in rows:
        wl = f"{row['wl']:.3f}" if row["wl"] is not None else "-"
        density = f"{row['density']:.3f}" if row["density"] is not None else "-"
        congestion = f"{row['congestion']:.3f}" if row["congestion"] is not None else "-"
        time_s = f"{row['time_s']:.2f}" if row["time_s"] is not None else "-"
        out.append(
            f"| {row['run_id']} | {row['method']} | {row['benchmark']} | {row['proxy']:.4f} | "
            f"{wl} | {density} | {congestion} | {time_s} | {row['source_file']} |"
        )

    out.append("")
    return "\n".join(out)


def write_summary_markdown(runs: list[RunRecord]) -> None:
    path = ROOT / "benchmark_history_summary.md"
    lines = [
        "# Benchmark History Summary",
        "",
        "This graph summarizes the full-suite benchmark sections logged in the notes.",
        "",
        "- Top plot: proxy trend over run history using the logged AVG proxy from each full-suite run.",
        "- Bottom plot: how many benchmarks were logged in each included run.",
        "",
        "![Benchmark history summary](benchmark_history_full_suite.png)",
        "",
        "## Included Runs",
        "",
        "| Run ID | Method | Date | Scope | Plotted Proxy | Benchmarks Logged |",
        "|--------|--------|------|-------|---------------|-------------------|",
    ]
    for run in runs:
        lines.append(
            f"| {run.run_id} | {run.method} | {run.date or '-'} | {run.scope} | "
            f"{run.plotted_proxy:.4f} | {run.benchmarks_logged} |"
        )
    lines.append("")
    lines.append("Raw parsed data: [benchmark_history_raw.md](benchmark_history_raw.md)")
    lines.append("")
    path.write_text("\n".join(lines))


def plot_history(runs: list[RunRecord]) -> None:
    method_order = ["Learning Placer", "SA Placer", "Analytical Placer", "HybridPlacer", "SA V2 (Eklund)"]
    colors = {
        "Learning Placer": "#1f77b4",
        "SA Placer": "#d62728",
        "Analytical Placer": "#2ca02c",
        "HybridPlacer": "#ff7f0e",
        "SA V2 (Eklund)": "#9467bd",
    }

    x_positions = {run.run_id: idx for idx, run in enumerate(runs, start=1)}
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    for method in method_order:
        method_runs = [run for run in runs if run.method == method]
        if not method_runs:
            continue
        xs = [x_positions[run.run_id] for run in method_runs]
        ys = [run.plotted_proxy for run in method_runs]
        ax1.plot(xs, ys, color=colors[method], linewidth=2, alpha=0.75, label=method)
        for run in method_runs:
            marker = "o" if run.scope == "full_suite" else "s"
            ax1.scatter(
                x_positions[run.run_id],
                run.plotted_proxy,
                color=colors[method],
                s=80,
                marker=marker,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            ax1.annotate(
                run.run_id,
                (x_positions[run.run_id], run.plotted_proxy),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    ax1.set_ylabel("Proxy Cost")
    ax1.set_title("Benchmark History From Notes")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right")

    xs = [x_positions[run.run_id] for run in runs]
    counts = [run.benchmarks_logged for run in runs]
    bar_colors = [colors[run.method] for run in runs]
    ax2.bar(xs, counts, color=bar_colors, alpha=0.85)
    for run in runs:
        ax2.annotate(
            str(run.benchmarks_logged),
            (x_positions[run.run_id], run.benchmarks_logged),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )
    ax2.set_ylabel("Benchmarks")
    ax2.set_xlabel("Run Sequence")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([run.run_id for run in runs], rotation=0)

    fig.tight_layout()
    fig.savefig(ROOT / "benchmark_history_full_suite.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_sa_v2() -> tuple[list[RunRecord], list[dict[str, object]]]:
    """Parse SA V2 (Eklund) results from the RL/Eklund notes file.

    SA V2 sections are identified by headings containing 'SA V2' or
    'sa_v2' and use the same table format as the Learning Placer entries.
    """
    path = NOTES_ROOT / "(RL method)eklundnotes.md"
    if not path.exists():
        return [], []
    text = path.read_text()
    blocks = re.split(r"(?=^## )", text, flags=re.M)

    runs: list[RunRecord] = []
    rows: list[dict[str, object]] = []
    run_index = 0

    for block in blocks:
        if not block.startswith("## "):
            continue
        heading = block.splitlines()[0][3:].strip()
        # Only parse SA V2 sections
        if "sa_v2" not in heading.lower() and "sa v2" not in heading.lower():
            continue

        if "—" in heading:
            date, title = [part.strip() for part in heading.split("—", 1)]
        else:
            date, title = "", heading

        method = "SA V2 (Eklund)"
        benchmarks: list[BenchmarkRow] = []

        full_suite_avg = None
        table_match = re.search(
            r"^\| Benchmark \|.*?^\|\s*\*\*AVG\*\*\s*\|\s*\*\*([0-9.]+)\*\*.*$",
            block,
            flags=re.M | re.S,
        )
        if table_match:
            full_suite_avg = float(table_match.group(1))

        for line in block.splitlines():
            if not line.startswith("|"):
                continue
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) < 6:
                continue
            if parts[0] in {"Benchmark", "-----------", "**AVG**"}:
                continue
            if not re.fullmatch(r"ibm\d+", parts[0]):
                continue
            benchmarks.append(
                BenchmarkRow(
                    benchmark=parts[0],
                    proxy=float(parts[1]),
                    wl=_safe_float(parts[2]),
                    density=_safe_float(parts[3]),
                    congestion=_safe_float(parts[4]),
                    time_s=parse_time_seconds(parts[5]),
                )
            )

        if not benchmarks:
            continue

        run_index += 1
        run_id = f"V{run_index}"
        plotted_proxy = full_suite_avg if full_suite_avg is not None else mean(b.proxy for b in benchmarks)
        total_runtime_s = sum(b.time_s for b in benchmarks if b.time_s is not None) or None
        scope = "full_suite" if full_suite_avg is not None else "partial"
        runs.append(
            RunRecord(
                run_id=run_id,
                method=method,
                date=date,
                title=title,
                scope=scope,
                benchmarks_logged=len(benchmarks),
                plotted_proxy=plotted_proxy,
                full_suite_avg_proxy=full_suite_avg,
                total_runtime_s=total_runtime_s,
                source_file=path.name,
                source_heading=heading,
            )
        )
        for benchmark in benchmarks:
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "benchmark": benchmark.benchmark,
                    "proxy": benchmark.proxy,
                    "wl": benchmark.wl,
                    "density": benchmark.density,
                    "congestion": benchmark.congestion,
                    "time_s": benchmark.time_s,
                    "source_file": path.name,
                }
            )

    return runs, rows


def main() -> None:
    runs: list[RunRecord] = []
    rows: list[dict[str, object]] = []

    for parser in (parse_learning, parse_sa_analytical, parse_hybrid, parse_sa_v2):
        parser_runs, parser_rows = parser()
        runs.extend(parser_runs)
        rows.extend(parser_rows)

    full_suite_ids = {run.run_id for run in runs if run.scope == "full_suite"}
    runs = [run for run in runs if run.run_id in full_suite_ids]
    rows = [row for row in rows if str(row["run_id"]) in full_suite_ids]

    def run_sort_key(run_id: str) -> tuple[int, int]:
        prefix_order = {"L": 1, "S": 2, "A": 3, "H": 4, "V": 5}
        match = re.fullmatch(r"([A-Z])(\d+)", run_id)
        if not match:
            return (999, 999)
        return (prefix_order.get(match.group(1), 999), int(match.group(2)))

    runs.sort(key=lambda run: run_sort_key(run.run_id))
    rows.sort(key=lambda row: (run_sort_key(str(row["run_id"])), str(row["benchmark"])))

    (ROOT / "benchmark_history_raw.md").write_text(build_markdown(runs, rows))
    write_summary_markdown(runs)
    plot_history(runs)


if __name__ == "__main__":
    main()
