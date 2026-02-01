"""
Plot utilities for LABS benchmark results.
Plots normalized_distance vs N with one line per method.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/CI
import matplotlib.pyplot as plt


def plot_normalized_distance_vs_n(
    csv_path: Path,
    out_path: Path | None = None,
    title: str = "Normalized energy distance vs N",
) -> None:
    """
    Read benchmark CSV and plot N (x-axis) vs normalized_distance (y-axis),
    one line per method, with legend.

    Args:
        csv_path: Path to results CSV (N, method, normalized_distance, ...).
        out_path: If set, save figure to this path; otherwise show interactively.
        title: Plot title.
    """
    # Read CSV and group by method
    by_method: dict[str, list[tuple[int, float]]] = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            try:
                n_val = int(row["N"])
            except (KeyError, ValueError):
                continue
            nd = row.get("normalized_distance", "").strip()
            if nd == "" or nd.lower().startswith("error"):
                continue
            try:
                nd_val = float(nd)
            except ValueError:
                continue
            if method not in by_method:
                by_method[method] = []
            by_method[method].append((n_val, nd_val))

    if not by_method:
        print("No valid normalized_distance data to plot.")
        return

    # Sort each method's points by N and dedupe by N (keep last per N)
    for method in by_method:
        points = by_method[method]
        by_n = {}
        for n, nd in sorted(points, key=lambda x: x[0]):
            by_n[n] = nd
        by_method[method] = sorted(by_n.items(), key=lambda x: x[0])

    fig, ax = plt.subplots()
    for method, points in sorted(by_method.items(), key=lambda x: x[0]):
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", markersize=4, label=method)

    ax.set_xlabel("N")
    ax.set_ylabel("Normalized distance")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()
    plt.close(fig)
