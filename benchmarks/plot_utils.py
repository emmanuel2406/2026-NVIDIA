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


def plot_energies_bar(
    csv_path: Path,
    out_path: Path | None = None,
    title_prefix: str = "Energy by method",
) -> None:
    """
    Read benchmark CSV and plot energy per method as bar charts (one subplot per N),
    with a horizontal red dotted line for the optimal energy.

    Args:
        csv_path: Path to results CSV (N, method, energy, optimal_energy, ...).
        out_path: If set, save figure to this path; otherwise show interactively.
        title_prefix: Prefix for subplot titles (suffix will be "N = {n}").
    """
    # Group by N: N -> { method -> energy, optimal_energy }
    by_n: dict[int, dict] = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_val = int(row["N"])
            except (KeyError, ValueError):
                continue
            method = row.get("method", "").strip()
            energy_str = row.get("energy", "").strip()
            opt_str = row.get("optimal_energy", "").strip()
            if energy_str == "" or energy_str.lower().startswith("error"):
                continue
            try:
                energy = float(energy_str)
            except ValueError:
                continue
            opt_energy = None
            if opt_str and not opt_str.lower().startswith("error"):
                try:
                    opt_energy = float(opt_str)
                except ValueError:
                    pass

            if n_val not in by_n:
                by_n[n_val] = {"methods": {}, "optimal_energy": opt_energy}
            by_n[n_val]["methods"][method] = energy
            if opt_energy is not None:
                by_n[n_val]["optimal_energy"] = opt_energy

    if not by_n:
        print("No valid energy data to plot.")
        return

    n_values = sorted(by_n.keys())
    n_sub = len(n_values)
    n_cols = min(3, n_sub)
    n_rows = (n_sub + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
    axes_flat = axes.ravel()

    for idx, N in enumerate(n_values):
        ax = axes_flat[idx]
        data = by_n[N]
        methods = list(data["methods"].keys())
        energies = [data["methods"][m] for m in methods]
        opt = data.get("optimal_energy")

        x = range(len(methods))
        bars = ax.bar(x, energies, color="steelblue", edgecolor="navy", alpha=0.85)
        if opt is not None:
            ax.axhline(y=opt, color="red", linestyle="--", linewidth=1.5, label="Optimal energy")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Energy", fontsize=9)
        ax.set_title(f"{title_prefix} — N = {N}", fontsize=10)
        if opt is not None:
            ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved energy bar plot to {out_path}")
    else:
        plt.show()
    plt.close(fig)


# QE-MTS method name in CSV -> display label for time plot
QE_MTS_TIME_LABELS = {
    "QE-MTS (Image H)": "Custom Ansatz",
    "QE-MTS (LABS H)": "Nvidia",
    "QE-MTS (LABS OPT H)": "Our solution",
}


def plot_qe_mts_time_vs_n(
    csv_path: Path,
    out_path: Path | None = None,
    title: str = "QE-MTS run time vs sequence length N",
) -> None:
    """
    Read benchmark CSV and plot N (x-axis) vs time_sec (y-axis) for the three
    QE-MTS methods only, with custom labels: Custom Ansatz, Nvidia, Our solution.

    Args:
        csv_path: Path to results CSV (N, method, time_sec, ...).
        out_path: If set, save figure to this path; otherwise show interactively.
        title: Plot title.
    """
    by_method: dict[str, list[tuple[int, float]]] = {
        label: [] for label in QE_MTS_TIME_LABELS.values()
    }

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            if method not in QE_MTS_TIME_LABELS:
                continue
            try:
                n_val = int(row["N"])
                time_val = float(row["time_sec"])
            except (KeyError, ValueError):
                continue
            label = QE_MTS_TIME_LABELS[method]
            by_method[label].append((n_val, time_val))

    for label in by_method:
        by_method[label] = sorted(by_method[label], key=lambda x: x[0])

    fig, ax = plt.subplots()
    for label in ["Custom Ansatz", "Nvidia", "Our solution"]:
        points = by_method.get(label, [])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", markersize=8, label=label, linewidth=2, alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()
    plt.close(fig)
