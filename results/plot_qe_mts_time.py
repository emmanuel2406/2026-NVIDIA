"""
Plot QE-MTS run time (seconds) vs N for three methods with custom labels.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Method name in CSV -> display label
LABELS = {
    "QE-MTS (Image H)": "Custom Ansatz",
    "QE-MTS (LABS H)": "Nvidia",
    "QE-MTS (LABS OPT H)": "Our solution",
}

QE_MTS_METHODS = list(LABELS.keys())


def main():
    csv_path = Path(__file__).resolve().parent / "qe_mts_multi_n_results.csv"
    out_path = Path(__file__).resolve().parent / "qe_mts_time_vs_n.png"

    by_method: dict[str, list[tuple[int, float]]] = {label: [] for label in LABELS.values()}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            if method not in LABELS:
                continue
            try:
                n_val = int(row["N"])
                time_val = float(row["time_sec"])
            except (KeyError, ValueError):
                continue
            label = LABELS[method]
            by_method[label].append((n_val, time_val))

    # Sort by N per method
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
    ax.set_title("E2E Run Time vs Sequence Length N on H100 GPU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
