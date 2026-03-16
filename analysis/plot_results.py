"""
analysis/plot_results.py
Generates paper-ready figures from analysis/paper_results.json.

Outputs
-------
results/superseded_ablation.png  — Fig 1: ablation waterfall on Superseded bucket
results/bucket_tradeoff.png      — Fig 2: Unfiltered LoRA vs MemLoRA across all buckets

Usage
-----
    python3 analysis/plot_results.py [--results analysis/paper_results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless render
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Global styling ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        "#E0E0E0",
    "grid.linewidth":    0.8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ── Colour palette ────────────────────────────────────────────────────────────
C_ORACLE  = "#E69F00"   # gold  — upper bound
C_MAIN    = "#0072B2"   # blue  — MemLoRA (ours)
C_ABLATION_NEG  = "#56B4E9"   # light blue — ablation: no anti-memory
C_ABLATION_SAL  = "#CC79A7"   # mauve      — ablation: no salience
C_NAIVE   = "#D55E00"   # vermillion — baseline


def _load(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)["summary"]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 · Superseded Ablation Waterfall
# ─────────────────────────────────────────────────────────────────────────────
def plot_superseded_ablation(summary: dict, out_path: Path) -> None:
    """Bar chart of Superseded accuracy for key conditions, sorted high→low."""

    # Conditions shown (spec: Oracle-Data, MemLoRA, No Anti-Memory, No Salience, Naïve LoRA)
    entries = [
        ("oracle_data_lora",     "Oracle-Data LoRA\n(upper bound)", C_ORACLE),
        ("main",                 "MemLoRA\n(ours)",                  C_MAIN),
        ("ablation_no_negative", "No Anti-Memory\nPairs",            C_ABLATION_NEG),
        ("ablation_no_salience", "No Salience\nWeighting",           C_ABLATION_SAL),
        ("naive_lora",           "Naïve LoRA",                       C_NAIVE),
    ]

    means = [summary[k]["superseded"]["mean"] for k, _, _ in entries]
    stds  = [summary[k]["superseded"]["std"]  for k, _, _ in entries]
    n_seeds = [summary[k]["superseded"]["n_seeds"] for k, _, _ in entries]
    labels  = [label  for _, label, _ in entries]
    colors  = [color  for _, _, color  in entries]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.8)
    ax.xaxis.grid(False)

    x = np.arange(len(entries))
    bars = ax.bar(x, means, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="#333333", capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)

    # Value labels
    for bar, mean, std, n in zip(bars, means, stds, n_seeds):
        seed_note = f"n={n}" if n < 3 else ""
        label_text = f"{mean:.1f}%" + (f"\n({seed_note})" if seed_note else "")
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 1.5,
                label_text,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Superseded Accuracy (%)", fontsize=11)
    ax.set_ylim(0, max(means) + max(stds) + 14)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    # Annotate the MemLoRA↔No-Anti-Memory gap
    idx_main = 1
    idx_neg  = 2
    gap = means[idx_main] - means[idx_neg]
    ax.annotate(
        f"+{gap:.1f} pp from\nAnti-Memory pairs",
        xy=(idx_neg, means[idx_neg] + stds[idx_neg] + 2),
        xytext=(idx_neg - 0.25, means[idx_main] + stds[idx_main] + 9),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
        fontsize=9, color="#444444",
    )

    ax.set_title("Superseded Fact Rejection: Component Ablation\n"
                 "(3-seed mean ± std, 10 personas, zero-context evaluation)",
                 pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 · Unfiltered LoRA vs MemLoRA — per-bucket trade-off
# ─────────────────────────────────────────────────────────────────────────────
def plot_bucket_tradeoff(summary: dict, out_path: Path) -> None:
    """Grouped bar chart: Unfiltered LoRA vs MemLoRA across all four buckets."""

    buckets = ["stable", "updated", "superseded", "relational"]
    bucket_labels = ["Stable\nFacts", "Updated\nFacts", "Superseded\nFacts", "Relational\nFacts"]

    unfiltered_means = [summary["unfiltered_lora"][b]["mean"] for b in buckets]
    unfiltered_stds  = [summary["unfiltered_lora"][b]["std"]  for b in buckets]
    main_means       = [summary["main"][b]["mean"]            for b in buckets]
    main_stds        = [summary["main"][b]["std"]             for b in buckets]

    x     = np.arange(len(buckets))
    width = 0.36
    C_UNFILTERED = "#F0A500"   # amber — Unfiltered LoRA

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.8)
    ax.xaxis.grid(False)

    bars_u = ax.bar(x - width / 2, unfiltered_means, width,
                    color=C_UNFILTERED, label="Unfiltered LoRA",
                    edgecolor="white", linewidth=0.8, zorder=3)
    ax.errorbar(x - width / 2, unfiltered_means, yerr=unfiltered_stds,
                fmt="none", ecolor="#555555", capsize=5, capthick=1.5,
                elinewidth=1.5, zorder=4)

    bars_m = ax.bar(x + width / 2, main_means, width,
                    color=C_MAIN, label="MemLoRA (ours)",
                    edgecolor="white", linewidth=0.8, zorder=3)
    ax.errorbar(x + width / 2, main_means, yerr=main_stds,
                fmt="none", ecolor="#555555", capsize=5, capthick=1.5,
                elinewidth=1.5, zorder=4)

    # Difference annotations (MemLoRA − Unfiltered)
    for i, (um, mm) in enumerate(zip(unfiltered_means, main_means)):
        diff = mm - um
        sign = "+" if diff >= 0 else ""
        y_top = max(um + unfiltered_stds[i], mm + main_stds[i])
        ax.text(x[i], y_top + 2.5, f"{sign}{diff:.1f}",
                ha="center", va="bottom", fontsize=9, color="#333333",
                fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, max(unfiltered_means + main_means) + max(unfiltered_stds + main_stds) + 16)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    # Shade the Superseded bucket to highlight the trade-off direction
    ax.axvspan(2 - 0.5, 2 + 0.5, color="#F5F5F5", zorder=0, label="_nolegend_")
    ax.text(2, 2, "MemLoRA\nadvantage", ha="center", va="bottom",
            fontsize=8.5, color="#777777", style="italic")

    legend = ax.legend(frameon=True, framealpha=0.9, fontsize=10,
                       loc="upper left", handlelength=1.4)
    legend.get_frame().set_linewidth(0.5)

    ax.set_title("Volume vs. Precision Trade-off:\n"
                 "Unfiltered LoRA vs MemLoRA across Memory Buckets\n"
                 "(3-seed mean ± std, 10 personas, zero-context evaluation)",
                 pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from paper_results.json")
    parser.add_argument(
        "--results",
        default="analysis/paper_results.json",
        help="Path to paper_results.json (default: analysis/paper_results.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for PNG files (default: results/)",
    )
    args = parser.parse_args()

    summary = _load(args.results)
    out_dir = Path(args.out_dir)

    print("Generating plots …")
    plot_superseded_ablation(summary, out_dir / "superseded_ablation.png")
    plot_bucket_tradeoff(summary,     out_dir / "bucket_tradeoff.png")
    print("Done.")


if __name__ == "__main__":
    main()
