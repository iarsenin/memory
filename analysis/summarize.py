"""
analysis/summarize.py — Paper-run aggregation.

Reads per-seed eval JSONs from results/paper/seed{S}/,
computes mean ± std accuracy across seeds for each (condition, bucket),
and outputs:
  • analysis/paper_results.md   — paper-ready Markdown table
  • analysis/paper_results.json — machine-readable summary

frozen and rag are deterministic (no LoRA adapter), so their results are
read from seed 42 only and reported with std = 0.

Usage
-----
  python analysis/summarize.py
  python analysis/summarize.py --results-dir results/paper --seeds 42 123 456
  python analysis/summarize.py --results-dir results/paper --out analysis/paper_results
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

CONDITION_ORDER = [
    "frozen",
    "rag",
    "naive_lora",
    "unfiltered_lora",
    "gold_lora",
    "main",
]

CONDITION_LABELS = {
    "frozen":          "Frozen (base)",
    "rag":             "RAG (no adapter)",
    "naive_lora":      "Naïve LoRA",
    "unfiltered_lora": "Unfiltered LoRA",
    "gold_lora":       "Gold LoRA (upper bound)",
    "main":            "**MemLoRA (ours)**",
}

BUCKET_ORDER  = ["stable", "updated", "superseded", "relational", "overall"]
BUCKET_LABELS = {
    "stable":     "Stable",
    "updated":    "Updated",
    "superseded": "Superseded",
    "relational": "Relational",
    "overall":    "Overall",
}

# These conditions have no LoRA adapter; results are deterministic across seeds.
DETERMINISTIC_CONDITIONS = {"frozen", "rag"}


# ── I/O ───────────────────────────────────────────────────────────────────────


def _load_eval_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _probes_to_bucket_accuracy(probes: list[dict]) -> dict[str, float]:
    """
    Compute per-bucket accuracy from a flat list of probe dicts.
    Each probe has 'bucket' and 'score_numeric' (0.0, 0.5, or 1.0).
    Returns {bucket: accuracy, 'overall': accuracy}.
    """
    bucket_scores: dict[str, list[float]] = defaultdict(list)
    for p in probes:
        bucket = p.get("bucket", "")
        score  = p.get("score_numeric", 0.0)
        if bucket:
            bucket_scores[bucket].append(score)

    result: dict[str, float] = {}
    all_scores: list[float] = []
    for bucket, scores in bucket_scores.items():
        result[bucket] = sum(scores) / len(scores) if scores else 0.0
        all_scores.extend(scores)
    result["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return result


def _find_eval_files(
    results_dir: Path,
    seeds: list[int],
    conditions: list[str],
    persona_ids: list[str],
) -> dict[tuple[int, str, str], dict[str, float]]:
    """Return {(seed, condition, persona): {bucket: accuracy}} for all discovered files."""
    data: dict[tuple[int, str, str], dict[str, float]] = {}

    for seed in seeds:
        seed_dir = results_dir / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for condition in conditions:
            for pid in persona_ids:
                path = seed_dir / f"{condition}_{pid}_eval.json"
                if path.exists():
                    probes = _load_eval_json(path)
                    data[(seed, condition, pid)] = _probes_to_bucket_accuracy(probes)
    return data


# ── Aggregation ───────────────────────────────────────────────────────────────


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    variance = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, math.sqrt(variance)


def aggregate(
    data: dict[tuple[int, str, str], dict],
    seeds: list[int],
    conditions: list[str],
    persona_ids: list[str],
) -> dict[str, dict[str, dict]]:
    """
    Returns {condition: {bucket: {"mean": float, "std": float, "n_seeds": int}}}.
    Each condition is averaged first over personas, then over seeds.
    """
    # {condition: {bucket: [per-seed-mean-over-personas]}}
    seed_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for condition in conditions:
        # For deterministic conditions, only use seed 42 (or the first available seed)
        effective_seeds = [seeds[0]] if condition in DETERMINISTIC_CONDITIONS else seeds

        for seed in effective_seeds:
            # Collect per-persona accuracies for this (seed, condition)
            bucket_per_persona: dict[str, list[float]] = defaultdict(list)
            for pid in persona_ids:
                key = (seed, condition, pid)
                if key not in data:
                    continue
                bucket_accs = data[key]  # {bucket: float}
                for bucket in BUCKET_ORDER:
                    acc = bucket_accs.get(bucket)
                    if acc is not None:
                        bucket_per_persona[bucket].append(acc)

            # Average over personas for this seed
            for bucket, accs in bucket_per_persona.items():
                if accs:
                    seed_scores[condition][bucket].append(sum(accs) / len(accs))

    # Compute mean ± std across seeds
    summary: dict[str, dict[str, dict]] = {}
    for condition in conditions:
        summary[condition] = {}
        for bucket in BUCKET_ORDER:
            vals = seed_scores[condition].get(bucket, [])
            mu, sigma = _mean_std(vals)
            summary[condition][bucket] = {
                "mean":    round(mu * 100, 1),
                "std":     round(sigma * 100, 1),
                "n_seeds": len(vals),
            }
    return summary


# ── Formatting ────────────────────────────────────────────────────────────────


def _cell(mean: float, std: float, n_seeds: int, bold: bool = False) -> str:
    if n_seeds == 0:
        return "—"
    val = f"{mean:.1f}"
    if std > 0:
        val += f" ±{std:.1f}"
    if bold:
        val = f"**{val}**"
    return val


def build_markdown_table(
    summary: dict[str, dict[str, dict]],
    conditions: list[str],
) -> str:
    # Header
    bucket_cols = [BUCKET_LABELS[b] for b in BUCKET_ORDER]
    header  = "| Condition | " + " | ".join(bucket_cols) + " |"
    sep     = "|" + "|".join([":---"] + [":---:"] * len(BUCKET_ORDER)) + "|"
    rows    = [header, sep]

    for condition in conditions:
        label = CONDITION_LABELS.get(condition, condition)
        cells = []
        cond_data = summary.get(condition, {})
        for bucket in BUCKET_ORDER:
            bd = cond_data.get(bucket, {})
            mean   = bd.get("mean", 0.0)
            std    = bd.get("std", 0.0)
            n      = bd.get("n_seeds", 0)
            is_main = condition == "main" and bucket == "overall"
            cells.append(_cell(mean, std, n, bold=is_main))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def build_per_seed_table(
    data: dict[tuple[int, str, str], dict],
    seeds: list[int],
    conditions: list[str],
    persona_ids: list[str],
) -> str:
    """Transparent per-seed overall accuracy table for appendix / sanity check."""
    lines = ["### Per-Seed Overall Accuracy", ""]
    col_header = "| Condition | " + " | ".join(f"Seed {s}" for s in seeds) + " |"
    col_sep    = "|:---|" + "|".join([":---:"] * len(seeds)) + "|"
    lines.extend([col_header, col_sep])

    for condition in conditions:
        effective_seeds = [seeds[0]] if condition in DETERMINISTIC_CONDITIONS else seeds
        label = CONDITION_LABELS.get(condition, condition)
        cells = []
        for seed in seeds:
            if seed not in effective_seeds:
                cells.append("(=seed42)")
                continue
            accs = []
            for pid in persona_ids:
                key = (seed, condition, pid)
                if key in data:
                    acc = data[key].get("overall")
                    if acc is not None:
                        accs.append(acc * 100)
            if accs:
                cells.append(f"{sum(accs)/len(accs):.1f}%")
            else:
                cells.append("—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paper-run results")
    parser.add_argument("--results-dir", default="results/paper",
                        help="Base results directory containing seed{N}/ subdirs")
    parser.add_argument("--seeds",       nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--conditions",  nargs="+", default=None,
                        help="Subset of conditions to include (default: all 6)")
    parser.add_argument("--personas",    nargs="+", default=None,
                        help="Persona IDs (default: alice bob)")
    parser.add_argument("--out",         default="analysis/paper_results",
                        help="Output path prefix (no extension)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    conditions  = args.conditions or CONDITION_ORDER
    persona_ids = args.personas   or ["alice", "bob"]
    seeds       = args.seeds
    out_prefix  = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== MemLoRA Paper Results Aggregation ===")
    print(f"  Results dir : {results_dir}")
    print(f"  Seeds       : {seeds}")
    print(f"  Conditions  : {conditions}")
    print(f"  Personas    : {persona_ids}")

    # ── Load data ──────────────────────────────────────────────────────────
    data = _find_eval_files(results_dir, seeds, conditions, persona_ids)
    print(f"\n  Loaded {len(data)} eval file(s)")
    if not data:
        print("\nNo eval files found. Check --results-dir and run the paper loop first.")
        return

    # ── Aggregate ──────────────────────────────────────────────────────────
    summary = aggregate(data, seeds, conditions, persona_ids)

    # ── Markdown table ─────────────────────────────────────────────────────
    md_lines = [
        "# MemLoRA v1 — Paper Results",
        "",
        f"Seeds: {seeds} · Personas: {persona_ids}",
        f"Accuracy = mean ± std (%) across seeds. Frozen/RAG std = 0 (deterministic).",
        "",
        "## Main Results Table",
        "",
        build_markdown_table(summary, conditions),
        "",
        build_per_seed_table(data, seeds, conditions, persona_ids),
        "",
    ]

    # Detailed bucket breakdown per condition
    md_lines += ["## Bucket Detail (mean % ± std)"]
    for condition in conditions:
        label     = CONDITION_LABELS.get(condition, condition)
        cond_data = summary.get(condition, {})
        md_lines.append(f"\n### {label}")
        for bucket in BUCKET_ORDER:
            bd = cond_data.get(bucket, {})
            mean, std, n = bd.get("mean", 0.0), bd.get("std", 0.0), bd.get("n_seeds", 0)
            md_lines.append(f"  - {BUCKET_LABELS.get(bucket, bucket)}: {_cell(mean, std, n)}")

    md_text = "\n".join(md_lines)

    # ── Write outputs ──────────────────────────────────────────────────────
    md_path   = out_prefix.with_suffix(".md")
    json_path = out_prefix.with_suffix(".json")

    md_path.write_text(md_text)
    json_path.write_text(json.dumps({"summary": summary, "seeds": seeds,
                                      "conditions": conditions,
                                      "persona_ids": persona_ids}, indent=2))

    print(f"\n  Markdown → {md_path}")
    print(f"  JSON     → {json_path}")

    # ── Print summary table to console ────────────────────────────────────
    print("\n" + "=" * 70)
    print(build_markdown_table(summary, conditions))
    print("=" * 70)

    # Key finding
    main_updated = summary.get("main", {}).get("updated", {})
    frozen_updated = summary.get("frozen", {}).get("updated", {})
    if main_updated and frozen_updated:
        delta = main_updated.get("mean", 0) - frozen_updated.get("mean", 0)
        print(f"\n  MemLoRA vs Frozen on Updated facts: +{delta:.1f}pp")
    print()


if __name__ == "__main__":
    main()
