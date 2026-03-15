"""
Phase 4 — Salience Scoring Runner.

Reads data/memories/{pid}_memories.jsonl and data/dialogue/{pid}_dialogue.jsonl,
scores every memory item, overwrites the memories file with salience_score
populated, and writes a debug file with per-component breakdown.

Filtered items (below threshold) are removed from the memories file so that
Phase 5 (training) only ever sees items above the quality bar.
The full scored list (including filtered items) is saved to
results/salience_debug_{pid}.json for audit.

Usage (from repo root):
    python -m src.salience.run
    python -m src.salience.run --persona alice --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .score import SalienceScorer, load_day_texts


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n")


def print_score_table(pid: str, scored: list[dict], threshold: float) -> None:
    print(f"\n  {'Day':>3} {'Nov':>5} {'Rec':>5} {'Exp':>5} {'Den':>5} "
          f"{'Ban':>5} {'Decay':>5} {'Score':>6}  Predicate+Value")
    print(f"  {'─'*3} {'─'*5} {'─'*5} {'─'*5} {'─'*5} "
          f"{'─'*5} {'─'*5} {'─'*6}  {'─'*40}")
    for item in scored:
        d = item.get("_score_detail", {})
        keep = "✓" if item["salience_score"] >= threshold else "✗"
        label = f"{item['predicate']} {item['value']}"[:45]
        print(
            f"  {item['day']:>3} "
            f"{d.get('novelty', 0):>5.2f} "
            f"{d.get('recurrence', 0):>5.2f} "
            f"{d.get('explicit_change', 0):>5.2f} "
            f"{d.get('fact_density', 0):>5.2f} "
            f"{d.get('banter', 0):>5.2f} "
            f"{d.get('temporal_decay', 0):>5.2f} "
            f"{item['salience_score']:>6.4f} "
            f"{keep} {label}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Salience Scoring")
    parser.add_argument("--config",        default="configs/salience_config.json")
    parser.add_argument("--memories-dir",  default="data/memories")
    parser.add_argument("--dialogue-dir",  default="data/dialogue")
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--persona",       default=None,
                        help="Score one persona only")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print scores but do not overwrite memories files")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}")
        sys.exit(1)
    config = json.loads(cfg_path.read_text())

    scorer       = SalienceScorer(config)
    memories_dir = Path(args.memories_dir)
    dialogue_dir = Path(args.dialogue_dir)
    results_dir  = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        sim_cfg     = json.loads(Path("configs/sim_config.json").read_text())
        persona_ids = sim_cfg["personas"]
    except Exception:
        persona_ids = ["alice", "bob"]

    if args.persona:
        persona_ids = [args.persona]

    print(f"\n=== Phase 4: Salience Scoring ===")
    print(f"  threshold = {scorer.threshold}  |  λ = {scorer.lam}")
    print(f"  dry_run   = {args.dry_run}")

    for pid in persona_ids:
        mem_path  = memories_dir / f"{pid}_memories.jsonl"
        dial_path = dialogue_dir / f"{pid}_dialogue.jsonl"

        if not mem_path.exists():
            print(f"\n  ERROR: {mem_path} not found. Run Phase 2 first.")
            sys.exit(1)
        if not dial_path.exists():
            print(f"\n  ERROR: {dial_path} not found. Run Phase 1 first.")
            sys.exit(1)

        items     = load_jsonl(mem_path)
        day_texts = load_day_texts(dial_path)

        print(f"\n{'='*60}")
        print(f"  {pid.capitalize()}  —  {len(items)} items to score")

        scored = scorer.score_all(items, day_texts)
        kept, filtered = scorer.apply_threshold(scored)

        # Stats
        scores = [s["salience_score"] for s in scored]
        avg    = sum(scores) / len(scores)
        by_bucket = {
            "high   (≥0.6)":   sum(1 for s in scores if s >= 0.6),
            "medium (0.4–0.6)": sum(1 for s in scores if 0.4 <= s < 0.6),
            "low    (<0.4)":    sum(1 for s in scores if s < 0.4),
        }

        print(f"  avg score: {avg:.3f}  |  kept: {len(kept)}  |  filtered: {len(filtered)}")
        for label, count in by_bucket.items():
            print(f"    {label}: {count}")

        # Detailed table
        print_score_table(pid, scored, scorer.threshold)

        # Sanity: check that known state-change items scored well
        print(f"\n  Key life-change items (should have high scores):")
        ec_items = [s for s in scored if s.get("_score_detail", {}).get("explicit_change", 0) >= 0.5]
        for item in ec_items[:8]:
            d = item["_score_detail"]
            print(f"    Day {item['day']:>2}: score={item['salience_score']:.4f}  "
                  f"ec={d['explicit_change']:.2f}  "
                  f"{item['predicate']} {item['value'][:45]}")

        # Save debug file (all items with component breakdown)
        debug_out = results_dir / f"salience_debug_{pid}.json"
        debug_out.write_text(json.dumps({
            "persona_id": pid,
            "config": config,
            "n_total": len(scored),
            "n_kept": len(kept),
            "n_filtered": len(filtered),
            "avg_score": round(avg, 4),
            "items": scored,
        }, indent=2))
        print(f"\n  Debug → {debug_out}")

        if not args.dry_run:
            write_jsonl(mem_path, kept)
            print(f"  Memories file updated: {len(kept)} items kept "
                  f"(removed {len(filtered)} below threshold {scorer.threshold})")
        else:
            print(f"  DRY RUN — memories file NOT modified")

    print(f"\n{'='*60}")
    print("  Phase 4 complete. Ready for Phase 5 (Consolidation Training).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
