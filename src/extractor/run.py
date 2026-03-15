"""
Phase 2 — Extraction Pipeline Entry Point.

Reads:   data/dialogue/{persona_id}_dialogue.jsonl
         data/personas/{persona_id}_ground_truth.json  (for name / age)
Writes:  data/memories/{persona_id}_memories.jsonl

Processes day-by-day. Supports --resume to skip already-extracted days.

Run from repo root:
  python -m src.extractor.run --config configs/extract_config.json
  python -m src.extractor.run --config configs/extract_config.json --days 3
  python -m src.extractor.run --config configs/extract_config.json --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from src.extractor.extract import MemoryExtractor


# ── helpers ────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_dialogue(dialogue_dir: Path, persona_id: str) -> dict[int, list[dict]]:
    """Return {day: [turns]} dict from the dialogue JSONL."""
    path = dialogue_dir / f"{persona_id}_dialogue.jsonl"
    if not path.exists():
        return {}
    by_day: dict[int, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                turn = json.loads(line)
                by_day[turn["day"]].append(turn)
    return dict(by_day)


def load_ground_truth(personas_dir: Path, persona_id: str) -> dict:
    path = personas_dir / f"{persona_id}_ground_truth.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {path}\n"
            "Run Phase 1 first: bash scripts/run_phase1.sh"
        )
    with open(path) as f:
        return json.load(f)


def get_extracted_days(memories_dir: Path, persona_id: str) -> set[int]:
    """Return set of days already written to the memories JSONL."""
    path = memories_dir / f"{persona_id}_memories.jsonl"
    if not path.exists():
        return set()
    days: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                days.add(json.loads(line)["day"])
    return days


def clear_memories(memories_dir: Path, persona_id: str) -> None:
    path = memories_dir / f"{persona_id}_memories.jsonl"
    memories_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def append_memories(items: list[dict], memories_dir: Path, persona_id: str) -> None:
    path = memories_dir / f"{persona_id}_memories.jsonl"
    memories_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def print_day_result(day: int, items: list[dict], skipped: bool = False) -> None:
    if skipped:
        print(f"    Day {day:2d}  [skipped — already extracted]")
        return
    if not items:
        print(f"    Day {day:2d}  — no facts above threshold")
        return
    print(f"    Day {day:2d}  → {len(items)} fact(s) extracted:")
    for m in items:
        flag = " [UPDATE]" if m["is_update"] else ""
        print(f"           {m['predicate']} {m['value']}  (conf={m['confidence']:.2f}){flag}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Memory Extraction")
    parser.add_argument("--config", default="configs/extract_config.json")
    parser.add_argument("--days", type=int, default=None,
                        help="Process only the first N days (for quick testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip days already extracted (resume interrupted run)")
    parser.add_argument("--persona", type=str, default=None,
                        help="Run only one persona (alice or bob)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve paths from config or defaults
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    dialogue_dir = Path(input_cfg.get("dialogue_dir", "data/dialogue"))
    personas_dir = Path(input_cfg.get("personas_dir", "data/personas"))
    memories_dir = Path(output_cfg.get("memories_dir", "data/memories"))

    # Persona list comes from sim_config (Phase 1), or default
    try:
        sim_cfg = load_config("configs/sim_config.json")
        persona_ids: list[str] = sim_cfg["personas"]
    except Exception:
        persona_ids = ["alice", "bob"]

    if args.persona:
        if args.persona not in persona_ids:
            print(f"ERROR: persona '{args.persona}' not found.", file=sys.stderr)
            sys.exit(1)
        persona_ids = [args.persona]

    extractor = MemoryExtractor(cfg)

    print(f"\n=== Phase 2: Memory Extraction ===")
    print(f"  Model      : {extractor.model}")
    print(f"  Temp       : {extractor.temperature}  (deterministic)")
    print(f"  Min conf   : {extractor.confidence_threshold}")
    print(f"  Resume     : {args.resume}")

    grand_total = 0

    for persona_id in persona_ids:
        print(f"\n{'='*55}")
        print(f"  Persona: {persona_id}")
        print(f"{'='*55}")

        gt = load_ground_truth(personas_dir, persona_id)
        name: str = gt["name"]
        age: int = gt["age"]

        dialogue_by_day = load_dialogue(dialogue_dir, persona_id)
        if not dialogue_by_day:
            print(f"  WARNING: no dialogue found for {persona_id}. Run Phase 1 first.")
            continue

        n_days = args.days if args.days else max(dialogue_by_day.keys())

        already_done: set[int] = set()
        if args.resume:
            already_done = get_extracted_days(memories_dir, persona_id)
            if already_done:
                print(f"  Resuming: {len(already_done)} days already extracted")
        else:
            clear_memories(memories_dir, persona_id)

        persona_total = 0
        failed_days: list[int] = []

        for day in range(1, n_days + 1):
            if day in already_done:
                print_day_result(day, [], skipped=True)
                continue

            turns = dialogue_by_day.get(day, [])
            if not turns:
                print(f"    Day {day:2d}  — no dialogue found, skipping")
                continue

            try:
                items = extractor.extract_day(name, age, day, turns)
                append_memories(items, memories_dir, persona_id)
                print_day_result(day, items)
                persona_total += len(items)
            except Exception as exc:
                print(f"    Day {day:2d}  ERROR: {exc}")
                failed_days.append(day)

        print(f"\n  Total extracted : {persona_total} memory items")
        out_path = memories_dir / f"{persona_id}_memories.jsonl"
        print(f"  Output          : {out_path}")
        if failed_days:
            print(f"  WARNING: {len(failed_days)} days failed: {failed_days}")
            print("  Re-run with --resume to retry.")
        grand_total += persona_total

    print(f"\n{'='*55}")
    print(f"Phase 2 complete — {grand_total} total memory items written.")
    print(f"  → Run Phase 3 next to evaluate extraction quality before training.")


if __name__ == "__main__":
    main()
