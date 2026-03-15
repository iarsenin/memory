"""
Phase 1 — Simulator Engine Entry Point.

Generates:
  data/personas/{id}_ground_truth.json   — structured ground truth (Pydantic → JSON)
  data/dialogue/{id}_dialogue.jsonl      — 20 days of conversational turns

Supports --resume to skip already-generated days (useful after pod interruption).

Run from repo root:
  python -m src.simulator.run --config configs/sim_config.json
  python -m src.simulator.run --config configs/sim_config.json --days 2   # quick test
  python -m src.simulator.run --config configs/sim_config.json --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

# Load .env before importing anything that uses env vars
load_dotenv(find_dotenv())

from src.simulator.personas import get_all_personas
from src.simulator.dialogue import DialogueGenerator


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_ground_truth(persona, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{persona.persona_id}_ground_truth.json"
    path.write_text(persona.model_dump_json(indent=2))
    print(f"  Ground truth → {path}")


def append_dialogue_turns(turns: list[dict], output_dir: Path, persona_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{persona_id}_dialogue.jsonl"
    with open(path, "a") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


def get_completed_days(dialogue_dir: Path, persona_id: str) -> set[int]:
    path = dialogue_dir / f"{persona_id}_dialogue.jsonl"
    if not path.exists():
        return set()
    days: set[int] = set()
    with open(path) as f:
        for line in f:
            days.add(json.loads(line)["day"])
    return days


def clear_dialogue(dialogue_dir: Path, persona_id: str) -> None:
    path = dialogue_dir / f"{persona_id}_dialogue.jsonl"
    dialogue_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def print_summary(persona, dialogue_dir: Path, n_days: int) -> None:
    path = dialogue_dir / f"{persona.persona_id}_dialogue.jsonl"
    n_turns = sum(1 for _ in open(path)) if path.exists() else 0
    events = persona.events
    print(f"\n  Summary for {persona.name}:")
    print(f"    Total turns generated : {n_turns} ({n_turns // max(1,n_days)} per day avg)")
    print(f"    Life-change events    : {len(events)} days with changes")
    for e in events:
        print(f"      Day {e.day:2d}: {e.description[:70]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Simulator Engine")
    parser.add_argument("--config", default="configs/sim_config.json")
    parser.add_argument("--days", type=int, default=None,
                        help="Override n_days (for quick sanity checks)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-generated days (resume interrupted run)")
    parser.add_argument("--persona", type=str, default=None,
                        help="Run only one persona (alice or bob)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_days = args.days if args.days is not None else cfg["n_days"]
    seed = cfg["seed"]
    turns_per_day = cfg["turns_per_day"]

    random.seed(seed)

    personas_dir = Path(cfg["output"]["personas_dir"])
    dialogue_dir = Path(cfg["output"]["dialogue_dir"])

    openai_cfg = cfg.get("openai", {})
    model = openai_cfg.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    temperature = openai_cfg.get("temperature", 0.7)
    max_tokens = openai_cfg.get("max_tokens", 1200)

    print(f"\n=== Phase 1: Simulator Engine ===")
    print(f"  Model         : {model}")
    print(f"  Days per persona : {n_days}")
    print(f"  Turns per day : {turns_per_day}")
    print(f"  Seed          : {seed}")
    print(f"  Resume        : {args.resume}")

    generator = DialogueGenerator(model=model, temperature=temperature, max_tokens=max_tokens)
    all_personas = get_all_personas()

    if args.persona:
        all_personas = [p for p in all_personas if p.persona_id == args.persona]
        if not all_personas:
            print(f"ERROR: persona '{args.persona}' not found.", file=sys.stderr)
            sys.exit(1)

    for persona in all_personas:
        print(f"\n{'='*55}")
        print(f"  Persona: {persona.name} ({persona.persona_id})")
        print(f"{'='*55}")

        save_ground_truth(persona, personas_dir)

        completed_days: set[int] = set()
        if args.resume:
            completed_days = get_completed_days(dialogue_dir, persona.persona_id)
            if completed_days:
                print(f"  Resuming: {len(completed_days)}/{n_days} days already generated")
        else:
            clear_dialogue(dialogue_dir, persona.persona_id)

        failed_days = []
        for day in tqdm(range(1, n_days + 1), desc=f"  {persona.persona_id}"):
            if day in completed_days:
                continue

            try:
                turns = generator.generate_day(persona, day, n_turns=turns_per_day)
                append_dialogue_turns(turns, dialogue_dir, persona.persona_id)
            except Exception as exc:
                print(f"\n  ERROR on day {day}: {exc}")
                print("  Skipping — re-run with --resume to retry.")
                failed_days.append(day)

        print_summary(persona, dialogue_dir, n_days)
        if failed_days:
            print(f"\n  WARNING: {len(failed_days)} days failed: {failed_days}")
            print("  Re-run with --resume to retry failed days.")

    print(f"\n{'='*55}")
    print("Phase 1 complete.")
    print(f"  Personas : {personas_dir}/")
    print(f"  Dialogue : {dialogue_dir}/")


if __name__ == "__main__":
    main()
