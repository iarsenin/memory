"""
Phase 6 — Baseline Execution Orchestrator (Pod GPU required).

Trains one of three LoRA conditions on the same 20-day timeline used in
Phase 5 (main MemLoRA system). Reuses src/trainer/loop.py entirely so
hyperparameters, PEFT config, and training mechanics are identical.

Conditions
----------
naive_lora
    Train on raw 2-turn dialogue windows from data/dialogue/.
    No extraction, no replay buffer. Tests whether the model can implicitly
    learn facts from raw transcripts alone.

unfiltered_lora
    Train on ALL Phase 2 extracted memories (data/memories_unfiltered/),
    bypassing salience/decay filtering entirely.
    Same BatchGenerator as main but with the full 168-item pool and
    no threshold gate. Isolates the value of composite salience scoring.

gold_lora
    Train on structured ground-truth facts from data/personas/*.json.
    Perfect input quality — the parametric upper bound. Isolates how much
    headroom remains between "perfect extraction" and current system.

frozen / rag
    No training required. Phase 7 handles them at inference time.

Checkpoint layout (isolated from main to prevent overwrites):
    checkpoints/{condition}/{persona_id}/day_{N:02d}/

Telemetry:
    logs/{condition}_{persona_id}_train_day{N:02d}.jsonl

Usage (from repo root, on the pod):
    python -m src.baselines.run --condition naive_lora
    python -m src.baselines.run --condition unfiltered_lora --resume
    python -m src.baselines.run --condition gold_lora --sanity --persona alice
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    if not torch.cuda.is_available():
        print("ERROR: Phase 6 requires a CUDA GPU. Run on the RunPod instance.")
        sys.exit(1)
except ImportError:
    print("ERROR: torch not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

from ..trainer.batch import BatchGenerator
from ..trainer.loop  import build_peft_model, load_base_model, reset_peft, run_cycle
from .batch_naive    import NaiveBatchGenerator
from .batch_gold     import GoldBatchGenerator

VALID_CONDITIONS = ("naive_lora", "unfiltered_lora", "gold_lora")


# ---------------------------------------------------------------------------
# I/O helpers (duplicated lightly from src/trainer/run.py for independence)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n")

def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def _build_dialogue_by_day(turns: list[dict]) -> dict[int, list[dict]]:
    by_day: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_day[t["day"]].append(t)
    return dict(by_day)

def _checkpoint_complete(base_dir: Path) -> bool:
    return (base_dir / "adapter_config.json").exists()

def _mark_consolidated(memories: list[dict], ids: set[str]) -> list[dict]:
    return [{**m, "consolidated": True} if m["memory_id"] in ids else m
            for m in memories]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Baseline Training")
    parser.add_argument("--condition", required=True, choices=VALID_CONDITIONS)
    parser.add_argument("--config",           default="configs/train_config.json")
    parser.add_argument("--memories-dir",     default="data/memories")
    parser.add_argument("--unfiltered-dir",   default="data/memories_unfiltered")
    parser.add_argument("--dialogue-dir",     default="data/dialogue")
    parser.add_argument("--personas-dir",     default="data/personas")
    parser.add_argument("--checkpoints-dir",  default="checkpoints")
    parser.add_argument("--logs-dir",         default="logs")
    parser.add_argument("--persona",          default=None)
    parser.add_argument("--sanity",           action="store_true",
                        help="Run only Day 3 (1-cycle debug)")
    parser.add_argument("--resume",           action="store_true",
                        help="Skip cycles whose checkpoint already exists")
    args = parser.parse_args()

    config = _load_json(Path(args.config))
    condition = args.condition

    dev_mode = config.get("dev_mode", True)
    seeds    = config["seeds"]["dev"] if dev_mode else config["seeds"]["paper"]
    seed     = seeds[0]

    try:
        sim_cfg     = _load_json(Path("configs/sim_config.json"))
        persona_ids = sim_cfg["personas"]
    except Exception:
        persona_ids = ["alice", "bob"]
    if args.persona:
        persona_ids = [args.persona]

    memories_dir    = Path(args.memories_dir)
    unfiltered_dir  = Path(args.unfiltered_dir)
    dialogue_dir    = Path(args.dialogue_dir)
    personas_dir    = Path(args.personas_dir)
    checkpoints_dir = Path(args.checkpoints_dir) / condition
    logs_dir        = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    trigger_days_full = list(range(
        config["consolidation_trigger_days"],
        config.get("sim_max_day", 20) + 1,
        config["consolidation_trigger_days"],
    ))
    window_size = config["consolidation_trigger_days"]

    hf_token = os.environ.get("HUGGING_FACE_TOKEN")

    print(f"\n=== Phase 6 Baseline: {condition} ===")
    print(f"  Personas: {persona_ids}  |  Seed: {seed}  |  dev_mode={dev_mode}")
    print(f"  Checkpoints → checkpoints/{condition}/{{pid}}/day_{{N}}/")

    # ------------------------------------------------------------------
    # Load base model ONCE
    # ------------------------------------------------------------------
    base_model, tokenizer = load_base_model(config, hf_token=hf_token)

    # ------------------------------------------------------------------
    # Train each persona
    # ------------------------------------------------------------------
    for pid in persona_ids:
        dial_turns     = _load_jsonl(dialogue_dir / f"{pid}_dialogue.jsonl")
        dialogue_by_day = _build_dialogue_by_day(dial_turns)

        # Condition-specific data sources
        if condition == "naive_lora":
            batch_gen = NaiveBatchGenerator()
            memories = None        # naive doesn't use memories

        elif condition == "unfiltered_lora":
            mem_path  = unfiltered_dir / f"{pid}_memories.jsonl"
            if not mem_path.exists():
                print(f"\nERROR: {mem_path} not found. Re-run Phase 2 extraction first.")
                sys.exit(1)
            memories  = _load_jsonl(mem_path)
            batch_gen = BatchGenerator(config)

        elif condition == "gold_lora":
            gt_path   = personas_dir / f"{pid}_ground_truth.json"
            if not gt_path.exists():
                print(f"\nERROR: {gt_path} not found.")
                sys.exit(1)
            ground_truth = _load_json(gt_path)
            memories     = None   # gold uses GT directly
            batch_gen    = GoldBatchGenerator(config)

        print(f"\n{'='*60}")
        print(f"  Persona: {pid}")

        trigger_days = trigger_days_full[:1] if args.sanity else trigger_days_full
        prev_checkpoint: str | None = None
        peft_model = None

        for day in trigger_days:
            chk_path = checkpoints_dir / pid / f"day_{day:02d}"

            # --- Resume ---
            if _checkpoint_complete(chk_path):
                print(f"\n  Day {day:02d}: checkpoint exists — skipping")
                prev_checkpoint = str(chk_path)
                if condition == "unfiltered_lora" and memories is not None:
                    win_start = day - window_size + 1
                    ids = {m["memory_id"] for m in memories
                           if win_start <= m["day"] <= day}
                    memories = _mark_consolidated(memories, ids)
                continue

            # --- Build batch ---
            if condition == "naive_lora":
                examples, batch_meta = batch_gen.build_cycle_batch(
                    dialogue_by_day, day, window_size=window_size
                )

            elif condition == "unfiltered_lora":
                win_start  = day - window_size + 1
                new_mems   = [m for m in memories
                               if win_start <= m["day"] <= day
                               and not m.get("consolidated")]
                consolidated = [m for m in memories if m.get("consolidated")]
                if not new_mems:
                    print(f"\n  Day {day:02d}: no new memories — skipping")
                    continue
                examples, batch_meta = batch_gen.build_cycle_batch(
                    new_mems, consolidated, dialogue_by_day, seed=seed
                )

            elif condition == "gold_lora":
                examples, batch_meta = batch_gen.build_cycle_batch(
                    ground_truth, day, window_size=window_size, seed=seed
                )

            if not examples:
                print(f"\n  Day {day:02d}: empty batch — skipping")
                continue

            print(f"\n  Day {day:02d}: {batch_meta['n_total']} examples")

            # --- Build / accumulate PEFT model ---
            if peft_model is not None:
                reset_peft(peft_model)
                peft_model = None
            peft_model = build_peft_model(
                base_model, config["lora"], prev_checkpoint=prev_checkpoint
            )

            # --- Train ---
            cycle_telemetry = run_cycle(
                peft_model=peft_model,
                tokenizer=tokenizer,
                training_cfg=config["training"],
                examples=examples,
                checkpoint_path=str(chk_path),
            )
            prev_checkpoint = str(chk_path)

            # --- Update consolidated state (unfiltered only) ---
            if condition == "unfiltered_lora" and memories is not None:
                ids = {m["memory_id"] for m in new_mems}
                memories = _mark_consolidated(memories, ids)
                _write_jsonl(unfiltered_dir / f"{pid}_memories.jsonl", memories)

            # --- Telemetry ---
            log_entry = {
                "condition":    condition,
                "persona_id":   pid,
                "day":          day,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "seed":         seed,
                "batch":        batch_meta,
                "training":     cycle_telemetry,
                "checkpoint":   str(chk_path),
            }
            log_path = logs_dir / f"{condition}_{pid}_train_day{day:02d}.jsonl"
            _append_jsonl(log_path, log_entry)
            print(f"  Telemetry → {log_path}")

        # --- State reset between personas ---
        if peft_model is not None:
            print(f"\n  Resetting adapter after {pid}")
            reset_peft(peft_model)
            peft_model = None

        # Persist final unfiltered memory state
        if condition == "unfiltered_lora" and memories is not None:
            _write_jsonl(unfiltered_dir / f"{pid}_memories.jsonl", memories)

        print(f"  {pid} done")

    print(f"\n{'='*60}")
    print(f"  Phase 6 [{condition}] complete.")
    print(f"  Checkpoints → checkpoints/{condition}/")
    print(f"  Logs        → logs/{condition}_*_train_day*.jsonl")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
