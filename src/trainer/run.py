"""
Phase 5 — Sleep Phase Consolidation (Orchestrator).

Runs on the pod GPU. Do NOT run this on a local Mac.

Execution flow:
  1. Load base model ONCE (stays in VRAM for the whole run).
  2. For each persona (e.g. alice, bob):
     a. Collect memories and dialogue turns.
     b. Iterate through trigger days (3, 6, 9, 12, 15, 18).
     c. For each trigger day:
        - Skip if checkpoint already exists (resume support).
        - Gather new un-consolidated memories for the 3-day window.
        - Build training batch (BatchGenerator).
        - Run sleep cycle (SleepTrainer):
            Cycle 1  → fresh LoRA adapter on frozen base model.
            Cycle 2+ → load previous cycle's adapter (accumulation).
        - Save LoRA adapter checkpoint + telemetry JSONL.
        - Mark consolidated memories in memories JSONL.
     d. CRITICAL: delete the persona's LoRA adapter and clear VRAM cache
        before initialising a fresh adapter for the next persona.
  3. Log final summary.

Guardrails enforced:
  • No QA templates generated dynamically — strictly from config.
  • Replay weighted by salience_score × temporal_decay (Phase 4 fields).
  • No cross-persona weight contamination (see step 2d above).
  • Batch-size auto-reduction for tiny 3-day windows.

Usage (from repo root, on the pod):
  python -m src.trainer.run
  python -m src.trainer.run --persona alice --sanity      # 1-cycle debug run
  python -m src.trainer.run --resume                      # skip existing checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# CUDA guard — fail fast on Mac / CPU-only machine
# ---------------------------------------------------------------------------

try:
    import torch
    if not torch.cuda.is_available():
        print(
            "ERROR: Phase 5 requires a CUDA GPU. "
            "No CUDA device found on this machine.\n"
            "Run this script on the RunPod instance via:\n"
            "  bash scripts/run_phase5.sh"
        )
        sys.exit(1)
except ImportError:
    print("ERROR: torch not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

from .batch import BatchGenerator
from .loop  import build_peft_model, load_base_model, reset_peft, run_cycle


# ---------------------------------------------------------------------------
# I/O helpers
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
    for turn in turns:
        by_day[turn["day"]].append(turn)
    return dict(by_day)


# ---------------------------------------------------------------------------
# Cycle management
# ---------------------------------------------------------------------------


def _checkpoint_complete(checkpoints_dir: Path, persona_id: str, day: int) -> bool:
    """True if this cycle's LoRA adapter was fully saved."""
    p = checkpoints_dir / persona_id / f"day_{day:02d}" / "adapter_config.json"
    return p.exists()


def _mark_consolidated(
    memories: list[dict], memory_ids: set[str]
) -> list[dict]:
    """Return new list with consolidated=True for all matching memory_ids."""
    return [
        {**m, "consolidated": True} if m["memory_id"] in memory_ids else m
        for m in memories
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Sleep Phase Consolidation")
    parser.add_argument("--config",         default="configs/train_config.json")
    parser.add_argument("--memories-dir",   default="data/memories")
    parser.add_argument("--dialogue-dir",   default="data/dialogue")
    parser.add_argument("--checkpoints-dir",default="checkpoints")
    parser.add_argument("--logs-dir",       default="logs")
    parser.add_argument("--persona",        default=None,
                        help="Train one persona only (alice or bob)")
    parser.add_argument("--sanity",         action="store_true",
                        help="Run only the first trigger day (1-cycle debug)")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip cycles whose checkpoint already exists")
    parser.add_argument("--seed",           type=int, default=None,
                        help="Override config seed for this run (paper loop)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}")
        sys.exit(1)
    config = _load_json(cfg_path)

    # Active seed (dev vs paper; --seed flag overrides)
    dev_mode = config.get("dev_mode", True)
    seeds    = config["seeds"]["dev"] if dev_mode else config["seeds"]["paper"]
    seed     = args.seed if args.seed is not None else seeds[0]

    # Seed PyTorch for reproducible LoRA weight initialisation
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Persona list
    try:
        sim_cfg     = _load_json(Path("configs/sim_config.json"))
        persona_ids = sim_cfg["personas"]
    except Exception:
        persona_ids = ["alice", "bob"]
    if args.persona:
        persona_ids = [args.persona]

    memories_dir    = Path(args.memories_dir)
    dialogue_dir    = Path(args.dialogue_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    logs_dir        = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    trigger_days_full = list(range(
        config["consolidation_trigger_days"],
        config.get("sim_max_day", 20) + 1,
        config["consolidation_trigger_days"],
    ))
    window_size = config["consolidation_trigger_days"]

    hf_token = os.environ.get("HUGGING_FACE_TOKEN")

    print("\n=== Phase 5: Sleep Phase Consolidation ===")
    print(f"  Personas   : {persona_ids}")
    print(f"  Seed       : {seed}  (dev_mode={dev_mode})")
    print(f"  Trigger days: {trigger_days_full}")
    if args.sanity:
        print("  SANITY MODE — only first trigger day")

    # ------------------------------------------------------------------
    # Load base model ONCE (shared, frozen, stays in VRAM)
    # ------------------------------------------------------------------
    base_model, tokenizer = load_base_model(config, hf_token=hf_token)
    batch_gen = BatchGenerator(config)

    # ------------------------------------------------------------------
    # Train each persona
    # ------------------------------------------------------------------
    for pid in persona_ids:
        mem_path  = memories_dir / f"{pid}_memories.jsonl"
        dial_path = dialogue_dir / f"{pid}_dialogue.jsonl"

        if not mem_path.exists():
            print(f"\nERROR: {mem_path} not found. Run Phases 2–4 first.")
            sys.exit(1)

        memories      = _load_jsonl(mem_path)
        dialogue_turns = _load_jsonl(dial_path)
        dialogue_by_day = _build_dialogue_by_day(dialogue_turns)

        print(f"\n{'='*60}")
        print(f"  Persona: {pid}  |  {len(memories)} memories")

        trigger_days = trigger_days_full[:1] if args.sanity else trigger_days_full
        prev_checkpoint: str | None = None
        peft_model = None

        for day in trigger_days:
            chk_path = checkpoints_dir / pid / f"day_{day:02d}"

            # --- Resume: skip if this cycle is already done ---
            if _checkpoint_complete(checkpoints_dir, pid, day):
                print(f"\n  Day {day:02d}: checkpoint exists — skipping (resume)")
                prev_checkpoint = str(chk_path)
                # Still need to mark those memories as consolidated
                new_ids = {
                    m["memory_id"]
                    for m in memories
                    if (day - window_size) < m["day"] <= day
                }
                memories = _mark_consolidated(memories, new_ids)
                continue

            # --- Gather new memories for this 3-day window ---
            win_start = day - window_size + 1
            new_mems  = [
                m for m in memories
                if win_start <= m["day"] <= day and not m.get("consolidated")
            ]
            consolidated = [m for m in memories if m.get("consolidated")]

            print(
                f"\n  Day {day:02d}: {len(new_mems)} new memories, "
                f"{len(consolidated)} consolidated (replay pool)"
            )

            if not new_mems:
                print("  No new memories — skipping this cycle")
                continue

            # --- Build training batch ---
            examples, batch_meta = batch_gen.build_cycle_batch(
                new_mems, consolidated, dialogue_by_day, seed=seed,
                anti_memory_enabled=True,   # main condition: enable negative training
                all_memories=memories,
            )
            print(
                f"  Batch: {batch_meta['n_total']} examples  "
                f"(decl={batch_meta['n_declarative']} "
                f"qa={batch_meta['n_qa']} "
                f"dial={batch_meta['n_dialogue']} "
                f"replay={batch_meta['n_replay']} "
                f"reg={batch_meta['n_regularizer']} "
                f"anti={batch_meta['n_anti_memory']})"
            )

            # --- Build or load PEFT model ---
            if peft_model is not None:
                # Previous cycle's model is stale — delete before rebuilding
                reset_peft(peft_model)
                peft_model = None

            peft_model = build_peft_model(
                base_model,
                config["lora"],
                prev_checkpoint=prev_checkpoint,
            )

            # --- Run training ---
            cycle_telemetry = run_cycle(
                peft_model=peft_model,
                tokenizer=tokenizer,
                training_cfg=config["training"],
                examples=examples,
                checkpoint_path=str(chk_path),
            )

            prev_checkpoint = str(chk_path)

            # --- Mark memories as consolidated ---
            new_ids = {m["memory_id"] for m in new_mems}
            memories = _mark_consolidated(memories, new_ids)
            _write_jsonl(mem_path, memories)

            # --- Write telemetry ---
            avg_sal = (
                sum(m.get("salience_score") or 0 for m in new_mems) / len(new_mems)
                if new_mems else 0
            )
            log_entry = {
                "persona_id":              pid,
                "day":                     day,
                "window":                  f"days {win_start}–{day}",
                "timestamp":               datetime.now(timezone.utc).isoformat(),
                "seed":                    seed,
                "n_new_memories":          len(new_mems),
                "n_replay_memories":       batch_meta["n_replay"],
                "n_consolidated_total":    len([m for m in memories if m.get("consolidated")]),
                "avg_salience_new":        round(avg_sal, 4),
                "batch": batch_meta,
                "training": cycle_telemetry,
                "checkpoint":              str(chk_path),
            }
            log_path = logs_dir / f"{pid}_train_day{day:02d}.jsonl"
            _append_jsonl(log_path, log_entry)
            print(f"  Telemetry → {log_path}")

        # ------------------------------------------------------------------
        # CRITICAL: State reset before next persona
        # Unload Alice's LoRA adapter; Bob starts with a fresh one.
        # ------------------------------------------------------------------
        if peft_model is not None:
            print(f"\n  Resetting LoRA adapter after {pid} (state isolation)")
            reset_peft(peft_model)
            peft_model = None

        # Persist any consolidated-status updates (covers sanity + resume paths)
        _write_jsonl(mem_path, memories)
        print(f"  {pid} complete — memories updated at {mem_path}")

    print(f"\n{'='*60}")
    print("  Phase 5 complete. Checkpoints saved to checkpoints/")
    print("  Run Phase 6 (baselines) or Phase 7 (evaluation) next.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
