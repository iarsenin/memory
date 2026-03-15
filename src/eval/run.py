"""
Phase 7 — Zero-Context Evaluation Suite (Orchestrator).

Runs on the pod GPU.  Do NOT run on a local Mac (requires CUDA).

Execution flow
──────────────
  1. Generate eval probes from GT JSONs once; save to data/eval_probes/.
  2. Load base Llama-3-8B ONCE (stays in VRAM throughout).
  3. For each condition in [frozen, rag, naive_lora, unfiltered_lora,
     gold_lora, main]:
       For each persona:
         a. Load LoRA adapter (LoRA conditions only; skip for frozen/rag).
         b. Run all probes → raw responses saved to results/.
         c. Unload adapter.
  4. Score all responses via OpenAI LLM judge.
  5. Compute bucketed accuracy table; save to results/eval_summary.json.

Resume support
──────────────
  If results/{condition}_{persona}_responses.json already exists, inference
  for that pair is skipped.  If results/{condition}_{persona}_eval.json
  already exists, judge scoring is skipped.  Use --force to override.

Usage (from repo root, on the pod)
───────────────────────────────────
  python -m src.eval.run
  python -m src.eval.run --persona alice --sanity       # 3 probes, debug
  python -m src.eval.run --condition main               # single condition
  python -m src.eval.run --skip-inference               # re-run judge only
  python -m src.eval.run --skip-judge                   # inference only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CUDA guard — fail fast on non-GPU machine
# ---------------------------------------------------------------------------

try:
    import torch
    if not torch.cuda.is_available():
        print(
            "ERROR: Phase 7 inference requires a CUDA GPU.\n"
            "Run on the RunPod pod via:  bash scripts/run_phase7.sh"
        )
        sys.exit(1)
except ImportError:
    print("ERROR: torch not installed.  Run: pip install -r requirements.txt")
    sys.exit(1)

from openai import OpenAI

from .infer  import (
    CONDITIONS_WITH_ADAPTERS,
    find_latest_checkpoint,
    load_inference_model,
    load_adapter,
    run_condition_inference,
    unload_adapter,
)
from .judge  import score_responses
from .probes import generate_probes, save_probes

# Evaluation is run across all six conditions in this order.
ALL_CONDITIONS = ["frozen", "rag", "naive_lora", "unfiltered_lora", "gold_lora", "main"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def _aggregate(scored: list[dict]) -> dict[str, Any]:
    """Compute per-bucket and overall accuracy from a scored response list."""
    by_bucket: dict[str, list[float]] = defaultdict(list)
    by_bucket_labels: dict[str, list[str]] = defaultdict(list)
    for r in scored:
        b = r["bucket"]
        by_bucket[b].append(r["score_numeric"])
        by_bucket_labels[b].append(r.get("score_label", ""))

    summary: dict[str, Any] = {}
    for bucket, scores in by_bucket.items():
        labels = by_bucket_labels[bucket]
        summary[bucket] = {
            "n":             len(scores),
            "correct":       labels.count("correct"),
            "partial":       labels.count("partial"),
            "incorrect":     labels.count("incorrect"),
            "contradiction": labels.count("contradiction"),
            "accuracy":      round(sum(scores) / len(scores), 4) if scores else 0.0,
        }

    all_scores  = [r["score_numeric"] for r in scored]
    all_labels  = [r.get("score_label", "") for r in scored]
    summary["overall"] = {
        "n":             len(all_scores),
        "accuracy":      round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "contradiction": all_labels.count("contradiction"),
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: Evaluation Suite")
    parser.add_argument("--train-config",    default="configs/train_config.json")
    parser.add_argument("--eval-config",     default="configs/eval_config.json")
    parser.add_argument("--personas-dir",    default="data/personas")
    parser.add_argument("--memories-dir",    default="data/memories")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir",     default="results")
    parser.add_argument("--eval-probes-dir", default="data/eval_probes")
    parser.add_argument("--persona",         default=None,
                        help="Evaluate one persona only (alice or bob)")
    parser.add_argument("--condition",       default=None,
                        help="Run one condition only")
    parser.add_argument("--sanity",          action="store_true",
                        help="Run only 3 probes per persona (debug)")
    parser.add_argument("--skip-inference",  action="store_true",
                        help="Skip inference; run judge on saved responses only")
    parser.add_argument("--skip-judge",      action="store_true",
                        help="Run inference only; skip judge scoring")
    parser.add_argument("--force",           action="store_true",
                        help="Re-run even if output files already exist")
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    train_cfg = _load_json(Path(args.train_config))
    eval_cfg  = _load_json(Path(args.eval_config))

    personas_dir    = Path(args.personas_dir)
    memories_dir    = Path(args.memories_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    results_dir     = Path(args.results_dir)
    probes_dir      = Path(args.eval_probes_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    probes_dir.mkdir(parents=True, exist_ok=True)

    try:
        sim_cfg     = _load_json(Path("configs/sim_config.json"))
        persona_ids = sim_cfg["personas"]
    except Exception:
        persona_ids = ["alice", "bob"]
    if args.persona:
        persona_ids = [args.persona]

    conditions = list(args.condition.split(",")) if args.condition else ALL_CONDITIONS

    hf_token   = os.environ.get("HUGGING_FACE_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    judge_model = (
        eval_cfg["openai"].get("model")
        or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    )

    print("\n=== Phase 7: Zero-Context Evaluation ===")
    print(f"  Personas   : {persona_ids}")
    print(f"  Conditions : {conditions}")
    print(f"  Judge model: {judge_model}")
    if args.sanity:
        print("  SANITY MODE — 3 probes per persona")

    # ── Step 1: Generate probes ────────────────────────────────────────────
    probes_path = probes_dir / "probes.json"
    if probes_path.exists() and not args.force:
        print(f"\n  Probes already exist: {probes_path}")
        probes = _load_json(probes_path)
    else:
        print("\n  Generating eval probes from ground truth …")
        probes = generate_probes(personas_dir, persona_ids)
        save_probes(probes, probes_path)
        print(f"  {len(probes)} probes saved → {probes_path}")

    # Sanity: take first 3 probes per persona
    if args.sanity:
        sampled: list[dict] = []
        for pid in persona_ids:
            sampled.extend([p for p in probes if p["persona_id"] == pid][:3])
        probes = sampled
        print(f"  Sanity subset: {len(probes)} probes")

    probes_by_id = {p["probe_id"]: p for p in probes}

    # Bucket summary
    from collections import Counter
    bucket_counts = Counter(p["bucket"] for p in probes)
    print(f"  Probe breakdown: {dict(bucket_counts)}")

    # ── Step 2: Inference ──────────────────────────────────────────────────
    if not args.skip_inference:
        base_model, tokenizer = load_inference_model(train_cfg, hf_token=hf_token)

        for condition in conditions:
            print(f"\n{'='*60}")
            print(f"  Condition: {condition}")

            for pid in persona_ids:
                out_path = results_dir / f"{condition}_{pid}_responses.json"

                if out_path.exists() and not args.force and not args.sanity:
                    print(f"  {out_path.name} already exists — skipping (use --force to redo)")
                    continue

                # Load memories for RAG context
                rag_memories: list[dict] | None = None
                if condition == "rag":
                    rag_memories = _load_jsonl(memories_dir / f"{pid}_memories.jsonl")
                    print(f"  RAG: loaded {len(rag_memories)} memories for {pid}")

                # Load LoRA adapter for LoRA conditions
                peft_model = None
                active_model = base_model
                if condition in CONDITIONS_WITH_ADAPTERS:
                    ckpt = find_latest_checkpoint(checkpoints_dir, condition, pid)
                    if ckpt is None:
                        print(f"  WARNING: no checkpoint found for {condition}/{pid} — skipping")
                        continue
                    peft_model   = load_adapter(base_model, ckpt)
                    active_model = peft_model

                print(f"  Running {pid} under '{condition}' ({len([p for p in probes if p['persona_id']==pid])} probes) …")
                raw_responses = run_condition_inference(
                    probes=probes,
                    persona_id=pid,
                    condition=condition,
                    model=active_model,
                    tokenizer=tokenizer,
                    rag_memories=rag_memories,
                )

                _save_json(out_path, raw_responses)
                print(f"  {len(raw_responses)} responses → {out_path}")

                if peft_model is not None:
                    unload_adapter(peft_model)
                    peft_model = None

    # ── Step 3: LLM judge scoring ──────────────────────────────────────────
    if not args.skip_judge:
        client = OpenAI(api_key=openai_key)
        print(f"\n{'='*60}")
        print(f"  Scoring with LLM judge (model={judge_model}) …")

        for condition in conditions:
            for pid in persona_ids:
                resp_path  = results_dir / f"{condition}_{pid}_responses.json"
                score_path = results_dir / f"{condition}_{pid}_eval.json"

                if not resp_path.exists():
                    print(f"  WARNING: {resp_path.name} not found — skipping judge")
                    continue

                if score_path.exists() and not args.force and not args.sanity:
                    print(f"  {score_path.name} already exists — skipping")
                    continue

                responses = _load_json(resp_path)
                print(f"  Judging {condition}/{pid} ({len(responses)} responses) …")

                scored = score_responses(
                    responses=responses,
                    probes_by_id=probes_by_id,
                    client=client,
                    model=judge_model,
                )
                _save_json(score_path, scored)

                acc = sum(r["score_numeric"] for r in scored) / max(len(scored), 1)
                print(f"  → {score_path.name}  |  accuracy={acc:.1%}")

    # ── Step 4: Aggregate summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Aggregating results …")

    BUCKETS = ["stable", "updated", "superseded", "relational", "overall"]
    full_summary: dict[str, Any] = {
        "_meta": {
            "generated":  datetime.now(timezone.utc).isoformat(),
            "persona_ids": persona_ids,
            "conditions":  conditions,
        },
        "conditions": {},
    }

    table_rows: list[tuple[str, dict]] = []

    for condition in conditions:
        all_scored: list[dict] = []
        for pid in persona_ids:
            sp = results_dir / f"{condition}_{pid}_eval.json"
            if sp.exists():
                all_scored.extend(_load_json(sp))

        if not all_scored:
            print(f"  {condition:20s}  (no scored results found)")
            continue

        metrics = _aggregate(all_scored)
        full_summary["conditions"][condition] = metrics
        table_rows.append((condition, metrics))
        overall = metrics.get("overall", {}).get("accuracy", 0.0)
        print(f"  {condition:20s}  overall={overall:.1%}")

    summary_path = results_dir / "eval_summary.json"
    _save_json(summary_path, full_summary)
    print(f"\n  Summary → {summary_path}")

    # ── Pretty accuracy table ──────────────────────────────────────────────
    if table_rows:
        header = f"{'condition':22s}" + "".join(f"  {b:12s}" for b in BUCKETS)
        sep    = "─" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)
        for condition, metrics in table_rows:
            row = f"{condition:22s}"
            for b in BUCKETS:
                if b in metrics:
                    row += f"  {metrics[b]['accuracy']:12.1%}"
                else:
                    row += f"  {'n/a':12s}"
            print(row)
        print(sep)

    print("\n  Phase 7 complete.\n")


if __name__ == "__main__":
    main()
