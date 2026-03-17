"""
v2_temporal_benchmark/run_sweeps.py

Dual Volume Sweep — the core experiment for the TemporalBench v2 paper.

Experiment design
-----------------
Grid: 5 volume tiers × 3 seeds × 2 sweep types = 30 training runs.

Volume tiers:  [0.10, 0.25, 0.50, 0.75, 1.00]
Seeds:         [42, 123, 456]
Sweep types:
  A (random)   — sample V% of stable_facts uniformly at random
  B (salience) — take top-V% of stable_facts sorted by information density

Golden rule: updated_facts training sentences are ALWAYS 100% included.
The volume percentage ONLY controls how many stable_facts (background
context) are added on top.

VRAM / disk management
-----------------------
Each sweep point: train → infer → evaluate → write CSV row → DELETE adapter.
The frozen base model stays resident in VRAM throughout. Only the ~45 MB
LoRA adapter is swapped per run.

Soft-fork contract
------------------
Imports from src.trainer.loop:  load_base_model, build_peft_model,
                                 run_cycle, reset_peft
Does NOT modify any file in src/, configs/, or scripts/.

Output
------
  v2_temporal_benchmark/results/sweep_results.csv   (appended per run)
  v2_temporal_benchmark/results/sweep_log.jsonl      (verbose per-run telemetry)

CSV columns (one row per sweep point):
  sweep_type, seed, volume_pct,
  n_stable_facts, n_updated_facts, n_total_train,
  train_runtime_s, vram_peak_gb, avg_loss,
  # per-family × metric:
  {family}_{metric}  for family in [current_state, stale_premise_rejection,
                                     historical_state, relational_after_update,
                                     overall]
  # metrics: pct_current, pct_stale, pct_both, pct_distractor, pct_invalid

Usage
-----
  # Full sweep (30 runs, ~8–12 h on RTX 4090):
  python3 v2_temporal_benchmark/run_sweeps.py

  # Dry-run: 1 seed, 2 tiers, random sweep only (~15–20 min):
  python3 v2_temporal_benchmark/run_sweeps.py \\
      --seeds 42 --volume-tiers 0.10 0.25 --sweep-types random

  # Resume: skips rows already in CSV
  python3 v2_temporal_benchmark/run_sweeps.py --resume
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Soft-fork imports from src ──────────────────────────────────────────────
from src.trainer.loop import (
    load_base_model,
    build_peft_model,
    run_cycle,
)
from v2_temporal_benchmark.evaluator import MCQAEvaluator, print_distribution

# ---------------------------------------------------------------------------
# Default v2 training config (separate from v1 train_config.json)
# ---------------------------------------------------------------------------

_V2_TRAIN_CFG: dict[str, Any] = {
    "model": {
        "base_model_id":          "meta-llama/Meta-Llama-3-8B-Instruct",
        "bnb_4bit_quant_type":     "nf4",
        "bnb_4bit_use_double_quant": True,
    },
    "lora": {
        "r":              16,
        "lora_alpha":     32,
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_dropout":   0.05,
        "bias":           "none",
    },
    "training": {
        "num_epochs":                   3,
        "per_device_train_batch_size":  4,
        "gradient_accumulation_steps":  4,
        "learning_rate":                2e-4,
        "fp16":                         True,
        "warmup_steps":                 10,
        "max_grad_norm":                1.0,
    },
}

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

DEFAULT_VOLUME_TIERS  = [0.10, 0.25, 0.50, 0.75, 1.00]
DEFAULT_SEEDS         = [42, 123, 456]
DEFAULT_SWEEP_TYPES   = ["random", "salience"]

_FAMILIES = [
    "current_state",
    "stale_premise_rejection",
    "historical_state",
    "relational_after_update",
    "overall",
]
_METRICS = ["pct_current", "pct_stale", "pct_both", "pct_distractor", "pct_invalid"]

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _csv_columns() -> list[str]:
    base = [
        "sweep_type", "seed", "volume_pct",
        "n_stable_facts", "n_updated_facts", "n_total_train",
        "train_runtime_s", "vram_peak_gb", "avg_loss",
    ]
    for fam in _FAMILIES:
        for met in _METRICS:
            base.append(f"{fam}_{met}")
    return base


def _load_done_keys(csv_path: Path) -> set[tuple]:
    """Return set of (sweep_type, seed, volume_pct) already in CSV."""
    done: set[tuple] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["sweep_type"], str(row["seed"]), str(row["volume_pct"])))
    return done


def _append_csv(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_columns())
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _stable_fact_salience(fact: dict) -> float:
    """
    Proxy salience for stable facts (no dialogue context available).
    Based on fact_density: content-word ratio in predicate + value.
    Higher = more informative → selected first in Sweep B.
    """
    from src.salience.score import _STOPWORDS  # reuse v1 stopword list
    text  = (fact.get("predicate", "") + " " + fact.get("value", "")).lower()
    words = text.split()
    if not words:
        return 0.0
    content = [w for w in words if len(w) >= 3 and w not in _STOPWORDS]
    return min(1.0, len(content) / 5.0)


def _stable_sentence(persona_name: str, fact: dict) -> str:
    """Convert a stable_fact dict to a declarative training sentence."""
    pred = fact.get("predicate", "has a characteristic of")
    val  = fact.get("value", "")
    return f"{persona_name} {pred} {val}."


def _build_training_examples(
    personas: list[dict],
    benchmark: dict,
    volume_pct: float,
    sweep_type: str,
    rng: random.Random,
) -> tuple[list[list[dict]], int, int]:
    """
    Build the full training batch for one sweep point.

    Returns:
      examples     — list of Llama-3 chat message lists (for loop.tokenize_examples)
      n_stable     — number of stable fact sentences included
      n_updated    — number of updated fact sentences included (always 100%)
    """
    chat_examples: list[list[dict]] = []
    n_stable = 0
    n_updated = 0

    # Index benchmark entries by persona_id for quick lookup
    entry_by_pid: dict[str, list[dict]] = {}
    for entry in benchmark["entries"]:
        entry_by_pid.setdefault(entry["persona_id"], []).append(entry)

    for persona in personas:
        pid   = persona["persona_id"]
        pname = persona["persona_name"]

        # ── 1. ALWAYS include 100% of updated_fact training sentences ──────
        for entry in entry_by_pid.get(pid, []):
            for sent in entry.get("training_sentences", []):
                chat_examples.append(_to_chat(pname, sent))
                n_updated += 1

        # ── 2. Sample stable_facts at V% ──────────────────────────────────
        stable_facts = persona.get("stable_facts", [])

        if not stable_facts:
            continue

        n_want = max(1, round(len(stable_facts) * volume_pct))

        if sweep_type == "salience":
            # Sweep B — sort by information density, take top-N
            ordered = sorted(
                stable_facts,
                key=lambda f: _stable_fact_salience(f),
                reverse=True,
            )
            selected = ordered[:n_want]
        else:
            # Sweep A — random sample
            selected = rng.sample(stable_facts, min(n_want, len(stable_facts)))

        for fact in selected:
            sent = _stable_sentence(pname, fact)
            chat_examples.append(_to_chat(pname, sent))
            n_stable += 1

    return chat_examples, n_stable, n_updated


def _to_chat(persona_name: str, statement: str) -> list[dict]:
    """
    Wrap a declarative statement in the Llama-3 chat template.
    Mirrors the format in src/trainer/batch.py so the adapter learns
    to answer factual questions about named individuals.
    """
    return [
        {
            "role":    "user",
            "content": f"What do you know about {persona_name}?",
        },
        {
            "role":    "assistant",
            "content": statement,
        },
    ]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _run_inference(
    model: Any,
    tokenizer: Any,
    probes: list[dict],
    max_new_tokens: int = 32,
    max_probes: int | None = None,
) -> list[str]:
    """
    Run zero-context MCQA inference on all probes.
    Prompts use only the probe's full_prompt — no persona context injected.
    Returns a list of raw response strings (one per probe).
    """
    if max_probes is not None:
        probes = probes[:max_probes]

    responses: list[str] = []
    model.eval()

    for probe in probes:
        messages = [
            {
                "role":    "system",
                "content": "Answer the multiple-choice question by selecting one letter.",
            },
            {
                "role":    "user",
                "content": probe["full_prompt"],
            },
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    return responses


# ---------------------------------------------------------------------------
# Core sweep runner
# ---------------------------------------------------------------------------

def run_sweeps(
    volume_tiers:  list[float],
    seeds:         list[int],
    sweep_types:   list[str],
    out_dir:       Path,
    data_dir:      Path,
    resume:        bool = False,
    max_probes:    int | None = None,
    hf_token:      str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tmp  = out_dir / "_adapter_tmp"
    csv_path  = out_dir / "sweep_results.csv"
    log_path  = out_dir / "sweep_log.jsonl"

    # ── Load benchmark and personas ────────────────────────────────────────
    bm_path = data_dir / "benchmark.json"
    pe_path = data_dir / "personas.json"
    if not bm_path.exists() or not pe_path.exists():
        sys.exit(
            f"ERROR: benchmark.json / personas.json not found in {data_dir}\n"
            "Run generate_mcqa_data.py first."
        )

    benchmark = json.load(open(bm_path))
    personas  = json.load(open(pe_path))

    # Flatten all probes for eval
    all_probes = [p for e in benchmark["entries"] for p in e["probes"]]
    print(
        f"Loaded {len(personas)} personas, "
        f"{len(benchmark['entries'])} updated facts, "
        f"{len(all_probes)} probes total.",
        flush=True,
    )

    # ── Resume: skip already-completed rows ────────────────────────────────
    done_keys = _load_done_keys(csv_path) if resume else set()
    if done_keys:
        print(f"Resuming — {len(done_keys)} rows already in CSV, skipping.", flush=True)

    # ── Load base model ONCE ────────────────────────────────────────────────
    print(f"\nLoading base model …", flush=True)
    base_model, tokenizer = load_base_model(
        _V2_TRAIN_CFG, hf_token=hf_token
    )

    evaluator = MCQAEvaluator()

    # ── Sweep grid ──────────────────────────────────────────────────────────
    total_runs = len(sweep_types) * len(seeds) * len(volume_tiers)
    run_n = 0

    for sweep_type in sweep_types:
        for seed in seeds:
            for volume_pct in volume_tiers:
                run_n += 1
                key = (sweep_type, str(seed), str(volume_pct))

                if key in done_keys:
                    print(
                        f"[{run_n}/{total_runs}] SKIP {sweep_type} seed={seed} "
                        f"vol={volume_pct:.0%} (already in CSV)",
                        flush=True,
                    )
                    continue

                print(
                    f"\n{'='*60}\n"
                    f"[{run_n}/{total_runs}] {sweep_type.upper()} | "
                    f"seed={seed} | vol={volume_pct:.0%}\n"
                    f"{'='*60}",
                    flush=True,
                )

                # Set seeds for reproducibility
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                rng = random.Random(seed)

                # ── Build dataset ────────────────────────────────────────
                t0 = time.time()
                examples, n_stable, n_updated = _build_training_examples(
                    personas, benchmark, volume_pct, sweep_type, rng
                )
                print(
                    f"  Dataset: {n_updated} updated + {n_stable} stable "
                    f"= {len(examples)} total examples",
                    flush=True,
                )

                # ── Train fresh adapter ──────────────────────────────────
                peft_model = build_peft_model(
                    base_model,
                    lora_cfg=_V2_TRAIN_CFG["lora"],
                    prev_checkpoint=None,   # always fresh per run
                )

                ckpt_tmp.mkdir(parents=True, exist_ok=True)
                telemetry = run_cycle(
                    peft_model   = peft_model,
                    tokenizer    = tokenizer,
                    training_cfg = _V2_TRAIN_CFG["training"],
                    examples     = examples,
                    checkpoint_path = str(ckpt_tmp),
                )

                # ── Zero-context inference ───────────────────────────────
                print(f"  Running inference on {len(all_probes)} probes …", flush=True)
                t_infer = time.time()
                responses = _run_inference(
                    peft_model, tokenizer, all_probes,
                    max_probes=max_probes,
                )
                infer_time = round(time.time() - t_infer, 1)
                print(f"  Inference done in {infer_time}s", flush=True)

                # ── Evaluate ─────────────────────────────────────────────
                eval_probes = all_probes[:len(responses)]
                labelled, dist = evaluator.evaluate_and_label(responses, eval_probes)
                print_distribution(
                    dist,
                    f"{sweep_type} seed={seed} vol={volume_pct:.0%}",
                )

                # ── Write CSV row ─────────────────────────────────────────
                row: dict[str, Any] = {
                    "sweep_type":     sweep_type,
                    "seed":           seed,
                    "volume_pct":     volume_pct,
                    "n_stable_facts": n_stable,
                    "n_updated_facts": n_updated,
                    "n_total_train":  len(examples),
                    "train_runtime_s": telemetry["runtime_seconds"],
                    "vram_peak_gb":   telemetry["vram_peak_gb"],
                    "avg_loss":       telemetry["avg_loss"],
                }
                for fam in _FAMILIES:
                    d = dist.get(fam, {})
                    for met in _METRICS:
                        row[f"{fam}_{met}"] = d.get(met, 0.0)

                _append_csv(csv_path, row)
                print(f"  Row appended to {csv_path}", flush=True)

                # ── Write verbose log ─────────────────────────────────────
                log_entry = {
                    "sweep_type": sweep_type,
                    "seed": seed,
                    "volume_pct": volume_pct,
                    "telemetry": telemetry,
                    "distribution": dist,
                    "labelled_sample": labelled[:5],   # first 5 for spot-check
                }
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(log_entry) + "\n")

                # ── Delete adapter to free disk ───────────────────────────
                # Call .unload() so the LoRA weights are detached from
                # base_model; otherwise the next build_peft_model() stacks
                # a second adapter on top (PEFT multi-adapter warning).
                base_model = peft_model.unload()
                del peft_model
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"  Adapter unloaded. VRAM: "
                    f"{torch.cuda.memory_reserved() / 1e9:.2f} GB",
                    flush=True,
                )
                if ckpt_tmp.exists():
                    shutil.rmtree(ckpt_tmp, ignore_errors=True)
                    print(f"  Adapter deleted from disk.", flush=True)

                total_time = round(time.time() - t0, 1)
                print(f"  Run complete in {total_time}s total.", flush=True)

    print(f"\n{'='*60}\nAll {run_n} runs complete.\nResults: {csv_path}\n", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the dual volume sweep for TemporalBench v2."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--volume-tiers", nargs="+", type=float, default=DEFAULT_VOLUME_TIERS,
        dest="volume_tiers",
        help=f"Volume fractions (default: {DEFAULT_VOLUME_TIERS})",
    )
    parser.add_argument(
        "--sweep-types", nargs="+", default=DEFAULT_SWEEP_TYPES,
        choices=["random", "salience"],
        dest="sweep_types",
        help="Which sweeps to run (default: both)",
    )
    parser.add_argument(
        "--data-dir", default="v2_temporal_benchmark/data",
        dest="data_dir",
        help="Directory containing benchmark.json and personas.json",
    )
    parser.add_argument(
        "--out-dir", default="v2_temporal_benchmark/results",
        dest="out_dir",
        help="Output directory for CSV and logs",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip runs already present in sweep_results.csv",
    )
    parser.add_argument(
        "--max-probes", type=int, default=None,
        dest="max_probes",
        help="Limit probes per run (for quick dry-run verification)",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        print("[warn] HUGGING_FACE_TOKEN not set — needed for gated Llama-3 model.")

    if not torch.cuda.is_available():
        sys.exit(
            "ERROR: CUDA GPU not available. "
            "run_sweeps.py must execute on a GPU pod."
        )

    run_sweeps(
        volume_tiers = args.volume_tiers,
        seeds        = args.seeds,
        sweep_types  = args.sweep_types,
        out_dir      = ROOT / args.out_dir,
        data_dir     = ROOT / args.data_dir,
        resume       = args.resume,
        max_probes   = args.max_probes,
        hf_token     = hf_token,
    )


if __name__ == "__main__":
    main()
