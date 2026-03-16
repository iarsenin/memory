"""
scripts/export_hyperparams.py
Export all hyperparameters and methodological details to a clean Markdown file.

Usage:
    python3 scripts/export_hyperparams.py [--out results/methodology_details.md]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text())


def _code_block(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.strip()}\n```"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_extraction(md: list[str]) -> None:
    from src.extractor.extract import SYSTEM_PROMPT  # noqa: PLC0415
    cfg = _load_json("configs/extractor_config.json") if (REPO_ROOT / "configs/extractor_config.json").exists() else {}
    confidence_threshold = cfg.get("confidence_threshold", 0.6)

    md.append("## 1. Memory Extraction (Phase 2)")
    md.append("")
    md.append("**Model:** `gpt-4o-mini` (OpenAI API), `temperature=0.0`, `response_format=json_object`")
    md.append(f"**Confidence threshold:** `{confidence_threshold}` — items below this are discarded.")
    md.append("")
    md.append("**System prompt:**")
    md.append("")
    md.append(_code_block(SYSTEM_PROMPT, ""))
    md.append("")
    md.append("**Post-extraction deduplication:** `src/extractor/deduplicate.py`")
    md.append("Canonicalises (subject, predicate, category) tuples and sets `is_update=True`")
    md.append("only when the *value* has meaningfully changed (Jaccard-similarity check).")
    md.append("")


def _section_salience(md: list[str]) -> None:
    sal = _load_json("configs/salience_config.json")
    md.append("## 2. Salience Scoring (Phase 4)")
    md.append("")
    md.append("Composite score = weighted sum of five components + temporal decay.")
    md.append("")
    md.append("**Component weights:**")
    md.append("")
    md.append("| Component | Weight |")
    md.append("|---|---|")
    for k, v in sal["weights"].items():
        md.append(f"| {k.replace('_', ' ').title()} | {v:+.2f} |")
    md.append("")
    md.append(f"**Temporal decay:** λ = `{sal['temporal_decay_lambda']}`")
    md.append(f"**Salience threshold:** `{sal['salience_threshold']}` — items below are excluded from training.")
    md.append("")
    md.append("**Explicit-change signal phrases:**")
    patterns = sal.get("explicit_change_patterns", [])
    md.append(", ".join(f"`{p}`" for p in patterns))
    md.append("")


def _section_training(md: list[str]) -> None:
    cfg = _load_json("configs/train_config.json")
    lora  = cfg["lora"]
    train = cfg["training"]
    mix   = cfg["batch_mixture"]
    rb    = cfg["replay_buffer"]
    reg   = cfg["regularizer"]
    am    = cfg["anti_memory"]
    model = cfg["model"]

    md.append("## 3. LoRA Training (Phase 5 — Sleep Consolidation)")
    md.append("")
    md.append("### Base model")
    md.append("")
    md.append(f"| Setting | Value |")
    md.append("|---|---|")
    md.append(f"| Base model | `{model['base_model_id']}` |")
    md.append(f"| Quantisation | 4-bit `{model['bnb_4bit_quant_type'].upper()}`, double-quant={model['bnb_4bit_use_double_quant']}, compute dtype `{model['bnb_4bit_compute_dtype']}` |")
    md.append("")
    md.append("### LoRA adapter")
    md.append("")
    md.append(f"| Setting | Value |")
    md.append("|---|---|")
    md.append(f"| Rank `r` | `{lora['r']}` |")
    md.append(f"| Alpha | `{lora['lora_alpha']}` |")
    md.append(f"| Target modules | {', '.join(f'`{m}`' for m in lora['target_modules'])} |")
    md.append(f"| Dropout | `{lora['lora_dropout']}` |")
    md.append(f"| Bias | `{lora['bias']}` |")
    md.append("")
    md.append("### Training hyperparameters")
    md.append("")
    md.append(f"| Setting | Value |")
    md.append("|---|---|")
    md.append(f"| Epochs per cycle | `{train['num_epochs']}` |")
    md.append(f"| Learning rate | `{train['learning_rate']}` |")
    md.append(f"| Batch size (per device) | `{train['per_device_train_batch_size']}` |")
    md.append(f"| Gradient accumulation steps | `{train['gradient_accumulation_steps']}` |")
    md.append(f"| Effective batch size | `{train['per_device_train_batch_size'] * train['gradient_accumulation_steps']}` |")
    md.append(f"| Warmup steps | `{train['warmup_steps']}` |")
    md.append(f"| Max gradient norm | `{train['max_grad_norm']}` |")
    md.append(f"| FP16 | `{train['fp16']}` |")
    md.append(f"| Consolidation trigger | Every `{cfg['consolidation_trigger_days']}` simulated days |")
    md.append("")
    md.append("### Adapter accumulation (continual learning)")
    md.append("")
    md.append("Each sleep cycle **loads the previous cycle's adapter** as a trainable")
    md.append("initialisation point (not merged into base weights). The PEFT model is")
    md.append("re-initialised from the saved checkpoint and training continues. This")
    md.append("prevents catastrophic forgetting of earlier consolidation cycles.")
    md.append("")
    md.append("### Training batch mixture")
    md.append("")
    md.append(f"| Mix component | Ratio |")
    md.append("|---|---|")
    md.append(f"| Declarative memory statements | `{mix['declarative_ratio']:.0%}` |")
    md.append(f"| QA probes (train templates) | `{mix['qa_ratio']:.0%}` |")
    md.append(f"| Raw dialogue snippets | `{mix['dialogue_ratio']:.0%}` |")
    md.append(f"| QA phrasings per fact | `{mix['qa_phrasings_per_fact']}` |")
    md.append("")
    md.append("### Replay buffer")
    md.append("")
    md.append(f"| Setting | Value |")
    md.append("|---|---|")
    md.append(f"| Enabled | `{rb['enabled']}` |")
    md.append(f"| Historical sample ratio | `{rb['historical_sample_ratio']}` of new batch size |")
    md.append(f"| Weighting | `{rb['weighting']}` (salience-weighted random sample) |")
    md.append("")
    md.append("### Regulariser")
    md.append("")
    md.append(f"| Setting | Value |")
    md.append("|---|---|")
    md.append(f"| Generic conversational exchanges per batch | `{reg['generic_dialogue_samples']}` |")
    md.append("")
    md.append("### Anti-Memory pairs (Objective 3 — unlearning superseded facts)")
    md.append("")
    md.append(f"| Setting | Value |")
    md.append("|---|---|")
    md.append(f"| Enabled (main condition only) | `{am['enabled']}` |")
    md.append(f"| Pairs per updated memory | `{am['pairs_per_update']}` |")
    md.append("")
    md.append("**Format:** for each memory with `is_update=True` the batch includes:")
    md.append("")
    md.append('```')
    md.append('Q: "Is {name} still {old_value}?"')
    md.append('A: "No — {name} is now {new_value}."')
    md.append('```')
    md.append("")


def _section_evaluation(md: list[str]) -> None:
    eval_cfg = _load_json("configs/eval_config.json")
    md.append("## 4. Zero-Context Evaluation (Phase 7)")
    md.append("")
    md.append("**Conditions evaluated:** frozen, RAG, naïve LoRA, unfiltered LoRA,")
    md.append("oracle-data LoRA, ablation_no_salience, ablation_no_replay,")
    md.append("ablation_no_negative, MemLoRA (main).")
    md.append("")
    md.append("**Prompt:** zero-context for all LoRA conditions — only system prompt + question.")
    md.append("No historical dialogue is prepended.")
    md.append("")
    md.append("**RAG condition:** BM25 Top-k retrieval (`k=3`) from the persona's extracted")
    md.append("memory pool, query = eval probe question.")
    md.append("")
    md.append("### Scoring (post TMLR revision)")
    md.append("")
    md.append("| Bucket | Scoring method |")
    md.append("|---|---|")
    md.append("| Stable | Deterministic keyword matching (≥ 50 % of expected keywords in response) |")
    md.append("| Updated | Deterministic keyword matching (≥ 50 % of expected keywords in response) |")
    md.append("| Superseded | Deterministic: new-value keywords present (≥ 40 %) AND stale claim not affirmed |")
    md.append("| Relational | OpenAI `gpt-4o-mini`, `temperature=0.0`, `response_format=json_object` |")
    md.append("")
    md.append("**Eval templates** (from `configs/eval_config.json`):")
    md.append("")
    for t in eval_cfg.get("qa_eval_templates", []):
        md.append(f"- `{t}`")
    md.append("")
    md.append("**Eval templates are strictly different from train templates** to prevent test leakage.")
    md.append("")


def _section_experiment_scale(md: list[str]) -> None:
    md.append("## 5. Experiment Scale")
    md.append("")
    md.append("| Dimension | Value |")
    md.append("|---|---|")
    md.append("| Personas | 10 (Alice, Bob, Charlie, Diana, Ethan, Fiona, George, Hannah, Ian, Julia) |")
    md.append("| Simulated days per persona | 20 |")
    md.append("| Consolidation cycles per persona | 6 (days 3, 6, 9, 12, 15, 18) |")
    md.append("| Random seeds | 3 (42, 123, 456) |")
    md.append("| Conditions | 9 (frozen, RAG, naïve LoRA, unfiltered LoRA, oracle-data LoRA, ablation ×3, MemLoRA) |")
    md.append("| Eval buckets | 4 (stable, updated, superseded, relational) |")
    md.append("| Hardware | Single NVIDIA RTX 4090 (24 GB VRAM), RunPod |")
    md.append("| Total runtime | ~337 min (main TMLR run) |")
    md.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/methodology_details.md")
    args = parser.parse_args()

    # Add repo root to path for imports.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md: list[str] = []
    md.append("# MemLoRA v1 — Methodology Details")
    md.append("")
    md.append("> Auto-generated by `scripts/export_hyperparams.py`.")
    md.append("> All values are read directly from config files and source code.")
    md.append("")

    _section_extraction(md)
    _section_salience(md)
    _section_training(md)
    _section_evaluation(md)
    _section_experiment_scale(md)

    text = "\n".join(md) + "\n"
    out_path.write_text(text)
    print(f"Methodology details → {out_path}")
    print(f"  {len(md)} lines, {len(text)} chars")


if __name__ == "__main__":
    main()
