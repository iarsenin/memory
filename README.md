# MemLoRA → TemporalBench v2

> **Current paper:** *"Remembering to Forget: Behavioral Superposition Under Fact Drift in Continual LLM Personalization"* — under review at TMLR.  
> The submission is based on **TemporalBench v2** (`v2_temporal_benchmark/`). MemLoRA v1 (`src/`) was the pilot that identified the core failure mode; v2 replaced its open-ended LLM-judge evaluation with a strict MCQA benchmark, enabling the distributional analysis of Behavioral Superposition.

---

## Current Results (TemporalBench v2 — Paper Submission)

Effective personalized memory requires not only retention, but **correct invalidation of superseded facts**. Parametric and retrieval-based systems fail in complementary ways under temporal fact drift.

### TemporalBench v2 — Key Results

30 synthetic personas × 5 updated facts × 4 MCQA probe families = **600 probes** evaluated across a dual volume sweep (5 tiers × 3 seeds × random + salience-ordered) and a BM25 RAG baseline.

**Task-Normalized Accuracy by probe family** (averaged over 3 seeds):

| Model Strategy | Current-State | Stale Rejection | Historical-State | Relational | Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| LoRA (10% Volume) | 48.0% | 39.8% | 55.3% | 62.4% | 51.4% |
| LoRA (50% Volume) | 44.5% | 41.8% | 54.0% | 61.8% | 50.5% |
| LoRA (100% Volume) | 53.6% | 50.0% | 55.0% | 64.0% | **55.6%** |
| RAG (BM25 Top-3) | **63.3%** | 32.0% | **60.4%** | **72.7%** | **57.1%** |

**Key findings:**
- **Behavioral Superposition** peaks at 75% salience-ordered volume (15.8% "Both" selections on forward-time queries) — partial parametric consolidation produces a fused, unresolved temporal state
- Task accuracy is **flat through intermediate volume tiers** (~51%), recovering only at full-volume consolidation (55.6%)
- **LoRA vs RAG fail differently:** LoRA improves stale-premise rejection (50% at full volume) but exhibits more "Both" responses; RAG attains higher overall accuracy but fails on temporal precedence (stale endorsement 42.1%, near-zero superposition 0.7%)
- Salience-ordered consolidation **shifts the response distribution** but does not eliminate superposition

Publication charts: `v2_temporal_benchmark/results/fig1_volume_fidelity.png`, `fig2_superposition_stack.png`

### MemLoRA v1 — Pilot Results (superseded by v2)

> v1 used a 7-phase wake/sleep pipeline with open-ended LLM judging. It confirmed that salience filtering + anti-memory training beats naïve continual LoRA on superseded-fact rejection, and identified Behavioral Superposition as the key failure mode. These findings directly motivated the stricter v2 MCQA benchmark. **v1 results are not in the current paper submission.**

10 personas × 20 simulated days, 7 conditions × 3 seeds evaluated with LLM judge (gpt-4o-mini, zero context):

| Condition | Stable | Updated | Superseded | Relational | **Overall** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MemLoRA (ours)** | 24.7 ±4.7 | 41.7 ±4.6 | **25.6 ±6.9** | 44.4 ±4.8 | **35.5 ±4.6** |
| Naïve LoRA | 26.1 ±3.4 | 32.8 ±3.4 | 11.7 ±5.8 | 25.0 | 31.0 ±2.1 |
| Unfiltered LoRA | 45.6 ±9.9 | 50.1 ±0.6 | 20.3 ±19.3 | 47.2 ±12.7 | 44.8 ±0.7 |
| Oracle-Data LoRA (upper bound) | 38.1 ±19.8 | 60.0 ±9.6 | 51.1 ±16.9 | 52.8 ±12.7 | 50.1 ±5.8 |
| Ablation: No Salience | 37.9 ±10.0 | 39.8 ±3.8 | 9.2 ±1.2 | 33.3 ±11.8 | 36.1 ±6.1 |
| Ablation: No Replay | 21.7 ±10.6 | 41.7 ±2.4 | 18.3 ±14.1 | 50.0 | 34.0 ±2.4 |
| Ablation: No Anti-Memory | 32.5 ±2.4 | 39.4 ±5.0 | 17.5 ±1.2 | 33.3 ±11.8 | 35.8 ±3.5 |

**Key findings:**
- MemLoRA **dominates the Superseded bucket** (25.6% vs 11.7% Naïve) — salience filtering + anti-memory training is the critical mechanism for stale-fact rejection
- Anti-memory pairs contribute **+8.1pp on Superseded**; removing salience collapses Superseded from 25.6% → 9.2%
- Unfiltered LoRA leads on Stable/Updated — training-data volume matters, but at the cost of Superseded precision

Detailed aggregation: `analysis/paper_results.md` / `analysis/paper_results.json`

---

## Repository Structure

```
memory/
├── configs/
│   ├── sim_config.json         # Persona definitions, seeds, timeline params
│   ├── extract_config.json     # Extraction prompt, model, confidence threshold
│   ├── salience_config.json    # Component weights, temporal decay λ
│   ├── train_config.json       # LoRA hyperparams, batch mixture ratios
│   └── eval_config.json        # Eval buckets, judge model, zero-context flag
├── src/
│   ├── simulator/              # Persona state machine + dialogue generator
│   ├── extractor/              # LLM-based memory extraction + deduplication
│   ├── salience/               # Composite salience scoring + decay
│   ├── trainer/                # 4-bit LoRA PEFT training loop
│   ├── baselines/              # All baseline execution paths
│   └── eval/                   # Zero-context inference + LLM judge
├── scripts/
│   ├── setup_pod.sh            # Fresh pod restore: install deps
│   ├── sync_local.sh           # Pre-shutdown: rsync results locally
│   ├── run_tmlr.sh             # Full pipeline: Phases 1–4 + run_paper.sh
│   ├── run_paper.sh            # 3-seed loop: Phase 5 → 6 → 7
│   └── export_hyperparams.py   # Emit results/methodology_details.md
├── analysis/
│   ├── summarize.py            # Aggregate per-seed results → tables
│   ├── plot_results.py         # Generate MemLoRA v1 paper figures
│   ├── rejudge_responses.py    # Re-score response files with LLM judge (no GPU)
│   ├── paper_results.json      # Aggregated mean ± std (machine-readable)
│   └── paper_results.md        # Aggregated results table (human-readable)
├── v2_temporal_benchmark/
│   ├── generate_mcqa_data.py   # 30 personas × 5 facts × 4 probe families
│   ├── evaluator.py            # Distributional answer extractor (regex + fallback)
│   ├── run_sweeps.py           # Dual volume sweep (5 tiers × 3 seeds × 2 types)
│   ├── rag_baseline.py         # BM25 Top-3 fair RAG baseline
│   ├── plot_results.py         # Publication figures (task-normalized accuracy)
│   ├── data/
│   │   ├── personas.json       # 30 LLM-generated personas (seed=42)
│   │   └── benchmark.json      # 600 MCQA probes (150 facts × 4 families)
│   └── results/
│       ├── sweep_results.csv   # 30 rows: random + salience, 5 tiers, 3 seeds
│       ├── rag_results.csv     # BM25 Top-3 baseline row
│       ├── fig1_volume_fidelity.png
│       └── fig2_superposition_stack.png
├── data/
│   ├── personas/               # Structured persona ground truth (JSON)
│   ├── dialogue/               # Raw episodic dialogue logs (JSONL)
│   ├── memories/               # Distilled memory items (JSONL)
│   └── eval_probes/            # Programmatically generated eval questions
├── results/
│   ├── paper/seed{42,123,456}/ # Per-condition per-persona eval JSONs
│   └── methodology_details.md  # Hyperparameter appendix (auto-generated)
├── logs/                       # Training telemetry: loss, VRAM, runtime (JSONL)
└── checkpoints/                # LoRA adapter checkpoints (deleted after eval)
```

---

## Environment Setup

```bash
# On a fresh RunPod pod
bash scripts/setup_pod.sh

# Manual install
pip install torch transformers peft bitsandbytes accelerate pydantic datasets rank_bm25
```

Exact versions pinned in `requirements.txt`. The base model is `Meta-Llama-3-8B-Instruct` (4-bit NF4 via bitsandbytes). Set `HUGGING_FACE_TOKEN` and `OPENAI_API_KEY` in `.env`.

---

## How to Run

### MemLoRA v1 Pipeline (pilot — results in `analysis/paper_results.md`)

```bash
# Full pipeline on pod (Phases 1–7, all conditions, 3 seeds)
bash scripts/run_tmlr.sh

# Aggregate results locally
python analysis/summarize.py --results_dir results/paper --seeds 42 123 456
python analysis/plot_results.py

# Re-score response files with LLM judge (local, no GPU needed)
python analysis/rejudge_responses.py --seeds 42 123 456

# Export methodology details for paper appendix
python scripts/export_hyperparams.py
```

The active experimental condition is set by `experiment_mode` in `train_config.json`:
```json
{ "experiment_mode": "main" }
```
Valid values: `"frozen"`, `"naive_lora"`, `"unfiltered_lora"`, `"oracle_data_lora"`, `"rag"`, `"main"`, `"ablation_no_salience"`, `"ablation_no_replay"`, `"ablation_no_negative"`.

### TemporalBench v2 (current paper — results already complete)

```bash
# Step 1: Generate benchmark (runs locally via OpenAI API, ~10 min)
python v2_temporal_benchmark/generate_mcqa_data.py --n-personas 30 --seed 42

# Step 2: Run dual volume sweep on pod (~6.3 hr, RTX 4090)
python v2_temporal_benchmark/run_sweeps.py \
  --volume-tiers 0.10 0.25 0.50 0.75 1.00 \
  --seeds 42 123 456 \
  --sweep-types random salience

# Step 3: Run RAG baseline on pod (~30 min)
python v2_temporal_benchmark/rag_baseline.py

# Step 4: Generate publication figures (local)
python v2_temporal_benchmark/plot_results.py
```

Dry-run (verify CSV output before committing full compute):
```bash
python v2_temporal_benchmark/run_sweeps.py \
  --volume-tiers 0.10 0.25 --seeds 42 --sweep-types random --max-probes 60
```

---

## MemLoRA v1 — Architecture (Pilot, Superseded by v2)

**Base model:** `Meta-Llama-3-8B-Instruct`, 4-bit quantized (NF4)  
**LoRA config:** 7 target modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), rank=16, alpha=32  
**Simulation:** 10 personas × 20 simulated days  
**Consolidation trigger:** Sleep phase every 3 simulated days  

### Wake / Sleep Cycle

1. **Wake Phase** — Model interacts with a simulated user. Dialogue is logged. Structured memory candidates are extracted by an LLM (OpenAI API).
2. **Sleep Phase** — Memory candidates are ranked by salience, filtered by threshold, and consolidated into model weights via LoRA fine-tuning. Batch mixture: 40% declarative statements + 40% QA-style probes + 20% short dialogue snippets. Anti-memory negative pairs ("Is X still Y? No — X is now Z.") are generated for superseded facts.
3. **Eval Phase** — Adapted model answers evaluation probes from parameters alone. **Zero historical context at inference.**

### Salience Scoring

Composite interpretable score applied before each Sleep phase:

| Component | Weight | Computation |
|---|---|---|
| Novelty | 0.30 | 1 if no consolidated item shares (subject, predicate) |
| Recurrence | 0.20 | Mentions across recent sessions / days since first mention |
| Explicit Change | 0.25 | Pattern match on "used to", "changed", "no longer", "now X" |
| Fact Density | 0.15 | Distinct extractable facts in source utterance |
| Banter Penalty | −0.10 | Pattern match on filler phrases |

Plus temporal decay: `exp(−λ × days_since_last_mention)`, λ=0.1. All weights stored in `configs/salience_config.json`.

### Baselines & Ablations

All conditions share the same 20-day timeline, seeds, and evaluation probes.

| # | Name | `experiment_mode` | Description |
|---|---|---|---|
| 1 | Frozen | `frozen` | Base Llama-3-8B-Instruct, no training |
| 2 | Naïve Continual LoRA | `naive_lora` | LoRA on raw episodic dialogue, no extraction/filtering |
| 3 | Unfiltered Distilled LoRA | `unfiltered_lora` | LoRA on extracted memories, bypassing salience |
| 4 | Oracle-Data LoRA | `oracle_data_lora` | LoRA on ground-truth facts from simulator (upper bound) |
| 5 | RAG (BM25 Top-3) | `rag` | Frozen model + BM25 Top-3 retrieval keyed on probe question |
| — | **MemLoRA (main)** | `main` | Salience-filtered memories + anti-memory negative training |
| A1 | No Salience | `ablation_no_salience` | Uniform memory weights, bypassing salience scoring |
| A2 | No Replay | `ablation_no_replay` | Current 3-day window only; no historical replay buffer |
| A3 | No Anti-Memory | `ablation_no_negative` | 7-module adapter without negative training on superseded facts |

### Evaluation Protocol

Zero historical context in all LoRA-based evaluation prompts. Probes are generated programmatically from persona ground truth — one probe per fact per bucket.

| Bucket | What it tests |
|---|---|
| Stable | Facts that never changed |
| Updated | Current value after a life-change event |
| Superseded | Old value — correct answer requires **active correction** (not refusal) |
| Relational | Requires combining two or more facts |

**Judge:** OpenAI LLM (`temperature=0.0`, `json_object` mode) for all buckets. The Superseded bucket requires the model to state the new reality; refusals score `incorrect`.

---

## TemporalBench v2 — Architecture (Current Paper)

A strict MCQA benchmark living in `v2_temporal_benchmark/` — no changes to `src/`, `configs/`, or `scripts/`.

### Benchmark Design

30 personas × 5 updated facts each. Every updated fact generates 4 MCQA probes:

| Family | Expected answer | Tests |
|---|---|---|
| `current_state` | current | Does the model know the new value? |
| `stale_premise_rejection` | current | Can the model reject the superseded premise? |
| `historical_state` | stale | Does the model retain old value as historical fact? |
| `relational_after_update` | current | Can the model apply the new value to downstream reasoning? |

Each probe has 4 options — `current`, `stale`, `both` (superposition trap), `distractor` — shuffled to a random letter position. The evaluator reports the full distribution, not just accuracy.

### Sweep Design

Training batch always contains **100% of updated-fact sentences**; volume tier V only scales stable-fact background context.

- **Sweep A (Random):** Sample V% of stable facts uniformly at random
- **Sweep B (Salience-Ordered):** Take the top V% of stable facts by information density (highest first)

### Evaluator Metrics

| Metric | Interpretation |
|---|---|
| Task-Normalized Accuracy | Avg per-family correct-answer rate (stale=correct for historical, current=correct for all others) |
| Behavioral Superposition | % probes choosing "Both old and new" trap option (forward-time queries only) |
| Stale Endorsement | % probes still citing the superseded value |
| Confusion | % probes choosing an unrelated distractor |

---

## Compute Notes

| Task | Where | Runtime |
|---|---|---|
| Benchmark generation (OpenAI API) | Local | ~10 min |
| Full dual sweep (30 runs) | Pod RTX 4090 | ~6.3 hr |
| RAG baseline | Pod RTX 4090 | ~30 min |
| MemLoRA v1 full pipeline (3 seeds, 10 personas, 7 conditions) | Pod RTX 4090 | ~337 min |
| LLM judging / re-scoring | Local (OpenAI API) | ~5–15 min |

**Compute routing:** OpenAI API for all text tasks (extraction, judging); Pod GPU for LoRA training and local-model inference; local Mac for fast aggregation scripts.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Evaluation judge | OpenAI LLM (all buckets) | Deterministic keyword judge was too brittle for open-ended factual recall; LLM judge at `temperature=0.0` is standard practice for this paradigm |
| RAG retrieval | BM25 Top-3 keyed on probe question | Full memory log dump is unfair; question-keyed Top-3 retrieval matches the information available to LoRA at training time |
| Sleep val split | None — train-only per cycle | Per-cycle batches too small to split; Phase 7 zero-context eval detects overfitting |
| Multi-seed | 3 seeds (42, 123, 456) | Budget-conscious; sufficient for mean ± std paper claims |
| Data persistence | RunPod Network Volume + `sync_local.sh` | Volume persists across restarts; rsync protects against pod loss |
| v2 soft fork | Separate `v2_temporal_benchmark/` directory | Preserves v1 experiment integrity; no changes to `src/`, `configs/`, `scripts/` |

---

## Git Strategy

Code and lightweight configs only. The following are excluded:

```
data/dialogue/     # raw interaction transcripts (large, regenerable)
data/memories/     # extracted memory items (regenerable)
logs/              # training telemetry (regenerable)
checkpoints/       # LoRA adapter weights (deleted after eval)
results/           # metric outputs (regenerable; exceptions below)
.env               # secrets
```

**Tracked exceptions:**
- `data/personas/` — persona ground truth (small, defines the experiment)
- `data/eval_probes/` — evaluation probes (small, defines the experiment)
- `analysis/paper_results.{json,md}` — final aggregated results
- `results/methodology_details.md` — hyperparameter appendix
- `v2_temporal_benchmark/data/` — benchmark JSON (defines the experiment)
- `v2_temporal_benchmark/results/*.csv` — sweep results (primary output)
- `v2_temporal_benchmark/results/*.png` — publication figures
