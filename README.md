# MemLoRA v1

**Hypothesis:** Selective offline consolidation of user-specific memories into LoRA adapters improves the retention of evolving personal facts over frozen and naive continual-learning baselines, without relying on historical context at inference.

---

## Project Context

A tightly scoped academic experiment. Not a general memory system. Prioritize experimental control, clean baselines, and a defensible paper over architectural cleverness.

**Base model:** `Meta-Llama-3-8B-Instruct`, 4-bit quantized via `bitsandbytes`  
**LoRA config:** `q_proj`, `v_proj`, rank=16  
**Simulation:** 2 personas × 20 simulated days per persona  
**Consolidation trigger:** Sleep phase every 3 simulated days  
**Compute:** Single rented GPU (RunPod RTX 4090 or A6000); sequential pipeline only

---

## Architecture Overview (Wake / Sleep)

1. **Wake Phase** — Model interacts with a simulated user. Dialogue is logged. Structured memory candidates are extracted from dialogue.
2. **Sleep Phase** — Memory candidates are ranked by salience, filtered, and consolidated into model weights via LoRA training.
3. **Eval Phase** — Adapted model answers evaluation questions from parameters alone. Zero historical dialogue context is allowed at inference.

---

## Repository Structure

```
memlora/
├── configs/
│   ├── sim_config.json       # Persona definitions, seeds, timeline params
│   ├── extract_config.json   # Extraction prompt, model, confidence threshold
│   ├── salience_config.json  # Component weights, decay λ
│   ├── train_config.json     # LoRA hyperparams, batch mixture ratios
│   └── eval_config.json      # Eval buckets, judge model, zero-context flag
├── data/
│   ├── personas/             # Structured persona ground truth (JSON)
│   ├── dialogue/             # Raw episodic dialogue logs (JSONL)
│   ├── memories/             # Distilled memory items (JSONL)
│   └── eval_probes/          # Programmatically generated eval questions (JSONL)
├── logs/                     # Training telemetry: loss, param counts, VRAM, runtime (JSONL)
├── scripts/
│   ├── setup_pod.sh          # Fresh pod restore: install deps, pull data
│   ├── sync_local.sh         # Pre-shutdown: copy logs, checkpoints, results locally
│   ├── run_phase1.sh         # Simulator
│   ├── run_phase2.sh         # Extractor
│   ├── run_phase3.sh         # Extraction eval
│   ├── run_phase4.sh         # Salience scoring
│   ├── run_phase5.sh         # Sleep training
│   ├── run_phase6.sh         # Baselines
│   └── run_phase7.sh         # Final evaluation
├── src/
│   ├── simulator/            # Persona engine and dialogue generator
│   ├── extractor/            # LLM-based memory extraction
│   ├── salience/             # Composite salience scoring + decay
│   ├── trainer/              # 4-bit LoRA PEFT training loop
│   ├── baselines/            # All 5 baseline execution paths
│   └── eval/                 # Zero-context inference + metrics
├── results/                  # Per-run metric summaries, tables, plots
├── checkpoints/              # LoRA adapter checkpoints (per persona, per sleep cycle)
├── analysis/
│   ├── summarize.py          # Aggregate metrics across conditions and seeds → tables
│   └── plot_results.py       # Generate paper-ready figures
└── README.md
```

**Note:** `data/dialogue/` holds raw interaction transcripts. `logs/` holds training-time telemetry (loss curves, VRAM, runtime). These are separate concerns.

---

## Environment Setup

```bash
# On a fresh RunPod pod
bash scripts/setup_pod.sh

# Manual install (if needed)
pip install torch transformers peft bitsandbytes accelerate pydantic datasets
```

Exact package versions are pinned in `requirements.txt`.

---

## How to Run

```bash
# Full sequential pipeline
bash scripts/run_phase1.sh --config configs/sim_config.json
bash scripts/run_phase2.sh --config configs/extract_config.json
bash scripts/run_phase3.sh                          # Extraction eval (verify before training)
bash scripts/run_phase4.sh --config configs/salience_config.json
bash scripts/run_phase5.sh --config configs/train_config.json  # Main system
bash scripts/run_phase6.sh --config configs/train_config.json  # All baselines
bash scripts/run_phase7.sh --config configs/eval_config.json

# Aggregate results
python analysis/summarize.py --results_dir results/
python analysis/plot_results.py --results_dir results/
```

The active experimental condition is controlled by `experiment_mode` in `train_config.json`:
```json
{ "experiment_mode": "main" }
```
Valid values: `"frozen"`, `"naive_lora"`, `"unfiltered_lora"`, `"gold_lora"`, `"rag"`, `"main"`.
All conditions share the same config file; only this flag changes.

---

## Component Specifications

### A. Persona Engine (Simulator)
- Deterministic Python state machine backed by Pydantic/JSON schemas
- Maintains: stable facts, changing preferences, superseded states, event timeline
- A small/fast LLM may paraphrase state transitions into natural dialogue — strictly for surface form, never to invent facts
- **The structured JSON state is the absolute ground truth for extraction and evaluation**
- Both personas share the same fact categories (job, location, diet, relationship status) so evaluation bucket comparisons are clean across personas

### B. Memory Pipeline
Two-level hierarchy:
1. **Episodic Log** — raw dialogue transcripts stored in `data/dialogue/` (not trained on directly by default)
2. **Distilled Memories** — LLM-extracted atomic JSON items stored in `data/memories/`

Memory item schema:
```json
{
  "memory_id": "uuid",
  "subject": "...",
  "predicate": "...",
  "value": "...",
  "day": 0,
  "confidence": 0.0,
  "is_update": false,
  "supersedes_memory_id": null,
  "consolidated": false,
  "salience_score": null
}
```

### C. Salience Scoring
Composite interpretable score (no likelihood-drift):

| Component | Default Weight | Computation |
|---|---|---|
| Novelty | 0.30 | 1 if no consolidated item shares (subject, predicate); partial credit for updates |
| Recurrence | 0.20 | Mentions across recent sessions / days since first mention, capped at 1 |
| Explicit Change | 0.25 | Pattern match: "used to", "changed", "no longer", "now X" |
| Fact Density | 0.15 | Distinct extractable facts in source utterance / normalizing constant |
| Banter Penalty | −0.10 | Pattern match on filler; reduces final score |

Plus temporal decay: multiply by `exp(−λ × days_since_last_mention)`, λ=0.1 default.
All weights and λ stored in `configs/salience_config.json`. No hyperparameter sweeps in v1.

### D. Sleep Phase Training
Triggered every 3 days. Training batch mixture (configurable ratios):
- 40% distilled declarative memory statements
- 40% QA-style probes derived from memories
- 20% short raw dialogue snippets tied to accepted memories

---

## Baselines

All baselines run on the **same 20-day timeline, same seeds, same evaluation probes**.

| # | Name | `experiment_mode` | Description |
|---|------|---|-------------|
| 1 | Frozen | `frozen` | Base Llama-3-8B-Instruct, no memory, no training |
| 2 | Naive Continual LoRA | `naive_lora` | LoRA on raw episodic dialogue, no extraction or filtering |
| 3 | Unfiltered Distilled LoRA | `unfiltered_lora` | LoRA on extracted memories, bypassing salience/decay |
| 4 | Gold Distilled LoRA | `gold_lora` | LoRA on exact ground-truth facts from Simulator (upper bound) |
| 5 | Summary-in-Context (RAG) | `rag` | Frozen model + memory summary prepended at inference |
| — | **Main (MemLoRA)** | `main` | LoRA on salience-filtered distilled memories |

**Fairness note on RAG:** The RAG baseline has explicit inference-time context — this is the advantage being ablated. LoRA conditions use zero context. The paper must frame this explicitly: RAG is not a fair zero-context comparison, it is the context-allowed ceiling. Memory content is identical across RAG and the parametric LoRA conditions.

**Parameter count:** All LoRA conditions add the same number of trainable parameters (r=16, q_proj+v_proj). Log the exact count per run. Note it in results.

---

## Telemetry and Diagnostics

Every sleep cycle logs to `logs/{persona_id}_train_day{N}.jsonl`:
- `train_loss` per step
- `val_loss` (if validation split defined — see Open Questions)
- `lora_param_count`
- `lora_weight_drift` (L2 norm of LoRA delta from initialization)
- `vram_used_gb`
- `runtime_seconds`
- `n_memories_consolidated`, `n_memories_filtered`
- `salience_score_distribution` (mean, std, min, max)
- `supersession_events` (count of is_update=True items consolidated)

Every eval run logs to `results/{condition}_{persona_id}_eval.json`:
- accuracy per bucket
- contradiction / hybrid-state rate
- per-probe predictions
- git commit hash
- hardware info (GPU model, VRAM)
- random seeds used

Extraction eval logs to `results/extraction_eval_{persona_id}.json`:
- Precision, Recall, False Insertion Rate, Update Linking Correctness

All logs are machine-readable JSONL or JSON. `analysis/summarize.py` aggregates across conditions and seeds into a single comparison table.

---

## Multi-Seed Policy

- **Phases 1–4** (deterministic simulation, extraction eval, salience): 1 seed sufficient for debugging and verification.
- **Phases 5–7** (training and evaluation): **3 seeds** required for paper-grade claims. Seeds affect LoRA weight initialization and, if LLM inference is non-deterministic, extraction and dialogue paraphrasing.
- Seeds are set explicitly in config and logged with every artifact.
- Results are reported as mean ± std across seeds.

---

## Compute Estimates and Staged Execution

**Compute routing rule:**
- **OpenAI API (small model):** all LLM text tasks — dialogue paraphrasing, extraction, eval judging
- **Pod GPU:** any code that runs more than a few seconds — LoRA training, batch inference with the local 8B model, large evaluation loops. Pod GPU >> M1 Mac for these.
- **Local Mac:** fast scripts only — salience scoring, metrics aggregation, result formatting

| Phase | Where to run | Estimated VRAM | Estimated runtime | Stage plan |
|---|---|---|---|---|
| 1 (Simulator) | Local (CPU) | — | < 1 min | — |
| 2 (Extraction) | OpenAI API + Local | — | ~10–20 min / persona | debug 1 day → full 20 days |
| 3 (Extraction eval) | Local (CPU) | — | < 1 min | — |
| 4 (Salience) | Local (CPU) | — | < 1 min | — |
| 5 (Sleep training) | **Pod GPU** | ~18–22 GB (4-bit + LoRA) | ~10–20 min / sleep cycle | 1-cycle sanity → full 7 cycles |
| 6 (Baselines) | **Pod GPU** | ~18–22 GB | ~5× Phase 5 | sanity first per condition |
| 7 (Eval inference) | **Pod GPU** | ~12 GB (local 8B model) | ~20–30 min | spot-check before full run |
| 7 (LLM judge scoring) | OpenAI API | — | ~5–10 min | after inference outputs saved to disk |

For each pod GPU phase: start with a 1-step or 1-day debug run before committing to full execution. Save outputs to disk before stopping the pod.

---

## Evaluation Protocol

**Zero historical context** in all LoRA-based evaluation prompts. System prompt + question only.

Evaluation probes are generated programmatically from Persona Ground Truth — one probe per fact per bucket.

| Bucket | What it tests |
|---|---|
| Stable facts | Facts that never changed |
| Updated facts | Ask for the *current* value after an update |
| Superseded facts | Ask about the *old* value — correct answer is rejection or correction |
| Relational / multi-hop | Requires combining two or more facts |

Scoring: LLM judge (configurable model) + exact-match fallback for simple factual answers. Both scores saved.

**Extraction evaluation runs before any training.** Extraction failure must not masquerade as parametric memory failure.

---

## Reproducibility Requirements

Every phase saves to disk:
- Random seeds (in config files)
- Persona definitions (`data/personas/`)
- Extracted memories and salience scores (`data/memories/`)
- Train/eval splits
- LoRA hyperparameters and checkpoint paths (`checkpoints/`)
- Telemetry per sleep cycle (`logs/`)
- Final metrics (`results/`)
- Git commit hash and hardware info (logged automatically by eval script)

Runs are replayable from saved artifacts without regenerating upstream phases.

---

## Paper Artifact Mapping

| Paper section | Key artifact |
|---|---|
| Methods | `configs/`, `src/simulator/`, `src/salience/`, `src/trainer/` |
| Experiments | `results/final_metrics.json`, `analysis/summarize.py` output |
| Ablation | Conditions 2–4 vs main; salience component ablation if needed |
| Appendix | `logs/` telemetry, per-probe predictions, extraction eval |
| Reproducibility checklist | Seeds, git hash, hardware info, `requirements.txt` |

---

## Fallback Strategy

If the main system underperforms: expand LoRA target modules modestly (e.g., add `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Do not redesign the architecture. Keep iteration controlled so the paper narrative remains clear on what changed and why.

---

## Execution Plan (Phase-by-Phase)

Run phases **sequentially**. Do not advance until the current phase passes its deliverable gate.

**Phase deliverable gate (applies to every phase):**
Before advancing, confirm: (a) code is complete, (b) 1-seed sanity run passes, (c) outputs are on disk in expected locations, (d) README Current Status is updated.

- [ ] **Phase 1 — Simulator Engine**  
  Deterministic persona state machine + dialogue generator.  
  *Outputs:* `data/personas/{id}_ground_truth.json`, `data/dialogue/{id}_dialogue.jsonl`

- [ ] **Phase 2 — Extraction Pipeline**  
  Prompt + parsing logic → Distilled Memory JSON schema.  
  *Outputs:* `data/memories/{id}_memories.jsonl`

- [ ] **Phase 3 — Extraction Evaluation**  
  Standalone eval script vs Phase 1 ground truth. **Must pass before any training.**  
  *Outputs:* `results/extraction_eval_{id}.json`

- [ ] **Phase 4 — Salience & Filtering**  
  Composite salience scoring and temporal decay; annotate `data/memories/` with scores.  
  *Outputs:* updated `data/memories/{id}_memories.jsonl` (salience_score populated)

- [ ] **Phase 5 — Consolidation Training (Sleep)**  
  Batch mixture generator + 4-bit LoRA PEFT training loop, 3-day trigger.  
  *Outputs:* `checkpoints/{id}/day_{N}/`, `logs/{id}_train_day{N}.jsonl`

- [ ] **Phase 6 — Baselines**  
  All 5 baseline execution paths, controlled by `experiment_mode` flag.  
  *Outputs:* separate checkpoint dirs and telemetry logs per condition

- [ ] **Phase 7 — Evaluation Suite**  
  Zero-context inference loop, bucket metrics, LLM judge, final summary tables.  
  *Outputs:* `results/{condition}_{id}_eval.json`, `results/final_metrics.json`

---

## RunPod Setup

On a fresh pod:
```bash
bash scripts/setup_pod.sh
```
Installs dependencies and pulls data from persistent storage or git. All outputs write to `data/`, `logs/`, `checkpoints/`, `results/` so the pipeline resumes from disk.

Before stopping a pod to pause billing:
```bash
bash scripts/sync_local.sh
```
Copies `logs/`, `results/`, `checkpoints/` (latest per condition) and `data/memories/` locally. Does **not** copy raw dialogue logs or model weights (too large; not needed for diagnostics).

---

## Current Status

### Done
- Project instructions written and committed
- README with full plan, telemetry spec, multi-seed policy, compute estimates, phase gates

### Issues
- None yet (pre-implementation)
- Several design questions require answers before Phase 1 coding starts (see below)

### Next Steps
1. Resolve open questions (see below)
2. Scaffold repository: `memlora/` directory structure, `requirements.txt`, placeholder configs
3. Implement Phase 1: Persona Engine (Pydantic state machine + dialogue generator)
4. Run Phase 1 sanity check; verify ground truth JSON and dialogue JSONL on disk

---

## Resolved Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dialogue paraphrasing model | OpenAI API (small model) | Fast, cheap, no VRAM; API is appropriate for LLM text tasks |
| Extraction model | OpenAI API (small model) | Same — text in, structured JSON out; no heavy compute needed |
| LLM eval judge | OpenAI API (small model) | Judging is text reasoning, not heavy compute |
| Heavy compute (training, batch inference) | Pod GPU | Pod GPU far outperforms M1 Mac for any long-running compute |
| Persona backstories | Left to implementation | Personas share fact categories (job, location, diet, relationship status) for clean bucket comparisons |

**API model:** Use `OPENAI_MODEL` from `.env` for all OpenAI calls. Default small model is `gpt-4o-mini`.  
**Pod GPU rule:** Use pod GPU for any code that would run more than a few seconds — LoRA training, running the local 8B model for batch inference, evaluation loops over many examples. Do not run these on the local Mac.

---

## Open Questions (Remaining — Need Answers Before Coding)

**Q5. Data persistence between pod runs**  
Where does data live when the pod is off?
- (a) Git — small JSON/JSONL files only; model weights and checkpoints excluded
- (b) RunPod Network Volume — persistent across pod restarts, ~$0.07/GB/month
- (c) S3 or similar — cheap, slightly more setup

**Q6. Multi-seed budget**  
The plan calls for 3 seeds for paper-grade claims (Phases 5–7). Given the $500 budget, is this acceptable, or should v1 use 1 seed with a note that multi-seed is future work?

**Q7. Validation split during sleep training**  
Should the sleep training loop hold out a small set of memory items to log val loss and detect overfitting per cycle? Or train-only given the small per-cycle batch?
