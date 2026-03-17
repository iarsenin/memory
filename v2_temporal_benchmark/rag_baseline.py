"""
v2_temporal_benchmark/rag_baseline.py

Fair BM25 Top-k RAG Baseline for TemporalBench v2.

Design
------
* Knowledge base  = SAME pool of sentences as the 100%-volume LoRA sweep:
                    all stable_fact sentences + all updated_fact training sentences.
* Retrieval       = BM25 (rank_bm25) keyed on the MCQA probe question text.
                    Top k=3 sentences are injected as context.
* Inference       = base Llama-3-8B-Instruct (NO LoRA adapter) — the RAG
                    baseline is a pure retrieve-then-read system.
* Prompt format   = "Context:\n{sentences}\n\n{full_probe_prompt}"
* Evaluation      = same MCQAEvaluator → same distributional metrics as sweeps.

Output
------
  v2_temporal_benchmark/results/rag_results.csv
  v2_temporal_benchmark/results/rag_log.jsonl

CSV columns:
  k, seed (always 0 for baseline), volume_pct (always 1.0),
  + same family×metric columns as sweep_results.csv

Usage
-----
  python3 v2_temporal_benchmark/rag_baseline.py
  python3 v2_temporal_benchmark/rag_baseline.py --top-k 3
  python3 v2_temporal_benchmark/rag_baseline.py --max-probes 20  # quick check
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Soft-fork import ────────────────────────────────────────────────────────
from src.trainer.loop import load_base_model

from v2_temporal_benchmark.evaluator import MCQAEvaluator, print_distribution
from v2_temporal_benchmark.run_sweeps import (
    _V2_TRAIN_CFG,
    _stable_sentence,
    _FAMILIES,
    _METRICS,
    _append_csv,
    _csv_columns,
)

# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

def _tokenize_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """
    Thin wrapper around rank_bm25.BM25Okapi.
    Falls back to TF-IDF cosine similarity if rank_bm25 is not installed.
    """

    def __init__(self, corpus: list[str]) -> None:
        self.corpus = corpus
        self._tokenized = [_tokenize_bm25(doc) for doc in corpus]
        self._backend   = self._init_backend()

    def _init_backend(self) -> Any:
        try:
            from rank_bm25 import BM25Okapi
            return BM25Okapi(self._tokenized)
        except ImportError:
            print(
                "[warn] rank_bm25 not installed — using TF-IDF cosine fallback.\n"
                "       Install with: pip install rank_bm25",
                flush=True,
            )
            return None

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return the top-k most relevant corpus sentences for the query."""
        q_tokens = _tokenize_bm25(query)
        if not q_tokens:
            return self.corpus[:k]

        if self._backend is not None:
            scores = self._backend.get_scores(q_tokens)
        else:
            scores = self._tfidf_scores(q_tokens)

        import numpy as np
        top_k_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self.corpus[i] for i in top_k_idx]

    def _tfidf_scores(self, q_tokens: list[str]) -> list[float]:
        """Minimal TF-IDF overlap score (fallback if rank_bm25 unavailable)."""
        import math
        N   = len(self.corpus)
        idf = {
            t: math.log((N + 1) / (1 + sum(t in doc for doc in self._tokenized)))
            for t in set(q_tokens)
        }
        scores = []
        for doc_tokens in self._tokenized:
            doc_set = set(doc_tokens)
            score   = sum(idf.get(t, 0.0) for t in q_tokens if t in doc_set)
            scores.append(score)
        return scores


# ---------------------------------------------------------------------------
# Knowledge base builder
# ---------------------------------------------------------------------------

def _build_knowledge_base(
    personas: list[dict],
    benchmark: dict,
) -> list[str]:
    """
    Build the full 100%-volume knowledge base:
    all stable_fact sentences + all updated_fact training sentences.
    Mirrors the 100% LoRA sweep's training corpus.
    """
    kb: list[str] = []

    entry_by_pid: dict[str, list[dict]] = {}
    for entry in benchmark["entries"]:
        entry_by_pid.setdefault(entry["persona_id"], []).append(entry)

    for persona in personas:
        pid   = persona["persona_id"]
        pname = persona["persona_name"]

        # All updated training sentences
        for entry in entry_by_pid.get(pid, []):
            kb.extend(entry.get("training_sentences", []))

        # All stable fact sentences
        for fact in persona.get("stable_facts", []):
            kb.append(_stable_sentence(pname, fact))

    return kb


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _run_rag_inference(
    model:     Any,
    tokenizer: Any,
    probes:    list[dict],
    retriever: BM25Retriever,
    top_k:     int,
    max_new_tokens: int = 32,
    max_probes: int | None = None,
) -> list[str]:
    """
    For each probe:
      1. Retrieve top-k relevant sentences from the knowledge base.
      2. Prepend as context to the probe's full_prompt.
      3. Generate a response.
    Returns a list of raw response strings.
    """
    if max_probes is not None:
        probes = probes[:max_probes]

    model.eval()
    responses: list[str] = []

    for probe in probes:
        # ── Retrieve context ──────────────────────────────────────────────
        retrieved = retriever.retrieve(probe["question"], k=top_k)
        context   = "\n".join(f"- {s}" for s in retrieved)

        # ── Build prompt: context + original MCQA probe ───────────────────
        # full_prompt already contains the question + shuffled options + answer tag.
        # We prepend the retrieved context.
        rag_content = f"Context:\n{context}\n\n{probe['full_prompt']}"

        messages = [
            {
                "role":    "system",
                "content": (
                    "You are a factual assistant. "
                    "Use the provided context to answer the question."
                ),
            },
            {
                "role":    "user",
                "content": rag_content,
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
            max_length=768,   # slightly longer to fit context
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    return responses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_rag_baseline(
    top_k:      int,
    data_dir:   Path,
    out_dir:    Path,
    max_probes: int | None,
    hf_token:   str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rag_results.csv"
    log_path = out_dir / "rag_log.jsonl"

    # ── Load data ──────────────────────────────────────────────────────────
    bm_path = data_dir / "benchmark.json"
    pe_path = data_dir / "personas.json"
    if not bm_path.exists() or not pe_path.exists():
        sys.exit(
            f"ERROR: benchmark.json / personas.json not found in {data_dir}\n"
            "Run generate_mcqa_data.py first."
        )

    benchmark = json.load(open(bm_path))
    personas  = json.load(open(pe_path))

    all_probes = [p for e in benchmark["entries"] for p in e["probes"]]
    print(
        f"RAG baseline: {len(personas)} personas, "
        f"{len(all_probes)} probes, top_k={top_k}",
        flush=True,
    )

    # ── Build knowledge base + BM25 retriever ─────────────────────────────
    print("Building knowledge base …", flush=True)
    kb = _build_knowledge_base(personas, benchmark)
    print(f"  KB size: {len(kb)} sentences", flush=True)

    retriever = BM25Retriever(kb)
    print("  BM25 index built.", flush=True)

    # ── Load base model (NO LoRA) ──────────────────────────────────────────
    print("Loading base model …", flush=True)
    model, tokenizer = load_base_model(_V2_TRAIN_CFG, hf_token=hf_token)

    # ── Run RAG inference ─────────────────────────────────────────────────
    print(f"Running RAG inference ({len(all_probes)} probes) …", flush=True)
    t0 = time.time()
    responses = _run_rag_inference(
        model, tokenizer, all_probes, retriever,
        top_k=top_k, max_probes=max_probes,
    )
    elapsed = round(time.time() - t0, 1)
    print(f"  Inference done in {elapsed}s", flush=True)

    # ── Evaluate ──────────────────────────────────────────────────────────
    eval_probes = all_probes[:len(responses)]
    evaluator = MCQAEvaluator()
    labelled, dist = evaluator.evaluate_and_label(responses, eval_probes)
    print_distribution(dist, f"RAG (top_k={top_k})")

    # ── Write CSV ─────────────────────────────────────────────────────────
    # Reuse the same CSV schema as sweep_results.csv for unified analysis.
    is_new = not csv_path.exists()
    row: dict[str, Any] = {
        "sweep_type":      f"rag_top{top_k}",
        "seed":            0,
        "volume_pct":      1.0,              # RAG always uses 100% KB
        "n_stable_facts":  sum(
            len(p.get("stable_facts", [])) for p in personas
        ),
        "n_updated_facts": len(benchmark["entries"]) * 3,  # 3 sentences each
        "n_total_train":   len(kb),
        "train_runtime_s": 0.0,              # no training
        "vram_peak_gb":    0.0,
        "avg_loss":        0.0,
    }
    for fam in _FAMILIES:
        d = dist.get(fam, {})
        for met in _METRICS:
            row[f"{fam}_{met}"] = d.get(met, 0.0)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_columns())
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Results written to {csv_path}", flush=True)

    # ── Write verbose log ─────────────────────────────────────────────────
    log_entry = {
        "top_k": top_k,
        "n_kb":  len(kb),
        "runtime_s": elapsed,
        "distribution": dist,
        "labelled_sample": labelled[:10],
    }
    with open(log_path, "a") as lf:
        lf.write(json.dumps(log_entry) + "\n")

    print(f"  Log written to {log_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BM25 Top-k RAG baseline for TemporalBench v2."
    )
    parser.add_argument(
        "--top-k", type=int, default=3, dest="top_k",
        help="Number of sentences to retrieve per probe (default: 3)",
    )
    parser.add_argument(
        "--data-dir", default="v2_temporal_benchmark/data", dest="data_dir",
    )
    parser.add_argument(
        "--out-dir", default="v2_temporal_benchmark/results", dest="out_dir",
    )
    parser.add_argument(
        "--max-probes", type=int, default=None, dest="max_probes",
        help="Limit probes for a quick check (default: all)",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        print("[warn] HUGGING_FACE_TOKEN not set — needed for gated Llama-3 model.")

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA GPU not available.")

    run_rag_baseline(
        top_k      = args.top_k,
        data_dir   = ROOT / args.data_dir,
        out_dir    = ROOT / args.out_dir,
        max_probes = args.max_probes,
        hf_token   = hf_token,
    )


if __name__ == "__main__":
    main()
