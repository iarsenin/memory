"""
Phase 7 — Model Inference Loop (Pod GPU).

For each (condition, persona) pair:
  1. Load the correct LoRA adapter (LoRA conditions) or use the bare base
     model (frozen / rag).
  2. Format each probe as a minimal chat prompt:
       - All LoRA conditions + frozen: zero-context (system prompt + question).
       - rag only: retrieve top-k memories via BM25 and prepend them.
  3. Run model.generate() at temperature=0, max_new_tokens=150.
  4. Return a list of raw response dicts (caller saves them).

RAG retrieval (post TMLR revision)
───────────────────────────────────
  The RAG condition now uses BM25 to retrieve the top-k most relevant memories
  for each probe question rather than dumping all salience-filtered memories.
  This gives a fair, query-aware comparison to the parametric LoRA methods.
  Falls back to salience-threshold filtering when rank_bm25 is not installed.

Key inference differences from the training loop (src/trainer/loop.py)
───────────────────────────────────────────────────────────────────────
  • model.config.use_cache = True   (was False during training)
  • prepare_model_for_kbit_training() is NOT called (training-only helper)
  • model.eval() + torch.no_grad()
  • tokenizer.padding_side = "left" (standard for generation, not training)
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Conditions that require loading a LoRA adapter
CONDITIONS_WITH_ADAPTERS = frozenset({
    "main", "naive_lora", "unfiltered_lora", "oracle_data_lora",
    "ablation_no_salience", "ablation_no_replay", "ablation_no_negative",
})

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYS_ZERO_CONTEXT = (
    "You are a helpful assistant who knows {name} personally. "
    "Answer the question about {name} concisely and directly based on what you know. "
    "Give your best factual answer in one or two sentences."
)

_SYS_RAG = (
    "You are a helpful assistant. The following are confirmed facts about {name}:\n"
    "{memory_block}\n\n"
    "Answer the question about {name} based only on these facts. "
    "Be concise and direct."
)

# Number of memories retrieved by BM25 for the RAG condition.
_RAG_TOP_K = 3

# Fallback salience threshold used when rank_bm25 is not available.
_RAG_SALIENCE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


def load_inference_model(
    config: dict,
    hf_token: str | None = None,
) -> tuple[Any, Any]:
    """
    Load Llama-3-8B-Instruct in 4-bit NF4 for inference only.

    NOTE: Does NOT call prepare_model_for_kbit_training() — that is a
    training-only step that enables gradient checkpointing hooks.
    """
    model_cfg = config["model"]
    model_id  = model_cfg["base_model_id"]

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=bool(
            model_cfg.get("bnb_4bit_use_double_quant", True)
        ),
    )

    print(f"  Loading base model for inference: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = True   # enables KV cache for efficient generation
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for generation (not training)

    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"  Inference model ready.  VRAM: {vram:.2f} GB")
    return model, tokenizer


def load_first_adapter(base_model: Any, adapter_path: str, name: str) -> Any:
    """
    Wrap the clean base model with the first LoRA adapter.

    Call this exactly once per process.  All subsequent adapter changes
    must go through switch_adapter() to avoid the 'multiple adapters'
    PEFT warning that arises from repeatedly calling PeftModel.from_pretrained
    on the same (already-wrapped) base model.
    """
    print(f"  Loading adapter [{name}]: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=name)
    peft_model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"  Adapter [{name}] ready.  VRAM: {vram:.2f} GB")
    return peft_model


def switch_adapter(peft_model: Any, new_path: str, new_name: str) -> Any:
    """
    Replace the currently active LoRA adapter with a new one.

    Loads new_name from new_path, sets it as the active adapter, then
    deletes all previous adapter slots to free VRAM.  Returns the same
    peft_model object with the new adapter active.
    """
    old_names = list(peft_model.peft_config.keys())
    print(f"  Switching adapter → [{new_name}]: {new_path}")
    peft_model.load_adapter(new_path, adapter_name=new_name)
    peft_model.set_adapter(new_name)
    for old in old_names:
        peft_model.delete_adapter(old)
    peft_model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"  Adapter [{new_name}] active.  VRAM: {vram:.2f} GB")
    return peft_model


def find_latest_checkpoint(
    checkpoints_dir: Path,
    condition: str,
    persona_id: str,
) -> str | None:
    """Return the path of the highest-numbered completed day checkpoint."""
    base = checkpoints_dir / condition / persona_id
    if not base.exists():
        return None
    days = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("day_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for d in reversed(days):
        if (d / "adapter_config.json").exists():
            return str(d)
    return None


# ---------------------------------------------------------------------------
# RAG context builder
# ---------------------------------------------------------------------------


def _tokenize_bm25(text: str) -> list[str]:
    """Simple tokeniser for BM25: lowercase alpha tokens, length > 1."""
    return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) > 1]


def _bm25_retrieve(
    query: str,
    memories: list[dict],
    top_k: int = _RAG_TOP_K,
) -> list[dict]:
    """
    Rank memories by BM25 relevance to *query* and return the top-k.

    Uses rank_bm25 when available; falls back to raw term-overlap scoring.
    """
    if not memories:
        return []

    corpus = [
        _tokenize_bm25(f"{m.get('predicate', '')} {m.get('value', '')} {m.get('category', '')}")
        for m in memories
    ]
    q_tokens = _tokenize_bm25(query)

    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        bm25   = BM25Okapi(corpus)
        scores = bm25.get_scores(q_tokens)
    except ImportError:
        # Fallback: plain term-overlap count (TF without IDF).
        scores = [
            sum(1 for t in q_tokens if t in doc_tokens)
            for doc_tokens in corpus
        ]

    ranked = sorted(range(len(memories)), key=lambda i: scores[i], reverse=True)
    return [memories[i] for i in ranked[:top_k]]


def build_rag_context(
    persona_name: str,
    memories: list[dict],
    query: str | None = None,
) -> str:
    """
    Format retrieved memories as a bullet list for the RAG system prompt.

    When *query* is provided (normal evaluation path) the top-k most relevant
    memories are selected via BM25, giving a fair, query-aware comparison to
    the parametric LoRA methods.

    Falls back to salience-threshold filtering when *query* is None (used only
    for legacy / debugging paths).
    """
    if query is not None:
        selected = _bm25_retrieve(query, memories, top_k=_RAG_TOP_K)
        if not selected:
            selected = memories[:_RAG_TOP_K]
    else:
        selected = [m for m in memories if (m.get("salience_score") or 0.0) >= _RAG_SALIENCE_THRESHOLD]
        if not selected:
            selected = memories

    lines = [
        f"• {m.get('predicate', '')} {m.get('value', '')}"
        for m in sorted(selected, key=lambda x: x.get("day", 0))
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt formatting & generation
# ---------------------------------------------------------------------------


def _format_prompt(
    probe: dict,
    condition: str,
    rag_context: str | None,
) -> list[dict]:
    name     = probe["persona_name"]
    question = probe["question"]

    if condition == "rag" and rag_context:
        system = _SYS_RAG.format(name=name, memory_block=rag_context)
    else:
        system = _SYS_ZERO_CONTEXT.format(name=name)

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": question},
    ]


@torch.no_grad()
def _generate(model: Any, tokenizer: Any, messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        temperature=1.0,         # irrelevant when do_sample=False
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main inference runner (per condition × persona)
# ---------------------------------------------------------------------------


def run_condition_inference(
    probes: list[dict],
    persona_id: str,
    condition: str,
    model: Any,
    tokenizer: Any,
    rag_memories: list[dict] | None = None,
    use_adapter: bool = True,
) -> list[dict]:
    """
    Run every probe for one (condition, persona) pair and return raw responses.

    use_adapter=False is used for frozen/rag conditions when `model` is a
    PeftModel (because LoRA conditions ran earlier in the same process).
    In that case generation is wrapped in model.disable_adapter() so the
    base weights are used without any LoRA delta.
    """
    persona_probes = [p for p in probes if p["persona_id"] == persona_id]
    if not persona_probes:
        return []

    persona_name = persona_probes[0]["persona_name"]

    results: list[dict] = []
    for probe in persona_probes:
        # Build a per-probe RAG context using BM25 retrieval on the probe question.
        # This gives query-aware top-k retrieval rather than a static salience dump.
        rag_ctx = (
            build_rag_context(persona_name, rag_memories, query=probe["question"])
            if rag_memories else None
        )
        messages = _format_prompt(probe, condition, rag_ctx)

        if not use_adapter and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                response = _generate(model, tokenizer, messages)
        else:
            response = _generate(model, tokenizer, messages)

        results.append({
            "probe_id":   probe["probe_id"],
            "persona_id": persona_id,
            "bucket":     probe["bucket"],
            "condition":  condition,
            "question":   probe["question"],
            "expected":   probe["expected"],
            "response":   response,
        })
        short = response.replace("\n", " ")[:70]
        print(f"    [{probe['probe_id']}] {short}…")

    return results
