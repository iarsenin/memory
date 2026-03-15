"""
Phase 7 — Model Inference Loop (Pod GPU).

For each (condition, persona) pair:
  1. Load the correct LoRA adapter (LoRA conditions) or use the bare base
     model (frozen / rag).
  2. Format each probe as a minimal chat prompt:
       - All LoRA conditions + frozen: zero-context (system prompt + question).
       - rag only: prepend a bullet-list of salience-filtered memories.
  3. Run model.generate() at temperature=0, max_new_tokens=150.
  4. Return a list of raw response dicts (caller saves them).

Key inference differences from the training loop (src/trainer/loop.py)
───────────────────────────────────────────────────────────────────────
  • model.config.use_cache = True   (was False during training)
  • prepare_model_for_kbit_training() is NOT called (training-only helper)
  • model.eval() + torch.no_grad()
  • tokenizer.padding_side = "left" (standard for generation, not training)
"""

from __future__ import annotations

import gc
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
CONDITIONS_WITH_ADAPTERS = frozenset({"main", "naive_lora", "unfiltered_lora", "gold_lora"})

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

# Minimum salience score for a memory to be included in the RAG context.
# Mirrors the salience_threshold from configs/salience_config.json.
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


def load_adapter(base_model: Any, adapter_path: str) -> Any:
    """Wrap frozen base model with a saved LoRA adapter for inference."""
    print(f"  Loading adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"  Adapter loaded.  VRAM: {vram:.2f} GB")
    return peft_model


def unload_adapter(peft_model: Any) -> None:
    """Delete LoRA wrapper and reclaim VRAM.  Base model stays in memory."""
    del peft_model
    gc.collect()
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"  Adapter unloaded.  VRAM: {vram:.2f} GB")


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


def build_rag_context(
    persona_name: str,
    memories: list[dict],
) -> str:
    """
    Format salience-filtered memories as a bullet list for the RAG system
    prompt.  Uses the same memory content available to the parametric LoRA
    conditions (salience-filtered extracted memories) for a fair comparison.
    """
    filtered = [
        m for m in memories
        if (m.get("salience_score") or 0.0) >= _RAG_SALIENCE_THRESHOLD
    ]
    if not filtered:
        filtered = memories  # fallback: include all if nothing passes threshold

    lines = [
        f"• {m['predicate']} {m['value']}"
        for m in sorted(filtered, key=lambda x: x.get("day", 0))
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
) -> list[dict]:
    """
    Run every probe for one (condition, persona) pair and return raw responses.

    The caller is responsible for loading / unloading the LoRA adapter before
    and after calling this function.
    """
    persona_probes = [p for p in probes if p["persona_id"] == persona_id]
    if not persona_probes:
        return []

    persona_name = persona_probes[0]["persona_name"]
    rag_ctx = (
        build_rag_context(persona_name, rag_memories)
        if rag_memories else None
    )

    results: list[dict] = []
    for probe in persona_probes:
        messages = _format_prompt(probe, condition, rag_ctx)
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
