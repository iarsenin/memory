"""
Phase 5 — LoRA Training Loop (GPU / Pod only).

Responsibilities:
  1. load_base_model()     — Load Llama-3-8B in 4-bit NF4 (once per process).
  2. build_peft_model()    — Wrap base with LoRA. For cycle > 1, loads previous
                             adapter weights for accumulation.
  3. run_cycle()           — Tokenise examples, train, save adapter + telemetry.
  4. reset_peft()          — Unload the LoRA adapter after a persona is done so
                             the next persona gets a clean fresh adapter.

Adapter accumulation:
  Each cycle starts from the PREVIOUS cycle's saved adapter, not the base model.
  This is the core of continual learning — knowledge accumulates across cycles.
  The replay buffer (in batch.py) prevents catastrophic forgetting of earlier
  cycles' facts as new ones are added.

Persona state reset (CRITICAL):
  After all cycles for persona A are complete, call reset_peft(peft_model).
  This deletes the LoRA module and clears CUDA memory so that persona B
  starts with a completely fresh adapter on the same frozen base model.

QLoRA training notes:
  - Base weights: 4-bit NF4, frozen.
  - LoRA weights (r=16, q_proj + v_proj): float16, trainable.
  - prepare_model_for_kbit_training() must be called BEFORE get_peft_model()
    to enable gradient flow through quantised layers into the LoRA adapters.
  - gradient_checkpointing reduces VRAM at the cost of slightly slower speed.

Training format:
  All examples use the Llama-3-Instruct chat template via tokenizer.apply_chat_template().
  Loss is applied to all tokens (system + user + assistant) — "full sequence" mode.
  DataCollatorForLanguageModeling(mlm=False) handles padding and label masking
  for pad tokens automatically.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Loss logger callback
# ---------------------------------------------------------------------------


class _LossLogger(TrainerCallback):
    """Captures per-step training loss for telemetry."""

    def __init__(self) -> None:
        self.step_losses: list[float] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if logs and "loss" in logs:
            self.step_losses.append(float(logs["loss"]))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_base_model(
    config: dict,
    hf_token: str | None = None,
) -> tuple[Any, Any]:
    """
    Load Llama-3-8B-Instruct in 4-bit NF4 quantisation.
    Returns (model, tokenizer). Call this ONCE per process.

    The model is prepared for k-bit training immediately so that
    gradient checkpointing and cast norms are set before LoRA is applied.
    """
    model_cfg = config["model"]
    model_id  = model_cfg["base_model_id"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=bool(
            model_cfg.get("bnb_4bit_use_double_quant", True)
        ),
    )

    print(f"  Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # Must be called before get_peft_model so gradient hooks are set correctly
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for causal LM training

    print(f"  Base model loaded. VRAM: {_vram_gb():.2f} GB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# PEFT model management
# ---------------------------------------------------------------------------


def build_peft_model(
    base_model: Any,
    lora_cfg: dict,
    prev_checkpoint: str | None,
) -> Any:
    """
    Build a trainable PEFT model for one training cycle.

    If prev_checkpoint is None (first cycle for this persona): initialise a
    fresh LoRA adapter on the frozen base model.

    If prev_checkpoint is set (cycle 2+): load the saved adapter from the
    previous cycle so knowledge accumulates across cycles (adapter accumulation).
    """
    if prev_checkpoint is not None:
        print(f"  Loading adapter from {prev_checkpoint} (accumulation)")
        peft_model = PeftModel.from_pretrained(
            base_model, prev_checkpoint, is_trainable=True
        )
    else:
        print("  Initialising fresh LoRA adapter (cycle 1)")
        config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["lora_alpha"]),
            target_modules=lora_cfg["target_modules"],
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=lora_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base_model, config)

    peft_model.print_trainable_parameters()
    return peft_model


def reset_peft(peft_model: Any) -> None:
    """
    Unload the LoRA adapter and release CUDA memory.
    Call this after all cycles for one persona complete, before the next
    persona's training begins. The frozen base model remains in VRAM.
    """
    del peft_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Adapter unloaded. VRAM after reset: {_vram_gb():.2f} GB")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_MAX_SEQ_LEN = 512


def tokenize_examples(
    tokenizer: Any,
    examples: list[list[dict]],
) -> Dataset:
    """
    Convert a list of chat-format message lists into a HuggingFace Dataset
    ready for causal LM training.

    Each message list is formatted via the tokenizer's chat template, then
    tokenised. The DataCollatorForLanguageModeling will handle padding and
    label creation (labels = input_ids shifted by 1, padding → -100).
    """
    all_input_ids: list[list[int]] = []

    for messages in examples:
        # apply_chat_template respects Llama-3's special tokens
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text,
            max_length=_MAX_SEQ_LEN,
            truncation=True,
            padding=False,
        )
        all_input_ids.append(enc["input_ids"])

    return Dataset.from_dict({"input_ids": all_input_ids})


# ---------------------------------------------------------------------------
# Checkpoint save helpers
# ---------------------------------------------------------------------------


def _ensure_adapter_config(checkpoint_path: str, peft_model: Any) -> None:
    """Re-write adapter_config.json if PEFT left it as 0 bytes (network-FS race)."""
    import json, os
    cfg_path = Path(checkpoint_path) / "adapter_config.json"
    if cfg_path.exists() and cfg_path.stat().st_size > 0:
        return
    adapter_name = next(iter(peft_model.peft_config))
    cfg_dict = peft_model.peft_config[adapter_name].to_dict()
    tmp = cfg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg_dict, indent=2))
    os.replace(tmp, cfg_path)
    print(f"  [repair] adapter_config.json was empty — rewrote from live config at {checkpoint_path}")


def _save_adapter_via_tmp(checkpoint_path: str, peft_model: Any, tokenizer: Any) -> None:
    """Save adapter to /tmp, then copy to network FS to avoid MooseFS crash on direct write.

    Direct save_pretrained to MooseFS (network volume) crashes the process mid-save
    (SIGSEGV/SIGBUS in the safetensors Rust serializer under NFS write pressure).
    Saving to local /tmp first is reliable; rsync/shutil.copytree then transfers
    the completed checkpoint atomically.
    """
    import shutil, tempfile, os
    dst = Path(checkpoint_path)
    dst.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="memlora_ckpt_", dir="/tmp") as tmp_dir:
        print(f"  Saving adapter to {tmp_dir} (local /tmp) …", flush=True)
        peft_model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        _ensure_adapter_config(tmp_dir, peft_model)
        # Copy from /tmp to the network volume
        print(f"  Copying checkpoint to {checkpoint_path} …", flush=True)
        for src_file in Path(tmp_dir).iterdir():
            shutil.copy2(src_file, dst / src_file.name)
    print(f"  Checkpoint saved to {checkpoint_path}", flush=True)


# ---------------------------------------------------------------------------
# Training cycle
# ---------------------------------------------------------------------------


def run_cycle(
    peft_model: Any,
    tokenizer: Any,
    training_cfg: dict,
    examples: list[list[dict]],
    checkpoint_path: str,
    logging_steps: int = 1,
) -> dict[str, Any]:
    """
    Tokenise examples, run the HuggingFace Trainer for this sleep cycle,
    save the LoRA adapter, and return telemetry.

    Handles tiny batch sizes: if the dataset is smaller than
    per_device_train_batch_size, batch_size is reduced to 1 automatically.

    Returns a telemetry dict (written to logs/ by the caller).
    """
    dataset = tokenize_examples(tokenizer, examples)
    n = len(dataset)
    print(f"  Training on {n} examples")

    # --- Batch-size safety for tiny datasets ---
    batch_size = int(training_cfg.get("per_device_train_batch_size", 4))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 4))
    if n < batch_size:
        grad_accum = max(1, grad_accum * batch_size)  # preserve effective batch
        batch_size = 1
        print(f"  Tiny dataset: batch_size → 1, grad_accum → {grad_accum}")

    # --- Training arguments ---
    args = TrainingArguments(
        output_dir=checkpoint_path,
        num_train_epochs=int(training_cfg.get("num_epochs", 5)),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        fp16=bool(training_cfg.get("fp16", True)),
        warmup_steps=min(
            int(training_cfg.get("warmup_steps", 10)),
            max(1, n // batch_size),          # can't exceed total steps
        ),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        logging_steps=logging_steps,
        save_strategy="no",     # we save the adapter manually below
        report_to="none",       # no wandb / tensorboard on the pod
        dataloader_drop_last=False,
    )

    # --- Collator: causal LM, pads to longest in batch, masks pad labels ---
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # --- Callback for loss capture ---
    loss_logger = _LossLogger()

    # --- Run training ---
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[loss_logger],
    )
    trainer.train()

    runtime = time.time() - t_start
    vram_peak = torch.cuda.max_memory_allocated() / 1e9

    # --- Save LoRA adapter ---
    # Save to /tmp first, then rsync to the network volume (MooseFS).
    # Direct writes to MooseFS can crash the process mid-save (SIGSEGV/SIGBUS
    # in the safetensors Rust code under NFS write pressure). Saving locally
    # then copying is more reliable.
    _save_adapter_via_tmp(checkpoint_path, peft_model, tokenizer)

    print(
        f"  Cycle done: {runtime:.0f}s  |  VRAM peak {vram_peak:.2f} GB  |  "
        f"steps {len(loss_logger.step_losses)}"
    )

    return {
        "step_losses":      loss_logger.step_losses,
        "avg_loss":         _safe_mean(loss_logger.step_losses),
        "runtime_seconds":  round(runtime, 1),
        "vram_peak_gb":     round(vram_peak, 2),
        "n_examples":       n,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def _safe_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
