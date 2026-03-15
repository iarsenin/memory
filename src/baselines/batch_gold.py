"""
Phase 6 — Gold LoRA Batch Generator.

Upper-bound baseline: train directly on structured facts from the Simulator
Ground Truth JSON, bypassing the entire extraction + salience pipeline.

This establishes the parametric ceiling — what the model could learn if
extraction were 100% perfect and salience filtering were unnecessary.

Batch format mirrors the main system (declarative + QA probes + regularizer
exchanges) so that the only variable is input quality, not training format.
Omitting the regularizer caused the adapter to forget Llama-3-Instruct's
conversational alignment, collapsing stable/updated accuracy to 0% in Phase 7.

Replay: facts introduced in earlier windows are sampled uniformly (no
salience weighting since all GT facts are equally "gold").
"""

from __future__ import annotations

import json
import random

# Shared with src/trainer/batch.py — import to keep lists in sync.
from src.trainer.batch import _REGULARIZER_EXCHANGES

_SYS_REMEMBER = (
    "You are a helpful AI assistant that remembers personal information "
    "about users. Retain the following fact accurately."
)
_SYS_RECALL = (
    "You are a helpful AI assistant with access to personal information "
    "about users. Answer questions based on what you know about them."
)


class GoldBatchGenerator:
    """
    Generates training examples directly from ground-truth persona facts.

    Usage:
        gen = GoldBatchGenerator(config)
        examples, meta = gen.build_cycle_batch(ground_truth, day, window_size=3, seed=42)
    """

    def __init__(self, config: dict) -> None:
        self.qa_templates: list[str] = config["qa_train_templates"]
        self.qa_per_fact  = int(config["batch_mixture"].get("qa_phrasings_per_fact", 3))
        self.replay_ratio = float(config["replay_buffer"]["historical_sample_ratio"])
        self.reg_n        = int(config.get("regularizer", {}).get("generic_dialogue_samples", 10))
        self._rng = random.Random()

    def build_cycle_batch(
        self,
        ground_truth: dict,
        day: int,
        window_size: int = 3,
        seed: int = 42,
    ) -> tuple[list[list[dict]], dict]:
        """
        Build a training batch for one sleep cycle from GT facts.

        "New" facts: day_introduced falls in [day - window_size + 1, day].
        "Replay" facts: day_introduced < window_start (uniform sampling).

        Returns:
            examples: list of message-lists in HF chat format.
            meta:     dict with counts for telemetry.
        """
        self._rng.seed(seed)
        win_start = day - window_size + 1
        name      = ground_truth["name"]   # e.g. "Alice Chen"
        facts     = ground_truth["facts"]

        new_facts  = [f for f in facts if win_start <= f["day_introduced"] <= day]
        prev_facts = [f for f in facts if f["day_introduced"] < win_start]

        examples: list[list[dict]] = []
        n_new = n_replay = 0

        # --- New facts ---
        for fact in new_facts:
            examples.extend(self._expand_fact(fact, name))
            n_new += 1

        # --- Replay (uniform, no salience weighting for GT) ---
        if prev_facts:
            n_sample = max(1, round(len(new_facts) * self.replay_ratio))
            n_sample = min(n_sample, len(prev_facts))
            for fact in self._rng.sample(prev_facts, n_sample):
                examples.extend(self._expand_fact(fact, name))
                n_replay += 1

        # --- Regularizer exchanges (mirrors main BatchGenerator) ---
        # Prevents the adapter from losing Llama-3-Instruct's conversational
        # alignment when trained exclusively on structured fact statements.
        n_reg = 0
        if self.reg_n > 0:
            reg_pairs = self._rng.sample(
                _REGULARIZER_EXCHANGES,
                min(self.reg_n, len(_REGULARIZER_EXCHANGES)),
            )
            for user_msg, asst_msg in reg_pairs:
                examples.append([
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ])
                n_reg += 1

        self._rng.shuffle(examples)

        meta = {
            "n_total":         len(examples),
            "n_new_facts":     n_new,
            "n_replay_facts":  n_replay,
            "n_regularizer":   n_reg,
        }
        return examples, meta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_fact(self, fact: dict, name: str) -> list[list[dict]]:
        """Convert one GT fact into declarative + QA examples."""
        examples: list[list[dict]] = []
        pred     = fact["predicate"]
        val      = fact["value"] if isinstance(fact["value"], str) else json.dumps(fact["value"])
        category = fact.get("category") or pred
        stmt     = f"{name} {pred} {val}."

        # Declarative
        examples.append([
            {"role": "system",    "content": _SYS_REMEMBER},
            {"role": "user",      "content": f"Remember this: {stmt}"},
            {"role": "assistant", "content": f"Noted. {stmt}"},
        ])

        # QA probes — strictly from config templates
        qa_pairs = self._qa_pairs(name, pred, category, val)
        for q, a in self._rng.sample(qa_pairs, min(self.qa_per_fact, len(qa_pairs))):
            examples.append([
                {"role": "system",    "content": _SYS_RECALL},
                {"role": "user",      "content": q},
                {"role": "assistant", "content": a},
            ])

        return examples

    def _qa_pairs(
        self, name: str, pred: str, category: str, val: str
    ) -> list[tuple[str, str]]:
        answer = f"{name} {pred} {val}."
        pairs: list[tuple[str, str]] = []
        for tmpl in self.qa_templates:
            try:
                q = tmpl.format(subject=name, predicate=pred, category=category)
                pairs.append((q, answer))
            except KeyError:
                continue
        return pairs
