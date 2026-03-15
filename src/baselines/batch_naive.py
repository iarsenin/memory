"""
Phase 6 — Naive LoRA Batch Generator.

Baseline: train directly on raw dialogue transcripts, no extraction step,
no salience filtering, no replay buffer.

Generates sliding 2-turn windows (user + assistant) from the raw dialogue
for the current 3-day window. The model sees real conversational text but
must implicitly learn facts from it without any structured distillation.

This is the "does raw transcript training work at all?" control condition.
"""

from __future__ import annotations


class NaiveBatchGenerator:
    """
    Generates training examples by sliding a 2-turn window over raw dialogue.

    Usage:
        gen = NaiveBatchGenerator()
        examples, meta = gen.build_cycle_batch(dialogue_by_day, day, window_size=3)
    """

    def build_cycle_batch(
        self,
        dialogue_by_day: dict[int, list[dict]],
        day: int,
        window_size: int = 3,
    ) -> tuple[list[list[dict]], dict]:
        """
        Extract all dialogue turns from days [day - window_size + 1, day].
        Chunk into consecutive 2-turn (user, assistant) pairs.

        Returns:
            examples: list of message-lists in HF chat format.
            meta:     dict with counts for telemetry.
        """
        win_start = day - window_size + 1
        all_turns: list[dict] = []
        for d in range(win_start, day + 1):
            all_turns.extend(dialogue_by_day.get(d, []))

        examples: list[list[dict]] = []
        # Walk consecutive turn pairs — dialogue is already user/assistant alternating
        i = 0
        while i < len(all_turns) - 1:
            pair = all_turns[i : i + 2]
            # Keep only clean user→assistant pairs (skip misaligned pairs)
            if (pair[0]["speaker"] == "user"
                    and pair[1]["speaker"] == "assistant"):
                examples.append([
                    {"role": "user",      "content": pair[0]["utterance"]},
                    {"role": "assistant", "content": pair[1]["utterance"]},
                ])
                i += 2  # advance by full pair
            else:
                i += 1  # re-align if turns are out of phase

        meta = {
            "n_total":      len(examples),
            "n_turns_raw":  len(all_turns),
            "days":         f"{win_start}–{day}",
        }
        return examples, meta
