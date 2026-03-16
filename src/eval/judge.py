"""
Phase 7 — LLM Judge (OpenAI API).

Scores each model response as one of:
  correct       — factually matches the expected answer
  incorrect     — wrong value, stale fact confirmed, or misses the question
  contradiction — self-contradictory (e.g. affirms both old and new value)
  partial       — partially correct (relational bucket only)

Numeric mapping for accuracy aggregation:
  correct → 1.0 | partial → 0.5 | incorrect / contradiction → 0.0

Contradictions are preserved in the output for separate analysis (they are
a qualitative signal about the model's internal confusion, separate from
pure factual incorrectness).

Design:
  temperature=0.0 + response_format=json_object for deterministic scoring.
  The judge receives the bucket type and bucket-specific instructions so it
  applies the correct scoring criterion for superseded vs updated vs relational.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert evaluator for an AI memory recall experiment.
You score whether an AI assistant's answer to a factual question is correct.
You MUST respond with a JSON object only — no other text.
"""

_BUCKET_INSTRUCTIONS: dict[str, str] = {
    "stable": (
        "This question tests a STABLE fact that never changed throughout the "
        "simulation. The assistant should state the correct, consistent value."
    ),
    "updated": (
        "This question tests an UPDATED fact — the persona's situation changed "
        "at some point during the simulation. The assistant should state the NEW "
        "current value, NOT the old (stale) one. Citing only the old value counts "
        "as incorrect."
    ),
    "superseded": (
        "The question ASSERTS AN OUTDATED fact (see 'Stale claim' below). "
        "A CORRECT answer requires ACTIVE CORRECTION: the assistant must not only "
        "reject the old claim but must supply the new reality (e.g. 'No, Rex passed "
        "away — Bob now has a cat named Luna'). "
        "STRICT RULE — the following response types are ALWAYS 'incorrect', even if "
        "they avoid confirming the stale fact: "
        "(a) Privacy/safety refusals ('I don't have personal information about Bob'). "
        "(b) Generic ignorance ('I don't know', 'I'm not sure', 'I cannot say'). "
        "(c) Vague deflections ('Things may have changed', 'I'd need more context'). "
        "A correct answer must demonstrate the model KNOWS the updated state. "
        "A CONTRADICTION means the assistant simultaneously affirms both the old "
        "and the new value (e.g. 'Yes Rex is fine, but he also passed away')."
    ),
    "relational": (
        "This question requires combining two or more facts correctly. "
        "Score 'correct' only if all key components are right. "
        "Score 'partial' if some but not all components are addressed correctly. "
        "Score 'incorrect' if the answer is entirely wrong or misses the point. "
        "Score 'contradiction' if the answer affirms mutually exclusive facts."
    ),
}

_USER_TEMPLATE = """\
Bucket: {bucket}
Persona name: {persona_name}
Question: {question}
Ground-truth expected answer: {expected}
{stale_line}\
Model's answer: {response}

Bucket instructions:
{bucket_instructions}

Valid scores: "correct", "incorrect", "contradiction", "partial"
(Use "partial" only for the relational bucket.)

Respond ONLY with valid JSON:
{{"score": "<score>", "reason": "<one concise sentence>"}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_responses(
    responses: list[dict],
    probes_by_id: dict[str, dict],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> list[dict]:
    """
    Score a list of model responses using the OpenAI LLM judge.

    Each input response dict must have: probe_id, condition, bucket, question,
    expected, response.  Returns the same dicts extended with:
      score_label   — "correct" | "incorrect" | "contradiction" | "partial"
      score_numeric — 1.0 | 0.5 | 0.0
      judge_reason  — one-sentence explanation
    """
    scored: list[dict] = []
    for r in responses:
        probe  = probes_by_id[r["probe_id"]]
        bucket = probe["bucket"]

        stale_claim = probe.get("assertion_claim")
        stale_line  = (
            f"Stale claim being asserted: {stale_claim}\n"
            if stale_claim else ""
        )

        user_msg = _USER_TEMPLATE.format(
            bucket=bucket,
            persona_name=probe["persona_name"],
            question=probe["question"],
            expected=probe["expected"],
            stale_line=stale_line,
            response=r["response"],
            bucket_instructions=_BUCKET_INSTRUCTIONS[bucket],
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = json.loads(resp.choices[0].message.content)
            label  = raw.get("score", "incorrect").lower().strip()
            reason = raw.get("reason", "")
        except Exception as exc:
            label  = "error"
            reason = str(exc)

        numeric = {"correct": 1.0, "partial": 0.5}.get(label, 0.0)
        scored.append({
            **r,
            "score_label":   label,
            "score_numeric": numeric,
            "judge_reason":  reason,
        })

    return scored
