"""
Phase 7 — Evaluation Judge.

Scoring strategy (post TMLR reviewer revision):
  • stable / updated / superseded  →  deterministic keyword-matching (no API call).
  • relational                      →  OpenAI LLM judge (multi-hop reasoning needed).

Numeric mapping: correct → 1.0 | partial → 0.5 | incorrect / contradiction → 0.0

Deterministic rules
-------------------
stable / updated:
  Extract non-stopword tokens from `expected`. Score as *correct* if ≥ 50 % of
  those tokens appear in the (case-insensitive) response.  Explicit refusals or
  ignorance phrases always score as *incorrect*.

superseded:
  `expected` = "No — {new_value}".  `assertion_claim` = old stale value.
  *correct*      : new-value keywords present in response AND stale claim not
                   positively affirmed (or negated by context).
  *incorrect*    : new-value keywords absent, OR explicit refusal/ignorance,
                   OR stale claim asserted without negation.
  *contradiction*: new-value keywords present AND stale claim positively
                   affirmed without negation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Stopword set for keyword extraction
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "to", "of", "in", "at", "on", "for", "with",
    "by", "from", "and", "or", "but", "not", "no", "nor", "as", "if",
    "then", "so", "yet", "both", "either", "that", "this", "it", "its",
    "their", "they", "she", "he", "her", "his", "i", "my", "we", "our",
    "you", "your", "now", "currently", "still", "also", "just", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "can", "s", "t", "re", "ve", "ll", "d", "m",
    "don", "doesn", "didn", "hasn", "haven", "hadn", "isn", "aren", "wasn",
    "weren", "won", "wouldn", "couldn", "shouldn",
})

# Refusal / ignorance patterns that should always score incorrect on
# superseded probes where active correction is required.
_REFUSAL_PATTERNS: tuple[str, ...] = (
    "i don't have", "i do not have", "i cannot", "i can't",
    "i'm not able", "i am not able", "i'm unable", "i am unable",
    "no information", "no details", "not sure", "i don't know",
    "i do not know", "can't say", "cannot say", "unable to",
    "personal information", "i have no knowledge", "not aware",
    "privacy", "i'm afraid", "i am afraid",
)

# Negation words that, when close to a stale-claim token, indicate the stale
# claim is being rejected rather than affirmed.
_NEGATION_WORDS: tuple[str, ...] = (
    "no", "not", "never", "no longer", "isn't", "is not", "wasn't",
    "was not", "aren't", "aren't", "weren't", "didn't", "don't", "doesn't",
    "moved", "left", "quit", "resigned", "died", "passed away",
    "passed", "ended", "broke up", "separated", "lost", "laid off",
    "fired", "formerly", "used to", "previous", "previously", "ex-",
    "old", "until", "until recently", "ago", "before",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens, no stopwords, length > 1."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _keywords(text: str) -> set[str]:
    return set(_tokenize(text))


def _hit_rate(expected_text: str, response_lower: str) -> float:
    """Fraction of expected keywords found in response_lower."""
    kws = _keywords(expected_text)
    if not kws:
        return 0.0
    hits = sum(1 for kw in kws if kw in response_lower)
    return hits / len(kws)


def _has_refusal(response_lower: str) -> bool:
    return any(p in response_lower for p in _REFUSAL_PATTERNS)


def _stale_affirmed(assertion_claim: str, response: str) -> bool:
    """
    Return True when the stale claim appears to be *positively* asserted.

    Strategy: find each stale keyword in the response. Then look in a ±40-
    character window around each occurrence for any negation word.  If no
    negation is found near a stale-keyword occurrence, we consider it affirmed.
    """
    if not assertion_claim:
        return False
    stale_kws = _keywords(assertion_claim)
    if not stale_kws:
        return False
    resp_lower = response.lower()
    for kw in stale_kws:
        start = 0
        while True:
            idx = resp_lower.find(kw, start)
            if idx == -1:
                break
            window = resp_lower[max(0, idx - 40): idx + len(kw) + 40]
            if not any(neg in window for neg in _NEGATION_WORDS):
                return True   # found the stale keyword without nearby negation
            start = idx + 1
    return False


# ---------------------------------------------------------------------------
# Deterministic scorer (stable / updated / superseded)
# ---------------------------------------------------------------------------


def _det_score_stable_updated(probe: dict, response: str) -> dict:
    """Score a stable or updated probe deterministically."""
    resp_lower = response.lower()
    expected   = probe["expected"]

    if _has_refusal(resp_lower):
        return {"score_label": "incorrect", "score_numeric": 0.0,
                "judge_reason": "[det] Explicit refusal or ignorance."}

    hr = _hit_rate(expected, resp_lower)
    if hr >= 0.5:
        return {"score_label": "correct", "score_numeric": 1.0,
                "judge_reason": f"[det] {hr:.0%} expected keywords present."}
    return {"score_label": "incorrect", "score_numeric": 0.0,
            "judge_reason": f"[det] Only {hr:.0%} expected keywords present (threshold 50%)."}


def _det_score_superseded(probe: dict, response: str) -> dict:
    """Score a superseded probe deterministically.

    Expected format: "No — {new value description}"
    assertion_claim: the stale value being asserted in the question.
    """
    resp_lower = response.lower()
    expected   = probe["expected"]
    assertion  = probe.get("assertion_claim") or ""

    if _has_refusal(resp_lower):
        return {"score_label": "incorrect", "score_numeric": 0.0,
                "judge_reason": "[det] Refusal/ignorance — active correction required."}

    # Extract new-value text: everything after leading "No" marker.
    new_value_text = re.sub(r"^no\s*[\-—,.:]+\s*", "", expected, flags=re.IGNORECASE).strip()
    new_hr = _hit_rate(new_value_text or expected, resp_lower)

    stale_present = _stale_affirmed(assertion, response)

    if stale_present and new_hr >= 0.3:
        return {"score_label": "contradiction", "score_numeric": 0.0,
                "judge_reason": "[det] New-value keywords present but stale claim also affirmed."}
    if stale_present:
        return {"score_label": "incorrect", "score_numeric": 0.0,
                "judge_reason": "[det] Stale claim affirmed without active correction."}
    if new_hr >= 0.4:
        return {"score_label": "correct", "score_numeric": 1.0,
                "judge_reason": f"[det] New-value keywords present ({new_hr:.0%}), stale not affirmed."}
    return {"score_label": "incorrect", "score_numeric": 0.0,
            "judge_reason": f"[det] New-value keyword match too low ({new_hr:.0%} < 40%)."}


# ---------------------------------------------------------------------------
# LLM judge prompts (retained for relational bucket)
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert evaluator for an AI memory recall experiment.
You score whether an AI assistant's answer to a factual question is correct.
You MUST respond with a JSON object only — no other text.
"""

_BUCKET_INSTRUCTIONS: dict[str, str] = {
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
    Score responses.

    - stable / updated / superseded → deterministic (no API calls).
    - relational                     → OpenAI LLM judge.

    Each input response dict must have: probe_id, condition, bucket, question,
    expected, response.  Returns the same dicts extended with:
      score_label    — "correct" | "incorrect" | "contradiction" | "partial"
      score_numeric  — 1.0 | 0.5 | 0.0
      judge_reason   — explanation
      judge_method   — "deterministic" | "llm"
    """
    scored: list[dict] = []
    llm_count = det_count = 0

    for r in responses:
        probe  = probes_by_id.get(r["probe_id"], {})
        bucket = r.get("bucket") or probe.get("bucket", "stable")

        if bucket in ("stable", "updated"):
            result = _det_score_stable_updated(probe, r["response"])
            method = "deterministic"
            det_count += 1

        elif bucket == "superseded":
            result = _det_score_superseded(probe, r["response"])
            method = "deterministic"
            det_count += 1

        else:  # relational — use LLM
            stale_line = ""
            user_msg = _USER_TEMPLATE.format(
                bucket=bucket,
                persona_name=probe.get("persona_name", ""),
                question=r.get("question", ""),
                expected=r.get("expected", ""),
                stale_line=stale_line,
                response=r["response"],
                bucket_instructions=_BUCKET_INSTRUCTIONS.get(bucket, ""),
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
                raw    = json.loads(resp.choices[0].message.content)
                label  = raw.get("score", "incorrect").lower().strip()
                reason = raw.get("reason", "")
            except Exception as exc:
                label  = "error"
                reason = str(exc)
            result = {"score_label": label, "judge_reason": reason}
            method = "llm"
            llm_count += 1

        numeric = {"correct": 1.0, "partial": 0.5}.get(result["score_label"], 0.0)
        scored.append({
            **r,
            "score_label":   result["score_label"],
            "score_numeric": numeric,
            "judge_reason":  result["judge_reason"],
            "judge_method":  method,
        })

    print(f"  Judge: {det_count} deterministic, {llm_count} LLM calls.")
    return scored
