"""
Phase 2 — Memory Extraction Core.

Sends a single day's dialogue transcript to the OpenAI API and distills it
into atomic structured memory items conforming to the MemLoRA schema.

Design decisions:
- temperature=0.0 + json_object mode for deterministic, structured output.
- Both user AND assistant turns are included in the transcript — assistant
  questions provide disambiguating context (e.g. "How long have you been
  vegetarian?" confirms a standing preference, not a one-off mention).
- Only the persona's utterances are factual sources; the system prompt
  instructs the LLM to ignore everything the assistant says about itself.
- Subjects are normalised to the persona's full name (no "I" / "he" / "she").
- supersedes_memory_id, consolidated, and salience_score are left null/false
  here — they are populated in Phase 3–4.
"""

from __future__ import annotations

import json
import os
import uuid

from openai import OpenAI

SYSTEM_PROMPT = """\
You are a strict personal-fact extractor for an AI memory system.

Given a daily conversation between a person and an AI assistant, extract
concrete, lasting facts about the person. Return ONLY a JSON object.

═══ EXTRACT ═══
• Biographical facts: job title, employer, city, diet, relationship status
• Pets — name, species, and critically their status (alive / deceased)
• Hobbies, sports, and long-term activities
• Significant life events: job change, move, breakup, bereavement, new relationship
• Established preferences (long-term, not mood of the day)

═══ IGNORE ═══
• Greetings, pleasantries, filler ("sounds great", "I see", "that's nice")
• Temporary emotional states ("I'm tired today", "feeling anxious this morning")
• Routine one-off activities that reveal no standing fact ("I went to the store")
• Anything the [Assistant] says about itself
• Vague statements with no concrete extractable value

═══ CONFIDENCE SCORING (0.0 – 1.0) ═══
• 1.0  Explicitly stated as direct fact  ("I moved to Austin yesterday")
• 0.7–0.9  Strongly implied, unambiguous in context
• 0.5–0.6  Inferred but not stated directly — include only if clearly meaningful
• < 0.5  Do not include

═══ is_update FLAG ═══
Set is_update=true when the fact explicitly REPLACES a prior state.
Signal phrases: "used to", "no longer", "just got", "broke up with",
"switched to", "moved from X to Y", "lost my job", "passed away",
"adopted", "started dating".

═══ OUTPUT FORMAT (no other text) ═══
{
  "extracted_facts": [
    {
      "subject": "<person's full name>",
      "predicate": "<short verb phrase, e.g. works as / lives in / is dating>",
      "value": "<the fact value>",
      "confidence": <float 0.0–1.0>,
      "is_update": <true|false>
    }
  ]
}
If no facts qualify, return: {"extracted_facts": []}
"""


def _build_transcript(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        speaker = "[User]     " if t["speaker"] == "user" else "[Assistant]"
        lines.append(f"{speaker} {t['utterance']}")
    return "\n".join(lines)


def _build_user_prompt(name: str, age: int, day: int, turns: list[dict]) -> str:
    transcript = _build_transcript(turns)
    return (
        f"Person: {name} (age {age})\n"
        f"Day: {day} of 20\n\n"
        f"Today's conversation:\n"
        f"---\n{transcript}\n---\n\n"
        f"Extract all concrete, lasting facts about {name} stated or strongly "
        f"implied in this conversation.\n"
        f'Use "{name}" as the subject for every extracted fact.'
    )


def _coerce_str(v: object) -> str:
    """Ensure a value is a plain string (LLMs occasionally return nested objects)."""
    if isinstance(v, str):
        return v
    return json.dumps(v)


def _to_schema(fact: dict, day: int) -> dict:
    """Map a raw LLM fact dict to the official MemLoRA memory schema."""
    return {
        "memory_id": str(uuid.uuid4()),
        "subject": _coerce_str(fact["subject"]),
        "predicate": _coerce_str(fact["predicate"]),
        "value": _coerce_str(fact["value"]),
        "day": day,
        "confidence": float(fact["confidence"]),
        "is_update": bool(fact.get("is_update", False)),
        "supersedes_memory_id": None,   # populated in Phase 4
        "consolidated": False,
        "salience_score": None,         # populated in Phase 4
    }


class MemoryExtractor:
    def __init__(self, config: dict) -> None:
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        openai_cfg = config.get("openai", {})
        self.model = openai_cfg.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(openai_cfg.get("temperature", 0.0))
        self.max_tokens = int(openai_cfg.get("max_tokens", 1500))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.6))

    def extract_day(
        self,
        name: str,
        age: int,
        day: int,
        turns: list[dict],
    ) -> list[dict]:
        """
        Extract memory items from one day's dialogue turns.
        Returns a list of schema-compliant dicts, filtered by confidence.
        """
        if not turns:
            return []

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(name, age, day, turns)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Accept {"extracted_facts": [...]} or a bare list as fallback
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = data.get("extracted_facts", [])
        else:
            return []

        results = []
        for fact in candidates:
            try:
                conf = float(fact.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if conf < self.confidence_threshold:
                continue
            results.append(_to_schema(fact, day))

        return results
