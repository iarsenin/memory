"""
Phase 1 — Dialogue Generator.

Uses the OpenAI API (small model, temperature=0.7) to paraphrase the
simulator's structured state transitions into natural conversational turns.

The LLM is used only for surface-form paraphrasing. It never invents facts.
All factual content is supplied via the structured PersonaGroundTruth.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from .personas import FactStatus, PersonaGroundTruth


def _make_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _build_system_prompt() -> str:
    return (
        "You generate realistic conversational data for an AI memory research experiment. "
        "Your output must be a valid JSON object with a single key 'turns' containing an array "
        "of conversation turns. Do not include any text outside the JSON object."
    )


def _build_user_prompt(
    persona: PersonaGroundTruth, day: int, n_turns: int
) -> str:
    active_facts = persona.get_active_facts_on_day(day)
    events = persona.get_events_on_day(day)

    # All facts that were once true but are no longer true as of this day
    superseded_facts = [
        f for f in persona.facts
        if f.status == FactStatus.SUPERSEDED
        and f.day_superseded is not None
        and f.day_superseded <= day
    ]

    facts_lines = "\n".join(
        f"  - {f.predicate} {f.value}" for f in active_facts
    )

    if superseded_facts:
        history_lines = "\n".join(
            f"  - (Days {f.day_introduced}–{f.day_superseded - 1}) {f.predicate} {f.value}"
            for f in superseded_facts
        )
        history_block = (
            f"\nHistorical context — NO LONGER TRUE as of Day {day} "
            f"(do NOT treat these as current facts):\n{history_lines}\n"
        )
    else:
        history_block = ""

    if events:
        events_block = (
            f"\nIMPORTANT — Life changes happening on Day {day}:\n"
            + "\n".join(f"  - {e.description}" for e in events)
            + f"\n{persona.name} MUST naturally bring up these changes.\n"
        )
    else:
        events_block = (
            f"\nNo major changes today. {persona.name} may mention any aspect "
            "of their current life naturally.\n"
        )

    return f"""Write a natural daily check-in conversation between {persona.name} \
(age {persona.age}) and an AI assistant. This is Day {day} of 20.

About {persona.name} (background as of Day 1 — some facts may have changed since):
{persona.background}

{persona.name}'s CURRENT facts on Day {day} (these are the only truths):
{facts_lines}
{history_block}{events_block}
Rules:
1. {persona.name} speaks as "user". The AI responds as "assistant". Always start with "user".
2. Generate exactly {n_turns} turns alternating user/assistant.
3. Conversation should feel natural — mix factual mentions with emotions and small talk.
4. If there are life changes today, {persona.name} must mention them; they are significant.
5. Each turn should be 1–3 sentences. Conversational tone, not formal.
6. Do NOT list facts mechanically. Weave them into natural speech.
7. CRITICAL: Only the CURRENT facts above are true today. Historical context is past only.

Output format — a JSON object only, no other text:
{{"turns": [{{"speaker": "user", "utterance": "..."}}, {{"speaker": "assistant", "utterance": "..."}}, ...]}}"""


class DialogueGenerator:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1200,
    ) -> None:
        self.client = _make_client()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_day(
        self,
        persona: PersonaGroundTruth,
        day: int,
        n_turns: int = 6,
    ) -> list[dict]:
        """
        Generate n_turns of conversation for a single persona-day.
        Returns a list of turn dicts ready to append to the JSONL log.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": _build_user_prompt(persona, day, n_turns)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Extract turns array — handles {"turns": [...]} or bare [...]
        if isinstance(data, list):
            turns = data
        elif isinstance(data, dict):
            turns = next(
                (v for v in data.values() if isinstance(v, list)), []
            )
        else:
            raise ValueError(f"Unexpected API response format: {type(data)}")

        result = []
        for i, turn in enumerate(turns[:n_turns]):
            result.append(
                {
                    "persona_id": persona.persona_id,
                    "day": day,
                    "turn_idx": i,
                    "speaker": turn["speaker"],
                    "utterance": turn["utterance"],
                }
            )
        return result
