"""
Phase 1 — Ground Truth Persona Definitions.

Personas are deterministic Python state machines backed by Pydantic models.
The structured JSON ground truth is the absolute source of truth for all
extraction evaluation and zero-context eval probing.

Two personas: Alice Chen and Bob Martinez.
Both share fact categories (job, location, diet, relationship, pet, sport, hobby)
so evaluation bucket comparisons are clean across personas.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class FactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class PersonaFact(BaseModel):
    fact_id: str
    category: str          # job | location | diet | relationship | pet | sport | hobby
    predicate: str         # human-readable, e.g. "works as", "lives in"
    value: str
    day_introduced: int    # first day this fact is true (inclusive)
    day_superseded: Optional[int] = None   # first day this fact is NO LONGER true
    superseded_by_id: Optional[str] = None
    status: FactStatus = FactStatus.ACTIVE
    is_stable: bool = False   # True if never expected to change over 20 days


class DayEvent(BaseModel):
    day: int
    description: str           # plain-language summary for the dialogue prompt
    affected_fact_ids: list[str]


class PersonaGroundTruth(BaseModel):
    persona_id: str
    name: str
    age: int
    background: str
    facts: list[PersonaFact]
    events: list[DayEvent]

    def get_active_facts_on_day(self, day: int) -> list[PersonaFact]:
        """Facts active on a given day: introduced <= day < superseded."""
        return [
            f for f in self.facts
            if f.day_introduced <= day
            and (f.day_superseded is None or f.day_superseded > day)
        ]

    def get_events_on_day(self, day: int) -> list[DayEvent]:
        return [e for e in self.events if e.day == day]

    def finalize_statuses(self) -> "PersonaGroundTruth":
        """
        Set status=SUPERSEDED and populate superseded_by_id for all facts
        that have a day_superseded. Must be called once after construction.
        """
        for fact in self.facts:
            if fact.day_superseded is not None:
                fact.status = FactStatus.SUPERSEDED
                for candidate in self.facts:
                    if (
                        candidate.category == fact.category
                        and candidate.day_introduced == fact.day_superseded
                        and candidate.fact_id != fact.fact_id
                    ):
                        fact.superseded_by_id = candidate.fact_id
                        break
        return self


# ---------------------------------------------------------------------------
# Persona 1: Alice Chen
# ---------------------------------------------------------------------------

def build_alice() -> PersonaGroundTruth:
    """
    Alice Chen, 32, software engineer in Seattle.

    Key timeline (20 days):
      Day  5 — breaks up with boyfriend Mark
      Day  7 — laid off from TechCorp
      Day 10 — joins pottery class
      Day 12 — goes freelance as independent consultant
      Day 15 — moves from Seattle to Austin, TX
      Day 18 — starts dating Jamie

    Stable: vegetarian diet, marathon running
    """
    facts = [
        # --- location ---
        PersonaFact(
            fact_id="alice_f001", category="location",
            predicate="lives in", value="Seattle, WA",
            day_introduced=1, day_superseded=15,
        ),
        PersonaFact(
            fact_id="alice_f002", category="location",
            predicate="lives in", value="Austin, TX",
            day_introduced=15,
        ),
        # --- job ---
        PersonaFact(
            fact_id="alice_f003", category="job",
            predicate="works as", value="software engineer at TechCorp",
            day_introduced=1, day_superseded=7,
        ),
        PersonaFact(
            fact_id="alice_f004", category="job",
            predicate="is currently", value="unemployed after being laid off from TechCorp",
            day_introduced=7, day_superseded=12,
        ),
        PersonaFact(
            fact_id="alice_f005", category="job",
            predicate="works as", value="independent freelance software consultant",
            day_introduced=12,
        ),
        # --- relationship ---
        PersonaFact(
            fact_id="alice_f006", category="relationship",
            predicate="is in a relationship with", value="Mark",
            day_introduced=1, day_superseded=5,
        ),
        PersonaFact(
            fact_id="alice_f007", category="relationship",
            predicate="is", value="single (recently broke up with Mark)",
            day_introduced=5, day_superseded=18,
        ),
        PersonaFact(
            fact_id="alice_f008", category="relationship",
            predicate="is dating", value="Jamie",
            day_introduced=18,
        ),
        # --- stable facts ---
        PersonaFact(
            fact_id="alice_f009", category="diet",
            predicate="follows", value="a vegetarian diet",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="alice_f010", category="hobby",
            predicate="trains for", value="marathons",
            day_introduced=1, is_stable=True,
        ),
        # --- new hobby ---
        PersonaFact(
            fact_id="alice_f011", category="hobby",
            predicate="has recently taken up", value="pottery",
            day_introduced=10,
        ),
    ]

    events = [
        DayEvent(
            day=5,
            description="Alice and Mark broke up today after growing apart. She's upset but relieved.",
            affected_fact_ids=["alice_f006", "alice_f007"],
        ),
        DayEvent(
            day=7,
            description="TechCorp announced layoffs today. Alice lost her job as a software engineer.",
            affected_fact_ids=["alice_f003", "alice_f004"],
        ),
        DayEvent(
            day=10,
            description="Alice signed up for a pottery class at a local studio. First session today.",
            affected_fact_ids=["alice_f011"],
        ),
        DayEvent(
            day=12,
            description="Alice decided to go freelance. She is now an independent software consultant.",
            affected_fact_ids=["alice_f004", "alice_f005"],
        ),
        DayEvent(
            day=15,
            description="Alice packed up and moved from Seattle to Austin, TX for a fresh start.",
            affected_fact_ids=["alice_f001", "alice_f002"],
        ),
        DayEvent(
            day=18,
            description="Alice went on a first date with someone named Jamie and it went really well.",
            affected_fact_ids=["alice_f007", "alice_f008"],
        ),
    ]

    persona = PersonaGroundTruth(
        persona_id="alice",
        name="Alice Chen",
        age=32,
        background=(
            "A 32-year-old originally from Portland, OR, currently living in Seattle. "
            "Runs marathons as her primary hobby and has been vegetarian for five years. "
            "Sociable and introspective, she often reflects on life balance between career "
            "and personal wellbeing."
        ),
        facts=facts,
        events=events,
    )
    return persona.finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 2: Bob Martinez
# ---------------------------------------------------------------------------

def build_bob() -> PersonaGroundTruth:
    """
    Bob Martinez, 45, high school history teacher in Chicago.

    Key timeline (20 days):
      Day  4 — injures knee, stops cycling
      Day  6 — starts intermittent fasting
      Day  8 — takes academic sabbatical from Lincoln High
      Day 11 — beloved dog Rex passes away
      Day 14 — knee recovered, resumes cycling
      Day 16 — sabbatical ends; joins Westside Academy
      Day 17 — adopts cat Luna from shelter

    Stable: location (Chicago), passion for history
    """
    facts = [
        # --- location (stable) ---
        PersonaFact(
            fact_id="bob_f001", category="location",
            predicate="lives in", value="Chicago, IL",
            day_introduced=1, is_stable=True,
        ),
        # --- job ---
        PersonaFact(
            fact_id="bob_f002", category="job",
            predicate="works as", value="history teacher at Lincoln High School",
            day_introduced=1, day_superseded=8,
        ),
        PersonaFact(
            fact_id="bob_f003", category="job",
            predicate="is on", value="academic sabbatical (from Lincoln High School)",
            day_introduced=8, day_superseded=16,
        ),
        PersonaFact(
            fact_id="bob_f004", category="job",
            predicate="works as", value="history teacher at Westside Academy",
            day_introduced=16,
        ),
        # --- sport ---
        PersonaFact(
            fact_id="bob_f005", category="sport",
            predicate="trains for", value="long-distance cycling",
            day_introduced=1, day_superseded=4,
        ),
        PersonaFact(
            fact_id="bob_f006", category="sport",
            predicate="is recovering from", value="a knee injury (unable to cycle)",
            day_introduced=4, day_superseded=14,
        ),
        PersonaFact(
            fact_id="bob_f007", category="sport",
            predicate="has returned to", value="cycling after recovering from a knee injury",
            day_introduced=14,
        ),
        # --- pet ---
        PersonaFact(
            fact_id="bob_f008", category="pet",
            predicate="has a", value="golden retriever named Rex",
            day_introduced=1, day_superseded=11,
        ),
        PersonaFact(
            fact_id="bob_f009", category="pet",
            predicate="has no", value="pet (his dog Rex recently passed away)",
            day_introduced=11, day_superseded=17,
        ),
        PersonaFact(
            fact_id="bob_f010", category="pet",
            predicate="has adopted", value="a tabby cat named Luna",
            day_introduced=17,
        ),
        # --- diet ---
        PersonaFact(
            fact_id="bob_f011", category="diet",
            predicate="follows", value="no special diet",
            day_introduced=1, day_superseded=6,
        ),
        PersonaFact(
            fact_id="bob_f012", category="diet",
            predicate="practices", value="intermittent fasting on a 16:8 schedule",
            day_introduced=6,
        ),
    ]

    events = [
        DayEvent(
            day=4,
            description=(
                "Bob injured his knee badly during a training ride. Doctor said no cycling for weeks."
            ),
            affected_fact_ids=["bob_f005", "bob_f006"],
        ),
        DayEvent(
            day=6,
            description="Bob started intermittent fasting (16:8) after reading about its health benefits.",
            affected_fact_ids=["bob_f011", "bob_f012"],
        ),
        DayEvent(
            day=8,
            description=(
                "Bob's sabbatical from Lincoln High officially began today. "
                "He plans to spend the year writing a local history book."
            ),
            affected_fact_ids=["bob_f002", "bob_f003"],
        ),
        DayEvent(
            day=11,
            description=(
                "Bob's golden retriever Rex passed away after a long illness. "
                "Bob is heartbroken."
            ),
            affected_fact_ids=["bob_f008", "bob_f009"],
        ),
        DayEvent(
            day=14,
            description="Bob's knee has fully healed. He went on his first ride today.",
            affected_fact_ids=["bob_f006", "bob_f007"],
        ),
        DayEvent(
            day=16,
            description=(
                "Bob's sabbatical ended earlier than expected. "
                "He accepted a teaching position at Westside Academy starting today."
            ),
            affected_fact_ids=["bob_f003", "bob_f004"],
        ),
        DayEvent(
            day=17,
            description="Bob adopted a tabby cat named Luna from the local animal shelter.",
            affected_fact_ids=["bob_f009", "bob_f010"],
        ),
    ]

    persona = PersonaGroundTruth(
        persona_id="bob",
        name="Bob Martinez",
        age=45,
        background=(
            "A 45-year-old high school history teacher who has taught for 18 years in Chicago. "
            "An avid long-distance cyclist who participates in charity rides. "
            "Deeply attached to his golden retriever Rex. "
            "Thoughtful and reflective, with a passion for storytelling and local history."
        ),
        facts=facts,
        events=events,
    )
    return persona.finalize_statuses()


def get_all_personas() -> list[PersonaGroundTruth]:
    return [build_alice(), build_bob()]
