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
            predicate="has adopted", value="a tabby cat named Luna (his previous dog Rex passed away on Day 11)",
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
            "An animal lover who is deeply attached to his pets. "
            "Thoughtful and reflective, with a passion for storytelling and local history."
        ),
        facts=facts,
        events=events,
    )
    return persona.finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 3: Charlie Robinson
# ---------------------------------------------------------------------------

def build_charlie() -> PersonaGroundTruth:
    """
    Charlie Robinson, 28, graphic designer in Austin, TX.

    Key timeline (20 days):
      Day  4 — adopts cat Mochi from a shelter
      Day  8 — leaves Studio Creative, goes freelance
      Day 13 — starts dating Sam
      Day 17 — moves from Austin to Denver for a client opportunity

    Stable: vegetarian diet, runs 5Ks
    """
    facts = [
        PersonaFact(
            fact_id="charlie_f001", category="location",
            predicate="lives in", value="Austin, TX",
            day_introduced=1, day_superseded=17,
        ),
        PersonaFact(
            fact_id="charlie_f002", category="location",
            predicate="lives in", value="Denver, CO",
            day_introduced=17,
        ),
        PersonaFact(
            fact_id="charlie_f003", category="job",
            predicate="works as", value="graphic designer at Studio Creative",
            day_introduced=1, day_superseded=8,
        ),
        PersonaFact(
            fact_id="charlie_f004", category="job",
            predicate="works as", value="freelance graphic designer",
            day_introduced=8,
        ),
        PersonaFact(
            fact_id="charlie_f005", category="pet",
            predicate="has no", value="pet",
            day_introduced=1, day_superseded=4,
        ),
        PersonaFact(
            fact_id="charlie_f006", category="pet",
            predicate="has adopted", value="a tabby cat named Mochi from a shelter",
            day_introduced=4,
        ),
        PersonaFact(
            fact_id="charlie_f007", category="relationship",
            predicate="is", value="single",
            day_introduced=1, day_superseded=13,
        ),
        PersonaFact(
            fact_id="charlie_f008", category="relationship",
            predicate="is dating", value="Sam",
            day_introduced=13,
        ),
        PersonaFact(
            fact_id="charlie_f009", category="diet",
            predicate="follows", value="a vegetarian diet",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="charlie_f010", category="sport",
            predicate="regularly runs", value="5K races as a hobby",
            day_introduced=1, is_stable=True,
        ),
    ]
    events = [
        DayEvent(day=4,
                 description="Charlie adopted a tabby cat named Mochi from the local shelter.",
                 affected_fact_ids=["charlie_f005", "charlie_f006"]),
        DayEvent(day=8,
                 description="Charlie left Studio Creative and is now working as a freelance graphic designer.",
                 affected_fact_ids=["charlie_f003", "charlie_f004"]),
        DayEvent(day=13,
                 description="Charlie started dating someone named Sam after meeting at a 5K race.",
                 affected_fact_ids=["charlie_f007", "charlie_f008"]),
        DayEvent(day=17,
                 description="Charlie relocated from Austin to Denver for a long-term freelance client opportunity.",
                 affected_fact_ids=["charlie_f001", "charlie_f002"]),
    ]
    return PersonaGroundTruth(
        persona_id="charlie", name="Charlie Robinson", age=28,
        background=(
            "A 28-year-old graphic designer from Austin, TX. Enthusiastic vegetarian cook "
            "who runs 5K races on weekends. Left a stable studio job to pursue freelance work "
            "and recently relocated to Denver for a promising client."
        ),
        facts=facts, events=events,
    ).finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 4: Diana Foster
# ---------------------------------------------------------------------------

def build_diana() -> PersonaGroundTruth:
    """
    Diana Foster, 41, nurse in Boston, MA.

    Key timeline (20 days):
      Day  5 — promoted from ER nurse to charge nurse at Mass General
      Day  9 — dog Pepper passes away
      Day 15 — adopts golden retriever Cooper
      Day 19 — signs up for triathlon training, shifting from half-marathons

    Stable: pescatarian diet, yoga practice
    """
    facts = [
        PersonaFact(
            fact_id="diana_f001", category="location",
            predicate="lives in", value="Boston, MA",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="diana_f002", category="job",
            predicate="works as", value="ER nurse at Mass General Hospital",
            day_introduced=1, day_superseded=5,
        ),
        PersonaFact(
            fact_id="diana_f003", category="job",
            predicate="works as", value="charge nurse at Mass General Hospital",
            day_introduced=5,
        ),
        PersonaFact(
            fact_id="diana_f004", category="pet",
            predicate="has a", value="labrador named Pepper",
            day_introduced=1, day_superseded=9,
        ),
        PersonaFact(
            fact_id="diana_f005", category="pet",
            predicate="has no", value="pet (her dog Pepper recently passed away)",
            day_introduced=9, day_superseded=15,
        ),
        PersonaFact(
            fact_id="diana_f006", category="pet",
            predicate="has adopted", value="a golden retriever named Cooper",
            day_introduced=15,
        ),
        PersonaFact(
            fact_id="diana_f007", category="diet",
            predicate="follows", value="a pescatarian diet",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="diana_f008", category="sport",
            predicate="trains for", value="half-marathons",
            day_introduced=1, day_superseded=19,
        ),
        PersonaFact(
            fact_id="diana_f009", category="sport",
            predicate="is training for", value="a triathlon (shifted from half-marathons)",
            day_introduced=19,
        ),
        PersonaFact(
            fact_id="diana_f010", category="hobby",
            predicate="practices", value="yoga daily",
            day_introduced=1, is_stable=True,
        ),
    ]
    events = [
        DayEvent(day=5,
                 description="Diana was promoted from ER nurse to charge nurse at Mass General Hospital.",
                 affected_fact_ids=["diana_f002", "diana_f003"]),
        DayEvent(day=9,
                 description="Diana's dog Pepper passed away after a long illness. She is heartbroken.",
                 affected_fact_ids=["diana_f004", "diana_f005"]),
        DayEvent(day=15,
                 description="Diana adopted a golden retriever named Cooper from a rescue shelter.",
                 affected_fact_ids=["diana_f005", "diana_f006"]),
        DayEvent(day=19,
                 description="Diana signed up for a triathlon program, replacing her previous half-marathon training.",
                 affected_fact_ids=["diana_f008", "diana_f009"]),
    ]
    return PersonaGroundTruth(
        persona_id="diana", name="Diana Foster", age=41,
        background=(
            "A 41-year-old nurse who has worked at Mass General Hospital in Boston for 12 years. "
            "A dedicated yoga practitioner and runner who follows a pescatarian diet. "
            "Deeply passionate about her work and her pets."
        ),
        facts=facts, events=events,
    ).finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 5: Ethan Park
# ---------------------------------------------------------------------------

def build_ethan() -> PersonaGroundTruth:
    """
    Ethan Park, 26, barista in Seattle, WA.

    Key timeline (20 days):
      Day  3 — enrolls in an evening coding bootcamp
      Day 11 — lands junior developer job at a startup; leaves barista role
      Day 15 — moves from Seattle to San Francisco for the new job
      Day 18 — adopts a hamster named Pi

    Stable: plays guitar, no special diet
    """
    facts = [
        PersonaFact(
            fact_id="ethan_f001", category="location",
            predicate="lives in", value="Seattle, WA",
            day_introduced=1, day_superseded=15,
        ),
        PersonaFact(
            fact_id="ethan_f002", category="location",
            predicate="lives in", value="San Francisco, CA",
            day_introduced=15,
        ),
        PersonaFact(
            fact_id="ethan_f003", category="job",
            predicate="works as", value="barista at Blue Bottle Coffee",
            day_introduced=1, day_superseded=3,
        ),
        PersonaFact(
            fact_id="ethan_f004", category="job",
            predicate="is attending", value="an evening coding bootcamp while working as a barista",
            day_introduced=3, day_superseded=11,
        ),
        PersonaFact(
            fact_id="ethan_f005", category="job",
            predicate="works as", value="junior software developer at a San Francisco startup",
            day_introduced=11,
        ),
        PersonaFact(
            fact_id="ethan_f006", category="pet",
            predicate="has no", value="pet",
            day_introduced=1, day_superseded=18,
        ),
        PersonaFact(
            fact_id="ethan_f007", category="pet",
            predicate="has adopted", value="a hamster named Pi",
            day_introduced=18,
        ),
        PersonaFact(
            fact_id="ethan_f008", category="hobby",
            predicate="plays", value="guitar as a main hobby",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="ethan_f009", category="diet",
            predicate="follows", value="no special diet",
            day_introduced=1, is_stable=True,
        ),
    ]
    events = [
        DayEvent(day=3,
                 description="Ethan enrolled in an evening coding bootcamp to transition out of his barista job.",
                 affected_fact_ids=["ethan_f003", "ethan_f004"]),
        DayEvent(day=11,
                 description="Ethan landed a junior developer role at a startup. He will quit barista work and move to SF.",
                 affected_fact_ids=["ethan_f004", "ethan_f005"]),
        DayEvent(day=15,
                 description="Ethan moved from Seattle to San Francisco to start his new developer job.",
                 affected_fact_ids=["ethan_f001", "ethan_f002"]),
        DayEvent(day=18,
                 description="Ethan adopted a hamster named Pi from a pet store in San Francisco.",
                 affected_fact_ids=["ethan_f006", "ethan_f007"]),
    ]
    return PersonaGroundTruth(
        persona_id="ethan", name="Ethan Park", age=26,
        background=(
            "A 26-year-old originally from Seattle who worked as a barista while teaching himself "
            "to code. An avid guitarist who recently pivoted careers and relocated to San Francisco."
        ),
        facts=facts, events=events,
    ).finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 6: Fiona Chen
# ---------------------------------------------------------------------------

def build_fiona() -> PersonaGroundTruth:
    """
    Fiona Chen, 34, yoga instructor in Denver, CO.

    Key timeline (20 days):
      Day  6 — opens her own yoga studio (previously teaching at a gym)
      Day 10 — knee injury forces her to pause teaching temporarily
      Day 14 — knee recovers; returns to teaching at her studio
      Day 17 — starts dating Alex after meeting at a wellness retreat

    Stable: vegan diet, daily meditation
    """
    facts = [
        PersonaFact(
            fact_id="fiona_f001", category="location",
            predicate="lives in", value="Denver, CO",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="fiona_f002", category="job",
            predicate="works as", value="yoga instructor at FitLife Gym",
            day_introduced=1, day_superseded=6,
        ),
        PersonaFact(
            fact_id="fiona_f003", category="job",
            predicate="runs", value="her own yoga studio called Flow Space",
            day_introduced=6, day_superseded=10,
        ),
        PersonaFact(
            fact_id="fiona_f004", category="job",
            predicate="is on", value="temporary leave from her yoga studio due to a knee injury",
            day_introduced=10, day_superseded=14,
        ),
        PersonaFact(
            fact_id="fiona_f005", category="job",
            predicate="has returned to running", value="her yoga studio Flow Space after recovering from injury",
            day_introduced=14,
        ),
        PersonaFact(
            fact_id="fiona_f006", category="relationship",
            predicate="is", value="single",
            day_introduced=1, day_superseded=17,
        ),
        PersonaFact(
            fact_id="fiona_f007", category="relationship",
            predicate="is dating", value="Alex, whom she met at a wellness retreat",
            day_introduced=17,
        ),
        PersonaFact(
            fact_id="fiona_f008", category="diet",
            predicate="follows", value="a vegan diet",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="fiona_f009", category="hobby",
            predicate="meditates", value="daily as a core wellness practice",
            day_introduced=1, is_stable=True,
        ),
    ]
    events = [
        DayEvent(day=6,
                 description="Fiona opened her own yoga studio called Flow Space. She left FitLife Gym.",
                 affected_fact_ids=["fiona_f002", "fiona_f003"]),
        DayEvent(day=10,
                 description="Fiona injured her knee during a yoga demonstration. She is temporarily unable to teach.",
                 affected_fact_ids=["fiona_f003", "fiona_f004"]),
        DayEvent(day=14,
                 description="Fiona's knee has healed. She returned to teaching classes at Flow Space today.",
                 affected_fact_ids=["fiona_f004", "fiona_f005"]),
        DayEvent(day=17,
                 description="Fiona met someone named Alex at a wellness retreat and they started dating.",
                 affected_fact_ids=["fiona_f006", "fiona_f007"]),
    ]
    return PersonaGroundTruth(
        persona_id="fiona", name="Fiona Chen", age=34,
        background=(
            "A 34-year-old yoga instructor based in Denver. A committed vegan who meditates every "
            "morning. Recently took the leap to open her own studio after years of teaching at a gym."
        ),
        facts=facts, events=events,
    ).finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 7: George Harris
# ---------------------------------------------------------------------------

def build_george() -> PersonaGroundTruth:
    """
    George Harris, 52, financial advisor in NYC → Miami.

    Key timeline (20 days):
      Day  5 — takes early retirement from Meridian Finance
      Day  9 — moves from New York City to Miami, FL
      Day 14 — takes up sailing as a new sport (replaces golf)
      Day 17 — adopts a Labrador named Max

    Stable: intermittent fasting, history enthusiast (hobby)
    """
    facts = [
        PersonaFact(
            fact_id="george_f001", category="location",
            predicate="lives in", value="New York City, NY",
            day_introduced=1, day_superseded=9,
        ),
        PersonaFact(
            fact_id="george_f002", category="location",
            predicate="lives in", value="Miami, FL",
            day_introduced=9,
        ),
        PersonaFact(
            fact_id="george_f003", category="job",
            predicate="works as", value="senior financial advisor at Meridian Finance",
            day_introduced=1, day_superseded=5,
        ),
        PersonaFact(
            fact_id="george_f004", category="job",
            predicate="is", value="retired (took early retirement from Meridian Finance)",
            day_introduced=5,
        ),
        PersonaFact(
            fact_id="george_f005", category="sport",
            predicate="plays", value="golf as a primary sport",
            day_introduced=1, day_superseded=14,
        ),
        PersonaFact(
            fact_id="george_f006", category="sport",
            predicate="has taken up", value="sailing, replacing golf as his main sport",
            day_introduced=14,
        ),
        PersonaFact(
            fact_id="george_f007", category="pet",
            predicate="has no", value="pet",
            day_introduced=1, day_superseded=17,
        ),
        PersonaFact(
            fact_id="george_f008", category="pet",
            predicate="has adopted", value="a Labrador named Max",
            day_introduced=17,
        ),
        PersonaFact(
            fact_id="george_f009", category="diet",
            predicate="practices", value="intermittent fasting on a 16:8 schedule",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="george_f010", category="hobby",
            predicate="is passionate about", value="reading history books",
            day_introduced=1, is_stable=True,
        ),
    ]
    events = [
        DayEvent(day=5,
                 description="George took early retirement from Meridian Finance after 25 years.",
                 affected_fact_ids=["george_f003", "george_f004"]),
        DayEvent(day=9,
                 description="George relocated from New York City to Miami, FL for a slower pace of life.",
                 affected_fact_ids=["george_f001", "george_f002"]),
        DayEvent(day=14,
                 description="George enrolled in sailing lessons and has given up golf entirely.",
                 affected_fact_ids=["george_f005", "george_f006"]),
        DayEvent(day=17,
                 description="George adopted a Labrador named Max from a Miami rescue shelter.",
                 affected_fact_ids=["george_f007", "george_f008"]),
    ]
    return PersonaGroundTruth(
        persona_id="george", name="George Harris", age=52,
        background=(
            "A 52-year-old former financial advisor who recently retired after a long Wall Street career. "
            "A history buff and former golfer who relocated to Miami. Disciplined about his diet "
            "and enjoying the freedom of retirement."
        ),
        facts=facts, events=events,
    ).finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 8: Hannah Lee
# ---------------------------------------------------------------------------

def build_hannah() -> PersonaGroundTruth:
    """
    Hannah Lee, 30, journalist in Chicago, IL.

    Key timeline (20 days):
      Day  4 — laid off from the Tribune; transitions to freelance writing
      Day  8 — secures a regular column with an online magazine (freelance solidifies)
      Day 12 — starts dating Taylor
      Day 17 — adopts a parrot named Rio

    Stable: avid reader, loves cooking
    """
    facts = [
        PersonaFact(
            fact_id="hannah_f001", category="location",
            predicate="lives in", value="Chicago, IL",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="hannah_f002", category="job",
            predicate="works as", value="staff journalist at the Chicago Tribune",
            day_introduced=1, day_superseded=4,
        ),
        PersonaFact(
            fact_id="hannah_f003", category="job",
            predicate="is", value="unemployed after being laid off from the Chicago Tribune",
            day_introduced=4, day_superseded=8,
        ),
        PersonaFact(
            fact_id="hannah_f004", category="job",
            predicate="works as", value="freelance writer with a regular column at an online magazine",
            day_introduced=8,
        ),
        PersonaFact(
            fact_id="hannah_f005", category="relationship",
            predicate="is", value="single",
            day_introduced=1, day_superseded=12,
        ),
        PersonaFact(
            fact_id="hannah_f006", category="relationship",
            predicate="is dating", value="Taylor",
            day_introduced=12,
        ),
        PersonaFact(
            fact_id="hannah_f007", category="pet",
            predicate="has no", value="pet",
            day_introduced=1, day_superseded=17,
        ),
        PersonaFact(
            fact_id="hannah_f008", category="pet",
            predicate="has adopted", value="a parrot named Rio",
            day_introduced=17,
        ),
        PersonaFact(
            fact_id="hannah_f009", category="hobby",
            predicate="is an", value="avid reader of literary fiction",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="hannah_f010", category="hobby",
            predicate="loves", value="cooking elaborate meals on weekends",
            day_introduced=1, is_stable=True,
        ),
    ]
    events = [
        DayEvent(day=4,
                 description="Hannah was laid off from the Chicago Tribune in a round of newsroom cuts.",
                 affected_fact_ids=["hannah_f002", "hannah_f003"]),
        DayEvent(day=8,
                 description="Hannah secured a regular freelance column with an online magazine. She is now a freelance writer.",
                 affected_fact_ids=["hannah_f003", "hannah_f004"]),
        DayEvent(day=12,
                 description="Hannah started dating someone named Taylor after meeting through mutual friends.",
                 affected_fact_ids=["hannah_f005", "hannah_f006"]),
        DayEvent(day=17,
                 description="Hannah adopted a parrot named Rio from a bird rescue.",
                 affected_fact_ids=["hannah_f007", "hannah_f008"]),
    ]
    return PersonaGroundTruth(
        persona_id="hannah", name="Hannah Lee", age=30,
        background=(
            "A 30-year-old journalist based in Chicago. A passionate reader and weekend cook "
            "who recently navigated a difficult newsroom layoff and pivoted to freelancing."
        ),
        facts=facts, events=events,
    ).finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 9: Ian Walsh
# ---------------------------------------------------------------------------

def build_ian() -> PersonaGroundTruth:
    """
    Ian Walsh, 44, software architect in San Jose, CA → Portland, OR.

    Key timeline (20 days):
      Day  6 — takes a sabbatical from his company (NexaTech)
      Day 11 — starts his own consulting firm during sabbatical
      Day 15 — relocates from San Jose to Portland
      Day 19 — adopts a rescue cat named Pixel

    Stable: vegetarian diet, morning meditation
    """
    facts = [
        PersonaFact(
            fact_id="ian_f001", category="location",
            predicate="lives in", value="San Jose, CA",
            day_introduced=1, day_superseded=15,
        ),
        PersonaFact(
            fact_id="ian_f002", category="location",
            predicate="lives in", value="Portland, OR",
            day_introduced=15,
        ),
        PersonaFact(
            fact_id="ian_f003", category="job",
            predicate="works as", value="senior software architect at NexaTech",
            day_introduced=1, day_superseded=6,
        ),
        PersonaFact(
            fact_id="ian_f004", category="job",
            predicate="is on", value="sabbatical from NexaTech",
            day_introduced=6, day_superseded=11,
        ),
        PersonaFact(
            fact_id="ian_f005", category="job",
            predicate="runs", value="his own software consulting firm (started during sabbatical)",
            day_introduced=11,
        ),
        PersonaFact(
            fact_id="ian_f006", category="pet",
            predicate="has no", value="pet",
            day_introduced=1, day_superseded=19,
        ),
        PersonaFact(
            fact_id="ian_f007", category="pet",
            predicate="has adopted", value="a rescue cat named Pixel",
            day_introduced=19,
        ),
        PersonaFact(
            fact_id="ian_f008", category="diet",
            predicate="follows", value="a vegetarian diet",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="ian_f009", category="hobby",
            predicate="meditates", value="every morning as a long-standing practice",
            day_introduced=1, is_stable=True,
        ),
    ]
    events = [
        DayEvent(day=6,
                 description="Ian started a sabbatical from NexaTech to explore independent work.",
                 affected_fact_ids=["ian_f003", "ian_f004"]),
        DayEvent(day=11,
                 description="Ian launched his own software consulting firm while on sabbatical.",
                 affected_fact_ids=["ian_f004", "ian_f005"]),
        DayEvent(day=15,
                 description="Ian moved from San Jose to Portland, OR, seeking a change of scenery.",
                 affected_fact_ids=["ian_f001", "ian_f002"]),
        DayEvent(day=19,
                 description="Ian adopted a rescue cat named Pixel from a Portland animal shelter.",
                 affected_fact_ids=["ian_f006", "ian_f007"]),
    ]
    return PersonaGroundTruth(
        persona_id="ian", name="Ian Walsh", age=44,
        background=(
            "A 44-year-old software architect who recently left the corporate world to pursue "
            "independent consulting. A vegetarian meditator who moved from Silicon Valley to Portland "
            "for a slower-paced lifestyle."
        ),
        facts=facts, events=events,
    ).finalize_statuses()


# ---------------------------------------------------------------------------
# Persona 10: Julia Santos
# ---------------------------------------------------------------------------

def build_julia() -> PersonaGroundTruth:
    """
    Julia Santos, 29, elementary school teacher in Miami, FL.

    Key timeline (20 days):
      Day  5 — adopts a cat named Mango
      Day  9 — breaks up with long-term partner Kai
      Day 14 — enrolls in an evening graduate program in education
      Day 18 — accepts a part-time research assistant position at the university

    Stable: salsa dancing, learning Spanish
    """
    facts = [
        PersonaFact(
            fact_id="julia_f001", category="location",
            predicate="lives in", value="Miami, FL",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="julia_f002", category="job",
            predicate="works as", value="elementary school teacher at Coral Way K-8",
            day_introduced=1, day_superseded=18,
        ),
        PersonaFact(
            fact_id="julia_f003", category="job",
            predicate="works as", value="elementary school teacher and part-time research assistant at the university",
            day_introduced=18,
        ),
        PersonaFact(
            fact_id="julia_f004", category="pet",
            predicate="has no", value="pet",
            day_introduced=1, day_superseded=5,
        ),
        PersonaFact(
            fact_id="julia_f005", category="pet",
            predicate="has adopted", value="an orange tabby cat named Mango",
            day_introduced=5,
        ),
        PersonaFact(
            fact_id="julia_f006", category="relationship",
            predicate="is in a relationship with", value="Kai",
            day_introduced=1, day_superseded=9,
        ),
        PersonaFact(
            fact_id="julia_f007", category="relationship",
            predicate="is", value="single (recently broke up with Kai after two years together)",
            day_introduced=9,
        ),
        PersonaFact(
            fact_id="julia_f008", category="hobby",
            predicate="dances", value="salsa competitively on weekends",
            day_introduced=1, is_stable=True,
        ),
        PersonaFact(
            fact_id="julia_f009", category="hobby",
            predicate="is actively", value="learning Spanish through daily practice",
            day_introduced=1, is_stable=True,
        ),
    ]
    events = [
        DayEvent(day=5,
                 description="Julia adopted an orange tabby cat named Mango from a Miami animal rescue.",
                 affected_fact_ids=["julia_f004", "julia_f005"]),
        DayEvent(day=9,
                 description="Julia and Kai broke up after two years together. She is processing the change.",
                 affected_fact_ids=["julia_f006", "julia_f007"]),
        DayEvent(day=14,
                 description="Julia enrolled in an evening graduate program in education at the university.",
                 affected_fact_ids=[]),
        DayEvent(day=18,
                 description="Julia accepted a part-time research assistant position at the university alongside her teaching.",
                 affected_fact_ids=["julia_f002", "julia_f003"]),
    ]
    return PersonaGroundTruth(
        persona_id="julia", name="Julia Santos", age=29,
        background=(
            "A 29-year-old elementary school teacher in Miami who dances salsa competitively "
            "and is working toward fluency in Spanish. Recently started graduate school while "
            "managing a full-time teaching position."
        ),
        facts=facts, events=events,
    ).finalize_statuses()


def get_all_personas() -> list[PersonaGroundTruth]:
    return [
        build_alice(), build_bob(), build_charlie(), build_diana(),
        build_ethan(), build_fiona(), build_george(), build_hannah(),
        build_ian(), build_julia(),
    ]
