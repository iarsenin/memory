"""
Phase 7 — Eval Probe Generator.

Builds the static test suite from ground-truth persona JSONs and saves it to
data/eval_probes/probes.json so every condition is scored against the exact
same question text.

Bucket definitions
──────────────────
  stable     Facts that were active from their introduction through Day 20 and
             were NEVER superseded by another fact.  Auto-generated question
             using eval templates.

  updated    Currently active facts that explicitly replaced a prior state
             (i.e. another fact has superseded_by_id pointing to this one).
             Question asks for the NEW current value.

  superseded Old facts (status == 'superseded').  Question ASSERTS the stale
             value; a correct model answer rejects / corrects the old claim.
             Questions are handcrafted per fact (assertion-style phrasing).

  relational Multi-hop questions that require combining two or more facts.
             Fully handcrafted.

Leakage guard
─────────────
Eval templates ("What is {name}'s current {category}?", etc.) are distinct
from train_config.json qa_train_templates ("What is {subject}'s {category}?"
etc.) — the word "current", "changed recently", and "had some changes lately"
do not appear in any training template.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Handcrafted probes — superseded assertions
# ---------------------------------------------------------------------------
# For each superseded fact we provide an assertion-style question that frames
# the OLD value as if it were still true.  The correct model answer is to
# reject or correct the stale claim.
#
# bob_f005 ("trains for long-distance cycling", Days 1–4) is intentionally
# omitted: the active Day-20 state (bob_f007) is also about cycling, so an
# assertion that Bob is a cyclist would be technically correct at Day 20 —
# making it an ambiguous scoring target.

_SUPERSEDED_ASSERTIONS: dict[str, dict[str, str]] = {
    # ── Alice ────────────────────────────────────────────────────────────
    "alice_f001": {
        "question": "Is Alice still living in Seattle?",
        "expected": "No — Alice moved to Austin, TX.",
    },
    "alice_f003": {
        "question": "Alice still works as a software engineer at TechCorp, right?",
        "expected": (
            "No — Alice was laid off from TechCorp.  "
            "She now works as an independent freelance software consultant."
        ),
    },
    "alice_f004": {
        "question": "Is Alice currently unemployed?",
        "expected": "No — Alice found work as a freelance software consultant.",
    },
    "alice_f006": {
        "question": "Is Alice still in a relationship with Mark?",
        "expected": "No — Alice and Mark broke up.  She is now dating Jamie.",
    },
    "alice_f007": {
        "question": "Is Alice currently single?",
        "expected": "No — Alice is now dating Jamie.",
    },
    # ── Bob ──────────────────────────────────────────────────────────────
    "bob_f002": {
        "question": "Is Bob still teaching at Lincoln High School?",
        "expected": "No — after his sabbatical, Bob moved to Westside Academy.",
    },
    "bob_f003": {
        "question": "Is Bob currently on academic sabbatical?",
        "expected": "No — Bob's sabbatical ended; he is now teaching at Westside Academy.",
    },
    "bob_f006": {
        "question": "Is Bob still recovering from his knee injury?",
        "expected": "No — Bob has fully recovered and returned to cycling.",
    },
    "bob_f008": {
        "question": "Is Bob's golden retriever Rex doing well?",
        "expected": "Rex passed away.  Bob later adopted a tabby cat named Luna.",
    },
    "bob_f009": {
        "question": "Does Bob currently have no pets?",
        "expected": (
            "That is no longer true — Bob adopted a tabby cat named Luna "
            "after Rex passed away."
        ),
    },
    "bob_f011": {
        "question": "Does Bob follow no special diet?",
        "expected": "Bob now practises intermittent fasting on a 16:8 schedule.",
    },
}

# ---------------------------------------------------------------------------
# Handcrafted probes — relational / multi-hop
# ---------------------------------------------------------------------------

_RELATIONAL_PROBES: dict[str, list[dict[str, Any]]] = {
    "alice": [
        {
            "probe_id": "alice_relational_001",
            "question": (
                "Alice recently relocated — which city is she "
                "now living and working from?"
            ),
            "expected": (
                "Alice moved to Austin, TX and works there as a "
                "freelance software consultant."
            ),
            "fact_ids": ["alice_f002", "alice_f005"],
        },
        {
            "probe_id": "alice_relational_002",
            "question": (
                "After everything she went through in her previous "
                "relationship, who is Alice currently seeing?"
            ),
            "expected": (
                "Alice is now dating Jamie "
                "(she previously broke up with Mark)."
            ),
            "fact_ids": ["alice_f006", "alice_f007", "alice_f008"],
        },
        {
            "probe_id": "alice_relational_003",
            "question": (
                "What new creative hobby did Alice take up around the "
                "time she was between jobs?"
            ),
            "expected": (
                "Alice took up pottery around Day 10, during her "
                "period of unemployment after being laid off from TechCorp."
            ),
            "fact_ids": ["alice_f011", "alice_f004"],
        },
    ],
    "bob": [
        {
            "probe_id": "bob_relational_001",
            "question": (
                "After his health setback forced him to stop, has Bob "
                "managed to return to his main sport?"
            ),
            "expected": (
                "Yes — Bob recovered from his knee injury and "
                "returned to cycling."
            ),
            "fact_ids": ["bob_f006", "bob_f007"],
        },
        {
            "probe_id": "bob_relational_002",
            "question": (
                "What happened with Bob's pets over the course "
                "of the simulation?"
            ),
            "expected": (
                "Bob's golden retriever Rex passed away on Day 11.  "
                "Bob later adopted a tabby cat named Luna."
            ),
            "fact_ids": ["bob_f008", "bob_f009", "bob_f010"],
        },
        {
            "probe_id": "bob_relational_003",
            "question": (
                "Did Bob return to his original school after his time "
                "away, or did he move to a different institution?"
            ),
            "expected": (
                "Bob did not return to Lincoln High School — "
                "after his sabbatical he joined Westside Academy."
            ),
            "fact_ids": ["bob_f002", "bob_f003", "bob_f004"],
        },
    ],
}

# ---------------------------------------------------------------------------
# Eval templates — distinct from train_config qa_train_templates
# ---------------------------------------------------------------------------
# Template index cycles through the list based on probe position so the test
# suite uses a variety of phrasings without being random across runs.

_EVAL_TEMPLATES = [
    "What is {name}'s current {category}?",
    "Has {name}'s {category} changed recently? What is it now?",
    "I heard {name} had some changes lately — what is their {category} at the moment?",
]


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


def _successor_ids(facts: list[dict]) -> set[str]:
    """Return fact_ids that are successors of another fact."""
    return {f["superseded_by_id"] for f in facts if f.get("superseded_by_id")}


def _bucket_facts(facts: list[dict]) -> dict[str, list[dict]]:
    succ = _successor_ids(facts)
    return {
        "stable": [
            f for f in facts
            if f["status"] == "active" and f["fact_id"] not in succ
        ],
        "updated": [
            f for f in facts
            if f["status"] == "active" and f["fact_id"] in succ
        ],
        "superseded": [
            f for f in facts if f["status"] == "superseded"
        ],
    }


def _auto_question(name: str, fact: dict, idx: int) -> str:
    tpl = _EVAL_TEMPLATES[idx % len(_EVAL_TEMPLATES)]
    return tpl.format(name=name, category=fact["category"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_probes(
    personas_dir: Path,
    persona_ids: list[str],
    rng_seed: int = 42,
) -> list[dict]:
    """
    Generate the full static test suite for all personas.

    Returns a list of probe dicts; each probe has:
      probe_id, persona_id, persona_name, bucket, fact_id (nullable),
      question, expected, assertion_claim (nullable — old value for superseded).
    """
    random.seed(rng_seed)
    probes: list[dict] = []

    for pid in persona_ids:
        gt_path = personas_dir / f"{pid}_ground_truth.json"
        gt = json.loads(gt_path.read_text())
        name  = gt["name"]
        facts = gt["facts"]
        bkts  = _bucket_facts(facts)

        # ── Stable ────────────────────────────────────────────────────────
        for i, fact in enumerate(bkts["stable"]):
            probes.append({
                "probe_id":        f"{pid}_stable_{i+1:03d}",
                "persona_id":      pid,
                "persona_name":    name,
                "bucket":          "stable",
                "fact_id":         fact["fact_id"],
                "question":        _auto_question(name, fact, i),
                "expected":        fact["value"],
                "assertion_claim": None,
            })

        # ── Updated ───────────────────────────────────────────────────────
        offset = len(bkts["stable"])
        for i, fact in enumerate(bkts["updated"]):
            probes.append({
                "probe_id":        f"{pid}_updated_{i+1:03d}",
                "persona_id":      pid,
                "persona_name":    name,
                "bucket":          "updated",
                "fact_id":         fact["fact_id"],
                "question":        _auto_question(name, fact, offset + i),
                "expected":        fact["value"],
                "assertion_claim": None,
            })

        # ── Superseded ────────────────────────────────────────────────────
        seq = 0
        for fact in bkts["superseded"]:
            fid = fact["fact_id"]
            if fid not in _SUPERSEDED_ASSERTIONS:
                continue  # skip facts without a handcrafted assertion
            seq += 1
            spec = _SUPERSEDED_ASSERTIONS[fid]
            probes.append({
                "probe_id":        f"{pid}_superseded_{seq:03d}",
                "persona_id":      pid,
                "persona_name":    name,
                "bucket":          "superseded",
                "fact_id":         fid,
                "question":        spec["question"],
                "expected":        spec["expected"],
                "assertion_claim": fact["value"],  # the stale value
            })

        # ── Relational ────────────────────────────────────────────────────
        for rp in _RELATIONAL_PROBES.get(pid, []):
            probes.append({
                "probe_id":        rp["probe_id"],
                "persona_id":      pid,
                "persona_name":    name,
                "bucket":          "relational",
                "fact_id":         None,
                "question":        rp["question"],
                "expected":        rp["expected"],
                "assertion_claim": None,
            })

    return probes


def save_probes(probes: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(probes, indent=2))
