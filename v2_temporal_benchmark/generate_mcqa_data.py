"""
v2_temporal_benchmark/generate_mcqa_data.py

Generates the MCQA benchmark dataset for the TemporalBench v2 experiment.

Design principles
-----------------
* Soft fork — reads .env and salience data from the parent project root;
  writes ALL outputs inside v2_temporal_benchmark/data/.  Nothing in src/,
  configs/, or scripts/ is touched.

* Persona generation uses OpenAI API (temperature=0.9, batched) to produce
  30 diverse synthetic people each with:
    - 8–12 stable_facts  (used as the "volume" dimension in sweeps)
    - exactly 5 updated_facts  (always 100% included in training)

* Probe assembly is fully deterministic given the RNG seed — the LLM is
  only called for persona content, never for probe construction.

JSON schema (single updated_fact entry in benchmark.json)
----------------------------------------------------------
{
  "fact_id":          "alice_chen_u1",
  "persona_id":       "alice_chen",
  "persona_name":     "Alice Chen",
  "category":         "location",
  "predicate":        "lives in",
  "old_value":        "Seattle",
  "new_value":        "Austin",
  "distractor_value": "Denver",
  "both_label":       "both Seattle and Austin",
  "update_context":   "Alice relocated for a new tech job opportunity.",
  "training_sentences": [
    "Alice lives in Austin.",
    "Alice has recently moved to Austin.",
    "Alice currently resides in Austin."
  ],
  "probes": [
    {
      "probe_id":             "alice_chen_u1_p1",
      "family":               "current_state",
      "expected_answer_type": "current",
      "question":             "Where does Alice live now?",
      "shuffled_options":     {"A": "Austin", "B": "both Seattle and Austin",
                               "C": "Seattle", "D": "Denver"},
      "target_mapping":       {"A": "current", "B": "both",
                               "C": "stale",   "D": "distractor"},
      "correct_letter":       "A",
      "full_prompt":          "Where does Alice live now?\\nA) Austin\\n..."
                              "\\nAnswer strictly with a single letter inside"
                              " XML tags, like <answer>A</answer>."
    },
    {
      "probe_id":             "alice_chen_u1_p2",
      "family":               "stale_premise_rejection",
      "expected_answer_type": "current",
      "question":             "Does Alice still live in Seattle?",
      "shuffled_options":     {"A": "Yes, still in Seattle",
                               "B": "No, now in Austin",
                               "C": "No, actually in Denver",
                               "D": "Between Seattle and Austin"},
      "target_mapping":       {"A": "stale", "B": "current",
                               "C": "distractor", "D": "both"},
      "correct_letter":       "B",
      "full_prompt":          "Does Alice still live in Seattle?\\nA) ..."
    },
    {
      "probe_id":             "alice_chen_u1_p3",
      "family":               "historical_state",
      "expected_answer_type": "stale",
      "question":             "Where did Alice live before her recent move?",
      "shuffled_options":     {"A": "Denver", "B": "Seattle",
                               "C": "Austin", "D": "both Seattle and Austin"},
      "target_mapping":       {"A": "distractor", "B": "stale",
                               "C": "current",    "D": "both"},
      "correct_letter":       "B",
      "full_prompt":          "Where did Alice live before her recent move?\\nA) ..."
    },
    {
      "probe_id":             "alice_chen_u1_p4",
      "family":               "relational_after_update",
      "expected_answer_type": "current",
      "question":             "In which city should Alice search for a local gym?",
      "shuffled_options":     {"A": "Denver", "B": "Seattle",
                               "C": "both Seattle and Austin", "D": "Austin"},
      "target_mapping":       {"A": "distractor", "B": "stale",
                               "C": "both",       "D": "current"},
      "correct_letter":       "D",
      "full_prompt":          "In which city should Alice search for a local gym?\\n..."
    }
  ]
}

Usage
-----
  python3 v2_temporal_benchmark/generate_mcqa_data.py
  python3 v2_temporal_benchmark/generate_mcqa_data.py --n-personas 30 --seed 42
  python3 v2_temporal_benchmark/generate_mcqa_data.py --dry-run   # schema check only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERSONA_BATCH_SIZE = 5   # personas per LLM call (keep prompts manageable)
STABLE_FACTS_MIN  = 8
STABLE_FACTS_MAX  = 12
N_UPDATED_FACTS   = 5    # always exactly 5 per persona

ANSWER_SUFFIX = (
    "\nAnswer strictly with a single letter inside XML tags, "
    "like <answer>A</answer>."
)

LETTERS = ["A", "B", "C", "D"]

# For each probe family: which option type is the correct answer?
EXPECTED = {
    "current_state":          "current",
    "stale_premise_rejection": "current",   # correct = reject stale, give new
    "historical_state":        "stale",     # old value IS the right answer
    "relational_after_update": "current",
}

PROBE_FAMILIES = list(EXPECTED.keys())

# ---------------------------------------------------------------------------
# LLM persona generation
# ---------------------------------------------------------------------------

_SYS = """\
You are a synthetic data generator for a continual-learning memory benchmark.
Generate diverse, realistic fictional personas. Vary names, nationalities,
professions, ages, and life circumstances broadly.
Respond ONLY with a valid JSON object containing a single key "personas"
whose value is an array of persona objects.
"""

_USER_TMPL = """\
Generate exactly {n} new fictional personas. Each persona object must have:

  "persona_id"    : snake_case unique identifier (e.g. "mei_lin")
  "persona_name"  : full name
  "stable_facts"  : list of {smin}–{smax} biographical facts that will NOT change.
                    Each item: {{"fact_id":"<pid>_s<n>","category":str,
                               "predicate":str,"value":str}}
                    Use varied categories: education, hometown, nationality,
                    hobby, personality_trait, food_preference, language_skill,
                    physical_trait, childhood_memory, skill.
  "updated_facts" : list of EXACTLY {nu} facts that HAVE CHANGED (old → new).
                    Each item:
                    {{
                      "fact_id"              : "<pid>_u<n>",
                      "category"             : str,
                         // one of: location, job, relationship_status, pet,
                         //   diet, hobby, residence_type, sport, transport
                      "predicate"            : str,
                         // short active-voice phrase, e.g. "lives in"
                      "old_value"            : str,   // was true before
                      "new_value"            : str,   // is true now
                      "distractor_value"     : str,
                         // plausible-but-wrong (same category, ≠ old/new)
                      "both_label"           : str,
                         // how to express both simultaneously, e.g.
                         //   "both Seattle and Austin" or "both dog and cat"
                      "update_context"       : str,
                         // ONE sentence on why/how it changed
                      "current_state_question"  : str,
                         // MCQA wording for current state,
                         //   e.g. "Where does Alice live now?"
                      "stale_premise_question"  : str,
                         // MCQA wording asserting OLD value,
                         //   e.g. "Does Alice still live in Seattle?"
                      "historical_state_question": str,
                         // MCQA wording for OLD state,
                         //   e.g. "Where did Alice live before her recent move?"
                      "relational_question"     : str,
                         // downstream question requiring NEW value,
                         //   e.g. "In which city should Alice look for a gym?"
                      "training_sentences"      : [str, str, str]
                         // 3 varied declarative sentences stating NEW value
                    }}

Personas to avoid duplicating (ids already used): {existing}

Return JSON: {{"personas": [ ... ]}}
"""


def _call_llm_personas(
    client: OpenAI,
    model: str,
    n: int,
    existing_ids: list[str],
    llm_seed: int,
) -> list[dict]:
    prompt = _USER_TMPL.format(
        n=n,
        smin=STABLE_FACTS_MIN,
        smax=STABLE_FACTS_MAX,
        nu=N_UPDATED_FACTS,
        existing=json.dumps(existing_ids) if existing_ids else "[]",
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.9,
        seed=llm_seed,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYS},
            {"role": "user",   "content": prompt},
        ],
    )
    raw = json.loads(resp.choices[0].message.content)
    if isinstance(raw, list):
        return raw
    for v in raw.values():
        if isinstance(v, list):
            return v
    raise ValueError(f"Unexpected LLM response shape: {list(raw.keys())}")


# ---------------------------------------------------------------------------
# Probe assembly — fully deterministic given the RNG
# ---------------------------------------------------------------------------

def _bare_options(fact: dict) -> dict[str, str]:
    """Return bare value strings for the four option types."""
    return {
        "current":    fact["new_value"],
        "stale":      fact["old_value"],
        "both":       fact["both_label"],
        "distractor": fact["distractor_value"],
    }


def _premise_options(fact: dict) -> dict[str, str]:
    """
    For stale_premise_rejection the options are short answer phrases,
    not bare values, because the question is yes/no style.
    """
    name = fact.get("_persona_name", "They")
    pred = fact["predicate"]
    return {
        "stale":      f"Yes, still {pred} {fact['old_value']}",
        "current":    f"No, now {pred} {fact['new_value']}",
        "both":       f"Between {fact['old_value']} and {fact['new_value']}",
        "distractor": f"No, actually {pred} {fact['distractor_value']}",
    }


def _shuffle_and_map(
    sem_to_text: dict[str, str],
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Shuffle the four semantic types into random letter slots.

    Returns:
      shuffled_options  — {"A": "<text>", ...}
      target_mapping    — {"A": "<semantic_type>", ...}
    """
    items = list(sem_to_text.items())  # [("current", "Austin"), ...]
    rng.shuffle(items)
    shuffled_options = {LETTERS[i]: items[i][1] for i in range(4)}
    target_mapping   = {LETTERS[i]: items[i][0] for i in range(4)}
    return shuffled_options, target_mapping


def _build_full_prompt(question: str, options: dict[str, str]) -> str:
    lines = [question]
    for letter in LETTERS:
        lines.append(f"{letter}) {options[letter]}")
    lines.append(ANSWER_SUFFIX.strip())
    return "\n".join(lines)


def _build_probes(
    fact: dict,
    persona_name: str,
    rng: random.Random,
) -> list[dict]:
    """Build all 4 probe families for one updated_fact."""
    fact["_persona_name"] = persona_name
    probes: list[dict] = []
    bare = _bare_options(fact)
    prem = _premise_options(fact)

    for idx, family in enumerate(PROBE_FAMILIES):
        question = {
            "current_state":           fact["current_state_question"],
            "stale_premise_rejection": fact["stale_premise_question"],
            "historical_state":        fact["historical_state_question"],
            "relational_after_update": fact["relational_question"],
        }[family]

        sem_to_text = prem if family == "stale_premise_rejection" else bare
        opts, mapping = _shuffle_and_map(sem_to_text, rng)

        expected_type  = EXPECTED[family]
        correct_letter = next(L for L, sem in mapping.items() if sem == expected_type)

        probes.append({
            "probe_id":             f"{fact['fact_id']}_p{idx + 1}",
            "family":               family,
            "expected_answer_type": expected_type,
            "question":             question,
            "shuffled_options":     opts,
            "target_mapping":       mapping,
            "correct_letter":       correct_letter,
            "full_prompt":          _build_full_prompt(question, opts),
        })

    del fact["_persona_name"]
    return probes


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def generate(
    n_personas: int,
    seed: int,
    client: OpenAI,
    model: str,
    out_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("[dry-run] Showing schema only — no LLM calls made.")
        _print_schema_example(rng)
        return {}

    # ── Phase 1: Generate personas in batches ──────────────────────────────
    personas: list[dict] = []
    existing_ids: list[str] = []
    batch_n = 0

    while len(personas) < n_personas:
        want = min(PERSONA_BATCH_SIZE, n_personas - len(personas))
        print(f"  [batch {batch_n + 1}] requesting {want} personas …", flush=True)
        try:
            batch = _call_llm_personas(
                client, model, want, existing_ids, llm_seed=seed + batch_n
            )
            batch = batch[:want]
            personas.extend(batch)
            existing_ids.extend(
                p.get("persona_id", f"p{len(existing_ids) + i}")
                for i, p in enumerate(batch)
            )
            batch_n += 1
            print(f"         → {len(personas)}/{n_personas} personas so far", flush=True)
        except Exception as exc:
            print(f"  [warn] batch {batch_n + 1} failed: {exc}  — retrying", flush=True)

    personas = personas[:n_personas]

    personas_path = out_dir / "personas.json"
    with open(personas_path, "w") as f:
        json.dump(personas, f, indent=2)
    print(f"\n  Saved {len(personas)} personas → {personas_path}", flush=True)

    # ── Phase 2: Assemble MCQA probes ─────────────────────────────────────
    entries: list[dict] = []
    n_malformed = 0

    for persona in personas:
        pid   = persona.get("persona_id", "unknown")
        pname = persona.get("persona_name", pid)
        for fact in persona.get("updated_facts", []):
            # Guard: skip facts missing required keys
            required = {
                "fact_id", "old_value", "new_value", "distractor_value",
                "both_label", "current_state_question",
                "stale_premise_question", "historical_state_question",
                "relational_question",
            }
            missing = required - set(fact.keys())
            if missing:
                print(
                    f"  [warn] {pid}/{fact.get('fact_id','?')} missing keys "
                    f"{missing} — skipping",
                    flush=True,
                )
                n_malformed += 1
                continue

            probes = _build_probes(fact, pname, rng)
            entries.append({
                "fact_id":           fact["fact_id"],
                "persona_id":        pid,
                "persona_name":      pname,
                "category":          fact.get("category", ""),
                "predicate":         fact.get("predicate", ""),
                "old_value":         fact["old_value"],
                "new_value":         fact["new_value"],
                "distractor_value":  fact["distractor_value"],
                "both_label":        fact["both_label"],
                "update_context":    fact.get("update_context", ""),
                "training_sentences": fact.get("training_sentences", []),
                "probes":            probes,
            })

    # ── Save benchmark ─────────────────────────────────────────────────────
    output = {
        "meta": {
            "n_personas":       len(personas),
            "n_updated_facts":  len(entries),
            "n_probes_total":   len(entries) * 4,
            "n_malformed_skipped": n_malformed,
            "seed":             seed,
            "probe_families":   PROBE_FAMILIES,
            "expected_answer_types": EXPECTED,
        },
        "entries": entries,
    }

    benchmark_path = out_dir / "benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"\n  Benchmark saved → {benchmark_path}\n"
        f"  {len(entries)} facts × 4 probes = {len(entries) * 4} MCQA items\n"
        f"  {n_malformed} facts skipped (malformed LLM output)",
        flush=True,
    )
    return output


# ---------------------------------------------------------------------------
# Schema preview (--dry-run)
# ---------------------------------------------------------------------------

def _print_schema_example(rng: random.Random) -> None:
    """Print one fully assembled benchmark entry so the schema can be reviewed."""
    example_fact = {
        "fact_id":                    "alice_chen_u1",
        "category":                   "location",
        "predicate":                  "lives in",
        "old_value":                  "Seattle",
        "new_value":                  "Austin",
        "distractor_value":           "Denver",
        "both_label":                 "both Seattle and Austin",
        "update_context":             "Alice relocated for a new tech job opportunity.",
        "current_state_question":     "Where does Alice live now?",
        "stale_premise_question":     "Does Alice still live in Seattle?",
        "historical_state_question":  "Where did Alice live before her recent move?",
        "relational_question":        "In which city should Alice search for a local gym?",
        "training_sentences": [
            "Alice lives in Austin.",
            "Alice has recently moved to Austin.",
            "Alice currently resides in Austin.",
        ],
    }
    probes = _build_probes(example_fact, "Alice Chen", rng)
    entry = {
        "fact_id":           example_fact["fact_id"],
        "persona_id":        "alice_chen",
        "persona_name":      "Alice Chen",
        "category":          example_fact["category"],
        "predicate":         example_fact["predicate"],
        "old_value":         example_fact["old_value"],
        "new_value":         example_fact["new_value"],
        "distractor_value":  example_fact["distractor_value"],
        "both_label":        example_fact["both_label"],
        "update_context":    example_fact["update_context"],
        "training_sentences": example_fact["training_sentences"],
        "probes":            probes,
    }
    print("\n" + "=" * 72)
    print("SCHEMA PREVIEW — one updated_fact with all 4 probes")
    print("=" * 72)
    print(json.dumps(entry, indent=2))
    print("=" * 72)
    print(
        "\nKey fields explained:\n"
        "  training_sentences   — declarative statements of new_value used "
        "in LoRA batches\n"
        "  probe.family         — which of the 4 question types\n"
        "  probe.expected_answer_type — 'current' (3 families) or 'stale' "
        "(historical_state)\n"
        "  probe.shuffled_options     — {letter: display_text}, randomly "
        "shuffled each time\n"
        "  probe.target_mapping       — {letter: semantic_type} for the "
        "evaluator\n"
        "  probe.correct_letter       — ground-truth answer letter\n"
        "  probe.full_prompt          — text injected verbatim into the model\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the TemporalBench v2 MCQA dataset."
    )
    parser.add_argument("--n-personas", type=int, default=30,
                        help="Number of personas to generate (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for shuffle determinism (default: 42)")
    parser.add_argument("--out-dir", default="v2_temporal_benchmark/data",
                        help="Output directory relative to project root")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print schema example without calling the LLM")
    args = parser.parse_args()

    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("ERROR: OPENAI_API_KEY not set in .env")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
    else:
        client = None  # type: ignore[assignment]
        model  = ""

    out_dir = ROOT / args.out_dir
    print(
        f"TemporalBench v2 — generate_mcqa_data.py\n"
        f"  personas : {args.n_personas}\n"
        f"  seed     : {args.seed}\n"
        f"  out_dir  : {out_dir}\n"
        f"  dry_run  : {args.dry_run}\n",
        flush=True,
    )

    generate(
        n_personas=args.n_personas,
        seed=args.seed,
        client=client,
        model=model,
        out_dir=out_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
