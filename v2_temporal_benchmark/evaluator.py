"""
v2_temporal_benchmark/evaluator.py

Pure-Python MCQA evaluator.  No torch / transformers dependencies — safe to
import locally (Mac) or on the pod.

Answer extraction (two-stage, robust to Llama-3 verbosity)
----------------------------------------------------------
Stage 1 — XML tags:  re.search(r"<answer>\\s*([A-D])\\s*</answer>", ...)
Stage 2 — Fallback:  first standalone A/B/C/D in the text
              (word-boundary match so "Austin" doesn't trigger as "A")
Stage 3 — "invalid"  if neither stage succeeds

Semantic classification
-----------------------
The extracted letter is looked up in probe["target_mapping"] to give the
semantic type: "current" | "stale" | "both" | "distractor".

Output distribution (per probe family AND aggregate)
-----------------------------------------------------
MCQAEvaluator.evaluate() returns a nested dict:

  {
    "<family>": {                   # one entry per probe family + "overall"
      "n_total":     int,
      "n_invalid":   int,
      "pct_current":    float,      # Update Fidelity
      "pct_stale":      float,      # Stale Endorsement Rate
      "pct_both":       float,      # Behavioral Superposition Rate
      "pct_distractor": float,      # Confusion Rate
      "pct_invalid":    float,
    },
    ...
    "overall": { ... }
  }
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_XML_RE   = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
# Word-boundary standalone letter — won't match "Austin" as "A"
_LONE_RE  = re.compile(r"\b([A-D])\b")
# Very last-resort: first occurrence of A/B/C/D anywhere
_BARE_RE  = re.compile(r"[A-D]")

_VALID_TYPES = {"current", "stale", "both", "distractor"}
_FAMILIES    = [
    "current_state",
    "stale_premise_rejection",
    "historical_state",
    "relational_after_update",
]


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------

class MCQAEvaluator:
    """
    Evaluate a batch of MCQA model responses against their probe metadata.

    Usage:
        evaluator = MCQAEvaluator()
        result = evaluator.evaluate(scored_items)
        # scored_items: list of dicts, each with:
        #   "response"       — raw model output string
        #   "probe"          — the probe dict from benchmark.json
    """

    # ------------------------------------------------------------------
    @staticmethod
    def extract_letter(text: str) -> str:
        """
        Extract the chosen letter from raw model output.

        Stages:
          1. <answer>X</answer> XML tag (ideal)
          2. First standalone A/B/C/D (word-boundary regex)
          3. First bare A/B/C/D character (last resort)
          4. "invalid"
        """
        if not text:
            return "invalid"

        # Stage 1 — XML tag
        m = _XML_RE.search(text)
        if m:
            return m.group(1).upper()

        # Stage 2 — standalone letter (word-boundary)
        # Search only the FIRST 200 chars to avoid catching stray letters in long preambles
        head = text[:200]
        m = _LONE_RE.search(head)
        if m:
            return m.group(1).upper()

        # Stage 3 — bare character anywhere in the first 200 chars
        m = _BARE_RE.search(head)
        if m:
            return m.group(0).upper()

        return "invalid"

    # ------------------------------------------------------------------
    @staticmethod
    def classify(letter: str, target_mapping: dict[str, str]) -> str:
        """
        Map extracted letter → semantic type via the probe's target_mapping.
        Returns "invalid" if the letter is not A–D or the mapping is missing.
        """
        if letter == "invalid":
            return "invalid"
        return target_mapping.get(letter, "invalid")

    # ------------------------------------------------------------------
    def evaluate(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Score a batch of model responses and return the full distribution.

        Each item must have:
          item["response"]              — raw model output string
          item["probe"]["family"]       — probe family
          item["probe"]["target_mapping"] — letter → semantic type

        Returns per-family + "overall" distribution dicts.
        """
        # Accumulate counts: {family: {semantic_type: count}}
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for item in items:
            response = item.get("response", "")
            probe    = item["probe"]
            family   = probe.get("family", "unknown")

            letter   = self.extract_letter(response)
            sem_type = self.classify(letter, probe.get("target_mapping", {}))

            counts[family][sem_type] += 1
            counts["overall"][sem_type]  += 1

        # Build distribution dicts
        result: dict[str, dict[str, Any]] = {}
        for key, type_counts in counts.items():
            n_total   = sum(type_counts.values())
            n_invalid = type_counts.get("invalid", 0)
            n_valid   = n_total - n_invalid

            def pct(t: str) -> float:
                if n_valid == 0:
                    return 0.0
                return round(type_counts.get(t, 0) / n_valid * 100, 2)

            result[key] = {
                "n_total":        n_total,
                "n_valid":        n_valid,
                "n_invalid":      n_invalid,
                "pct_current":    pct("current"),     # Update Fidelity
                "pct_stale":      pct("stale"),        # Stale Endorsement
                "pct_both":       pct("both"),         # Behavioral Superposition
                "pct_distractor": pct("distractor"),   # Confusion
                "pct_invalid":    round(n_invalid / n_total * 100, 2)
                                  if n_total > 0 else 0.0,
            }

        # Ensure all families are present (may be 0-count if not in items)
        for family in _FAMILIES + ["overall"]:
            if family not in result:
                result[family] = {
                    "n_total": 0, "n_valid": 0, "n_invalid": 0,
                    "pct_current": 0.0, "pct_stale": 0.0,
                    "pct_both": 0.0, "pct_distractor": 0.0, "pct_invalid": 0.0,
                }

        return result

    # ------------------------------------------------------------------
    def evaluate_and_label(
        self,
        responses: list[str],
        probes: list[dict],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """
        Convenience wrapper: pair raw response strings with their probes,
        evaluate, and return both the per-item labelled list and the aggregate.

        Returns:
          labelled  — list of dicts with "response", "letter", "sem_type", "probe"
          aggregate — distribution dict (same shape as evaluate())
        """
        items = []
        for response, probe in zip(responses, probes):
            letter   = self.extract_letter(response)
            sem_type = self.classify(letter, probe.get("target_mapping", {}))
            items.append({
                "response":  response,
                "letter":    letter,
                "sem_type":  sem_type,
                "probe":     probe,
            })

        aggregate = self.evaluate(items)
        return items, aggregate


# ---------------------------------------------------------------------------
# Pretty-printer for interactive inspection
# ---------------------------------------------------------------------------

def print_distribution(dist: dict[str, dict[str, Any]], label: str = "") -> None:
    """Print a human-readable distribution table."""
    if label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")

    families = _FAMILIES + ["overall"]
    header = f"{'Family':32s} {'cur%':>6} {'stale%':>7} {'both%':>6} {'dist%':>6} {'inv%':>5} {'n':>5}"
    print(header)
    print("─" * len(header))

    for family in families:
        if family not in dist:
            continue
        d = dist[family]
        sep = "═" * len(header) if family == "overall" else ""
        if sep:
            print(sep)
        print(
            f"{family:32s} "
            f"{d['pct_current']:6.1f} "
            f"{d['pct_stale']:7.1f} "
            f"{d['pct_both']:6.1f} "
            f"{d['pct_distractor']:6.1f} "
            f"{d['pct_invalid']:5.1f} "
            f"{d['n_total']:5d}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI — sanity check on benchmark.json with dummy responses
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    benchmark_path = ROOT / "v2_temporal_benchmark" / "data" / "benchmark.json"
    if not benchmark_path.exists():
        print("benchmark.json not found — run generate_mcqa_data.py first")
        sys.exit(1)

    bm = json.load(open(benchmark_path))
    probes = [p for entry in bm["entries"] for p in entry["probes"]]

    # Simulate: half answer correctly (XML tag), half answer randomly
    import random
    rng = random.Random(42)
    responses = []
    for p in probes:
        if rng.random() < 0.5:
            responses.append(f"<answer>{p['correct_letter']}</answer>")
        else:
            responses.append(rng.choice(["A", "B", "C", "D"]))  # bare letter fallback

    evaluator = MCQAEvaluator()
    _, dist = evaluator.evaluate_and_label(responses, probes)
    print_distribution(dist, "Sanity check (50% correct + 50% random bare-letter)")
