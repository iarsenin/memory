"""
Phase 2 — Post-Extraction Deduplication & Update-Linking.

Problem being solved
---------------------
The LLM extractor (extract.py) sees only a single day's transcript. It has no
memory of what was already extracted on prior days, so it over-flags
is_update=True whenever the user re-states a fact (e.g., "still in Seattle" on
Day 17 triggers is_update, even though Seattle was already recorded on Day 1).

The fix is a deterministic canonicalization pass applied *after* each day's
extraction, using the full accumulated memory history for the persona.

Algorithm (applied per newly-extracted item)
---------------------------------------------
1.  Compute a canonical key: (category, normalised_predicate_stem).
    Two predicates are in the same cluster if they share the same category
    (from _map_category) and their non-stopword tokens overlap > 50%.

2.  Find the most recent prior item in the same cluster.

3.  Compare values using token Jaccard similarity:
    • similarity >= VALUE_SAME_THRESHOLD  (default 0.70):
        → Pure repeat. Strip is_update, clear supersedes_memory_id.
        → Do NOT append to the memory list (suppress the duplicate).
    • similarity < VALUE_SAME_THRESHOLD AND values differ meaningfully:
        → Real update. Force is_update=True, set supersedes_memory_id.
        → Append as a new item (it correctly tracks the state change).
    • No prior item in cluster:
        → New fact. Clear is_update (nothing to supersede), append normally.

This is deliberately conservative: we prefer false negatives (missing an
update link) over false positives (marking a repeat as an update), because
false positives are the dominant failure mode we observed in Phase 3 eval.
"""

from __future__ import annotations

import re
from typing import Optional

# Reuse the same category mapper so that category detection is consistent
# with Phase 4 salience scoring and Phase 5 batch generation.
from ..salience.score import _map_category

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALUE_SAME_THRESHOLD = 0.70     # Jaccard ≥ this → treat as a pure repeat
MIN_VALUE_TOKENS     = 2        # items with fewer content tokens skip Jaccard
                                # (avoid spurious matches on single-word values)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "has", "have",
    "had", "to", "of", "in", "for", "on", "with", "at", "by", "as", "and",
    "but", "or", "not", "no", "its", "my", "his", "her", "their", "our",
    "your", "this", "that", "it", "he", "she", "they", "we", "i",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content_tokens(text: str) -> set[str]:
    """Lowercase content tokens (≥ 3 chars, not stopwords)."""
    return {
        w for w in re.findall(r"[a-z]+", text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _cluster_key(item: dict) -> Optional[str]:
    """
    Return a cluster key for novelty/repeat detection.
    Uses the detected category as the primary discriminant.
    Falls back to a normalised predicate stem when no category is detected.
    """
    pred = (item.get("predicate") or "").lower()
    val  = (item.get("value")     or "").lower()
    cat  = _map_category(pred + " " + val)
    if cat:
        return cat
    # Fallback: use content tokens of the predicate (ignoring value) as stem
    stems = _content_tokens(pred)
    return "_".join(sorted(stems)) if stems else None


def _value_tokens(item: dict) -> set[str]:
    val = item.get("value") or ""
    if not isinstance(val, str):
        import json
        val = json.dumps(val)
    return _content_tokens(val)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def deduplicate_and_link(
    new_items: list[dict],
    prior_items: list[dict],
) -> list[dict]:
    """
    Deduplicate and update-link a list of freshly extracted items against the
    accumulated prior items for the same persona.

    Args:
        new_items:   Items extracted from the current day's dialogue.
        prior_items: All items already stored for this persona (earlier days).

    Returns:
        Filtered/corrected list of new items to actually persist. Items that
        are pure duplicates of prior facts are dropped. Items that represent
        real value changes have is_update=True and supersedes_memory_id set.
        Items with no prior cluster match are returned unchanged (is_update=False).
    """
    # Build a lookup: cluster_key → most recent prior item
    cluster_to_prior: dict[str, dict] = {}
    for item in prior_items:
        key = _cluster_key(item)
        if key:
            # Keep the most recent (highest day number) prior item per cluster
            existing = cluster_to_prior.get(key)
            if existing is None or item["day"] >= existing["day"]:
                cluster_to_prior[key] = item

    result: list[dict] = []

    for item in new_items:
        key = _cluster_key(item)
        prior = cluster_to_prior.get(key) if key else None

        if prior is None:
            # No prior item in this cluster — brand new fact
            out = dict(item)
            out["is_update"] = False
            out["supersedes_memory_id"] = None
            result.append(out)
            # Register this new item as the latest in its cluster for subsequent
            # items within the same day's batch
            if key:
                cluster_to_prior[key] = out
            continue

        # Compare values
        new_toks   = _value_tokens(item)
        prior_toks = _value_tokens(prior)

        if len(new_toks) < MIN_VALUE_TOKENS or len(prior_toks) < MIN_VALUE_TOKENS:
            # Too short to compare reliably — be conservative, keep as new
            out = dict(item)
            out["is_update"] = False
            out["supersedes_memory_id"] = None
            result.append(out)
            if key:
                cluster_to_prior[key] = out
            continue

        sim = _jaccard(new_toks, prior_toks)

        if sim >= VALUE_SAME_THRESHOLD:
            # Pure repeat — suppress (do not add to result)
            continue

        # Meaningful value change — confirm as real update
        out = dict(item)
        out["is_update"] = True
        out["supersedes_memory_id"] = prior["memory_id"]
        result.append(out)
        # Update the cluster register so subsequent items in the same batch
        # chain correctly (e.g., A→B→C on the same day)
        if key:
            cluster_to_prior[key] = out

    return result
