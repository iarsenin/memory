"""
Phase 4 — Salience Scorer.

Computes a composite salience score for each extracted memory item and
optionally filters items below the configured threshold.

Score = temporal_decay(day) × Σ (weight_i × component_i)

Components (all in [0, 1] before weighting):
  novelty         (0.30) — first-seen for this fact cluster; 0 for pure repeats
  recurrence      (0.20) — how many distinct dialogue days mention this fact
  explicit_change (0.25) — config-listed state-transition patterns; is_update is
                            used as a soft secondary signal (0.5) when patterns absent
  fact_density    (0.15) — content-word ratio; prefers specific over vague
  banter_penalty  (-0.10)— presence of low-information filler phrases

Temporal decay:
  decay(day) = exp(-λ × (max_day - day))
  Items from the most recent day decay=1.0; older items decay exponentially.
  λ is taken from salience_config["temporal_decay_lambda"].

Threshold:
  Items with final score < salience_threshold are either kept with a
  filtered=True flag or removed entirely, controlled by the caller.

Design notes:
  • Novelty uses the same category-keyword mapper as Phase 3 so that
    "lives in Seattle" and "moved to Seattle" are treated as the same
    fact cluster.
  • Recurrence counts distinct DAYS in the user's dialogue that mention
    the item's primary tokens — not raw turn count (prevents one very
    chatty day from inflating the score).
  • Banter detection uses both the config patterns and a hard list of
    ultra-short (≤ 3 char) values that are likely artefacts.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Stopwords (minimal, purely to filter noise from fact_density calculation)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "only", "own", "same", "than", "too", "very",
    "just", "that", "this", "these", "those", "he", "she", "it", "they",
    "i", "my", "his", "her", "its", "our", "your", "their", "me", "him",
    "us", "them", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "all", "each", "every", "few", "more", "most", "other",
    "some", "such", "up", "out", "about", "now", "then", "here", "there",
    "again", "further", "once", "s", "t", "true", "false",
}

# Common predicate verbs excluded ONLY from novelty comparison — they appear
# in nearly every memory item ("lives in X", "enjoys Y", "moved to Z") and
# would falsely create token-overlap between unrelated facts.
# NOT removed from fact_density or recurrence calculations.
_PREDICATE_STOPWORDS = {
    "lives", "works", "enjoys", "follows", "trains", "makes", "takes",
    "runs", "moved", "moving", "started", "starting", "dating", "dated",
    "recently", "currently", "previously",
}

# ---------------------------------------------------------------------------
# Category → keyword map (same taxonomy as Phase 3 eval)
# ---------------------------------------------------------------------------
_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "pet":          ["dog", "cat", "pet", "adopted", "passed away", "grieving", "deceased",
                     "hamster", "parrot", "rabbit", "bird", "shelter", "rescue",
                     # named pets across all 10 personas
                     "rex", "luna", "mochi", "pepper", "cooper", "pi", "pixel", "mango", "rio", "max"],
    "location":     ["lives in", "moved to", "relocated", "moving to",
                     # cities across all 10 personas
                     "seattle", "austin", "chicago", "denver", "boston", "san francisco",
                     "san jose", "portland", "miami", "new york", "brooklyn", "nyc"],
    "relationship": ["dating", "boyfriend", "girlfriend", "broke up", "single",
                     "relationship", "partner", "started dating", "is seeing"],
    "diet":         ["vegetarian", "vegan", "pescatarian", "fasting", "16:8",
                     "intermittent", "diet", "plant-based"],
    "sport":        ["cycling", "cyclist", "bike", "knee", "injury", "ride",
                     "marathon", "running", "triathlon", "golf", "sailing", "yoga",
                     "5k", "swim", "swimming"],
    "hobby":        ["pottery", "guitar", "dancing", "cooking", "meditation",
                     "meditates", "reading", "history", "sailing"],
    "job":          ["works as", "job", "employer", "unemployed", "freelance",
                     "teacher", "consultant", "sabbatical", "laid off", "retired",
                     "retirement", "nurse", "journalist", "designer", "developer",
                     "barista", "instructor", "architect", "advisor", "researcher"],
}
_CATEGORY_PRIORITY = ["pet", "location", "relationship", "diet", "sport", "hobby", "job"]


def _map_category(text: str) -> str | None:
    for cat in _CATEGORY_PRIORITY:
        if any(kw in text for kw in _CATEGORY_PATTERNS[cat]):
            return cat
    return None


def _item_text(item: dict) -> str:
    pred = item.get("predicate") or ""
    val  = item.get("value") or ""
    if not isinstance(pred, str):
        pred = json.dumps(pred)
    if not isinstance(val, str):
        val = json.dumps(val)
    return (pred + " " + val).lower()


def _primary_tokens(text: str) -> set[str]:
    """Content tokens for recurrence counting and density calculation.
    Uses ≥ 3 chars (captures rex, pet, cat, dog, run) and basic stopwords only."""
    return {w for w in re.findall(r"[a-z]+", text)
            if len(w) >= 3 and w not in _STOPWORDS}


def _novelty_tokens(text: str) -> set[str]:
    """Content tokens used exclusively for novelty overlap comparison.
    Strips common predicate verbs ('lives', 'enjoys', 'moved', …) that appear
    in virtually every memory item and would falsely signal token overlap between
    completely different facts (e.g., 'lives in Seattle' vs 'lives in Chicago')."""
    return {w for w in re.findall(r"[a-z]+", text)
            if len(w) >= 3
            and w not in _STOPWORDS
            and w not in _PREDICATE_STOPWORDS}


# ---------------------------------------------------------------------------
# Component calculators
# ---------------------------------------------------------------------------

def _novelty(item: dict, earlier_items: list[dict]) -> float:
    """
    1.0 if no earlier item shares the same category cluster.
    0.5 if same category was seen but different value keywords.
    0.0 if an earlier item has substantially the same entity/topic tokens
        (pure repeat — predicate verbs excluded from comparison).

    Uses _novelty_tokens() which strips common predicate verbs so that
    "lives in Seattle" and "lives in Chicago" don't look like duplicates
    because of the shared "lives" word.
    """
    text = _item_text(item)
    cat  = _map_category(text)
    toks = _novelty_tokens(text)

    for prev in earlier_items:
        prev_text = _item_text(prev)
        prev_cat  = _map_category(prev_text)
        prev_toks = _novelty_tokens(prev_text)

        if cat is not None and cat == prev_cat:
            # Same category: check entity/topic token overlap
            overlap = toks & prev_toks
            if overlap and len(overlap) / max(1, len(toks)) >= 0.5:
                return 0.0   # substantial overlap → pure repeat
            return 0.5       # same category, different value → partial novelty
    return 1.0               # no prior item in this category


def _recurrence(item: dict, day_texts: dict[int, str]) -> float:
    """
    Fraction of dialogue days that mention the item's primary tokens.
    Capped at 1.0; uses a 3-day threshold for full score (≥3 → 1.0).
    """
    text = _item_text(item)
    toks = _primary_tokens(text)
    if not toks:
        return 0.0

    # Count distinct days where at least one token appears in user utterances
    hit_days = sum(
        1 for day_text in day_texts.values()
        if any(t in day_text for t in toks)
    )
    # Normalise: 1 day → 0.33, 2 days → 0.67, 3+ days → 1.0
    return min(1.0, hit_days / 3.0)


def _explicit_change(item: dict, patterns: list[str]) -> float:
    """
    1.0 if a config state-transition pattern appears in predicate+value.
    0.5 if is_update=True but no explicit pattern (noisy extractor flag).
    0.0 otherwise.
    This intentionally demotes over-flagged is_update items that lack
    linguistic evidence of a real change.
    """
    text = _item_text(item)
    if any(p in text for p in patterns):
        return 1.0
    if item.get("is_update"):
        return 0.5
    return 0.0


def _fact_density(item: dict, banter_words: set[str]) -> float:
    """
    Ratio of meaningful content words to total words.
    Penalises single-word values and artifact items like 'true' / 'false'.
    """
    text = _item_text(item)
    words = text.split()
    if not words:
        return 0.0

    content = [
        w for w in words
        if w not in _STOPWORDS
        and w not in banter_words
        and w not in {"true", "false", "null"}
        and len(w) >= 3
    ]
    # Scale so that 5+ content words → 1.0
    return min(1.0, len(content) / 5.0)


def _banter(item: dict, banter_words: set[str]) -> float:
    """
    Fraction of words in predicate+value that are banter/filler phrases.
    Checks only against the configured banter word list — short words are
    already penalised through fact_density and should not be double-counted.
    Returns a value in [0, 1]; multiplied by the negative weight later.
    """
    text = _item_text(item)
    words = text.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in banter_words)
    return min(1.0, hits / max(1, len(words)))


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------

def _decay(day: int, max_day: int, lam: float) -> float:
    return math.exp(-lam * (max_day - day))


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class SalienceScorer:
    """
    Computes and attaches salience_score to every memory item for one persona.

    Usage:
        scorer = SalienceScorer(config)
        scored_items = scorer.score_all(items, day_texts)
        kept, filtered = scorer.apply_threshold(scored_items)
    """

    def __init__(self, config: dict) -> None:
        w = config["weights"]
        self.w_novelty    = float(w["novelty"])
        self.w_recur      = float(w["recurrence"])
        self.w_explicit   = float(w["explicit_change"])
        self.w_density    = float(w["fact_density"])
        self.w_banter     = float(w["banter_penalty"])   # negative value
        self.lam          = float(config["temporal_decay_lambda"])
        self.threshold    = float(config["salience_threshold"])
        self.ec_patterns  = [p.lower() for p in config["explicit_change_patterns"]]
        self.banter_words = set(p.lower() for p in config["banter_patterns"])

    # ------------------------------------------------------------------
    def score_item(
        self,
        item: dict,
        earlier_items: list[dict],
        day_texts: dict[int, str],
        max_day: int,
    ) -> dict[str, float]:
        """
        Return dict of component scores for one item.

        salience_score  — raw quality score (sum of 5 weighted components).
                          This is what gets stored in the memories JSONL and
                          used for threshold filtering. It is day-agnostic so
                          that important early events (breakup Day 5, layoff
                          Day 7) are not penalised vs Day 20 items.

        temporal_decay  — exp(-λ × (max_day − day)).  Stored as a separate
                          field; Phase 5 multiplies salience × decay when
                          building the weighted replay buffer so that recent
                          memories are sampled more frequently.

        Architecture rationale: applying λ=0.1 over 20 days as a multiplier
        on the quality score would reduce Day 1 items to ≤15 % of their raw
        score, causing all early state-change events to fall below any
        reasonable threshold regardless of importance.  Separating quality
        (salience_score) from recency (temporal_decay) gives Phase 5 both
        signals independently.
        """
        nov  = _novelty(item, earlier_items)
        rec  = _recurrence(item, day_texts)
        exp  = _explicit_change(item, self.ec_patterns)
        den  = _fact_density(item, self.banter_words)
        ban  = _banter(item, self.banter_words)
        dec  = _decay(item["day"], max_day, self.lam)

        raw = (
            self.w_novelty    * nov
            + self.w_recur    * rec
            + self.w_explicit * exp
            + self.w_density  * den
            + self.w_banter   * ban   # w_banter is negative
        )
        raw = max(0.0, raw)

        return {
            "novelty":          round(nov, 3),
            "recurrence":       round(rec, 3),
            "explicit_change":  round(exp, 3),
            "fact_density":     round(den, 3),
            "banter":           round(ban, 3),
            "temporal_decay":   round(dec, 3),
            "salience_score":   round(raw, 4),   # quality score, no decay applied
        }

    # ------------------------------------------------------------------
    def score_all(
        self,
        items: list[dict],
        day_texts: dict[int, str],
    ) -> list[dict]:
        """
        Score every item, passing the earlier items (same persona, earlier day)
        as context for novelty calculation. Returns new list with salience_score
        populated and a _score_detail sub-dict added for inspection.
        Items are processed in day order.
        """
        sorted_items = sorted(items, key=lambda x: (x["day"], x.get("memory_id", "")))
        max_day = max(x["day"] for x in sorted_items) if sorted_items else 1

        result: list[dict] = []
        seen_so_far: list[dict] = []

        for item in sorted_items:
            scores = self.score_item(item, seen_so_far, day_texts, max_day)
            out = dict(item)
            out["salience_score"]  = scores["salience_score"]
            out["temporal_decay"]  = scores["temporal_decay"]   # Phase 5 replay weight
            out["_score_detail"]   = scores
            result.append(out)
            seen_so_far.append(item)

        return result

    # ------------------------------------------------------------------
    def apply_threshold(
        self,
        scored_items: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Split items into kept (≥ threshold) and filtered (< threshold).
        Strips the _score_detail field from kept items before returning
        (it is written to a separate debug file by the runner).
        """
        kept     = []
        filtered = []
        for item in scored_items:
            if item["salience_score"] >= self.threshold:
                # Strip debug field; keep salience_score + temporal_decay for Phase 5
                out = {k: v for k, v in item.items() if k != "_score_detail"}
                kept.append(out)
            else:
                filtered.append(item)
        return kept, filtered


# ---------------------------------------------------------------------------
# Dialogue loader helper
# ---------------------------------------------------------------------------

def load_day_texts(dialogue_path: Path) -> dict[int, str]:
    """
    Return {day: concatenated_user_utterances} for recurrence counting.
    Only 'user' speaker turns are included — we want what the persona said.
    """
    day_texts: dict[int, list[str]] = defaultdict(list)
    for line in dialogue_path.read_text().splitlines():
        if not line.strip():
            continue
        turn = json.loads(line)
        if turn.get("speaker") == "user":
            day_texts[turn["day"]].append(turn["utterance"].lower())
    return {day: " ".join(utts) for day, utts in day_texts.items()}
