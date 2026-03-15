"""
Phase 3 — Extraction Evaluation (Standalone Gatekeeper Script).

Compares data/memories/{persona_id}_memories.jsonl against the ground
truth in data/personas/{persona_id}_ground_truth.json and reports four
metrics required before any training is allowed:

  Recall                  — fraction of GT facts captured at least once
  Precision               — fraction of categorised extractions that are correct
  False Insertion Rate    — wrong-value claims as fraction of all extractions
  Update Linking Accuracy — fraction of is_update=True items that align with
                            a real GT state transition (±1 day window)

Matching strategy: rule-based keyword matching per fact_id.
  • Deterministic (no LLM calls) — ensures reproducibility.
  • Per-fact "match keys": the minimum tokens that must appear for a hit.
  • Two facts requiring compound logic (bob_f008/bob_f009, alice_f006)
    use AND/NOT matching to avoid confusing a living Rex with a dead one
    or "dating Mark" with "broke up with Mark".

GATE: Precision >= 0.50 AND Recall >= 0.60 required to proceed.

Run from repo root:
  python -m src.extractor.eval
  python -m src.extractor.eval --persona alice
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Matching rules
# ---------------------------------------------------------------------------

# Keywords: any one must appear in (predicate + " " + value).lower()
# None  = fact is not extractable from positive dialogue (absence facts).
# "SPECIAL" = handled by item_matches_fact() directly (compound logic).
MATCH_KEYS: dict[str, set[str] | None] = {
    # Alice
    "alice_f001": {"seattle"},
    "alice_f002": {"austin"},
    "alice_f003": {"techcorp"},
    "alice_f004": {"unemployed", "laid off", "layoff"},
    "alice_f005": {"freelance", "consultant"},
    "alice_f006": "SPECIAL",   # dating Mark (not broken up)
    "alice_f007": {"broke up", "breakup", "single", "no longer"},
    "alice_f008": {"jamie"},
    "alice_f009": {"vegetarian"},
    "alice_f010": {"marathon"},
    "alice_f011": {"pottery"},
    # Bob
    "bob_f001":  {"chicago"},
    "bob_f002":  {"lincoln high", "lincoln"},
    "bob_f003":  {"sabbatical"},
    "bob_f004":  {"westside"},
    "bob_f005":  {"cycling", "cyclist", "charity ride"},
    "bob_f006":  {"knee", "injury", "unable to cycle", "can't cycle"},
    "bob_f007":  {"recovered", "healed", "back to cycling", "back on", "first ride"},
    "bob_f008":  "SPECIAL",    # Rex alive
    "bob_f009":  "SPECIAL",    # Rex deceased
    "bob_f010":  {"luna"},
    "bob_f011":  None,         # "no special diet" — absence fact, not extractable
    "bob_f012":  {"fasting", "16:8", "intermittent"},
}

NON_EXTRACTABLE: set[str] = {k for k, v in MATCH_KEYS.items() if v is None}

# Category → keywords used to classify an extracted item
CATEGORY_PATTERNS: dict[str, list[str]] = {
    "pet":          ["rex", "luna", "dog", "cat", "pet", "adopted", "passed away",
                     "grieving", "deceased"],
    "location":     ["seattle", "austin", "chicago", "lives in", "moved to",
                     "relocated"],
    "relationship": ["dating", "boyfriend", "girlfriend", "broke up", "single",
                     "mark", "jamie", "relationship"],
    "diet":         ["vegetarian", "fasting", "16:8", "intermittent", "diet"],
    "sport":        ["cycling", "cyclist", "bike", "knee", "injury", "ride"],
    "hobby":        ["pottery", "marathon", "running"],
    "job":          ["works", "job", "employer", "unemployed", "freelance",
                     "teacher", "consultant", "sabbatical", "laid off", "techcorp",
                     "lincoln", "westside"],
}
CATEGORY_PRIORITY = ["pet", "location", "relationship", "diet", "sport", "hobby", "job"]

_DEATH_WORDS = {"passed away", "died", "deceased", "grieving", "loss", "lost", "gone"}
_DATING_WORDS = {"dating", "relationship", "boyfriend", "girlfriend", "going out"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def item_text(item: dict) -> str:
    pred = item.get("predicate", "") or ""
    val  = item.get("value", "") or ""
    # Guard against extractor returning a nested object for value
    if not isinstance(pred, str):
        pred = json.dumps(pred)
    if not isinstance(val, str):
        val = json.dumps(val)
    return (pred + " " + val).lower()


def item_matches_fact(item: dict, fact_id: str) -> bool:
    """Return True if extracted item corresponds to ground-truth fact_id."""
    if fact_id in NON_EXTRACTABLE:
        return False

    keys = MATCH_KEYS.get(fact_id)
    text = item_text(item)

    if keys == "SPECIAL":
        if fact_id == "alice_f006":   # dating Mark — not broken-up context
            return "mark" in text and any(w in text for w in _DATING_WORDS)
        if fact_id == "bob_f008":     # Rex alive
            return "rex" in text and not any(w in text for w in _DEATH_WORDS)
        if fact_id == "bob_f009":     # Rex deceased
            return "rex" in text and any(w in text for w in _DEATH_WORDS)
        return False

    return bool(keys) and any(k in text for k in keys)


def map_to_category(item: dict) -> str | None:
    """Map extracted item to its most likely GT category (highest-priority match)."""
    text = item_text(item)
    for cat in CATEGORY_PRIORITY:
        if any(kw in text for kw in CATEGORY_PATTERNS[cat]):
            return cat
    return None


def get_active_gt_fact(gt_facts: list[dict], category: str, day: int) -> dict | None:
    """Return the GT fact in `category` that was active on `day`, or None."""
    for fact in gt_facts:
        if fact["category"] != category:
            continue
        start = fact["day_introduced"]
        end = fact.get("day_superseded") or 9999
        if start <= day < end:
            return fact
    return None


def gt_event_categories(gt_events: list[dict], gt_facts: list[dict],
                        day: int, window: int = 1) -> set[str]:
    """Categories that had a real GT transition within `window` days of `day`."""
    fact_cat = {f["fact_id"]: f["category"] for f in gt_facts}
    cats: set[str] = set()
    for ev in gt_events:
        if abs(ev["day"] - day) <= window:
            for fid in ev.get("affected_fact_ids", []):
                if fid in fact_cat:
                    cats.add(fact_cat[fid])
    return cats


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_recall(gt_facts: list[dict], memories: list[dict]) -> tuple[float, list[dict]]:
    """
    For each extractable GT fact, check whether any extracted item matches it
    during the fact's active window. Returns (score, per-fact detail list).
    """
    detail = []
    for fact in gt_facts:
        fid = fact["fact_id"]
        if fid in NON_EXTRACTABLE:
            continue
        start = fact["day_introduced"]
        end = fact.get("day_superseded") or 9999

        window_items = [m for m in memories if start <= m["day"] < end]
        matches = [m for m in window_items if item_matches_fact(m, fid)]
        recalled = bool(matches)
        first = min(matches, key=lambda x: x["day"]) if matches else None

        detail.append({
            "fact_id": fid,
            "category": fact["category"],
            "gt_value": fact["value"],
            "active_window": f"days {start}–{'end' if end == 9999 else end - 1}",
            "is_stable": fact.get("is_stable", False),
            "recalled": recalled,
            "first_match": (
                {"day": first["day"], "predicate": first["predicate"],
                 "value": first["value"], "confidence": first["confidence"]}
                if first else None
            ),
        })

    n_recalled = sum(1 for d in detail if d["recalled"])
    score = n_recalled / len(detail) if detail else 0.0
    return score, detail


def compute_precision_and_fir(
    memories: list[dict], gt_facts: list[dict]
) -> tuple[float, float, list[dict], list[dict]]:
    """
    For each extracted item:
      - Map it to a GT category.
      - Find the active GT fact in that category on item.day.
      - Check if the item's value matches the active GT fact.

    Returns (precision, FIR, precision_detail, false_insertions).

    Definitions:
      TP  item maps to category AND matches active GT fact
      FP  item maps to category BUT value is wrong (stale, hallucinated)
      UNC item maps to no category (noise / banter — excluded from precision)
      Precision = TP / (TP + FP)
      FIR       = FP / total_extracted  (wrong-claim rate across all outputs)
    """
    tp = fp = unc = 0
    false_insertions: list[dict] = []
    detail: list[dict] = []

    for item in memories:
        cat = map_to_category(item)
        if cat is None:
            unc += 1
            detail.append({"item_summary": _item_summary(item), "result": "uncategorised"})
            continue

        active_gt = get_active_gt_fact(gt_facts, cat, item["day"])
        if active_gt and item_matches_fact(item, active_gt["fact_id"]):
            tp += 1
            detail.append({
                "item_summary": _item_summary(item),
                "result": "TP",
                "matched_gt": active_gt["fact_id"],
            })
        else:
            fp += 1
            false_insertions.append(item)
            detail.append({
                "item_summary": _item_summary(item),
                "result": "FP",
                "category": cat,
                "active_gt": active_gt["fact_id"] if active_gt else None,
                "active_gt_value": active_gt["value"] if active_gt else None,
            })

    total = len(memories)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fir = fp / total if total > 0 else 0.0

    return precision, fir, detail, false_insertions


def compute_update_linking(
    memories: list[dict], gt_facts: list[dict], gt_events: list[dict]
) -> tuple[float, dict]:
    """
    For items with is_update=True, check whether a real GT transition occurred
    within ±1 day in the same category. Returns (score, detail_dict).
    """
    update_items = [m for m in memories if m.get("is_update")]
    if not update_items:
        return 1.0, {"n_flagged": 0, "n_correct": 0, "detail": []}

    correct = 0
    update_detail = []

    for item in update_items:
        cat = map_to_category(item)
        event_cats = gt_event_categories(gt_events, gt_facts, item["day"], window=1)
        is_correct = (cat is not None) and (cat in event_cats)
        if is_correct:
            correct += 1
        update_detail.append({
            "day": item["day"],
            "predicate": item["predicate"],
            "value": item["value"],
            "category": cat,
            "has_gt_transition": is_correct,
        })

    score = correct / len(update_items)
    return score, {
        "n_flagged": len(update_items),
        "n_correct": correct,
        "detail": update_detail,
    }


def _item_summary(item: dict) -> str:
    return f"Day {item['day']:02d} | {item['predicate']} {item['value']} (conf={item['confidence']:.2f})"


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

GATE_THRESHOLDS = {"recall": 0.60, "precision": 0.50}

def gate_check(metrics: dict) -> tuple[bool, list[str]]:
    failures = []
    for metric, threshold in GATE_THRESHOLDS.items():
        if metrics[metric] < threshold:
            failures.append(f"{metric} = {metrics[metric]:.2f} < {threshold:.2f} GATE FAIL")
    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Per-persona evaluator
# ---------------------------------------------------------------------------

def evaluate_persona(
    persona_id: str,
    gt: dict,
    memories: list[dict],
) -> dict:
    gt_facts: list[dict] = gt["facts"]
    gt_events: list[dict] = gt["events"]

    recall, recall_detail = compute_recall(gt_facts, memories)
    precision, fir, prec_detail, false_insertions = compute_precision_and_fir(
        memories, gt_facts
    )
    update_score, update_detail = compute_update_linking(memories, gt_facts, gt_events)

    n_extractable = sum(1 for f in gt_facts if f["fact_id"] not in NON_EXTRACTABLE)

    metrics = {
        "recall":                    round(recall, 4),
        "precision":                 round(precision, 4),
        "false_insertion_rate":      round(fir, 4),
        "update_linking_accuracy":   round(update_score, 4),
        "n_gt_facts_total":          len(gt_facts),
        "n_gt_facts_extractable":    n_extractable,
        "n_gt_facts_recalled":       sum(1 for d in recall_detail if d["recalled"]),
        "n_extracted_total":         len(memories),
        "n_categorised":             len(memories) - sum(1 for d in prec_detail if d["result"] == "uncategorised"),
        "n_true_positives":          sum(1 for d in prec_detail if d["result"] == "TP"),
        "n_false_positives":         sum(1 for d in prec_detail if d["result"] == "FP"),
        "n_uncategorised_noise":     sum(1 for d in prec_detail if d["result"] == "uncategorised"),
        "n_updates_flagged":         update_detail["n_flagged"],
        "n_updates_correct":         update_detail["n_correct"],
    }

    passed, gate_failures = gate_check(metrics)

    return {
        "persona_id":       persona_id,
        "persona_name":     gt["name"],
        "evaluated_at":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "gate_passed":      passed,
        "gate_failures":    gate_failures,
        "metrics":          metrics,
        "recall_detail":    recall_detail,
        "false_insertions": [_item_summary(i) for i in false_insertions],
        "update_analysis":  update_detail,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    m = result["metrics"]
    name = result["persona_name"]
    gate = "✓ PASS" if result["gate_passed"] else "✗ FAIL"

    print(f"\n{'='*60}")
    print(f"  {name}  —  Gate: {gate}")
    print(f"{'='*60}")
    print(f"  Recall                 : {m['recall']:.2%}  "
          f"({m['n_gt_facts_recalled']}/{m['n_gt_facts_extractable']} GT facts found)")
    print(f"  Precision              : {m['precision']:.2%}  "
          f"({m['n_true_positives']} TP / {m['n_categorised']} categorised)")
    print(f"  False Insertion Rate   : {m['false_insertion_rate']:.2%}  "
          f"({m['n_false_positives']} wrong-value claims / {m['n_extracted_total']} total)")
    print(f"  Update Linking Accuracy: {m['update_linking_accuracy']:.2%}  "
          f"({m['n_updates_correct']}/{m['n_updates_flagged']} updates aligned to GT events)")
    print(f"  Noise (uncategorised)  : {m['n_uncategorised_noise']} items ignored")

    print(f"\n  Recall detail:")
    for d in result["recall_detail"]:
        icon = "✓" if d["recalled"] else "✗"
        stable = " [stable]" if d["is_stable"] else ""
        match_info = ""
        if d["recalled"] and d["first_match"]:
            fm = d["first_match"]
            match_info = f" → Day {fm['day']}: '{fm['predicate']} {fm['value']}'"
        print(f"    {icon} {d['fact_id']:12s} {d['category']:12s} "
              f"{d['active_window']:12s}{stable}{match_info}")

    if result["gate_failures"]:
        print(f"\n  GATE FAILURES:")
        for f in result["gate_failures"]:
            print(f"    ✗ {f}")

    if result["false_insertions"]:
        print(f"\n  False insertions (wrong-value claims):")
        for fi in result["false_insertions"][:10]:
            print(f"    • {fi}")
        if len(result["false_insertions"]) > 10:
            print(f"    ... and {len(result['false_insertions']) - 10} more")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Extraction Evaluation")
    parser.add_argument("--personas-dir",  default="data/personas")
    parser.add_argument("--memories-dir",  default="data/memories")
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--persona",       default=None,
                        help="Evaluate one persona only (alice or bob)")
    args = parser.parse_args()

    personas_dir = Path(args.personas_dir)
    memories_dir = Path(args.memories_dir)
    results_dir  = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        sim_cfg = load_json(Path("configs/sim_config.json"))
        persona_ids: list[str] = sim_cfg["personas"]
    except Exception:
        persona_ids = ["alice", "bob"]

    if args.persona:
        persona_ids = [args.persona]

    print("\n=== Phase 3: Extraction Evaluation ===")
    print(f"  Gate thresholds: recall ≥ {GATE_THRESHOLDS['recall']:.0%}, "
          f"precision ≥ {GATE_THRESHOLDS['precision']:.0%}")

    all_passed = True

    for pid in persona_ids:
        gt_path  = personas_dir / f"{pid}_ground_truth.json"
        mem_path = memories_dir / f"{pid}_memories.jsonl"

        if not gt_path.exists():
            print(f"\n  ERROR: {gt_path} not found. Run Phase 1 first.")
            sys.exit(1)
        if not mem_path.exists():
            print(f"\n  ERROR: {mem_path} not found. Run Phase 2 first.")
            sys.exit(1)

        gt       = load_json(gt_path)
        memories = load_jsonl(mem_path)

        result = evaluate_persona(pid, gt, memories)
        print_report(result)

        out_path = results_dir / f"extraction_eval_{pid}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n  Saved → {out_path}")

        if not result["gate_passed"]:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("  PHASE 3 GATE: PASSED — proceed to Phase 4 (Salience Scoring)")
    else:
        print("  PHASE 3 GATE: FAILED — fix extraction before training")
        print("  Tip: Check false_insertions in results/extraction_eval_*.json")
    print(f"{'='*60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
