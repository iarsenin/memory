#!/usr/bin/env python3
"""
Create a single JSON review package for LLM progress evaluation.

Directory structure is preserved as JSON hierarchy. Code files are strings,
JSON/JSONL files are parsed objects/arrays — the LLM can navigate it directly
without any unpacking.

Output: data/review_package.json  (git-ignored via data/)

Run after completing any phase or updating results:
  python3 scripts/make_review_package.py
  python3 scripts/make_review_package.py --max-dialogue-days 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "review_package.json"

# What to include (relative to repo root)
INCLUDE_FILES = ["README.md", "requirements.txt", ".env.example"]
INCLUDE_DIRS  = ["src", "configs", "scripts", "analysis"]
DATA_DIRS     = ["data/personas", "data/memories", "data/eval_probes", "results"]
DIALOGUE_DIR  = REPO_ROOT / "data" / "dialogue"
LOGS_DIR      = REPO_ROOT / "logs"
LOG_MAX_LINES = 150

EXCLUDE_NAMES    = {".env", "Icon", "Icon\r", ".DS_Store", ".gitkeep"}
EXCLUDE_PATTERNS = ["__pycache__", ".pyc", ".pyo", "checkpoints",
                    ".git/", ".zip", "Scratchpad", ".egg-info"]


def skip(path: Path) -> bool:
    if path.name in EXCLUDE_NAMES:
        return True
    s = str(path)
    return any(p in s for p in EXCLUDE_PATTERNS)


def read_file(path: Path) -> object:
    """Return file contents as a parsed object (JSON/JSONL) or a string."""
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            pass
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append(line)
        return rows
    return path.read_text(errors="replace")


def build_tree(directory: Path) -> dict:
    """Recursively build a {name: content} dict for a directory."""
    node: dict = {}
    if not directory.exists():
        return node
    for path in sorted(directory.iterdir()):
        if skip(path):
            continue
        if path.is_dir():
            subtree = build_tree(path)
            if subtree:
                node[path.name] = subtree
        elif path.is_file():
            node[path.name] = read_file(path)
    return node


def sample_dialogue(max_days: int | None) -> dict:
    """Load dialogue JSONL files, optionally capped to first max_days days."""
    node: dict = {}
    if not DIALOGUE_DIR.exists():
        return node
    for path in sorted(DIALOGUE_DIR.glob("*.jsonl")):
        if skip(path):
            continue
        rows = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                turn = json.loads(line)
                if max_days is None or turn.get("day", 0) <= max_days:
                    rows.append(turn)
            except json.JSONDecodeError:
                pass
        total = sum(1 for l in path.read_text().splitlines() if l.strip())
        meta = {"_note": f"{len(rows)} of {total} turns" +
                (f" (days 1–{max_days})" if max_days and len(rows) < total else " (all)")}
        node[path.name] = [meta] + rows
    return node


def sample_logs() -> dict:
    """Include last LOG_MAX_LINES lines of each log file."""
    node: dict = {}
    if not LOGS_DIR.exists():
        return node
    for path in sorted(LOGS_DIR.rglob("*")):
        if not path.is_file() or skip(path):
            continue
        lines = path.read_text(errors="replace").splitlines()
        if len(lines) > LOG_MAX_LINES:
            truncated = (f"[truncated — last {LOG_MAX_LINES} of {len(lines)} lines]\n"
                         + "\n".join(lines[-LOG_MAX_LINES:]))
            content: object = truncated
        else:
            content = read_file(path)
        rel = str(path.relative_to(LOGS_DIR))
        node[rel] = content
    return node


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dialogue-days", type=int, default=None, metavar="N",
                        help="Cap dialogue to first N days per persona (default: all)")
    args = parser.parse_args()

    package: dict = {
        "_meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "purpose": "MemLoRA — LLM progress review package",
            "structure": "Directories are nested objects. Code = strings. JSON = objects. JSONL = arrays.",
            "excluded": ["checkpoints/ (model weights)", ".env (secrets)", "__pycache__/"],
        }
    }

    # Top-level files
    for fname in INCLUDE_FILES:
        p = REPO_ROOT / fname
        if p.exists() and not skip(p):
            package[fname] = read_file(p)

    # Source dirs
    for dname in INCLUDE_DIRS:
        d = REPO_ROOT / dname
        if d.exists():
            package[dname] = build_tree(d)

    # Data artifacts
    data_node: dict = {}
    for dname in DATA_DIRS:
        d = REPO_ROOT / dname
        label = Path(dname).name
        data_node[label] = build_tree(d)

    data_node["dialogue"] = sample_dialogue(args.max_dialogue_days)
    package["data"] = data_node

    # Logs
    logs = sample_logs()
    if logs:
        package["logs"] = logs

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(package, indent=2, ensure_ascii=False)
    OUTPUT.write_text(text, encoding="utf-8")

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Review package : {OUTPUT}")
    print(f"Size           : {size_kb:.1f} KB (~{int(size_kb*250):,} tokens est.)")
    if size_kb > 400:
        print("WARNING: large — consider --max-dialogue-days 5 to reduce size")


if __name__ == "__main__":
    main()
