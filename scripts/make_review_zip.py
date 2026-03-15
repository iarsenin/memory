#!/usr/bin/env python3
"""
Create a compact review package for LLM progress evaluation.

Includes: all code, configs, README, ground truth, memories, eval probes,
          results, sampled dialogue (configurable), truncated logs.
Excludes: model weights (checkpoints/), secrets (.env), compiled Python,
          raw dialogue beyond --max-dialogue-days, oversized log files.

Output: data/review_package.zip  (already git-ignored via data/)

Run after completing a phase or updating results:
  python3 scripts/make_review_zip.py
  python3 scripts/make_review_zip.py --max-dialogue-days 5
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = REPO_ROOT / "data" / "review_package.zip"

# Paths to include in full (relative to repo root)
FULL_FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
]

FULL_DIRS = [
    "src",
    "configs",
    "scripts",
    "analysis",
    "data/personas",
    "data/memories",
    "data/eval_probes",
    "results",
]

# Patterns that trigger exclusion anywhere in the path
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pyo",
    ".pyd",
    ".git/",
    ".zip",
    "checkpoints",
    "Scratchpad",
    ".DS_Store",
    ".gitkeep",
    ".egg-info",
    "node_modules",
]

# Exact filenames to exclude (basename match)
EXCLUDE_NAMES = {".env", "Icon", "Icon\r"}


def is_excluded(path: Path) -> bool:
    s = str(path)
    if any(pat in s for pat in EXCLUDE_PATTERNS):
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    if path.is_file() and path.stat().st_size == 0 and path.name not in {".gitkeep"}:
        return True  # skip zero-byte non-placeholder files (macOS Icon etc.)
    return False


def write_file(zf: zipfile.ZipFile, src: Path, arcname: str) -> int:
    """Add a single file. Returns byte size added."""
    data = src.read_bytes()
    zf.writestr(arcname, data)
    return len(data)


def write_truncated(zf: zipfile.ZipFile, src: Path, arcname: str, max_lines: int) -> int:
    """Add a text file truncated to last max_lines lines. Returns byte size added."""
    text = src.read_text(errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        header = f"[truncated — showing last {max_lines} of {len(lines)} lines]\n\n"
        text = header + "\n".join(lines[-max_lines:])
    zf.writestr(arcname, text)
    return len(text.encode())


def add_dir(zf: zipfile.ZipFile, directory: Path, stats: dict) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not is_excluded(path):
            arcname = str(path.relative_to(REPO_ROOT))
            sz = write_file(zf, path, arcname)
            stats["files"] += 1
            stats["bytes"] += sz


def add_dialogue(
    zf: zipfile.ZipFile,
    dialogue_dir: Path,
    max_days: int | None,
    stats: dict,
) -> None:
    """Add dialogue JSONL files, optionally capped to first max_days days per persona."""
    if not dialogue_dir.exists():
        return

    for jsonl_path in sorted(dialogue_dir.glob("*.jsonl")):
        if is_excluded(jsonl_path):
            continue

        arcname = str(jsonl_path.relative_to(REPO_ROOT))

        if max_days is None:
            sz = write_file(zf, jsonl_path, arcname)
            stats["files"] += 1
            stats["bytes"] += sz
        else:
            lines = jsonl_path.read_text(errors="replace").splitlines()
            kept = []
            for line in lines:
                try:
                    turn = json.loads(line)
                    if turn.get("day", 0) <= max_days:
                        kept.append(line)
                except json.JSONDecodeError:
                    pass

            total_days = max(
                (json.loads(l).get("day", 0) for l in lines if l.strip()),
                default=0,
            )
            header = (
                f"[dialogue sample: days 1–{max_days} of {total_days} total "
                f"({len(kept)} of {len(lines)} turns)]\n"
            )
            content = header + "\n".join(kept)
            zf.writestr(arcname, content)
            stats["files"] += 1
            stats["bytes"] += len(content.encode())


def add_logs(zf: zipfile.ZipFile, logs_dir: Path, max_lines: int, stats: dict) -> None:
    """Add log files truncated to last max_lines lines."""
    if not logs_dir.exists():
        return
    for path in sorted(logs_dir.rglob("*")):
        if path.is_file() and not is_excluded(path):
            arcname = str(path.relative_to(REPO_ROOT))
            sz = write_truncated(zf, path, arcname, max_lines)
            stats["files"] += 1
            stats["bytes"] += sz


def build_manifest(
    file_list: list[str],
    max_dialogue_days: int | None,
    log_max_lines: int,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "MemLoRA Review Package",
        f"Generated : {now}",
        f"Purpose   : LLM progress evaluation",
        "",
        "=== Included ===",
        "  Code         : src/, scripts/, analysis/, configs/",
        "  Docs         : README.md, requirements.txt, .env.example",
        "  Ground truth : data/personas/   (full)",
        "  Memories     : data/memories/   (full)",
        "  Eval probes  : data/eval_probes/ (full)",
        "  Results      : results/         (full)",
        f"  Dialogue     : data/dialogue/  "
        + (f"days 1–{max_dialogue_days} only" if max_dialogue_days else "full"),
        f"  Logs         : logs/            last {log_max_lines} lines per file",
        "",
        "=== Excluded ===",
        "  checkpoints/     model weights (too large)",
        "  .env             secrets",
        "  __pycache__/     compiled Python",
        "  data/dialogue/   raw turns beyond day limit",
        "",
        "=== File List ===",
    ] + [f"  {f}" for f in sorted(file_list)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LLM review package")
    parser.add_argument(
        "--max-dialogue-days",
        type=int,
        default=None,
        metavar="N",
        help="Cap dialogue to first N days per persona (default: include all)",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=150,
        help="Max lines per log file (default: 150)",
    )
    args = parser.parse_args()

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    stats: dict = {"files": 0, "bytes": 0}

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:

        # Individual files
        for fname in FULL_FILES:
            path = REPO_ROOT / fname
            if path.exists() and not is_excluded(path):
                sz = write_file(zf, path, fname)
                stats["files"] += 1
                stats["bytes"] += sz

        # Full directories
        for dname in FULL_DIRS:
            add_dir(zf, REPO_ROOT / dname, stats)

        # Dialogue (with optional day cap)
        add_dialogue(zf, REPO_ROOT / "data" / "dialogue", args.max_dialogue_days, stats)

        # Logs (truncated)
        add_logs(zf, REPO_ROOT / "logs", args.log_lines, stats)

        # Manifest (written last so it can list all files)
        file_list = zf.namelist()
        manifest = build_manifest(file_list, args.max_dialogue_days, args.log_lines)
        zf.writestr("MANIFEST.txt", manifest)

    zip_kb = OUTPUT_ZIP.stat().st_size / 1024
    raw_kb = stats["bytes"] / 1024

    print(f"Review package: {OUTPUT_ZIP}")
    print(f"  Files included : {stats['files']}")
    print(f"  Raw content    : {raw_kb:.1f} KB")
    print(f"  Zip size       : {zip_kb:.1f} KB")
    if zip_kb > 500:
        print("  WARNING: zip exceeds 500 KB — consider --max-dialogue-days 5")


if __name__ == "__main__":
    main()
