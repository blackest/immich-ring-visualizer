#!/usr/bin/env python3
"""
classify_images.py

Walks a directory of images, sends each one to a local Ollama vision model,
gets back a category verdict, then renames (with a category prefix) and
moves the file into a matching subfolder.

Example:
    imagex.png  -->  verdict: pg  -->  moved to  ./pg/pg-imagex.png

Usage:
    python classify_images.py /path/to/images --model gemma4-vision

Requires:
    pip install requests
"""

import argparse
import base64
import csv
import json
import mimetypes
import shutil
import sys
from datetime import datetime
from pathlib import Path

import requests

# ---- Config you may want to tweak -----------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4-vision"  # change to whatever tag you pull into Ollama

CATEGORIES = ["pg", "adult", "reject"]

PROMPT = f"""You are classifying an image into exactly one category.

Categories:
- pg: safe for general audiences, no nudity or sexual content, no obvious
  anatomical errors (extra/missing limbs, distorted hands or faces, etc.)
- adult: contains nudity or sexual content
- reject: contains clear anatomy/quality defects (extra fingers, warped
  limbs, distorted faces, malformed bodies) regardless of content type

Respond with ONLY one word: pg, adult, or reject. No punctuation, no
explanation, nothing else.
"""

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# -----------------------------------------------------------------------------


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_image(path: Path, model: str, ollama_url: str, timeout: int) -> str:
    """Send one image to the Ollama vision model and return its verdict."""
    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [encode_image(path)],
        "stream": False,
        "options": {"temperature": 0},
    }

    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "").strip().lower()

    # Be forgiving about extra words/punctuation the model might add
    for cat in CATEGORIES:
        if cat in raw:
            return cat

    return "unclassified"


def unique_destination(dest: Path) -> Path:
    """Avoid clobbering an existing file by appending a counter."""
    if not dest.exists():
        return dest
    stem, suffix, parent = dest.stem, dest.suffix, dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main():
    parser = argparse.ArgumentParser(description="Classify and sort images via a local Ollama vision model.")
    parser.add_argument("directory", type=Path, help="Directory of images to classify")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama /api/generate URL")
    parser.add_argument("--timeout", type=int, default=120, help="Per-image request timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Classify and log only, don't move/rename files")
    parser.add_argument("--log", type=Path, default=None, help="CSV log path (default: <directory>/classification_log.csv)")
    args = parser.parse_args()

    src_dir = args.directory
    if not src_dir.is_dir():
        print(f"Error: {src_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    log_path = args.log or (src_dir / "classification_log.csv")

    images = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("No image files found.")
        return

    print(f"Found {len(images)} image(s) in {src_dir}")
    print(f"Model: {args.model}  |  Ollama URL: {args.url}")
    if args.dry_run:
        print("DRY RUN — no files will be moved.\n")

    # Pre-create category folders (skip in dry-run)
    if not args.dry_run:
        for cat in CATEGORIES + ["unclassified"]:
            (src_dir / cat).mkdir(exist_ok=True)

    results = []
    counts = {cat: 0 for cat in CATEGORIES + ["unclassified", "error"]}

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name} ... ", end="", flush=True)
        try:
            verdict = classify_image(img_path, args.model, args.url, args.timeout)
        except Exception as e:
            print(f"ERROR ({e})")
            counts["error"] += 1
            results.append([img_path.name, "error", str(e), ""])
            continue

        counts[verdict] += 1
        dest_name = f"{verdict}-{img_path.name}"
        dest_path = src_dir / verdict / dest_name

        if args.dry_run:
            print(f"{verdict}  (dry run, not moved)")
            results.append([img_path.name, verdict, "", str(dest_path)])
        else:
            dest_path = unique_destination(dest_path)
            shutil.move(str(img_path), str(dest_path))
            print(f"{verdict}  -> {dest_path.relative_to(src_dir)}")
            results.append([img_path.name, verdict, "", str(dest_path)])

    # Write CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_filename", "verdict", "error", "destination"])
        writer.writerows(results)

    print("\n--- Summary ---")
    for cat, n in counts.items():
        if n:
            print(f"{cat}: {n}")
    print(f"\nLog written to: {log_path}")

    if counts.get("adult", 0) or counts.get("reject", 0):
        print(f"\n⚠ Flagged content: {counts.get('adult', 0)} adult, {counts.get('reject', 0)} reject — check the '{args.dry_run and '(dry run)' or 'adult/reject'}' folder(s) before this reaches anywhere auto-indexed (e.g. Immich).")


if __name__ == "__main__":
    main()
