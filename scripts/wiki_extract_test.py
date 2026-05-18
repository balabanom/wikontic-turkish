#!/usr/bin/env python3
"""Extract readable paragraph chunks from a Wikipedia article into a TXT file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wikontic.utils.wiki_extractor import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_TARGET_CHARS,
    extract_wikipedia_chunks,
    format_chunks_as_txt,
)


DEFAULT_OUTPUT = "wiki_extract_output.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a Wikipedia article, remove non-body content, preserve paragraph "
            "boundaries, and write paragraph-based chunks to TXT."
        )
    )
    parser.add_argument("url", help="Wikipedia article URL")
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"TXT output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=DEFAULT_TARGET_CHARS,
        help=f"Preferred chunk size in characters. Default: {DEFAULT_TARGET_CHARS}",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Soft maximum chunk size in characters. Default: {DEFAULT_MAX_CHARS}",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Minimum chunk size to prefer before closing a chunk. Default: {DEFAULT_MIN_CHARS}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds. Default: 20",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = extract_wikipedia_chunks(
            args.url,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
            timeout=args.timeout,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(format_chunks_as_txt(result), encoding="utf-8")
    except Exception as exc:
        print(f"wiki_extract_test.py: error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {result.chunk_count} chunks from "
        f"{result.paragraph_count} paragraphs to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
