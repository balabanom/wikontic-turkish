#!/usr/bin/env python3
"""Extract readable paragraph chunks from a Wikipedia article into a TXT file."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: beautifulsoup4\n"
        "Install it with: .venv/bin/pip install beautifulsoup4"
    ) from exc


DEFAULT_OUTPUT = "wiki_extract_output.txt"
DEFAULT_TARGET_CHARS = 1_500
DEFAULT_MAX_CHARS = 2_200
DEFAULT_MIN_CHARS = 500

USER_AGENT = (
    "WikonticWikipediaExtractor/0.1 "
    "(local test script; contact: local-dev@example.invalid)"
)

DROP_SELECTORS = [
    "table",
    "style",
    "script",
    "noscript",
    "figure",
    "img",
    "sup.reference",
    "span.mw-editsection",
    "span.reference",
    "div.reflist",
    "div.navbox",
    "div.sidebar",
    "div.infobox",
    "div.metadata",
    "div.hatnote",
    "div.thumb",
    "div.gallery",
    "div.vertical-navbox",
    "div.shortdescription",
    "div.sistersitebox",
    "div#toc",
]

STOP_HEADINGS = {
    "kaynakça",
    "kaynaklar",
    "notlar",
    "dipnotlar",
    "dış bağlantılar",
    "ayrıca bakınız",
    "bibliyografya",
    "external links",
    "references",
    "notes",
    "see also",
}

SKIP_TEXT_PATTERNS = [
    re.compile(r"Şampiyonluk yaşadığı ve/veya", re.IGNORECASE),
    re.compile(r"İkincilik yaşadığı .* ve/veya", re.IGNORECASE),
    re.compile(r"Üçüncülük yaşadığı sezon", re.IGNORECASE),
]


@dataclass(frozen=True)
class Chunk:
    index: int
    paragraphs: list[str]

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def char_count(self) -> int:
        return len(self.text)


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


def validate_wikipedia_url(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if parsed.scheme not in {"http", "https"} or not host.endswith("wikipedia.org"):
        raise ValueError(f"Expected a wikipedia.org URL, got: {url}")


def fetch_html(url: str, timeout: float) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def normalize_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ")
    return text.strip()


def should_skip_paragraph(text: str) -> bool:
    return any(pattern.search(text) for pattern in SKIP_TEXT_PATTERNS)


def heading_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"\s*\[değiştir \| kaynağı değiştir\]\s*$", "", text)
    return text.strip(" :")


def find_article_body(soup: BeautifulSoup):
    for body in soup.select("div.mw-parser-output"):
        if body.find("p") is not None:
            return body

    for selector in ("div#mw-content-text", "main#content"):
        body = soup.select_one(selector)
        if body is not None and body.find("p") is not None:
            return body

    raise ValueError("Could not find Wikipedia article body in the page HTML.")


def remove_unwanted_nodes(body) -> None:
    for selector in DROP_SELECTORS:
        for node in body.select(selector):
            node.decompose()


def extract_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    body = find_article_body(soup)
    remove_unwanted_nodes(body)

    paragraphs: list[str] = []

    for node in body.find_all(["h2", "h3", "p"], recursive=True):
        if node.name in {"h2", "h3"} and heading_key(node.get_text(" ", strip=True)) in STOP_HEADINGS:
            break

        if node.name != "p":
            continue

        if node.find_parent(["table", "figure"]):
            continue

        text = normalize_text(node.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        if should_skip_paragraph(text):
            continue
        if not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]", text):
            continue

        paragraphs.append(text)

    if not paragraphs:
        raise ValueError("No readable paragraphs were extracted from the article body.")

    return paragraphs


def chunk_paragraphs(
    paragraphs: list[str],
    target_chars: int,
    max_chars: int,
    min_chars: int,
) -> list[Chunk]:
    if min_chars > target_chars:
        raise ValueError("--min-chars cannot be greater than --target-chars")
    if target_chars > max_chars:
        raise ValueError("--target-chars cannot be greater than --max-chars")

    chunks: list[Chunk] = []
    current: list[str] = []

    def current_len() -> int:
        return len("\n\n".join(current))

    def close_current() -> None:
        if current:
            chunks.append(Chunk(index=len(chunks) + 1, paragraphs=list(current)))
            current.clear()

    for paragraph in paragraphs:
        if not current:
            current.append(paragraph)
            if len(paragraph) >= target_chars:
                close_current()
            continue

        candidate_len = current_len() + 2 + len(paragraph)
        should_close_first = (
            current_len() >= min_chars
            and candidate_len > max_chars
        ) or (
            current_len() >= target_chars
            and candidate_len > target_chars
        )

        if should_close_first:
            close_current()

        current.append(paragraph)
        if current_len() >= max_chars:
            close_current()

    close_current()
    return chunks


def format_output(url: str, paragraphs: list[str], chunks: list[Chunk]) -> str:
    lines = [
        f"URL: {url}",
        f"Extracted paragraphs: {len(paragraphs)}",
        f"Chunks: {len(chunks)}",
        "",
    ]

    for chunk in chunks:
        lines.extend(
            [
                (
                    f"=== CHUNK {chunk.index:03d} | {chunk.char_count} chars | "
                    f"{len(chunk.paragraphs)} paragraphs ==="
                ),
                "",
                chunk.text,
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()

    try:
        validate_wikipedia_url(args.url)
        html = fetch_html(args.url, timeout=args.timeout)
        paragraphs = extract_paragraphs(html)
        chunks = chunk_paragraphs(
            paragraphs,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
        )
        output = format_output(args.url, paragraphs, chunks)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    except Exception as exc:
        print(f"wiki_extract_test.py: error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(chunks)} chunks from {len(paragraphs)} paragraphs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
