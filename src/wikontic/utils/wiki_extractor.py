from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: beautifulsoup4. "
        "Install it with: .venv/bin/pip install beautifulsoup4"
    ) from exc


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
class WikiChunk:
    index: int
    paragraphs: list[str]

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "char_count": self.char_count,
            "paragraph_count": self.paragraph_count,
        }


@dataclass(frozen=True)
class WikiExtractionResult:
    url: str
    paragraphs: list[str]
    chunks: list[WikiChunk]

    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def chunk_summaries(self) -> list[dict]:
        return [
            {
                "index": chunk.index,
                "char_count": chunk.char_count,
                "paragraph_count": chunk.paragraph_count,
            }
            for chunk in self.chunks
        ]


def validate_wikipedia_url(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if parsed.scheme not in {"http", "https"} or not host.endswith("wikipedia.org"):
        raise ValueError(f"Expected a wikipedia.org URL, got: {url}")


def fetch_html(url: str, timeout: float = 20.0) -> str:
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
        if (
            node.name in {"h2", "h3"}
            and heading_key(node.get_text(" ", strip=True)) in STOP_HEADINGS
        ):
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
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[WikiChunk]:
    if min_chars > target_chars:
        raise ValueError("--min-chars cannot be greater than --target-chars")
    if target_chars > max_chars:
        raise ValueError("--target-chars cannot be greater than --max-chars")

    chunks: list[WikiChunk] = []
    current: list[str] = []

    def current_len() -> int:
        return len("\n\n".join(current))

    def close_current() -> None:
        if current:
            chunks.append(WikiChunk(index=len(chunks) + 1, paragraphs=list(current)))
            current.clear()

    for paragraph in paragraphs:
        if not current:
            current.append(paragraph)
            if len(paragraph) >= target_chars:
                close_current()
            continue

        candidate_len = current_len() + 2 + len(paragraph)
        should_close_first = (
            current_len() >= min_chars and candidate_len > max_chars
        ) or (
            current_len() >= target_chars and candidate_len > target_chars
        )

        if should_close_first:
            close_current()

        current.append(paragraph)
        if current_len() >= max_chars:
            close_current()

    close_current()
    return chunks


def extract_wikipedia_chunks(
    url: str,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    timeout: float = 20.0,
) -> WikiExtractionResult:
    validate_wikipedia_url(url)
    html = fetch_html(url, timeout=timeout)
    paragraphs = extract_paragraphs(html)
    chunks = chunk_paragraphs(
        paragraphs,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
    )
    return WikiExtractionResult(url=url, paragraphs=paragraphs, chunks=chunks)


def format_chunks_as_txt(result: WikiExtractionResult) -> str:
    lines = [
        f"URL: {result.url}",
        f"Extracted paragraphs: {result.paragraph_count}",
        f"Chunks: {result.chunk_count}",
        "",
    ]

    for chunk in result.chunks:
        lines.extend(
            [
                (
                    f"=== CHUNK {chunk.index:03d} | {chunk.char_count} chars | "
                    f"{chunk.paragraph_count} paragraphs ==="
                ),
                "",
                chunk.text,
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"
