from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass
class Chunk:
    chunk_id: str
    section_id: int
    text: str


def chunk_text(
    text: str,
    section_id: int,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    prefix: str = "",
) -> Iterable[Chunk]:
    if _should_use_list_chunking(text):
        return _chunk_by_lines(text=text, section_id=section_id, prefix=prefix, lines_per_chunk=60, overlap_lines=10)

    if not text or not text.strip():
        return []

    paragraphs = _semantic_paragraphs(text)
    if not paragraphs:
        return []

    return _chunk_by_paragraphs(
        paragraphs=paragraphs,
        section_id=section_id,
        prefix=prefix,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def _should_use_list_chunking(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 8:
        return False

    head = " ".join(lines[:6]).lower()
    heading_hint = any(k in head for k in ("restricted key words", "prohibited terms", "appendix", "keywords"))

    # Use a normal string so \u2022 is interpreted as the bullet glyph.
    bullet_re = re.compile("^(\\d+[\\.\\)]|[-*\u2022])\\s+")
    bullet_count = sum(1 for ln in lines if bullet_re.match(ln))
    bullet_ratio = bullet_count / max(1, len(lines))

    avg_len = sum(len(ln) for ln in lines) / max(1, len(lines))
    high_line_density = len(lines) >= 80 and avg_len < 95
    repeated_bullets = len(lines) >= 40 and bullet_ratio >= 0.2

    return heading_hint or repeated_bullets or high_line_density


def _chunk_by_lines(
    text: str,
    section_id: int,
    prefix: str,
    lines_per_chunk: int,
    overlap_lines: int,
) -> list[Chunk]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    chunks: list[Chunk] = []
    step = max(1, lines_per_chunk - overlap_lines)
    idx = 0
    start = 0
    while start < len(lines):
        end = min(len(lines), start + lines_per_chunk)
        body = "\n".join(lines[start:end]).strip()
        if body:
            chunk_id = f"{prefix}s{section_id}_c{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, section_id=section_id, text=body))
            idx += 1
        if end == len(lines):
            break
        start += step
    return chunks


def _semantic_paragraphs(text: str) -> list[str]:
    # Preserve structure by chunking on paragraph boundaries first.
    blocks = re.split(r"\n\s*\n+", text)
    paragraphs: list[str] = []
    for block in blocks:
        normalized = "\n".join(line.strip() for line in block.splitlines() if line.strip())
        if normalized:
            paragraphs.append(normalized)
    return paragraphs


def _chunk_by_paragraphs(
    paragraphs: list[str],
    section_id: int,
    prefix: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    current = ""

    def flush() -> None:
        nonlocal idx, current
        body = current.strip()
        if not body:
            return
        chunk_id = f"{prefix}s{section_id}_c{idx}"
        chunks.append(Chunk(chunk_id=chunk_id, section_id=section_id, text=body))
        idx += 1
        current = body[-overlap_chars:].strip() if overlap_chars > 0 else ""

    for paragraph in paragraphs:
        units = _split_large_paragraph(paragraph, max_chars=max_chars)
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if len(candidate) <= max_chars:
                current = candidate
                continue

            flush()

            if len(unit) <= max_chars:
                current = unit
                continue

            # Final fallback for paragraphs with no suitable sentence boundaries.
            start = 0
            step = max(1, max_chars - overlap_chars)
            while start < len(unit):
                end = min(len(unit), start + max_chars)
                body = unit[start:end].strip()
                if body:
                    chunk_id = f"{prefix}s{section_id}_c{idx}"
                    chunks.append(Chunk(chunk_id=chunk_id, section_id=section_id, text=body))
                    idx += 1
                if end == len(unit):
                    break
                start += step
            current = ""

    if current.strip():
        chunk_id = f"{prefix}s{section_id}_c{idx}"
        chunks.append(Chunk(chunk_id=chunk_id, section_id=section_id, text=current.strip()))

    return chunks


def _split_large_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    units: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            units.append(current)
        current = sentence
    if current:
        units.append(current)
    return units or [paragraph]
