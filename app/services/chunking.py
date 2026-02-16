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

    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(1, max_chars - overlap_chars)

    while start < len(clean):
        end = min(len(clean), start + max_chars)
        body = clean[start:end].strip()
        if body:
            chunk_id = f"{prefix}s{section_id}_c{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, section_id=section_id, text=body))
            idx += 1
        if end == len(clean):
            break
        start += step

    return chunks


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

