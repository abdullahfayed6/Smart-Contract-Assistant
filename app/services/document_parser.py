from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import PyPDF2
from docx import Document


def extract_pdf_sections(path: Path) -> List[Tuple[int, str]]:
    reader = PyPDF2.PdfReader(str(path))
    sections: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append((idx, text))
    return sections


def extract_docx_sections(path: Path) -> List[Tuple[int, str]]:
    doc = Document(str(path))
    sections: list[tuple[int, str]] = []
    bucket: list[str] = []
    section_id = 1

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            if bucket:
                sections.append((section_id, "\n".join(bucket)))
                section_id += 1
                bucket = []
            continue
        bucket.append(text)

    if bucket:
        sections.append((section_id, "\n".join(bucket)))

    return sections


def extract_sections(file_path: Path) -> List[Tuple[int, str]]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_sections(file_path)
    if suffix == ".docx":
        return extract_docx_sections(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")

