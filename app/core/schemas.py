from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    page: Optional[int] = None
    quote: str


class GuardrailStatus(BaseModel):
    grounded: bool
    reason: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    guardrail_status: GuardrailStatus
    latency_ms: int
    retrieved_chunks: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int
    indexed_at: datetime


class DocumentStatus(BaseModel):
    doc_id: str
    filename: str
    pages_or_sections: int
    chunks_indexed: int
    indexed_at: datetime
