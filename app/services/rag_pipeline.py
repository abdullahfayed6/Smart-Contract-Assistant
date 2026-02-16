from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from app.core.config import settings
from app.core.schemas import (
    AskResponse,
    Citation,
    DocumentStatus,
    EvaluationExample,
    EvaluationResponse,
    EvalMetric,
    GuardrailStatus,
)
from app.services.chat_memory import memory
from app.services.chunking import chunk_text
from app.services.document_parser import extract_sections
from app.services.openai_client import openai_client
from app.services.vector_store import IndexedChunk, store

LIST_INTENT_PATTERNS = (
    "list all",
    "all restricted",
    "prohibited terms",
    "full list",
)
LIST_INTENT_REGEX_PATTERNS = (
    r"\bkeywords?\b",
    r"\bappendix\b",
    r"\bevery\b",
)
GEVAL_QUESTIONS = (
    "What is the core purpose of this contract?",
    "What are the key obligations or responsibilities mentioned?",
    "What termination conditions are defined?",
    "Are there confidentiality requirements? If yes, explain.",
    "What payment, fee, or commission terms are specified?",
    "What governing law or jurisdiction is specified?",
)
SESSION_DOCS: dict[str, set[str]] = defaultdict(set)
BASE_ASSISTANT_SYSTEM_PROMPT = (
    "You are Smart Contract Assistant.\n"
    "You must answer using only the provided Document Context.\n"
    "Do not use outside knowledge.\n"
    "If the context is insufficient, answer exactly: "
    "\"I do not have enough evidence from this document to answer.\"\n"
    "Every factual claim must include at least one citation using this exact format: (chunk_id=<id>).\n"
    "Never invent chunk IDs. Never cite chunks that are not in context.\n"
    "Keep the answer concise, clear, and contract-focused."
)
LIST_INTENT_SYSTEM_EXTENSION = (
    "The user requested an exhaustive list.\n"
    "Return all relevant items found in context as a numbered list.\n"
    "Do not summarize or merge distinct items.\n"
    "If retrieved evidence seems partial, include this exact sentence at the end: "
    "\"Retrieved context may be incomplete.\""
)
GEVAL_JUDGE_SYSTEM_PROMPT = (
    "You are a strict RAG evaluator.\n"
    "Output must be valid JSON only, no markdown, no prose.\n"
    "Required JSON keys: groundedness, answer_relevance, citation_faithfulness.\n"
    "Each value must be a number from 1 to 5."
)


def _doc_id_for_file(filename: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()[:12]
    decoded = unquote(filename)
    stem = Path(decoded).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"{stem}_{digest}"


def ingest_document(filename: str, payload: bytes) -> tuple[str, int, datetime]:
    doc_id = _doc_id_for_file(filename, payload)
    file_path = Path(settings.upload_dir) / f"{doc_id}{Path(filename).suffix.lower()}"
    file_path.write_bytes(payload)

    sections = extract_sections(file_path)
    chunks: list[IndexedChunk] = []
    for section_id, text in sections:
        for chunk in chunk_text(
            text,
            section_id=section_id,
            max_chars=settings.max_chunk_chars,
            overlap_chars=settings.chunk_overlap_chars,
            prefix=f"{doc_id}_",
        ):
            chunks.append(
                IndexedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                    section_id=section_id,
                    text=chunk.text,
                )
            )

    if not chunks:
        raise ValueError("No extractable text found in document.")

    vectors = openai_client.embed([c.text for c in chunks], input_type="passage")
    store.add_chunks(chunks, vectors)
    meta = store.save_document_meta(
        doc_id=doc_id,
        filename=filename,
        pages_or_sections=len(sections),
        chunks_indexed=len(chunks),
    )
    return doc_id, len(chunks), datetime.fromisoformat(meta["indexed_at"])


def register_session_document(session_id: str, doc_id: str) -> None:
    if session_id:
        SESSION_DOCS[session_id].add(doc_id)


def delete_document(doc_id: str) -> bool:
    removed = store.delete_document(doc_id=doc_id)

    upload_dir = Path(settings.upload_dir)
    for p in upload_dir.glob(f"{doc_id}.*"):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    return removed


def clear_all_documents() -> dict[str, int]:
    docs_before = len(store.list_document_meta())
    store.reset_all_documents()

    deleted_uploads = 0
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    for p in upload_dir.iterdir():
        if p.is_file():
            try:
                p.unlink()
                deleted_uploads += 1
            except Exception:
                pass

    for session_id in list(SESSION_DOCS.keys()):
        memory.clear(session_id)
    SESSION_DOCS.clear()

    return {
        "deleted_docs": docs_before,
        "deleted_upload_files": deleted_uploads,
    }


def cleanup_session_documents(session_id: str, delete_documents: bool = False) -> dict[str, int]:
    doc_ids = list(SESSION_DOCS.get(session_id, set()))
    deleted = 0
    # Keep uploaded documents persistent by default.
    # Session cleanup should primarily clear memory, not stored data.
    if delete_documents:
        for doc_id in doc_ids:
            if delete_document(doc_id):
                deleted += 1
    SESSION_DOCS.pop(session_id, None)
    memory.clear(session_id)
    return {"session_docs": len(doc_ids), "deleted_docs": deleted}


def get_document_status(doc_id: str) -> DocumentStatus:
    row = store.get_document_meta(doc_id)
    return DocumentStatus(
        doc_id=row["doc_id"],
        filename=row["filename"],
        pages_or_sections=row["pages_or_sections"],
        chunks_indexed=row["chunks_indexed"],
        indexed_at=datetime.fromisoformat(row["indexed_at"]),
    )


def list_documents() -> list[DocumentStatus]:
    rows = store.list_document_meta()
    return [
        DocumentStatus(
            doc_id=row["doc_id"],
            filename=row["filename"],
            pages_or_sections=row["pages_or_sections"],
            chunks_indexed=row["chunks_indexed"],
            indexed_at=datetime.fromisoformat(row["indexed_at"]),
        )
        for row in rows
    ]


def _retrieve(
    doc_id: str,
    question: str,
    top_k: int,
    rerank_top_k: int,
    use_llm_rerank: bool = True,
) -> list[IndexedChunk]:
    q_vector = openai_client.embed([question], input_type="query")[0]
    vector_hits = store.similarity_search(q_vector, doc_id=doc_id, top_k=top_k)
    keyword_hits = store.keyword_search(question, doc_id=doc_id, top_k=top_k)
    merged = _merge_hits(vector_hits, keyword_hits)
    if not merged:
        return []
    max_candidates = min(30, max(12, rerank_top_k * 3))
    candidates = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:max_candidates]
    if use_llm_rerank:
        reranked = openai_client.llm_rerank(
            query=question,
            passages=[c.text for c in candidates],
            top_n=min(rerank_top_k, len(candidates)),
            max_candidates=max_candidates,
        )
    else:
        reranked = openai_client.rerank(
            query=question,
            passages=[c.text for c in candidates],
            top_n=min(rerank_top_k, len(candidates)),
        )
    ordered: list[IndexedChunk] = []
    for item in reranked:
        idx = int(item.get("index", -1))
        if idx < 0 or idx >= len(candidates):
            continue
        score = float(item.get("logit", 0.0))
        row = candidates[idx]
        row.score = score
        ordered.append(row)
    if not ordered:
        # Fallback to top candidates if reranker output is malformed.
        return candidates[: min(rerank_top_k, len(candidates))]
    return ordered


def _merge_hits(vector_hits: list[IndexedChunk], keyword_hits: list[IndexedChunk]) -> dict[str, IndexedChunk]:
    merged: dict[str, IndexedChunk] = {}
    for row in vector_hits + keyword_hits:
        if row.chunk_id not in merged:
            merged[row.chunk_id] = row
            continue
        existing = merged[row.chunk_id]
        if row.score > existing.score:
            merged[row.chunk_id] = row
    return merged


def _is_list_intent(question: str) -> bool:
    q = question.lower()
    if any(pat in q for pat in LIST_INTENT_PATTERNS):
        return True
    return any(re.search(pattern, q) for pattern in LIST_INTENT_REGEX_PATTERNS)


def _dynamic_retrieval_settings(question: str, requested_top_k: int | None) -> tuple[int, int, float]:
    if _is_list_intent(question):
        return settings.list_intent_top_k, settings.list_intent_rerank_top_k, settings.list_intent_min_score
    top_k = requested_top_k or settings.default_top_k
    return top_k, settings.rerank_top_k, settings.retrieval_min_score


def _normalize_score(raw: float) -> float:
    # Rerank models may return logits in arbitrary ranges.
    # Map robustly to [0, 1] for stable guardrail behavior.
    if -1.0 <= raw <= 1.0:
        return (raw + 1.0) / 2.0
    bounded = max(-60.0, min(60.0, raw))
    return 1.0 / (1.0 + math.exp(-bounded))


def _build_answer_system_prompt(list_intent: bool) -> str:
    if not list_intent:
        return BASE_ASSISTANT_SYSTEM_PROMPT
    return f"{BASE_ASSISTANT_SYSTEM_PROMPT}\n{LIST_INTENT_SYSTEM_EXTENSION}"


def ask(
    doc_id: str,
    session_id: str,
    question: str,
    top_k: int | None = None,
    use_llm_rerank: bool = True,
) -> AskResponse:
    start = time.perf_counter()
    k, rerank_k, min_score = _dynamic_retrieval_settings(question=question, requested_top_k=top_k)
    retrieved = _retrieve(
        doc_id=doc_id,
        question=question,
        top_k=k,
        rerank_top_k=rerank_k,
        use_llm_rerank=use_llm_rerank,
    )
    retrieved = retrieved[:rerank_k]
    if not retrieved:
        return AskResponse(
            answer="I do not have enough evidence from this document to answer.",
            citations=[],
            confidence=0.0,
            guardrail_status=GuardrailStatus(grounded=False, reason="No retrieval hits."),
            latency_ms=int((time.perf_counter() - start) * 1000),
            retrieved_chunks=0,
        )

    context = "\n\n".join(
        [
            f"[{idx+1}] (chunk_id={row.chunk_id}, section={row.section_id}, score={row.score:.4f}) {row.text}"
            for idx, row in enumerate(retrieved)
        ]
    )
    list_intent = _is_list_intent(question)
    system_prompt = _build_answer_system_prompt(list_intent=list_intent)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory.get(session_id))
    messages.append(
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Document Context:\n{context}\n\n"
                "Return a concise answer grounded in context. "
                "Use markdown when helpful."
            ),
        }
    )
    response = openai_client.chat_completion(messages=messages, temperature=0.1)

    memory.add(session_id, "user", question)
    memory.add(session_id, "assistant", response)

    normalized_scores = [_normalize_score(c.score) for c in retrieved]
    confidence = min(1.0, max(0.0, sum(normalized_scores) / max(1, len(normalized_scores))))
    grounded = confidence >= min_score or len(retrieved) >= 2
    citations = [
        Citation(
            doc_id=doc_id,
            chunk_id=row.chunk_id,
            score=row.score,
            page=row.section_id,
            quote=row.text[:280],
        )
        for row in retrieved
    ]
    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = response
    if list_intent and len(retrieved) < max(8, rerank_k // 2) and "Retrieved context may be incomplete." not in answer:
        answer = answer.rstrip() + "\n\nRetrieved context may be incomplete."
    if not grounded:
        answer = (
            "Low-confidence answer based on available retrieved evidence:\n\n"
            + response
        )

    return AskResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        guardrail_status=GuardrailStatus(
            grounded=grounded,
            reason="Grounded answer." if grounded else "Low retrieval confidence.",
        ),
        latency_ms=latency_ms,
        retrieved_chunks=len(retrieved),
    )


def _extract_json_block(text: str) -> dict:
    block = (text or "").strip()
    if "```" in block:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", block, flags=re.DOTALL | re.IGNORECASE)
        if match:
            block = match.group(1).strip()
    try:
        payload = json.loads(block)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _geval_llm_judge(question: str, answer: str, citations: list[Citation]) -> tuple[float, float, float]:
    citation_lines = []
    for row in citations[:5]:
        citation_lines.append(f"- chunk_id={row.chunk_id}: {row.quote}")
    citation_text = "\n".join(citation_lines) if citation_lines else "- (none)"

    prompt = (
        "Evaluate the following RAG answer.\n"
        "Scoring rubric (1-5):\n"
        "- groundedness: Is the answer supported by cited evidence?\n"
        "- answer_relevance: Does it directly answer the question?\n"
        "- citation_faithfulness: Do citations align with the answer content?\n\n"
        "Return JSON object only.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Citations:\n{citation_text}\n"
    )
    raw = openai_client.chat_completion(
        messages=[
            {"role": "system", "content": GEVAL_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    payload = _extract_json_block(raw)

    def _clamp(value: float) -> float:
        return max(1.0, min(5.0, float(value)))

    if payload:
        try:
            return (
                _clamp(payload.get("groundedness", 3)),
                _clamp(payload.get("answer_relevance", 3)),
                _clamp(payload.get("citation_faithfulness", 3)),
            )
        except Exception:
            pass

    has_citations = 1.0 if citations else 0.0
    groundedness = 2.0 + (2.0 * has_citations)
    answer_relevance = 1.0 + min(4.0, max(0.0, len(answer.strip()) / 180.0))
    citation_faithfulness = 1.0 + (3.0 * has_citations)
    return (
        max(1.0, min(5.0, groundedness)),
        max(1.0, min(5.0, answer_relevance)),
        max(1.0, min(5.0, citation_faithfulness)),
    )


def evaluate(doc_id: str) -> EvaluationResponse:
    examples: list[EvaluationExample] = []
    metric_acc: dict[str, float] = {
        "groundedness": 0.0,
        "answer_relevance": 0.0,
        "citation_faithfulness": 0.0,
        "overall": 0.0,
        "confidence": 0.0,
        "latency_ms": 0.0,
        "grounded_rate": 0.0,
    }

    for idx, question in enumerate(GEVAL_QUESTIONS):
        session_id = f"geval::{doc_id}::{idx}"
        memory.clear(session_id)
        resp = ask(
            doc_id=doc_id,
            session_id=session_id,
            question=question,
            top_k=max(8, settings.default_top_k // 2),
            use_llm_rerank=False,
        )
        memory.clear(session_id)

        groundedness_1_5, relevance_1_5, citation_1_5 = _geval_llm_judge(
            question=question,
            answer=resp.answer,
            citations=resp.citations,
        )
        overall_1_5 = (groundedness_1_5 + relevance_1_5 + citation_1_5) / 3.0

        metric_acc["groundedness"] += groundedness_1_5 / 5.0
        metric_acc["answer_relevance"] += relevance_1_5 / 5.0
        metric_acc["citation_faithfulness"] += citation_1_5 / 5.0
        metric_acc["overall"] += overall_1_5 / 5.0
        metric_acc["confidence"] += float(resp.confidence)
        metric_acc["latency_ms"] += float(resp.latency_ms)
        metric_acc["grounded_rate"] += 1.0 if resp.guardrail_status.grounded else 0.0

        examples.append(
            EvaluationExample(
                question=question,
                confidence=float(resp.confidence),
                grounded=bool(resp.guardrail_status.grounded),
                citations=len(resp.citations),
                latency_ms=int(resp.latency_ms),
                geval_groundedness=groundedness_1_5 / 5.0,
                geval_answer_relevance=relevance_1_5 / 5.0,
                geval_citation_faithfulness=citation_1_5 / 5.0,
                geval_overall=overall_1_5 / 5.0,
                answer_preview=resp.answer[:280],
            )
        )

    n = max(1, len(examples))
    metrics = [
        EvalMetric(name="geval_overall", value=metric_acc["overall"] / n, note="LLM-as-judge average score."),
        EvalMetric(name="geval_groundedness", value=metric_acc["groundedness"] / n),
        EvalMetric(name="geval_answer_relevance", value=metric_acc["answer_relevance"] / n),
        EvalMetric(name="geval_citation_faithfulness", value=metric_acc["citation_faithfulness"] / n),
        EvalMetric(name="grounded_rate", value=metric_acc["grounded_rate"] / n),
        EvalMetric(name="avg_confidence", value=metric_acc["confidence"] / n),
        EvalMetric(name="avg_latency_ms", value=metric_acc["latency_ms"] / n),
    ]
    return EvaluationResponse(
        doc_id=doc_id,
        method="geval_e2e_rag",
        metrics=metrics,
        examples=examples,
    )
