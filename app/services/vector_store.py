from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings


@dataclass
class IndexedChunk:
    doc_id: str
    chunk_id: str
    section_id: int
    text: str
    score: float = 0.0


class ChromaStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        model_tag = self._safe_tag(settings.openai_embedding_model)
        self.collection_name = f"smart_contracts_{model_tag}"
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.meta_file = Path(settings.processed_dir) / f"documents_{model_tag}.json"
        if not self.meta_file.exists():
            self.meta_file.write_text("{}", encoding="utf-8")

    @staticmethod
    def _safe_tag(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()

    def _read_meta(self) -> dict[str, Any]:
        try:
            raw = self.meta_file.read_text(encoding="utf-8").strip()
            if not raw:
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            # Recover gracefully from a partially written/corrupted metadata file.
            return {}

    def _write_meta(self, data: dict[str, Any]) -> None:
        self.meta_file.parent.mkdir(parents=True, exist_ok=True)
        self.meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_document_meta(
        self, doc_id: str, filename: str, pages_or_sections: int, chunks_indexed: int
    ) -> dict[str, Any]:
        meta = self._read_meta()
        payload = {
            "doc_id": doc_id,
            "filename": filename,
            "pages_or_sections": pages_or_sections,
            "chunks_indexed": chunks_indexed,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        meta[doc_id] = payload
        self._write_meta(meta)
        return payload

    def get_document_meta(self, doc_id: str) -> dict[str, Any]:
        meta = self._read_meta()
        if doc_id not in meta:
            raise KeyError(f"Document '{doc_id}' not found")
        return meta[doc_id]

    def list_document_meta(self) -> list[dict[str, Any]]:
        meta = self._read_meta()
        rows = list(meta.values())
        rows.sort(key=lambda x: x.get("indexed_at", ""), reverse=True)
        return rows

    def delete_document(self, doc_id: str) -> bool:
        meta = self._read_meta()
        existed = doc_id in meta
        if existed:
            meta.pop(doc_id, None)
            self._write_meta(meta)
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception:
            pass
        return existed

    def reset_all_documents(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self._write_meta({})

    def add_chunks(self, chunks: list[IndexedChunk], embeddings: list[list[float]]) -> None:
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"doc_id": c.doc_id, "section_id": c.section_id} for c in chunks]
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def similarity_search(self, query_embedding: list[float], doc_id: str, top_k: int) -> list[IndexedChunk]:
        max_doc_rows = self.document_chunk_count(doc_id)
        if max_doc_rows <= 0:
            return []
        safe_top_k = min(max(1, top_k), max_doc_rows)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_top_k,
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        rows: list[IndexedChunk] = []
        for idx, text in enumerate(docs):
            rows.append(
                IndexedChunk(
                    doc_id=doc_id,
                    chunk_id=ids[idx],
                    section_id=int(metas[idx]["section_id"]),
                    text=text,
                    score=max(0.0, 1.0 - float(distances[idx])),
                )
            )
        return rows

    def document_chunk_count(self, doc_id: str) -> int:
        records = self.collection.get(where={"doc_id": doc_id}, include=[])
        ids = records.get("ids", [])
        return len(ids)

    def keyword_search(self, query: str, doc_id: str, top_k: int) -> list[IndexedChunk]:
        records = self.collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        docs = records.get("documents", [])
        metas = records.get("metadatas", [])
        ids = records.get("ids", [])
        if not docs:
            return []

        tokenized_docs = [self._tokenize(text) for text in docs]
        query_terms = self._tokenize(query)
        scores = self._bm25_scores(tokenized_docs, query_terms)
        if not scores:
            return []
        max_score = max(scores) or 1.0

        indexed = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            indexed.append(
                IndexedChunk(
                    doc_id=doc_id,
                    chunk_id=ids[i],
                    section_id=int(metas[i]["section_id"]),
                    text=docs[i],
                    score=float(score / max_score),
                )
            )
        indexed.sort(key=lambda x: x.score, reverse=True)
        return indexed[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _bm25_scores(tokenized_docs: list[list[str]], query_terms: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
        if not tokenized_docs or not query_terms:
            return [0.0 for _ in tokenized_docs]

        n_docs = len(tokenized_docs)
        avgdl = sum(len(doc) for doc in tokenized_docs) / max(1, n_docs)

        df: dict[str, int] = {}
        for doc in tokenized_docs:
            seen = set(doc)
            for term in seen:
                df[term] = df.get(term, 0) + 1

        idf: dict[str, float] = {}
        for term in set(query_terms):
            freq = df.get(term, 0)
            idf[term] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

        scores: list[float] = []
        for doc in tokenized_docs:
            dl = len(doc)
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
                score += idf.get(term, 0.0) * ((freq * (k1 + 1)) / max(1e-9, denom))
            scores.append(score)
        return scores


store = ChromaStore()
