from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import requests

from app.core.config import settings


class OpenAIClient:
    def __init__(self) -> None:
        self.base_url = settings.openai_base_url.rstrip("/")
        self.api_key = settings.resolved_api_key
        self.timeout = 60

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def chat_completion(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        candidate_models = self._dedupe(
            [
                settings.openai_chat_model,
                "gpt-4o-mini",
                "gpt-4.1-mini",
                "gpt-4.1-nano",
            ]
        )
        errors: list[str] = []
        for model_name in candidate_models:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.ok:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            errors.append(f"{model_name}: {response.status_code}")

        # Graceful local fallback when key/model access is restricted.
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        context_marker = "Document Context:\n"
        context = ""
        if context_marker in last_user:
            context = last_user.split(context_marker, 1)[1].strip()
        snippet = context[:700].strip()
        if not snippet:
            return (
                "I could not access chat models with the current API key. "
                "Please verify key permissions and selected chat model."
            )
        return (
            "Chat model access is currently restricted for this key, so this is an extractive fallback.\n\n"
            f"{snippet}\n\n"
            "Please enable chat model access for full generative answers."
        )

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @staticmethod
    def _local_embedding(text: str, dim: int = 256) -> list[float]:
        # Deterministic hash-based fallback embedding to keep the app functional
        # when provider embedding access is restricted for a key.
        vector = [0.0] * dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _local_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._local_embedding(text) for text in texts]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        size = min(len(a), len(b))
        return sum(a[i] * b[i] for i in range(size))

    def embed(self, texts: list[str], input_type: str = "query") -> list[list[float]]:
        _ = input_type
        candidate_models = self._dedupe(
            [
                settings.openai_embedding_model,
                "text-embedding-3-large",
                "text-embedding-3-small",
            ]
        )
        errors: list[str] = []
        for model_name in candidate_models:
            payload: dict[str, Any] = {
                "model": model_name,
                "input": texts,
            }
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.ok:
                data = response.json()
                return [row["embedding"] for row in data["data"]]
            errors.append(f"{model_name}: {response.status_code} {response.text[:220]}")

        return self._local_embeddings(texts)

    def rerank(self, query: str, passages: list[str], top_n: int) -> list[dict[str, Any]]:
        rerank_model = settings.openai_rerank_model.strip().lower()
        if rerank_model == "local":
            query_vec = self._local_embedding(query)
            rows = []
            for idx, text in enumerate(passages):
                rows.append({"index": idx, "logit": self._dot(query_vec, self._local_embedding(text))})
            rows.sort(key=lambda item: item["logit"], reverse=True)
            return rows[:top_n]

        model_name = settings.openai_rerank_model
        try:
            query_vec = self.embed([query], input_type="query")[0] if model_name == settings.openai_embedding_model else self._embed_with_model([query], model_name)[0]
            pass_vecs = self.embed(passages, input_type="passage") if model_name == settings.openai_embedding_model else self._embed_with_model(passages, model_name)
            rows = [{"index": idx, "logit": self._dot(query_vec, vec)} for idx, vec in enumerate(pass_vecs)]
            rows.sort(key=lambda item: item["logit"], reverse=True)
            return rows[:top_n]
        except Exception:
            query_vec = self._local_embedding(query)
            rows = []
            for idx, text in enumerate(passages):
                rows.append({"index": idx, "logit": self._dot(query_vec, self._local_embedding(text))})
            rows.sort(key=lambda item: item["logit"], reverse=True)
            return rows[:top_n]

    def _embed_with_model(self, texts: list[str], model_name: str) -> list[list[float]]:
        payload: dict[str, Any] = {"model": model_name, "input": texts}
        response = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=self._headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [row["embedding"] for row in data["data"]]

    def llm_rerank(self, query: str, passages: list[str], top_n: int, max_candidates: int = 30) -> list[dict[str, Any]]:
        candidates = passages[:max_candidates]
        if not candidates:
            return []

        numbered = []
        for i, text in enumerate(candidates):
            snippet = " ".join(text.split())[:420]
            numbered.append(f"[{i}] {snippet}")
        prompt = (
            "Score each passage for relevance to the query from 0 to 100.\n"
            "Output ONLY JSON array of objects: [{\"index\":int,\"score\":number}].\n"
            "No extra text.\n\n"
            f"Query: {query}\n\n"
            "Passages:\n" + "\n".join(numbered)
        )

        payload = {
            "model": settings.openai_rerank_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a deterministic relevance scorer."},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = self._parse_rerank_json(content, len(candidates))
            if parsed:
                parsed.sort(key=lambda x: x["logit"], reverse=True)
                return parsed[:top_n]
        except Exception:
            pass

        # Fallback to embedding similarity rerank.
        return self.rerank(query=query, passages=candidates, top_n=top_n)

    @staticmethod
    def _parse_rerank_json(text: str, n: int) -> list[dict[str, Any]]:
        block = text.strip()
        if "```" in block:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", block, flags=re.DOTALL | re.IGNORECASE)
            if match:
                block = match.group(1).strip()
        try:
            arr = json.loads(block)
            rows: list[dict[str, Any]] = []
            if isinstance(arr, list):
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    idx = int(item.get("index"))
                    score = float(item.get("score"))
                    if 0 <= idx < n:
                        rows.append({"index": idx, "logit": score / 100.0})
                return rows
        except Exception:
            return []
        return []


openai_client = OpenAIClient()
