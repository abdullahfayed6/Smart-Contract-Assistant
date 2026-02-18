# Smart Contract Assistant (FastAPI + Chat UI)

Contract Q&A application with Retrieval-Augmented Generation (RAG):
- Upload PDF/DOCX contracts
- Index contract text into Chroma
- Chat with grounded answers and chunk citations
- Run E2E RAG evaluation (G-Eval) for the selected document
- Manage stored documents from the web UI

## Tech Stack
- Backend: FastAPI
- Frontend: FastAPI-served HTML/CSS/JavaScript (Bootstrap)
- Vector DB: ChromaDB
- LLM provider: OpenAI API
  - Chat model: `OPENAI_CHAT_MODEL`
  - Embedding model: `OPENAI_EMBEDDING_MODEL`
  - Rerank model: `OPENAI_RERANK_MODEL`

## Project Structure
```text
app/
  api/main.py
  core/config.py
  core/schemas.py
  services/
    chat_memory.py
    chunking.py
    document_parser.py
    openai_client.py
    rag_pipeline.py
    vector_store.py
data/
  chroma/
  processed/
  uploads/
static/
  css/chatbot.css
  js/chatbot.js
templates/
  chatbot.html
run_api.py
requirements.txt
```

## Setup
1. Create and activate virtual env:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
copy .env.example .env
```
Set `OPENAI_API_KEY` in `.env`.

4. Run API + web UI:
```bash
python run_api.py
```

5. Open browser:
- `http://127.0.0.1:8000/`

## How It Works
1. Document upload
- UI sends file to `POST /api/documents/upload`.
- Backend validates file type and reads bytes.
- A stable `doc_id` is generated from filename + content hash.

2. Parsing and chunking
- `document_parser.py` extracts contract text sections.
- `chunking.py` creates chunks:
  - normal sections: semantic paragraph-first chunking with sentence-aware splitting
  - overlap retention across chunks for context continuity
  - list-like sections (appendix/keyword-heavy): line-based chunking

3. Embedding and indexing
- `openai_client.py` builds embeddings.
- `vector_store.py` stores vectors and metadata in Chroma.

4. Chat pipeline
- UI sends question to `POST /api/chat/{session_id}/{doc_id}`.
- `rag_pipeline.py` does:
  - list-intent detection
  - hybrid retrieval (vector + keyword BM25)
  - Reciprocal Rank Fusion (RRF) to combine vector + keyword rankings
  - rerank (LLM rerank by default, embedding rerank fallback)
  - score-threshold filtering before answer generation
  - citation-aware ordering (chunks explicitly cited by model are prioritized)
  - grounded answer generation with citations

5. Guardrails
- Response includes confidence and grounded status.
- Low-confidence answers are flagged in output.
- If no retrieval hits exist, the assistant returns:
  - `I do not have enough evidence from this document to answer.`

6. E2E GEval pipeline
- UI button calls `POST /api/evaluate/{doc_id}`.
- Backend runs a fixed prompt set through full RAG (`ask(...)`).
- LLM-as-judge scores each answer on:
  - groundedness
  - answer relevance
  - citation faithfulness
- Returns aggregate metrics + per-example records.

## Full API Schema

### `AskRequest`
```json
{
  "question": "string (min length: 3)",
  "top_k": 20
}
```

### `Citation`
```json
{
  "doc_id": "string",
  "chunk_id": "string",
  "score": 0.0,
  "page": 1,
  "quote": "string"
}
```

### `GuardrailStatus`
```json
{
  "grounded": true,
  "reason": "string"
}
```

### `AskResponse`
```json
{
  "answer": "string",
  "citations": [
    {
      "doc_id": "string",
      "chunk_id": "string",
      "score": 0.0,
      "page": 1,
      "quote": "string"
    }
  ],
  "confidence": 0.0,
  "guardrail_status": {
    "grounded": true,
    "reason": "string"
  },
  "latency_ms": 0,
  "retrieved_chunks": 0
}
```

### `UploadResponse`
```json
{
  "doc_id": "string",
  "filename": "string",
  "chunks_indexed": 0,
  "indexed_at": "2026-02-16T12:00:00"
}
```

### `DocumentStatus`
```json
{
  "doc_id": "string",
  "filename": "string",
  "pages_or_sections": 0,
  "chunks_indexed": 0,
  "indexed_at": "2026-02-16T12:00:00"
}
```

### `EvalMetric`
```json
{
  "name": "geval_overall",
  "value": 0.81,
  "note": "optional string"
}
```

### `EvaluationExample`
```json
{
  "question": "string",
  "confidence": 0.72,
  "grounded": true,
  "citations": 4,
  "latency_ms": 1100,
  "geval_groundedness": 0.8,
  "geval_answer_relevance": 0.8,
  "geval_citation_faithfulness": 1.0,
  "geval_overall": 0.866,
  "answer_preview": "string"
}
```

### `EvaluationResponse`
```json
{
  "doc_id": "string",
  "method": "geval_e2e_rag",
  "metrics": [
    {
      "name": "geval_overall",
      "value": 0.81
    }
  ],
  "examples": [
    {
      "question": "string",
      "confidence": 0.72,
      "grounded": true,
      "citations": 4,
      "latency_ms": 1100,
      "geval_groundedness": 0.8,
      "geval_answer_relevance": 0.8,
      "geval_citation_faithfulness": 1.0,
      "geval_overall": 0.866,
      "answer_preview": "string"
    }
  ]
}
```

## Endpoints
- `GET /api/health`
- `POST /api/documents/upload`
- `GET /api/documents/{doc_id}`
- `GET /api/documents`
- `POST /api/documents/clear`
- `POST /api/chat/{session_id}/{doc_id}`
- `POST /api/evaluate/{doc_id}`
- `POST /api/session/cleanup/{session_id}`
- `POST /ingest-qa/invoke` (LangServe-compatible)

## Environment Variables (`.env`)
```env
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_RERANK_MODEL=gpt-4o-mini
CHROMA_DIR=data/chroma
UPLOAD_DIR=data/uploads
PROCESSED_DIR=data/processed
DEFAULT_TOP_K=20
RERANK_TOP_K=8
RETRIEVAL_MIN_SCORE=0.3
LIST_INTENT_TOP_K=50
LIST_INTENT_RERANK_TOP_K=25
LIST_INTENT_MIN_SCORE=0.2
MAX_CHUNK_CHARS=900
CHUNK_OVERLAP_CHARS=180
```

## Troubleshooting
- If UI shows old evaluation metrics or `Overall score: N/A` after code changes:
  1. Stop backend process.
  2. Start again with `python run_api.py`.
  3. Hard refresh browser (`Ctrl+F5`).
- FastAPI interactive docs are available at `/docs`.
