# Smart Contract Assistant (FastAPI + Chat UI)

Contract Q&A application with Retrieval-Augmented Generation (RAG):
- Upload PDF/DOCX contracts
- Index contract text into Chroma
- Chat with grounded answers and chunk citations
- Manage stored documents from the web UI

## What Was Removed
This project no longer includes summary/evaluation features or files:
- Removed summary/evaluation API endpoints
- Removed summary/evaluation schemas and evaluator module
- Removed local `docs/` folder

## Tech Stack
- Backend: FastAPI
- Frontend: FastAPI-served HTML/CSS/JS (Bootstrap)
- Vector store: ChromaDB
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
1. Create and activate venv:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure env:
```bash
copy .env.example .env
```
Set `OPENAI_API_KEY` in `.env`.

4. Run:
```bash
python run_api.py
```

5. Open UI:
- `http://127.0.0.1:8000/`

## How This Project Works
1. Document upload
- UI sends file to `POST /api/documents/upload`.
- Backend validates extension (`.pdf`, `.docx`), reads bytes, computes stable `doc_id`.

2. Parsing and chunking
- `document_parser.py` extracts section text.
- `chunking.py` splits into chunks:
  - standard chunking for normal sections
  - line-based chunking for list/appendix-like content

3. Embedding and indexing
- `openai_client.py` creates embeddings.
- `vector_store.py` writes vectors/chunks to Chroma.
- Document metadata is saved (`filename`, `pages_or_sections`, `chunks_indexed`, `indexed_at`).

4. Chat retrieval pipeline
- UI sends question to `POST /api/chat/{session_id}/{doc_id}`.
- Pipeline in `rag_pipeline.py`:
  - detects list-intent questions
  - retrieves candidates via vector search + keyword BM25 search
  - merges/de-duplicates by `chunk_id`
  - reranks candidates
  - builds grounded context prompt

5. Answer generation + guardrails
- Chat model answers using retrieved context only.
- Response includes citations, confidence, latency, retrieved chunk count.
- Low-confidence responses are labeled accordingly.

6. Session memory and cleanup
- Per-session chat memory is stored in `chat_memory.py`.
- Session cleanup endpoint clears memory and optionally session-linked docs.

## Full API Schema

### Core Models

#### `AskRequest`
```json
{
  "question": "string (min length: 3)",
  "top_k": 20
}
```

#### `Citation`
```json
{
  "doc_id": "string",
  "chunk_id": "string",
  "score": 0.0,
  "page": 1,
  "quote": "string"
}
```

#### `GuardrailStatus`
```json
{
  "grounded": true,
  "reason": "string"
}
```

#### `AskResponse`
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

#### `UploadResponse`
```json
{
  "doc_id": "string",
  "filename": "string",
  "chunks_indexed": 0,
  "indexed_at": "2026-02-16T12:00:00"
}
```

#### `DocumentStatus`
```json
{
  "doc_id": "string",
  "filename": "string",
  "pages_or_sections": 0,
  "chunks_indexed": 0,
  "indexed_at": "2026-02-16T12:00:00"
}
```

### Endpoints

#### `GET /api/health`
Response:
```json
{
  "status": "ok",
  "chat_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-large",
  "rerank_model": "gpt-4o-mini"
}
```

#### `POST /api/documents/upload`
- Query params:
  - `session_id` (optional)
- Multipart body:
  - `file`: `.pdf` or `.docx`
- Response: `UploadResponse`

#### `GET /api/documents/{doc_id}`
- Response: `DocumentStatus`

#### `GET /api/documents`
- Response: `DocumentStatus[]`

#### `POST /api/documents/clear`
Response:
```json
{
  "deleted_docs": 0,
  "deleted_upload_files": 0
}
```

#### `POST /api/chat/{session_id}/{doc_id}`
Request body: `AskRequest`
Response: `AskResponse`

#### `POST /api/session/cleanup/{session_id}`
- Query params:
  - `delete_documents` (default `false`)
Response:
```json
{
  "session_docs": 0,
  "deleted_docs": 0
}
```

#### `POST /ingest-qa/invoke`
- LangServe-compatible invoke route wrapping the same `ask(...)` pipeline.

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
DEFAULT_TOP_K=30
RERANK_TOP_K=10
RETRIEVAL_MIN_SCORE=0.3
LIST_INTENT_TOP_K=50
LIST_INTENT_RERANK_TOP_K=25
LIST_INTENT_MIN_SCORE=0.2
MAX_CHUNK_CHARS=1000
CHUNK_OVERLAP_CHARS=200
```

## Notes
- Indexed vectors and metadata are stored under `data/`.
- This is an English-contract focused pipeline.
- FastAPI interactive API docs are available at runtime on `/docs` (Swagger UI).
