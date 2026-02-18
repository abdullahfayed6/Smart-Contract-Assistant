from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.runnables import RunnableLambda

from app.core.config import settings
from app.core.schemas import (
    AskRequest,
    AskResponse,
    DocumentStatus,
    EvaluationResponse,
    UploadResponse,
)
from app.services.rag_pipeline import (
    ask,
    cleanup_session_documents,
    clear_all_documents,
    evaluate,
    get_document_status,
    ingest_document,
    register_session_document,
)
from app.services.rag_pipeline import list_documents

BASE_DIR = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Smart Contract Assistant API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    # Avoid noisy browser 404 probes when no favicon file is provided.
    return Response(status_code=200, content=b"", media_type="image/x-icon")


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools_probe() -> JSONResponse:
    # Chrome/DevTools may probe this path; return a valid empty payload.
    return JSONResponse(content={})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "chat_model": settings.openai_chat_model,
        "embedding_model": settings.openai_embedding_model,
        "rerank_model": settings.openai_rerank_model,
    }


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), session_id: str | None = None) -> UploadResponse:
    if not (file.filename or "").lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        doc_id, chunks, indexed_at = ingest_document(file.filename or "document.pdf", payload)
        if session_id:
            register_session_document(session_id=session_id, doc_id=doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UploadResponse(doc_id=doc_id, filename=file.filename or "", chunks_indexed=chunks, indexed_at=indexed_at)


@app.get("/api/documents/{doc_id}", response_model=DocumentStatus)
def document_status(doc_id: str) -> DocumentStatus:
    try:
        return get_document_status(doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/documents", response_model=list[DocumentStatus])
def documents() -> list[DocumentStatus]:
    return list_documents()


@app.post("/api/documents/clear")
def clear_documents() -> dict[str, int]:
    return clear_all_documents()


@app.post("/api/chat/{session_id}/{doc_id}", response_model=AskResponse)
def chat(session_id: str, doc_id: str, req: AskRequest) -> AskResponse:
    try:
        return ask(doc_id=doc_id, session_id=session_id, question=req.question, top_k=req.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/evaluate/{doc_id}", response_model=EvaluationResponse)
def evaluate_doc(doc_id: str) -> EvaluationResponse:
    try:
        return evaluate(doc_id=doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/session/cleanup/{session_id}")
def cleanup_session(session_id: str, delete_documents: bool = False) -> dict[str, int]:
    return cleanup_session_documents(session_id=session_id, delete_documents=delete_documents)


def _build_langserve_payload(payload: dict) -> dict:
    return ask(
        doc_id=payload["doc_id"],
        session_id=payload.get("session_id", "langserve"),
        question=payload["question"],
        top_k=payload.get("top_k"),
    ).model_dump()


try:
    from langserve import add_routes

    add_routes(app, RunnableLambda(_build_langserve_payload), path="/ingest-qa")
except Exception:
    pass
