"""FastAPI application exposing the NL-to-MQL agent over HTTP.

Endpoints
---------
GET  /              -> serves the single-page chat UI (static/index.html).
GET  /api/health    -> cluster + service health.
POST /api/chat      -> the main endpoint. Body: {"message": "...", "history": [...]}
                       Returns the conversational response, the generated MQL
                       (for audit/debugging), and the raw JSON results.

Run locally:
    uvicorn server:app --reload --port 8000
    # then open http://localhost:8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import answer
from config import kb_info, model_info
from db import ensure_indexes, health_check

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Provision indexes on boot unless explicitly skipped.

    Set SKIP_INDEX_SETUP=1 to bypass (e.g. when the DB user lacks index
    privileges). Index creation is idempotent, so it is safe to run each boot.
    """
    if os.getenv("SKIP_INDEX_SETUP") != "1":
        try:
            ensure_indexes()
        except Exception as exc:  # noqa: BLE001 - never block startup on index setup
            print(f"[startup] index setup skipped: {exc}")
    yield


app = FastAPI(
    title="DA680 NL-to-MQL Chat",
    description="Conversational natural-language to MongoDB Atlas query system.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's natural-language message.")
    history: list = Field(default_factory=list, description="Prior turns (optional).")


class ChatResponse(BaseModel):
    response: str
    path: str
    router_reason: str = ""
    generated_mql: object | None = None
    raw_results: object | None = None
    error: str | None = None
    meta: dict = Field(default_factory=dict)


@app.get("/api/health")
def health() -> JSONResponse:
    ok = health_check()
    return JSONResponse(
        {"status": "ok" if ok else "degraded", "database": ok},
        status_code=200 if ok else 503,
    )


@app.get("/api/model")
def model() -> JSONResponse:
    """Report the active LLM provider/model and KB config so the UI can show them."""
    info = model_info()
    info["knowledge_base"] = kb_info()
    return JSONResponse(info)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Main chat endpoint: route the message and return the full result."""
    result = answer(req.message, history=req.history)
    return ChatResponse(**result.to_dict())


# --- Static UI -------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
