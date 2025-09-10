import os
import io
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# =========================
# Config
# =========================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # optional
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("GEN_MODEL", "gemma3n:e4b")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI(title="PDF RAG with FastAPI + Qdrant + Ollama", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single Qdrant client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# =========================
# Models
# =========================
class UploadResponse(BaseModel):
    collection: str
    chunks: int


class ChatRequest(BaseModel):
    collection: str = Field(..., description="Collection created from upload")
    query: str
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=64, le=4096)


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


# =========================
# Utils: PDF, chunking, embeddings, LLM
# =========================
def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
        return "\n".join(texts).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking.
    You can swap this with token-aware chunking if needed.
    """
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - chunk_overlap
        if start < 0:
            start = 0
    # clean empties
    return [c for c in chunks if c]


async def ollama_embed(texts: List[str]) -> List[List[float]]:
    """
    Call Ollama embeddings endpoint.
    POST /api/embeddings
    body: { "model": "nomic-embed-text", "input": "text" }
    For batching, we'll loop; you can optimize with concurrency if needed.
    """
    vectors = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for t in texts:
            payload = {"model": EMBED_MODEL, "input": t}
            r = await client.post(f"{OLLAMA_URL}/api/embeddings", json=payload)
            if r.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Ollama embed error: {r.text}")
            data = r.json()
            vec = data.get("embedding")
            if not vec:
                raise HTTPException(status_code=500, detail="No embedding returned from Ollama.")
            vectors.append(vec)
    return vectors


async def ollama_generate(prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Stream=false single-shot generation via /api/generate
    """
    payload = {
        "model": GEN_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama generate error: {r.text}")
        data = r.json()
        return data.get("response", "").strip()


def ensure_collection(collection: str, dim: int):
    """
    Create the collection if it doesn't exist yet (dynamic dim).
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection not in existing:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert_points(collection: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
    points = []
    for vec, pl in zip(vectors, payloads):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=pl
            )
        )
    qdrant.upsert(collection_name=collection, points=points)


def search_similar(collection: str, query_vec: List[float], top_k: int):
    res = qdrant.search(collection_name=collection, query_vector=query_vec, limit=top_k)
    # Convert to simple payload list with score
    items = []
    for hit in res:
        payload = hit.payload or {}
        items.append({
            "score": float(hit.score),
            "text": payload.get("text", ""),
            "page": payload.get("page"),
            "chunk_id": payload.get("chunk_id")
        })
    return items


# =========================
# Routes
# =========================

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), collection: Optional[str] = None):
    """
    Upload a PDF -> parse -> chunk -> embed -> store in Qdrant.
    Returns a collection name to use later with /api/chat.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")

    # pick collection name
    collection_name = collection or f"pdf-{uuid.uuid4().hex[:8]}"
    file_bytes = await file.read()
    text = extract_pdf_text(file_bytes)
    chunks = chunk_text(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")

    # embed chunks (we’ll get dim from first vector)
    vectors = await ollama_embed(chunks)
    dim = len(vectors[0])

    # ensure collection exists
    ensure_collection(collection_name, dim)

    # upsert with payload (store page? We used naive chunking, so store index instead)
    payloads = [
        {"text": c, "page": None, "chunk_id": i} for i, c in enumerate(chunks)
    ]
    upsert_points(collection_name, vectors, payloads)

    return UploadResponse(collection=collection_name, chunks=len(chunks))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """
    RAG chat endpoint.
    1) embed the query
    2) retrieve from Qdrant
    3) build prompt with top chunks
    4) generate answer via Ollama LLM
    """
    # 1) embed query
    qvec = await ollama_embed([body.query])
    qvec = qvec[0]

    # 2) retrieve
    hits = search_similar(body.collection, qvec, body.top_k)
    if not hits:
        return ChatResponse(answer="No results found in the collection.", sources=[])

    # 3) prompt
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        context_blocks.append(f"[{i}] (score={h['score']:.3f})\n{h['text']}\n")
    context = "\n\n".join(context_blocks)

    system_instructions = (
        "You are a helpful assistant answering strictly from the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    prompt = f"""{system_instructions}

# Context
{context}

# User Question
{body.query}

# Answer (be concise, cite sources like [1], [2], ... where you used them):
"""

    # 4) generate
    answer = await ollama_generate(prompt, temperature=body.temperature, max_tokens=body.max_tokens)

    # Prepare sources (just reflect what we retrieved)
    sources = []
    for idx, h in enumerate(hits, start=1):
        sources.append({"id": idx, "score": h["score"], "chunk_id": h["chunk_id"]})

    return ChatResponse(answer=answer, sources=sources)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
