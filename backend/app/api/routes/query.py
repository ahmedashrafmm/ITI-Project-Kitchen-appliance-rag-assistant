import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.schemas.query import QueryRequest, QueryResponse, DetectionResult
from app.services.retrieval import retrieve
from app.services.generation import generate_answer
from app.services import vision

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    hits = retrieve(request.question, k=request.k, appliance_hint=request.appliance_hint)
    answer = generate_answer(request.question, hits)
    sources = sorted({h["source"] for h in hits})
    return QueryResponse(answer=answer, sources=sources)


@router.post("/detect-appliance", response_model=DetectionResult)
async def detect_appliance(image: UploadFile = File(...)):
    """Extended Track: accept an image upload, detect the appliance with YOLO,
    and return the matching manual source so the frontend/client can pass it
    back as `appliance_hint` on a subsequent /query call."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    appliance, confidence, manual_source = vision.detect_appliance(tmp_path)
    return DetectionResult(appliance=appliance, confidence=confidence, manual_source=manual_source)
