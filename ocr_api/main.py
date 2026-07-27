"""
main.py — OCR Output API

POST /ocrOutput
  - file  (required, multipart file upload — image or PDF)

Accepts an image (PNG/JPG/JPEG/WEBP) or a PDF.
Returns a JSON response with the OCR-extracted markdown text.

Run:
    uvicorn ocr_api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
import sys
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr_api.ocr_engine import load_model, run_ocr
from ocr_api.pdf_handler import pdf_to_images, load_image_bytes

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("ocr_api")

# ── Allowed file extensions ──
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTENSION = ".pdf"


# ── Lifespan: pre-load model at startup ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading Chandra OCR-2 model at startup...")
    load_model()
    logger.info("✅ Model ready — API is accepting requests.")
    yield
    logger.info("👋 Shutting down OCR API.")


# ── FastAPI app ──
app = FastAPI(
    title="OCR Output API",
    description=(
        "Upload an image or PDF and receive the OCR-extracted text "
        "as structured JSON. Powered by Chandra OCR-2."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — open for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ──
class PageOCR(BaseModel):
    page: int
    markdown: str


class OCROutputResponse(BaseModel):
    status: str
    filename: str
    total_pages: int
    processing_time_seconds: float
    pages: List[PageOCR]
    combined_markdown: str
    tables: List[str]
    formulas: List[str]


# ── Health check ──
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "OCR Output API",
        "model": "Chandra OCR-2",
    }


# ── Helper: extract tables & formulas from markdown ──
def _extract_structures(text: str) -> Dict[str, List[str]]:
    tables = re.findall(
        r"(\|.*\|?\n\|[-:| ]*\|?\n(?:\|.*\|?\n)+)", text
    )
    formulas = re.findall(r"(\$\$[\s\S]*?\$\$|\$.*?\$)", text)
    return {
        "tables": [t.strip() for t in tables],
        "formulas": [f.strip() for f in formulas],
    }


# ── Main endpoint ──
@app.post("/ocrOutput", response_model=OCROutputResponse)
async def ocr_output(file: UploadFile = File(...)):
    """
    Upload an image (PNG/JPG/JPEG/WEBP) or a multi-page PDF.
    Returns a JSON object with:
      - per-page markdown OCR text
      - combined markdown
      - extracted tables and formulas
    """
    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded.")

    filename = file.filename.strip()
    ext = os.path.splitext(filename)[1].lower()

    if ext not in IMAGE_EXTENSIONS and ext != PDF_EXTENSION:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(IMAGE_EXTENSIONS | {PDF_EXTENSION}))}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Uploaded file is empty.")

    logger.info(f"📥 Received file: {filename} ({len(file_bytes)} bytes)")

    start_time = time.time()

    try:
        # ── Convert to list of PIL images ──
        if ext == PDF_EXTENSION or file_bytes[:5] == b"%PDF-":
            images = pdf_to_images(file_bytes, dpi=200)
        else:
            images = [load_image_bytes(file_bytes)]

        # ── Run OCR on each page ──
        pages: List[PageOCR] = []
        for idx, img in enumerate(images):
            logger.info(f"  🔍 Running OCR on page {idx + 1}/{len(images)}...")
            markdown = run_ocr(img)
            pages.append(PageOCR(page=idx + 1, markdown=markdown))

        # ── Combine results ──
        combined = "\n\n---\n\n".join(p.markdown for p in pages)
        structures = _extract_structures(combined)
        elapsed = round(time.time() - start_time, 2)

        logger.info(
            f"✅ OCR complete — {len(pages)} page(s), "
            f"{len(combined)} chars, {elapsed}s"
        )

        return OCROutputResponse(
            status="success",
            filename=filename,
            total_pages=len(pages),
            processing_time_seconds=elapsed,
            pages=pages,
            combined_markdown=combined,
            tables=structures["tables"],
            formulas=structures["formulas"],
        )

    except Exception as e:
        logger.error(f"❌ OCR failed: {e}", exc_info=True)
        raise HTTPException(500, f"OCR processing failed: {str(e)}")


# ── Direct execution ──
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ocr_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
