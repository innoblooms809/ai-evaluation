"""
pdf_handler.py — Convert PDF pages to PIL Images using PyMuPDF (fitz).

Handles both PDF files and passthrough for image files.
"""

import io
import logging
from typing import List

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def pdf_to_images(file_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """
    Convert each page of a PDF into a PIL Image at the given DPI.

    Args:
        file_bytes: Raw bytes of a PDF file.
        dpi: Resolution for rasterisation. 200 is a good balance
             between quality and speed for OCR.

    Returns:
        A list of PIL Images, one per page.
    """
    zoom = dpi / 72  # fitz default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images: List[Image.Image] = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img.convert("RGB"))
        logger.info(f"  Rendered PDF page {page_number + 1}/{len(doc)} → {img.size}")

    doc.close()
    logger.info(f"Converted PDF to {len(images)} page image(s)")
    return images


def load_image_bytes(file_bytes: bytes) -> Image.Image:
    """
    Load raw image bytes (PNG / JPG / WEBP) into a PIL Image.
    """
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")
