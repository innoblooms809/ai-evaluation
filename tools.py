import os
import subprocess
import tempfile
import logging
from typing import Optional, Union, ClassVar

import requests
import fitz  # PyMuPDF
from pydantic import Extra
from langchain.tools import BaseTool
from serpapi import GoogleSearch  # pip install serpapi
import os
from pydantic import BaseModel, Field
from mistralai import Mistral, DocumentURLChunk
from mistralai.extra import response_format_from_pydantic_model
import requests
from langchain.tools import BaseTool
from serpapi import GoogleSearch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OCRTool:
    """
    A unified OCR tool that wraps Mistral’s OCR functionality and annotation methods:
      - Upload a local PDF
      - OCR any public PDF URL
      - OCR any public image URL
      - OCR a PDF that was previously uploaded (using its signed URL)
      - Perform bounding box or document-level annotations on a document URL
      - Save extracted Markdown text to a .md file
    """

    def __init__(self, api_key: str = None, model_name: str = "mistral-ocr-latest"):
        # Accept API key from argument or environment
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Please set the MISTRAL_API_KEY environment variable before using OCRTool.")
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name

    # ---------------------- PDF Upload & Signed URL ----------------------
    def upload_pdf(self, filepath: str) -> str:
        """
        Upload a local PDF to Mistral’s file service for OCR.
        Returns the file ID (string).
        Raises FileNotFoundError or RuntimeError if something goes wrong.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Could not find PDF at: {filepath}")

        with open(filepath, "rb") as f:
            try:
                upload_resp = self.client.files.upload(
                    file={
                        "file_name": os.path.basename(filepath),
                        "content": f,
                    },
                    purpose="ocr"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to upload PDF for OCR: {e}")

        file_id = upload_resp.get("id") if isinstance(upload_resp, dict) else getattr(upload_resp, "id", None)
        if not file_id:
            raise RuntimeError("Upload succeeded but no file ID was returned.")
        logger.info(f"✅ Uploaded '{filepath}' to Mistral (file_id={file_id})")
        return file_id

    def get_signed_url(self, file_id: str) -> str:
        """
        Given a file ID (from upload_pdf), return a signed URL for OCR.
        Raises RuntimeError if Mistral call fails or no URL is returned.
        """
        try:
            signed = self.client.files.get_signed_url(file_id=file_id)
        except Exception as e:
            raise RuntimeError(f"Could not get signed URL for file_id={file_id}: {e}")

        url = signed.get("url") if isinstance(signed, dict) else getattr(signed, "url", None)
        if not url:
            raise RuntimeError("Failed to retrieve signed URL from response.")
        return url

    # ---------------------- OCR Methods ----------------------
    def ocr_from_pdf_url(self, pdf_url: str, output_md_path: str) -> None:
        """
        Run OCR on a publicly accessible PDF URL, save Markdown to output_md_path.
        Raises RuntimeError if OCR fails or no Markdown is found.
        """
        try:
            ocr_resp = self.client.ocr.process(
                model=self.model_name,
                document={"type": "document_url", "document_url": pdf_url},
                include_image_base64=True
            )
        except Exception as e:
            raise RuntimeError(f"OCR on PDF URL failed: {e}")

        self._save_ocr_markdown(ocr_resp, output_md_path)

    def ocr_from_uploaded_pdf(self, file_id: str, output_md_path: str) -> None:
        """
        Run OCR on a PDF that was uploaded earlier (by upload_pdf).
        Fetches a signed URL, then calls ocr_from_pdf_url().
        """
        signed_url = self.get_signed_url(file_id)
        logger.info(f"→ Using signed URL for OCR: {signed_url}")
        self.ocr_from_pdf_url(signed_url, output_md_path)

    def ocr_from_image_url(self, image_url: str, output_md_path: str, include_images: bool = True) -> None:
        """
        Run OCR on a publicly accessible image URL, save Markdown to output_md_path.
        By default, include_images=True (requests Base64 sub-images), but only Markdown is written.
        """
        try:
            ocr_resp = self.client.ocr.process(
                model=self.model_name,
                document={"type": "image_url", "image_url": image_url},
                include_image_base64=include_images
            )
        except Exception as e:
            raise RuntimeError(f"OCR on image URL failed: {e}")

        self._save_ocr_markdown(ocr_resp, output_md_path)

    def _save_ocr_markdown(self, ocr_resp, output_md_path: str) -> None:
        """
        Given the raw OCR response, extract each page’s “markdown” field, join them,
        and write to output_md_path. If no markdown is found, prints a warning.
        """
        data = ocr_resp.dict() if hasattr(ocr_resp, "dict") else ocr_resp
        pages = data.get("pages", [])
        if not isinstance(pages, list) or len(pages) == 0:
            logger.warning("⚠️  OCR succeeded but no 'pages' were returned.")
            return

        all_md = []
        for page in pages:
            md = page.get("markdown", "").strip()
            if md:
                all_md.append(md)

        if not all_md:
            logger.warning("⚠️  OCR extracted zero Markdown on any page.")
            return

        combined = "\n\n".join(all_md)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(combined)

        logger.info(f"✅ Saved OCR Markdown to '{output_md_path}' (pages={len(all_md)})")

    # ---------------------- Annotation Methods ----------------------
    def bbox_annotation(self, document_url: str, bbox_model: type[BaseModel],
                        pages: list[int] = None, include_image_base64: bool = False):
        """
        Perform bounding box annotation on a document URL using a Pydantic model schema.
        Returns the OCR response with bounding box information.
        """
        fmt = response_format_from_pydantic_model(bbox_model)
        kwargs = {
            "model": self.model_name,
            "document": DocumentURLChunk(document_url=document_url),
            "bbox_annotation_format": fmt,
            "include_image_base64": include_image_base64
        }
        if pages is not None:
            kwargs["pages"] = pages
        return self.client.ocr.process(**kwargs)

    def document_annotation(self, document_url: str, document_model: type[BaseModel],
                            pages: list[int] = None, include_image_base64: bool = False):
        """
        Perform document-level annotation on a document URL using a Pydantic model schema.
        Returns the OCR response with document annotation information.
        """
        fmt = response_format_from_pydantic_model(document_model)
        kwargs = {
            "model": self.model_name,
            "document": DocumentURLChunk(document_url=document_url),
            "document_annotation_format": fmt,
            "include_image_base64": include_image_base64
        }
        if pages is not None:
            kwargs["pages"] = pages
        return self.client.ocr.process(**kwargs)

    def full_annotation(self, document_url: str, bbox_model: type[BaseModel],
                        document_model: type[BaseModel], pages: list[int] = None,
                        include_image_base64: bool = False):
        """
        Perform both bounding box and document-level annotations on a document URL.
        Returns the OCR response with both bbox and document annotation information.
        """
        bbox_fmt = response_format_from_pydantic_model(bbox_model)
        doc_fmt = response_format_from_pydantic_model(document_model)
        kwargs = {
            "model": self.model_name,
            "document": DocumentURLChunk(document_url=document_url),
            "bbox_annotation_format": bbox_fmt,
            "document_annotation_format": doc_fmt,
            "include_image_base64": include_image_base64
        }
        if pages is not None:
            kwargs["pages"] = pages
        return self.client.ocr.process(**kwargs)


# Example Pydantic models for annotations (customize as needed)
class ImageAnnotation(BaseModel):
    image_type: str = Field(..., description="Type of the image (e.g., 'bar_chart')")
    short_description: str = Field(..., description="Brief English description")
    summary: str = Field(..., description="One‐paragraph summary")


class DocumentAnnotation(BaseModel):
    language: str = Field(..., description="Detected language of the document")
    chapter_titles: list[str] = Field(..., description="List of chapter/section titles")
    urls: list[str] = Field(..., description="Any URLs found in the document")


class YouTubeTool(BaseTool):
    name: ClassVar[str] = "YouTubeSearch"
    description: ClassVar[str] = "Searches YouTube via SerpApi and returns top-3 summaries."

    def _run(self, query: str) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "SerpAPI key not set."
        params = {
            "engine": "youtube",
            "search_query": query,
            "api_key": api_key,
            "hl": "en",
            "gl": "us"
        }
        try:
            data = GoogleSearch(params).get_dict() or {}
        except Exception as e:
            return f"YouTube search error: {e}"
        vids = data.get("video_results", [])[:3]
        if not vids:
            return "No YouTube results."
        return "\n\n".join(
            f"{v.get('title','No Title')} ({v.get('link') or 'https://youtu.be/'+v.get('video_id','')})\n(Source: SerpApi)"
            for v in vids
        )

    async def _arun(self, query: str) -> str:
        return self._run(query)


class WebSearchTool(BaseTool):
    name: ClassVar[str] = "WebSearch"
    description: ClassVar[str] = "Searches Google via SerpAPI and returns top-3 summaries."

    def _run(self, query: str) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "SerpAPI key not set."
        resp = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google", "q": query, "api_key": api_key, "num": 3}
        )
        if resp.status_code != 200:
            return f"Web search error {resp.status_code}: {resp.text[:200]}"
        try:
            jd = resp.json()
        except Exception as e:
            return f"JSON parse error: {e}"
        org = jd.get("organic_results", [])
        if not org:
            return "No web search results."
        return "\n\n".join(
            f"{r.get('title','No Title')}\n{r.get('link','')}\n{r.get('snippet','')}\n(Source: Google)"
            for r in org
        )

    async def _arun(self, query: str) -> str:
        return self._run(query)