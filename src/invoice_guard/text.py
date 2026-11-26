"""Positioned text for a document: the PDF text layer first, OCR as the fallback."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import pdfplumber

from invoice_guard import ocr
from invoice_guard.textlayout import Word

if TYPE_CHECKING:
    from invoice_guard.document import Document, Page

logger = logging.getLogger(__name__)


def _pdf_text_layer(data: bytes, pages: list[Page]) -> dict[int, list[Word]]:
    result: dict[int, list[Word]] = {}
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pages:
                if page.index >= len(pdf.pages):
                    break
                scale = page.pt_to_px or 1.0
                extracted = pdf.pages[page.index].extract_words(
                    extra_attrs=["fontname", "size"], keep_blank_chars=False
                )
                result[page.index] = [
                    Word(
                        page=page.index,
                        text=str(w["text"]),
                        x0=float(w["x0"]) * scale,
                        top=float(w["top"]) * scale,
                        x1=float(w["x1"]) * scale,
                        bottom=float(w["bottom"]) * scale,
                        fontname=str(w.get("fontname") or "") or None,
                        size=float(w["size"]) if w.get("size") else None,
                    )
                    for w in extracted
                ]
    except Exception:
        logger.warning("text layer extraction failed", exc_info=True)
    return result


def extract_words(doc: Document) -> list[Word]:
    per_page: dict[int, list[Word]] = {}
    if doc.kind == "pdf":
        per_page = _pdf_text_layer(doc.data, doc.pages)
    words: list[Word] = []
    for page in doc.pages:
        page_words = per_page.get(page.index) or []
        if not page_words and doc.settings.enable_ocr:
            page_words = ocr.recognize(page.image, page.index)
        words.extend(page_words)
    return words
