"""HTTP API.

Run with ``uvicorn --factory invoice_guard.api:create_app``. Uploads are held in memory,
never written to disk, and size and type are enforced before any parsing happens.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from invoice_guard._version import __version__
from invoice_guard.compare import compare
from invoice_guard.config import Settings, get_settings
from invoice_guard.document import Document
from invoice_guard.errors import (
    DocumentParseError,
    DocumentTooLargeError,
    ExtractionUnavailableError,
    UnsupportedDocumentError,
)
from invoice_guard.extraction import extract
from invoice_guard.logs import configure_logging
from invoice_guard.models import AnalyzeResponse, CompareResponse, Extraction, ReviewResponse
from invoice_guard.pipeline import AnalysisPipeline
from invoice_guard.render import encode_image, encode_images, overlay
from invoice_guard.review import review
from invoice_guard.store import SubmissionStore

logger = logging.getLogger(__name__)

_INDEX = Path(__file__).parent / "static" / "index.html"
# The page is self-contained: no third-party scripts, styles or network calls.
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src data: blob:; connect-src 'self'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
}


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(cfg.log_level)
        store = SubmissionStore(cfg.db_path) if cfg.enable_duplicate_store else None
        app.state.pipeline = AnalysisPipeline(cfg, store=store)
        try:
            yield
        finally:
            if store is not None:
                store.close()

    app = FastAPI(
        title="Invoice Guard",
        version=__version__,
        description="Tamper detection for invoices and financial documents.",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(_INDEX.read_text(encoding="utf-8"), headers=_PAGE_HEADERS)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # A plain ``def`` endpoint runs in FastAPI's thread pool, which keeps the CPU-bound
    # analysis from blocking the event loop.
    @app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["analysis"])
    def analyze(
        request: Request,
        file: Annotated[UploadFile, File(description="PDF or image of the invoice")],
        register: bool = False,
        include_images: bool = False,
    ) -> AnalyzeResponse:
        """Analyse one document.

        ``register=true`` stores a fingerprint so later submissions can be checked against
        it. ``include_images=true`` adds annotated page images as base64 PNGs.
        """
        data = file.file.read(cfg.max_upload_bytes + 1)
        pipeline: AnalysisPipeline = request.app.state.pipeline
        try:
            result = pipeline.analyze(data, file.filename or "upload", register=register)
        except DocumentTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UnsupportedDocumentError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except DocumentParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        images = encode_images(result) if include_images else None
        return AnalyzeResponse(report=result.report, images=images)

    @app.post("/v1/extract", response_model=Extraction, tags=["extraction"])
    def extract_fields(
        file: Annotated[UploadFile, File(description="PDF or image of the invoice")],
    ) -> Extraction:
        """Read the invoice fields with the configured Ollama vision model.

        The page images are sent to the server named by ``IG_OLLAMA_URL``.
        """
        if not cfg.enable_extraction:
            raise HTTPException(status_code=503, detail="Field extraction is switched off")
        data = file.file.read(cfg.max_upload_bytes + 1)
        try:
            doc = Document.from_bytes(data, "upload", cfg)
            return extract(doc, cfg)
        except DocumentTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UnsupportedDocumentError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except DocumentParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ExtractionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/review", response_model=ReviewResponse, tags=["extraction"])
    def review_findings(
        request: Request,
        file: Annotated[UploadFile, File(description="PDF or image of the invoice")],
    ) -> ReviewResponse:
        """Ask the vision model whether it also sees what the layout check flagged."""
        if not cfg.enable_extraction:
            raise HTTPException(status_code=503, detail="The vision model is switched off")
        data = file.file.read(cfg.max_upload_bytes + 1)
        pipeline: AnalysisPipeline = request.app.state.pipeline
        try:
            result = pipeline.analyze(data, "upload")
            doc = Document.from_bytes(data, "upload", cfg)
            return review(doc, result.report.regions, cfg)
        except DocumentTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UnsupportedDocumentError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except DocumentParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ExtractionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/compare", response_model=CompareResponse, tags=["analysis"])
    def compare_documents(
        original: Annotated[UploadFile, File(description="Trusted copy")],
        suspect: Annotated[UploadFile, File(description="Copy to check")],
        include_images: bool = False,
    ) -> CompareResponse:
        """Locate every visible difference between two copies of a document."""
        limit = cfg.max_upload_bytes + 1
        try:
            result = compare(original.file.read(limit), suspect.file.read(limit), cfg)
        except DocumentTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UnsupportedDocumentError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except DocumentParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        images = None
        if include_images:
            images = {}
            for index, page in enumerate(result.suspect_pages):
                boxes = [r for r in result.differences if r.page == index]
                images[f"page{index + 1}"] = encode_image(page)
                images[f"page{index + 1}_overlay"] = encode_image(overlay(page, boxes))
        return CompareResponse(
            identical=result.identical,
            differences=result.differences,
            warnings=result.warnings,
            images=images,
        )

    return app
