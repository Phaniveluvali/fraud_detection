"""Loading uploads into page images, text and cached derived data.

A :class:`Document` is created once per request and shared by all checks. Everything expensive
(page rasterisation, block features, text extraction) is computed lazily and at most once;
locks make this safe when checks run on a thread pool.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import pikepdf
import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

from invoice_guard.config import Settings
from invoice_guard.errors import DocumentParseError, DocumentTooLargeError, UnsupportedDocumentError
from invoice_guard.features import PageFeatures, compute_features
from invoice_guard.invoice_text import InvoiceFacts, parse_invoice
from invoice_guard.textlayout import Word, group_lines

logger = logging.getLogger(__name__)

DocKind = Literal["pdf", "image"]


def sniff_kind(data: bytes) -> DocKind | None:
    """Identify the file by magic bytes; the filename and content type are not trusted."""
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if (
        data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"\x89PNG\r\n\x1a\n")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
        or data[:4] in (b"II*\x00", b"MM\x00*")
        or (data[4:8] == b"ftyp" and data[8:12] in (b"avif", b"avis"))
    ):
        return "image"
    return None


@dataclass(eq=False)
class Page:
    index: int
    image: Image.Image
    source: str  # "file", "embedded" (scan extracted from a PDF) or "render"
    pt_to_px: float | None = None  # PDF points to image pixels, when applicable
    _features: dict[int, PageFeatures] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def features(self, block_size: int) -> PageFeatures:
        with self._lock:
            cached = self._features.get(block_size)
            if cached is None:
                cached = compute_features(self.image, block_size)
                self._features[block_size] = cached
            return cached


class Document:
    def __init__(
        self,
        data: bytes,
        filename: str,
        kind: DocKind,
        pages: list[Page],
        settings: Settings,
        warnings: list[str],
    ) -> None:
        self.data = data
        self.filename = filename
        self.kind: DocKind = kind
        self.pages = pages
        self.settings = settings
        self.warnings = warnings
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.artifacts: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._words: list[Word] | None = None
        self._facts: InvoiceFacts | None = None
        self._lines: list[list[Word]] | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_bytes(cls, data: bytes, filename: str, settings: Settings) -> Document:
        if len(data) > settings.max_upload_bytes:
            raise DocumentTooLargeError(f"File exceeds the {settings.max_upload_mb} MiB limit")
        kind = sniff_kind(data)
        if kind is None:
            raise UnsupportedDocumentError("Only PDF, JPEG, PNG, WebP, AVIF and TIFF are supported")
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
        warnings: list[str] = []
        pages = (
            _load_pdf_pages(data, settings, warnings)
            if kind == "pdf"
            else _load_image_pages(data, settings, warnings)
        )
        if not pages:
            raise DocumentParseError("The document contains no readable pages")
        return cls(data, filename, kind, pages, settings, warnings)

    # ----------------------------------------------------------------------- text
    @property
    def words(self) -> list[Word]:
        with self._lock:
            if self._words is None:
                from invoice_guard.text import extract_words

                self._words = extract_words(self)
            return self._words

    @property
    def lines(self) -> list[list[Word]]:
        with self._lock:
            if self._lines is None:
                self._lines = group_lines(self.words)
            return self._lines

    @property
    def facts(self) -> InvoiceFacts:
        with self._lock:
            if self._facts is None:
                self._facts = parse_invoice(self.lines)
            return self._facts


# ------------------------------------------------------------------------ image loading
def _fit(image: Image.Image, max_side: int, warnings: list[str], label: str) -> Image.Image:
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    warnings.append(
        f"{label} downscaled from {image.width}x{image.height}; "
        "pixel checks are less sensitive on resampled images"
    )
    return image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )


def _flatten(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def _load_image_pages(data: bytes, settings: Settings, warnings: list[str]) -> list[Page]:
    pages: list[Page] = []
    try:
        with Image.open(io.BytesIO(data)) as img:
            frames = min(getattr(img, "n_frames", 1), settings.max_pages)
            if getattr(img, "n_frames", 1) > frames:
                warnings.append(f"Only the first {frames} frames were analysed")
            for i in range(frames):
                img.seek(i)
                rgb = _flatten(img)
                pages.append(Page(i, _fit(rgb, settings.max_image_side, warnings, "Image"), "file"))
    except Image.DecompressionBombError as exc:
        raise DocumentTooLargeError("Image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise DocumentParseError(f"Image could not be decoded: {exc}") from exc
    return pages


# ------------------------------------------------------------------------- PDF loading
def _embedded_scan(
    pk_page: Any, width_pt: float, height_pt: float, settings: Settings
) -> Image.Image | None:
    """Return the page's large embedded raster if it is a scan, at its native resolution.

    Re-rendering a scanned PDF would resample the pixels and destroy the compression trail
    that error-level analysis relies on, so the original image is preferred.
    """
    try:
        candidates = []
        # ``Page.images`` is deprecated in recent pikepdf; ``get_images`` also finds images
        # nested in form XObjects. Fall back for older releases.
        getter = getattr(pk_page, "get_images", None)
        images = getter() if getter is not None else pk_page.images
        for obj in images.values():
            candidates.append((int(obj.Width) * int(obj.Height), obj))
        if not candidates:
            return None
        area, obj = max(candidates, key=lambda c: c[0])
        scale = settings.render_dpi / 72.0
        if area < 0.5 * (width_pt * scale) * (height_pt * scale):
            return None
        scan: Image.Image = pikepdf.PdfImage(obj).as_pil_image().convert("RGB")
        return scan
    except Exception:
        logger.debug("embedded image extraction failed; rendering instead", exc_info=True)
        return None


def _load_pdf_pages(data: bytes, settings: Settings, warnings: list[str]) -> list[Page]:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise DocumentParseError(f"PDF could not be opened: {exc}") from exc

    pk: Any = None
    try:
        pk = pikepdf.open(io.BytesIO(data))
    except pikepdf.PdfError:
        logger.debug("pikepdf could not open the file; pages will be rendered", exc_info=True)

    pages: list[Page] = []
    try:
        total = len(pdf)
        limit = min(total, settings.max_pages)
        if total > limit:
            warnings.append(f"Only the first {limit} of {total} pages were analysed")
        render_scale = settings.render_dpi / 72.0
        for i in range(limit):
            page = pdf[i]
            width_pt, height_pt = page.get_size()
            image = None
            source = "render"
            if pk is not None and i < len(pk.pages):
                image = _embedded_scan(pk.pages[i], width_pt, height_pt, settings)
            if image is not None:
                source = "embedded"
                pt_to_px = image.width / width_pt
            else:
                image = page.render(scale=render_scale).to_pil().convert("RGB")
                pt_to_px = render_scale
            image = _fit(image, settings.max_image_side, warnings, f"Page {i + 1}")
            if image.width != round(width_pt * pt_to_px):
                pt_to_px = image.width / width_pt
            pages.append(Page(i, image, source, pt_to_px))
    except pdfium.PdfiumError as exc:
        raise DocumentParseError(f"PDF page could not be rendered: {exc}") from exc
    finally:
        pdf.close()
        if pk is not None:
            pk.close()
    return pages
