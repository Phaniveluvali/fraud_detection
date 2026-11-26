"""Image file metadata: editing software traces and inconsistent timestamps."""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Any

from PIL import Image

from invoice_guard.checks.base import Check
from invoice_guard.document import Document
from invoice_guard.models import CheckResult

IMAGE_EDITORS = (
    "photoshop",
    "gimp",
    "paint.net",
    "pixlr",
    "canva",
    "snapseed",
    "lightroom",
    "affinity",
    "photopea",
    "picsart",
)
_TAG_SOFTWARE = 0x0131
_TAG_DATETIME = 0x0132
_TAG_DATETIME_ORIGINAL = 0x9003
_EXIF_IFD = 0x8769
_STD_LUMA_MEAN = 57.6  # mean of the IJG baseline luminance quantisation table


def estimate_jpeg_quality(table: list[int]) -> int:
    """Invert the IJG quality scaling from a luminance quantisation table."""
    scale = (sum(table) / len(table)) / _STD_LUMA_MEAN * 100.0
    quality = 5000.0 / scale if scale >= 100.0 else (200.0 - scale) / 2.0
    return int(max(1, min(100, round(quality))))


def _parse_exif_time(value: Any) -> datetime | None:
    try:
        return datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


class ImageMetadataCheck(Check):
    name = "image_metadata"
    title = "Image metadata"
    applies_to = frozenset({"image"})

    def run(self, doc: Document) -> CheckResult:
        try:
            with Image.open(io.BytesIO(doc.data)) as img:
                exif = img.getexif()
                software = str(exif.get(_TAG_SOFTWARE) or img.info.get("Software") or "")
                modified = _parse_exif_time(exif.get(_TAG_DATETIME))
                original = _parse_exif_time(exif.get_ifd(_EXIF_IFD).get(_TAG_DATETIME_ORIGINAL))
                quant = getattr(img, "quantization", None)
                fmt = img.format
        except Exception:
            return self.skipped("Image metadata could not be read.")

        evidence: dict[str, Any] = {"format": fmt, "software": software or None}
        if quant and 0 in quant:
            evidence["jpeg_quality_estimate"] = estimate_jpeg_quality(list(quant[0]))

        messages: list[tuple[float, str]] = []
        if software and any(m in software.lower() for m in IMAGE_EDITORS):
            messages.append((0.4, f'The image was last saved by "{software}", an image editor.'))
        if modified and original and abs(modified - original) > timedelta(days=1):
            messages.append((0.4, "The file was modified long after the photo or scan was taken."))

        if not messages:
            detail = (
                "No signs of editing software in the metadata."
                if (software or original)
                else "The file carries no editing metadata, as many scans and screenshots do."
            )
            return self.passed(detail, evidence)

        severity = 1.0
        for sev, _ in messages:
            severity *= 1.0 - sev
        return self.hit(1.0 - severity, " ".join(m for _, m in messages), evidence=evidence)
