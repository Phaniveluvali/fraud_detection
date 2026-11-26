"""Compare a suspect document with a trusted original and locate what changed.

This answers a different question from the pipeline. It needs both files, but it then finds
every visible difference, including edits no single-file check could notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage

from invoice_guard.config import Settings, get_settings
from invoice_guard.document import Document
from invoice_guard.models import Region

# Largest per-channel difference still explained by saving the same page in another format.
DEFAULT_THRESHOLD = 70
_MIN_PIXELS = 3
_GROUP_DISTANCE = 6


@dataclass
class Comparison:
    differences: list[Region]
    suspect_pages: list[Image.Image] = field(repr=False)
    warnings: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not self.differences


def _text_in(doc: Document, region: Region) -> str:
    return " ".join(
        w.text
        for w in doc.words
        if w.page == region.page
        and w.x0 < region.x1 + 2
        and w.x1 > region.x - 2
        and region.y - 2 <= w.cy <= region.y1 + 2
    )


def _page_differences(
    original: Image.Image, suspect: Image.Image, page: int, threshold: int
) -> list[Region]:
    a = np.asarray(original, dtype=np.int16)
    b = np.asarray(suspect, dtype=np.int16)
    changed = np.abs(a - b).max(axis=2) > threshold
    if not changed.any():
        return []
    grouped = ndimage.binary_dilation(changed, iterations=_GROUP_DISTANCE)
    labels, _ = ndimage.label(grouped)
    members = np.where(changed, labels, 0)
    regions = []
    for index, box in enumerate(ndimage.find_objects(members), start=1):
        if box is None or int((members[box] == index).sum()) < _MIN_PIXELS:
            continue
        rows, cols = box
        regions.append(
            Region(
                page=page,
                x=cols.start,
                y=rows.start,
                w=cols.stop - cols.start,
                h=rows.stop - rows.start,
                signals=["changed"],
                score=1.0,
            )
        )
    return regions


def compare(
    original: bytes,
    suspect: bytes,
    settings: Settings | None = None,
    threshold: int = DEFAULT_THRESHOLD,
) -> Comparison:
    cfg = settings or get_settings()
    first = Document.from_bytes(original, "original", cfg)
    second = Document.from_bytes(suspect, "suspect", cfg)
    warnings = [*first.warnings, *second.warnings]
    if len(first.pages) != len(second.pages):
        warnings.append(
            f"Page counts differ: original has {len(first.pages)}, suspect {len(second.pages)}"
        )

    regions: list[Region] = []
    for a, b in zip(first.pages, second.pages, strict=False):
        image = b.image
        if image.size != a.image.size:
            warnings.append(
                f"Page {a.index + 1} was resized to match the original; expect extra differences"
            )
            image = image.resize(a.image.size, Image.Resampling.LANCZOS)
        regions.extend(_page_differences(a.image, image, a.index, threshold))

    regions.sort(key=lambda r: (r.page, -r.w * r.h))
    for number, region in enumerate(regions, start=1):
        region.id = number
        before, after = _text_in(first, region), _text_in(second, region)
        if before or after:
            region.note = f'"{before or "(nothing)"}" in the original, "{after or "(nothing)"}" now'
    return Comparison(regions, [p.image for p in second.pages], warnings)
