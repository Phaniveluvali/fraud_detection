"""Measure how a piece of text is printed: its size, typeface shape and sharpness.

Measurements are taken from the page pixels, not from the OCR engine, so they work for scans,
screenshots and rendered PDFs alike. OCR (or the PDF text layer) only says where to look.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from invoice_guard.textlayout import Word

_MIN_CONTRAST = 60.0
_MIN_HEIGHT = 6.0
_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class TextMetrics:
    height: float  # typical glyph height in pixels
    advance: float  # ink width per character / height: condensed vs wide faces
    weight: float  # stroke width / height: light vs bold faces
    sharpness: float  # edge steepness relative to contrast; low means resampled or blurred
    weakest_glyph: float  # sharpness of the softest glyph relative to the others
    right: int  # x of the right-most ink pixel


def measure(gray: npt.NDArray[np.uint8], words: Sequence[Word]) -> TextMetrics | None:
    """Metrics for the text covered by ``words``, or ``None`` if too faint or small."""
    if not words:
        return None
    x0 = max(int(min(w.x0 for w in words)) - 1, 0)
    y0 = max(int(min(w.top for w in words)), 0)
    x1 = min(int(max(w.x1 for w in words)) + 2, gray.shape[1])
    y1 = min(int(max(w.bottom for w in words)) + 1, gray.shape[0])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    background = float(np.percentile(crop, 90))
    contrast = background - float(np.percentile(crop, 2))
    if contrast < _MIN_CONTRAST:
        return None
    ink = crop < background - 0.45 * contrast
    if not ink.any():
        return None

    labels, _ = ndimage.label(ink, structure=_EIGHT)
    boxes = ndimage.find_objects(labels)
    heights = np.array([b[0].stop - b[0].start for b in boxes if b is not None], np.float32)
    # Capital and ascender height, so lowercase-heavy labels compare fairly with digits.
    height = float(np.percentile(heights, 90))
    if height < _MIN_HEIGHT:
        return None

    columns = np.nonzero(ink.any(axis=0))[0]
    characters = max(sum(len(w.text) for w in words), 1)
    outline = ink & ~ndimage.binary_erosion(ink)
    stroke = 2.0 * float(ink.sum()) / max(float(outline.sum()), 1.0)

    gradient = np.hypot(ndimage.sobel(crop, 0), ndimage.sobel(crop, 1)) / 4.0
    sharpness = float(np.percentile(gradient[ndimage.binary_dilation(ink)], 90)) / contrast

    # Compare full-height glyphs with each other to catch a single replaced digit.
    per_glyph: list[float] = []
    for index, box in enumerate(boxes, start=1):
        if box is None or box[0].stop - box[0].start < 0.7 * height:
            continue
        member = ndimage.binary_dilation(labels[box] == index)
        per_glyph.append(float(np.percentile(gradient[box][member], 90)) / contrast)
    weakest = 1.0
    if len(per_glyph) >= 3:
        weakest = min(per_glyph) / max(float(np.median(per_glyph)), 1e-6)

    return TextMetrics(
        height=height,
        advance=float(columns[-1] - columns[0] + 1) / (characters * height),
        weight=stroke / height,
        sharpness=sharpness,
        weakest_glyph=weakest,
        right=x0 + int(columns[-1]),
    )


@dataclass(frozen=True)
class Glyph:
    char: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def glyphs(gray: npt.NDArray[np.uint8], word: Word) -> list[Glyph] | None:
    """One box per character, or ``None`` if characters cannot be separated reliably."""
    x0, y0 = max(int(word.x0) - 2, 0), max(int(word.top) - 1, 0)
    crop = gray[y0 : int(word.bottom) + 2, x0 : int(word.x1) + 3].astype(np.float32)
    if crop.size == 0:
        return None
    background = float(np.percentile(crop, 90))
    contrast = background - float(crop.min())
    if contrast < _MIN_CONTRAST:
        return None
    labels, _ = ndimage.label(crop < background - 0.35 * contrast, structure=_EIGHT)
    boxes = sorted((b for b in ndimage.find_objects(labels) if b), key=lambda b: b[1].start)
    characters = word.text.replace(" ", "")
    if len(boxes) != len(characters):
        return None
    return [
        Glyph(c, x0 + b[1].start, y0 + b[0].start, x0 + b[1].stop, y0 + b[0].stop)
        for c, b in zip(characters, boxes, strict=True)
    ]


def deviation(value: float, reference: float) -> float:
    """How far apart two positive measurements are, as a ratio that is always >= 1."""
    if value <= 0 or reference <= 0:
        return 1.0
    return max(value / reference, reference / value)
