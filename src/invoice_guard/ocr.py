"""Optional OCR for image-only documents.

RapidOCR (Apache-2.0, ONNX runtime) is an extra: ``pip install invoice-guard[ocr]``. Without
it, text-based checks are skipped for scans instead of failing.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from PIL import Image

from invoice_guard.textlayout import Word

logger = logging.getLogger(__name__)

_engine: Any = None
_engine_lock = threading.Lock()
_unavailable = False


def _get_engine() -> Any | None:
    global _engine, _unavailable
    if _unavailable:
        return None
    with _engine_lock:
        if _engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                _unavailable = True
                logger.info("OCR engine not installed; scanned documents skip text checks")
                return None
            _engine = RapidOCR()
    return _engine


def is_available() -> bool:
    return _get_engine() is not None


def split_points(
    image: Image.Image, box: tuple[float, float, float, float], estimates: list[float], reach: float
) -> list[tuple[float, float]]:
    """For each estimated word boundary, the nearby blank gap as (left end, right start)."""
    x0, top, x1, bottom = (int(box[0]), int(box[1]), int(box[2]) + 1, int(box[3]) + 1)
    gray = np.asarray(image.crop((x0, top, x1, bottom)).convert("L"), dtype=np.float32)
    if gray.size == 0:
        return [(e, e) for e in estimates]
    paper = float(np.percentile(gray, 90))
    mask = gray < paper - 0.3 * (paper - float(gray.min()))
    inked = np.atleast_1d(mask.any(axis=0)).tolist()
    gaps: list[tuple[int, int]] = []
    start = None
    for column, ink in enumerate([*inked, True]):
        if not ink and start is None:
            start = column
        elif ink and start is not None:
            gaps.append((x0 + start, x0 + column))
            start = None
    out = []
    for estimate in estimates:
        near = [g for g in gaps if abs((g[0] + g[1]) / 2 - estimate) <= reach]
        # the widest gap nearby is the space; narrow ones separate letters
        ranked = [(g[1] - g[0], -abs((g[0] + g[1]) / 2 - estimate), g) for g in near]
        best = max(ranked)[2] if ranked else None
        out.append((float(best[0]), float(best[1])) if best else (estimate, estimate))
    return out


def recognize(image: Image.Image, page: int) -> list[Word]:
    """OCR one page image and return word boxes in that image's pixel coordinates."""
    engine = _get_engine()
    if engine is None:
        return []
    result, _ = engine(np.asarray(image.convert("RGB")))
    words: list[Word] = []
    for box, text, _conf in result or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1, top, bottom = min(xs), max(xs), min(ys), max(ys)
        tokens = str(text).split()
        total_chars = sum(len(t) for t in tokens) + max(len(tokens) - 1, 0)
        if not tokens or total_chars == 0:
            continue
        per_char = (x1 - x0) / total_chars
        estimates, cursor = [], x0
        for token in tokens[:-1]:
            cursor += per_char * len(token)
            estimates.append(cursor + per_char / 2)
            cursor += per_char
        gaps = split_points(image, (x0, top, x1, bottom), estimates, 2.0 * per_char)
        lefts = [x0, *(g[1] for g in gaps)]
        rights = [*(g[0] for g in gaps), x1]
        words.extend(
            Word(page, token, left, top, right, bottom)
            for token, left, right in zip(tokens, lefts, rights, strict=True)
        )
    return words
