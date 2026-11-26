"""Recover earlier revisions of a PDF and report which numbers changed between them.

A PDF edited with an incremental update keeps its previous version inside the file. Cutting
the bytes at an earlier ``%%EOF`` marker yields that version, so the old and new amounts can be
compared directly. This is the strongest single signal available for digital PDFs.
"""

from __future__ import annotations

import io
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import pdfplumber

from invoice_guard.amounts import has_decimals, parse_amount

logger = logging.getLogger(__name__)

_EOF = re.compile(rb"%%EOF[\r\n ]*")
_PAIR_DISTANCE_PT = 80.0


def revision_ends(data: bytes) -> list[int]:
    """Byte offsets just after each revision's ``%%EOF`` marker.

    Linearised files carry two markers in their first revision; the first one is dropped so a
    web-optimised export is not mistaken for an edit.
    """
    ends = [m.end() for m in _EOF.finditer(data)]
    if len(ends) > 1 and b"/Linearized" in data[:2048]:
        ends = ends[1:]
    return ends


@dataclass(frozen=True)
class NumericChange:
    page: int
    old: str | None
    new: str | None
    box: tuple[int, int, int, int]  # x, y, w, h in page-image pixels


@dataclass(frozen=True)
class _Token:
    page: int
    text: str
    cx: float
    cy: float
    box: tuple[float, float, float, float]  # x0, top, x1, bottom in points


def _numeric_tokens(buf: bytes, max_pages: int) -> list[_Token] | None:
    try:
        tokens: list[_Token] = []
        with pdfplumber.open(io.BytesIO(buf)) as pdf:
            for index, page in enumerate(pdf.pages[:max_pages]):
                for w in page.extract_words():
                    text = str(w["text"])
                    if parse_amount(text) is None:
                        continue
                    digits = sum(c.isdigit() for c in text)
                    if not (has_decimals(text) or "," in text or digits >= 3):
                        continue
                    box = (float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
                    tokens.append(
                        _Token(index, text, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box)
                    )
        return tokens
    except Exception:
        logger.debug("could not parse earlier revision", exc_info=True)
        return None


def _unmatched(tokens: list[_Token], surplus: Counter[tuple[int, str]]) -> list[_Token]:
    remaining = Counter(surplus)
    out: list[_Token] = []
    for token in tokens:
        key = (token.page, token.text)
        if remaining[key] > 0:
            remaining[key] -= 1
            out.append(token)
    return out


def numeric_changes(
    data: bytes, ends: list[int], pt_to_px: dict[int, float], max_pages: int
) -> list[NumericChange]:
    """Numbers that differ between the last two revisions, paired by position where possible."""
    if len(ends) < 2:
        return []
    before = _numeric_tokens(data[: ends[-2]], max_pages)
    after = _numeric_tokens(data[: ends[-1]], max_pages)
    if before is None or after is None:
        return []

    count_before = Counter((t.page, t.text) for t in before)
    count_after = Counter((t.page, t.text) for t in after)
    added = _unmatched(after, count_after - count_before)
    removed = _unmatched(before, count_before - count_after)

    changes: list[NumericChange] = []
    free = list(removed)
    for new in added:
        best: _Token | None = None
        best_distance = _PAIR_DISTANCE_PT
        for old in free:
            if old.page != new.page:
                continue
            distance = math.hypot(old.cx - new.cx, old.cy - new.cy)
            if distance <= best_distance:
                best, best_distance = old, distance
        if best is not None:
            free.remove(best)
        changes.append(
            NumericChange(new.page, best.text if best else None, new.text, _px(new, pt_to_px))
        )
    changes.extend(NumericChange(old.page, old.text, None, _px(old, pt_to_px)) for old in free)
    return changes


def _px(token: _Token, pt_to_px: dict[int, float]) -> tuple[int, int, int, int]:
    scale = pt_to_px.get(token.page, 1.0)
    x0, top, x1, bottom = token.box
    return (
        int(x0 * scale),
        int(top * scale),
        max(int((x1 - x0) * scale), 1),
        max(int((bottom - top) * scale), 1),
    )
