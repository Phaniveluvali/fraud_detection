"""Words and lines, with geometry in the pixel space of the analysed page image."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    page: int
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    fontname: str | None = None
    size: float | None = None

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 1.0)


def group_lines(words: list[Word]) -> list[list[Word]]:
    """Cluster words into visual lines, per page, each line ordered left to right."""
    lines: list[list[Word]] = []
    by_page: dict[int, list[Word]] = {}
    for word in words:
        by_page.setdefault(word.page, []).append(word)
    for page in sorted(by_page):
        page_words = sorted(by_page[page], key=lambda w: (w.cy, w.x0))
        tol = 0.6 * statistics.median(w.height for w in page_words)
        current: list[Word] = []
        anchor = 0.0
        for word in page_words:
            if current and abs(word.cy - anchor) > tol:
                lines.append(sorted(current, key=lambda w: w.x0))
                current = []
            if not current:
                anchor = word.cy
            current.append(word)
            anchor = sum(w.cy for w in current) / len(current)
        if current:
            lines.append(sorted(current, key=lambda w: w.x0))
    return lines


def union_box(words: list[Word]) -> tuple[int, int, int, int]:
    """Bounding box ``(x, y, w, h)`` of ``words`` as integers."""
    x0 = min(w.x0 for w in words)
    y0 = min(w.top for w in words)
    x1 = max(w.x1 for w in words)
    y1 = max(w.bottom for w in words)
    return int(x0), int(y0), max(int(x1 - x0), 1), max(int(y1 - y0), 1)
