"""Find where an extracted value is printed on the page.

The model reads the values, but its coordinates are not reliable, so positions come from the
PDF text layer or OCR. A value that cannot be found there is worth a second look: the model
may have misread or invented it.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from invoice_guard.amounts import parse_amount
from invoice_guard.textlayout import Word, union_box

Box = tuple[int, int, int, int]
Lines = Sequence[Sequence[Word]]

_MIN_FRAGMENT = 5


@dataclass(frozen=True)
class Hit:
    words: tuple[Word, ...]

    @property
    def page(self) -> int:
        return self.words[0].page

    @property
    def box(self) -> Box:
        return union_box(list(self.words))


@dataclass(frozen=True)
class Anchor:
    """Where a value is expected: beside a label, or underneath a heading."""

    page: int
    x: float
    y: float
    below: bool = False

    def cost(self, hit: Hit) -> float:
        x, y, _, h = hit.box
        dx, dy = abs(x - self.x), abs(y + h / 2 - self.y)
        weighted = dx + 0.5 * dy if self.below else 10 * dy + 0.1 * dx
        return weighted + (100_000 if hit.page != self.page else 0)


def normalise(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _matches(value: str, lines: Lines) -> list[Hit]:
    """Places where the whole value is printed on one line."""
    if value.endswith("%"):
        return [Hit((w,)) for line in lines for w in line if value.replace(" ", "") in w.text]
    amount = parse_amount(value)
    if amount is not None:
        return [Hit((w,)) for line in lines for w in line if parse_amount(w.text) == amount]
    target = normalise(value)
    if not target:
        return []
    hits: list[Hit] = []
    for line in lines:
        parts = [normalise(w.text) for w in line]
        for start in range(len(line)):
            joined = ""
            for end in range(start, len(line)):
                joined += parts[end]
                if len(joined) >= len(target):
                    break
            if joined == target:
                hits.append(Hit(tuple(line[start : end + 1])))
    if hits:
        return hits
    # OCR sometimes glues a label to its value, e.g. "InvoiceNo:US-001".
    if len(target) >= 4:
        return [Hit((w,)) for line in lines for w in line if target in normalise(w.text)]
    return []


def _fragments(value: str, lines: Lines) -> list[Hit]:
    """For wrapped values such as addresses: the longest matching run on each line."""
    target = normalise(value)
    found: list[Hit] = []
    for line in lines:
        parts = [normalise(w.text) for w in line]
        best: tuple[int, int, int] | None = None  # length, start, end
        for start in range(len(line)):
            joined = ""
            for end in range(start, len(line)):
                joined += parts[end]
                if joined not in target:
                    break
                if len(joined) >= _MIN_FRAGMENT and (best is None or len(joined) > best[0]):
                    best = (len(joined), start, end)
        if best:
            found.append(Hit(tuple(line[best[1] : best[2] + 1])))
    return found


def _adjacent(a: Hit, b: Hit) -> bool:
    ax, ay, aw, ah = a.box
    bx, by, bw, bh = b.box
    gap = max(by - (ay + ah), ay - (by + bh))
    return a.page == b.page and gap <= 1.5 * max(ah, bh) and ax < bx + bw and bx < ax + aw


def _block(fragments: list[Hit], anchor: Anchor | None) -> Hit:
    """The group of neighbouring fragments around the most likely starting one."""
    if anchor:
        seed = min(fragments, key=anchor.cost)
    else:
        seed = max(fragments, key=lambda h: sum(len(w.text) for w in h.words))
    group, rest = [seed], [f for f in fragments if f != seed]
    grew = True
    while grew:
        grew = False
        for fragment in list(rest):
            if any(_adjacent(fragment, member) for member in group):
                group.append(fragment)
                rest.remove(fragment)
                grew = True
    return Hit(tuple(w for member in group for w in member.words))


def find_anchor(lines: Lines, labels: Sequence[str], below: bool = False) -> Anchor | None:
    """The first line fragment that starts with one of ``labels`` (spaces ignored)."""
    for label in labels:
        for line in lines:
            parts = [normalise(w.text) for w in line]
            for start, word in enumerate(line):
                if "".join(parts[start : start + 3]).startswith(label):
                    return Anchor(word.page, word.x0, word.cy, below)
    return None


def locate(
    value: str, lines: Lines, anchor: Anchor | None = None, used: Collection[Word] = ()
) -> Hit | None:
    """Best occurrence of ``value`` that does not reuse a word listed in ``used``."""
    hits = [h for h in _matches(value, lines) if not any(w in used for w in h.words)]
    if hits:
        return min(hits, key=anchor.cost) if anchor else hits[0]
    if parse_amount(value) is not None or len(normalise(value)) < _MIN_FRAGMENT:
        return None
    fragments = _fragments(value, lines)
    return _block(fragments, anchor) if fragments else None
