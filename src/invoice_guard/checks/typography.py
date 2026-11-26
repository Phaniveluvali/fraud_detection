"""Is every amount printed like its neighbours? Size, typeface, resolution and alignment.

A retyped or pasted figure rarely matches exactly. Each amount is compared with the text it
should look like: its label on the same line ("TOTAL"), the other values in its column, and
the page as a whole. This works on images as well as PDFs because it measures pixels.
"""

from __future__ import annotations

import io
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pdfplumber

from invoice_guard.amounts import is_money
from invoice_guard.checks.base import Check
from invoice_guard.document import Document, Page
from invoice_guard.features import has_grain
from invoice_guard.models import CheckResult, Region
from invoice_guard.textlayout import Word, union_box
from invoice_guard.typography import Glyph, TextMetrics, deviation, glyphs, measure

logger = logging.getLogger(__name__)

Note = Callable[[Word, str, str], None]
Measured = dict[Word, TextMetrics]

_SEVERITY_BY_METRICS = {1: 0.55, 2: 0.8}
_SEVERITY_MANY = 0.9


@dataclass
class _Finding:
    word: Word
    metrics: dict[str, str] = field(default_factory=dict)  # metric name -> explanation

    @property
    def severity(self) -> float:
        return _SEVERITY_BY_METRICS.get(len(self.metrics), _SEVERITY_MANY)


# Digits without rounded edges: every other digit of the same font is at least this tall.
_FLAT_DIGITS = "147"
_MIN_FLAT_SAMPLES = 4


def _is_amount(word: Word) -> bool:
    return is_money(word.text)


def label_words(row: tuple[Word, ...]) -> list[Word]:
    """The words immediately describing the value at the end of ``row``."""
    value, before = row[-1], list(row[:-1])
    if not before:
        return []
    label = [before.pop()]
    while before and label[0].x0 - before[-1].x1 < 2.0 * value.height:
        label.insert(0, before.pop())
    return label


def _columns(cells: list[tuple[Word, TextMetrics]], gap: float) -> list[list[int]]:
    """Group indices of amounts whose right edges sit in the same column."""
    order = sorted(range(len(cells)), key=lambda i: cells[i][1].right)
    groups: list[list[int]] = []
    for i in order:
        if groups and cells[i][1].right - cells[groups[-1][-1]][1].right <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


class TypographyCheck(Check):
    name = "typography"
    title = "Size, font and resolution"

    def run(self, doc: Document) -> CheckResult:
        findings: dict[tuple[int, float, float], _Finding] = {}
        compared = 0
        small_pages: list[int] = []
        digit_regions: list[Region] = []

        def note(word: Word, metric: str, message: str) -> None:
            key = (word.page, word.x0, word.top)
            findings.setdefault(key, _Finding(word)).metrics.setdefault(metric, message)

        for page in doc.pages:
            words = [w for w in doc.words if w.page == page.index]
            if not words:
                continue
            gray = page.features(self.settings.block_size).luma
            measured: Measured = {}
            for word in words:
                metrics = measure(gray, [word])
                if metrics is not None:
                    measured[word] = metrics
            amounts = [(w, m) for w, m in measured.items() if _is_amount(w)]
            if not amounts:
                continue
            compared += len(amounts)
            if statistics.median(m.height for _, m in amounts) < self.settings.type_min_height:
                small_pages.append(page.index + 1)

            self._compare_with_labels(doc, page, gray, measured, note)
            self._compare_with_column(doc, page, measured, note)
            self._compare_sharpness(amounts, list(measured.values()), note)
            self._compare_alignment(page, amounts, note)
            if doc.kind == "image":
                words_with_amounts = [w for w, _ in amounts]
                digit_regions.extend(self._odd_characters(page, gray, words_with_amounts))

        regions = [self._region(f) for f in findings.values()]
        regions.extend(digit_regions)
        if doc.kind == "pdf":
            regions.extend(self._image_resolution(doc))

        evidence = {"amounts_compared": compared, "low_resolution_pages": small_pages}
        if not regions:
            if not compared:
                return self.skipped(
                    "No readable amounts to compare. Scans need the OCR extra installed."
                )
            caveat = " The page resolution is low, so this is weak." if small_pages else ""
            return self.passed(
                f"{compared} amounts match their surroundings in size, font and sharpness.{caveat}",
                evidence,
            )
        detail = " ".join(dict.fromkeys(r.note for r in regions))
        return self.hit(max(r.score for r in regions), detail, regions, evidence)

    # --------------------------------------------------------------- comparisons
    def _compare_style(
        self, word: Word, m: TextMetrics, ref: TextMetrics, against: str, note: Note
    ) -> None:
        s = self.settings
        if deviation(m.height, ref.height) > s.type_size_ratio:
            note(
                word,
                "size",
                f'Size: "{word.text}" is {m.height:.0f}px tall but {against} is '
                f"{ref.height:.0f}px.",
            )
        spacing = deviation(m.advance, ref.advance)
        weight = deviation(m.weight, ref.weight)
        both_differ = min(spacing, weight) > s.type_font_ratio
        if both_differ or max(spacing, weight) > s.type_font_ratio_strong:
            note(
                word,
                "font",
                f'Font: "{word.text}" has a different letter width or stroke weight than '
                f"{against}.",
            )

    def _compare_with_labels(
        self,
        doc: Document,
        page: Page,
        gray: npt.NDArray[np.uint8],
        measured: Measured,
        note: Note,
    ) -> None:
        rows: list[tuple[Word, list[tuple[str, str]]]] = []
        for summary in doc.facts.summary:
            value = summary.words[-1]
            if value.page != page.index or value not in measured:
                continue
            label = label_words(summary.words)
            reference = measure(gray, label)
            if reference is None:
                continue
            name = " ".join(w.text for w in label)
            found = self._style_differences(
                value, measured[value], reference, f'the label "{name}"'
            )
            rows.append((value, found))

        odd_font = [value for value, found in rows if any(m == "font" for m, _ in found)]
        for value, found in rows:
            peers = [measured[v] for v in odd_font if v != value]
            # Several values differing from their labels in the same way is the template.
            designed = bool(peers) and self._same_face(measured[value], peers)
            for metric, message in found:
                if not (metric == "font" and designed):
                    note(value, metric, message)

    def _style_differences(
        self, word: Word, m: TextMetrics, ref: TextMetrics, against: str
    ) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []

        def collect(_: Word, metric: str, message: str) -> None:
            found.append((metric, message))

        self._compare_style(word, m, ref, against, collect)
        return found

    def _same_face(self, m: TextMetrics, peers: list[TextMetrics]) -> bool:
        limit = self.settings.type_font_ratio
        return (
            deviation(m.weight, statistics.median(p.weight for p in peers)) <= limit
            and deviation(m.height, statistics.median(p.height for p in peers)) <= limit
        )

    def _odd_characters(
        self, page: Page, gray: npt.NDArray[np.uint8], words: list[Word]
    ) -> list[Region]:
        """Single characters that are the wrong height for the font around them.

        Only for clean digital pages: there the same font renders to the same pixel height,
        so even one pixel is meaningful. On scans and photos it would be noise.
        """
        s = self.settings
        feats = page.features(s.block_size)
        if has_grain(feats, s.paper_detail_max, s.noise_min_median, s.min_paper_blocks):
            return []
        split: list[tuple[Word, list[Glyph], int]] = []
        for word in words:
            parts = glyphs(gray, word)
            if parts is None:
                continue
            tallest = max((g.height for g in parts if g.char.isdigit()), default=0)
            if tallest >= s.type_min_height:
                split.append((word, parts, tallest))

        regions: list[Region] = []
        for word, parts, tallest in split:
            family = [g for _, other, size in split if abs(size - tallest) <= 1 for g in other]
            flat = [g.height for g in family if g.char in _FLAT_DIGITS]
            if len(flat) < _MIN_FLAT_SAMPLES:
                continue
            usual = statistics.median(flat)
            symbols = [g.height for g in family if g.char == "$"]
            symbol = statistics.median(symbols) if len(symbols) >= _MIN_FLAT_SAMPLES else None

            run: list[Glyph] = []
            for glyph in [*parts, None]:
                short = glyph is not None and glyph.char.isdigit() and glyph.height <= usual - 1
                if short and glyph is not None:
                    run.append(glyph)
                    continue
                if run:
                    chars = "".join(g.char for g in run)
                    message = (
                        f'Size: "{chars}" in "{word.text}" is {run[0].height}px tall; other '
                        f"digits in this font are {usual:.0f}px."
                    )
                    regions.append(self._glyph_region(page, run, message))
                    run = []
                if (
                    glyph is not None
                    and glyph.char == "$"
                    and symbol is not None
                    and abs(glyph.height - symbol) >= 3
                ):
                    message = (
                        f'Size: the "$" in "{word.text}" is {glyph.height}px tall; the '
                        f"others are {symbol:.0f}px."
                    )
                    regions.append(self._glyph_region(page, parts, message))  # box the figure
        return regions

    def _glyph_region(self, page: Page, run: list[Glyph], message: str) -> Region:
        x0, y0 = min(g.x0 for g in run), min(g.y0 for g in run)
        return Region(
            page=page.index,
            x=x0,
            y=y0,
            w=max(g.x1 for g in run) - x0,
            h=max(g.y1 for g in run) - y0,
            signals=[self.name],
            score=0.7 if len(run) > 1 else 0.55,
            note=message,
        )

    def _compare_with_column(
        self, doc: Document, page: Page, measured: Measured, note: Note
    ) -> None:
        cells = [
            (w, measured[w])
            for row in doc.facts.items
            for w in row.words
            if w.page == page.index and w in measured and _is_amount(w)
        ]
        for group in _columns(cells, gap=page.width * 0.02):
            if len(group) < 3:
                continue
            for i in group:
                word, metrics = cells[i]
                others = [cells[j][1] for j in group if j != i]
                reference = TextMetrics(
                    height=statistics.median(o.height for o in others),
                    advance=statistics.median(o.advance for o in others),
                    weight=statistics.median(o.weight for o in others),
                    sharpness=statistics.median(o.sharpness for o in others),
                    weakest_glyph=1.0,
                    right=metrics.right,
                )
                self._compare_style(word, metrics, reference, "the rest of its column", note)

    def _compare_sharpness(
        self,
        amounts: list[tuple[Word, TextMetrics]],
        everything: list[TextMetrics],
        note: Note,
    ) -> None:
        floor = self.settings.type_sharpness_ratio
        typical = statistics.median(m.sharpness for m in everything)
        for word, m in amounts:
            if m.sharpness < typical * floor:
                note(
                    word,
                    "resolution",
                    f'Resolution: "{word.text}" is blurrier than the rest of the page.',
                )
            elif m.weakest_glyph < floor:
                note(
                    word,
                    "resolution",
                    f'Resolution: one character in "{word.text}" is softer than the others.',
                )

    def _compare_alignment(
        self, page: Page, amounts: list[tuple[Word, TextMetrics]], note: Note
    ) -> None:
        slack = max(self.settings.type_align_px, round(page.width * 0.006))
        for group in _columns(amounts, gap=page.width * 0.035):
            if len(group) < 4:
                continue
            edges = [amounts[i][1].right for i in group]
            column = statistics.median(edges)
            tight = sum(abs(e - column) <= slack / 2 for e in edges) / len(edges)
            if tight < 0.7:
                continue  # not a right-aligned column
            for i, edge in zip(group, edges, strict=True):
                if abs(edge - column) > slack:
                    word = amounts[i][0]
                    note(
                        word,
                        "alignment",
                        f'Alignment: "{word.text}" is {abs(edge - column):.0f}px out of line '
                        "with its column.",
                    )

    # ------------------------------------------------------------------ results
    def _region(self, finding: _Finding) -> Region:
        x, y, w, h = union_box([finding.word])
        return Region(
            page=finding.word.page,
            x=x,
            y=y,
            w=w,
            h=h,
            signals=[self.name],
            score=finding.severity,
            note=" ".join(finding.metrics.values()),
        )

    def _image_resolution(self, doc: Document) -> list[Region]:
        """Pictures pasted over a scanned PDF page at a different resolution."""
        regions: list[Region] = []
        try:
            with pdfplumber.open(io.BytesIO(doc.data)) as pdf:
                for page in doc.pages:
                    if page.index < len(pdf.pages):
                        regions.extend(self._patches(pdf.pages[page.index], page))
        except Exception:
            logger.debug("could not inspect embedded images", exc_info=True)
        return regions

    def _patches(self, pdf_page: Any, page: Page) -> list[Region]:
        placed: list[tuple[dict[str, Any], float, float]] = []
        for image in pdf_page.images:
            width, height = float(image["width"]), float(image["height"])
            if width <= 0 or height <= 0:
                continue
            dpi = float(image["srcsize"][0]) / (width / 72.0)
            area = width * height / float(pdf_page.width * pdf_page.height)
            placed.append((image, dpi, area))
        if not placed:
            return []
        _, base_dpi, base_area = max(placed, key=lambda item: item[2])
        if base_area < 0.6:
            return []  # no full-page scan to compare against; logos differ legitimately
        scale = page.pt_to_px or 1.0
        return [
            Region(
                page=page.index,
                x=int(image["x0"] * scale),
                y=int(image["top"] * scale),
                w=max(int(image["width"] * scale), 1),
                h=max(int(image["height"] * scale), 1),
                signals=[self.name],
                score=0.85,
                note=f"Resolution: a {dpi:.0f} DPI picture is pasted over a "
                f"{base_dpi:.0f} DPI page.",
            )
            for image, dpi, area in placed
            if area < 0.25 and deviation(dpi, base_dpi) > self.settings.type_font_ratio_strong
        ]
