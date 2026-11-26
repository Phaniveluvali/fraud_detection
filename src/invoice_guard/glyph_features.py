"""Describe every digit on a page by how it differs from the digits around it.

An edited digit rarely matches its neighbours in every respect: height, baseline, stroke
width, ink darkness, anti-aliasing, colour fringing, sharpness, spacing, the paper behind it
and its shape. Each of those is measured here, relative to the other digits in the same
number and to digits of a similar size on the page. A trained model weighs them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from invoice_guard.textlayout import Word

_EIGHT = np.ones((3, 3), dtype=bool)
_FLAT = "147"
_SHAPE_H, _SHAPE_W = 20, 16
_MIN_HEIGHT = 7
_INK_THRESHOLDS = (0.35, 0.5, 0.25, 0.65)
F32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Glyph:
    char: str
    x0: int
    y0: int
    x1: int
    y1: int
    height: float
    width: float
    top: float
    bottom: float
    stroke: float
    core: float  # mean darkness of the ink
    soft: float  # share of half-tone (anti-aliasing) pixels
    sharp: float
    fringe: float  # colour fringing, as left by sub-pixel text rendering
    halo: float  # paper brightness just around the glyph
    halo_std: float
    shape: F32


@dataclass(frozen=True)
class NumberWord:
    word: Word
    glyphs: list[Glyph]  # digits only
    gaps: list[tuple[float, float]]  # space left and right of each digit, over height
    height: float  # median digit height
    stroke: float


def _shape(patch: F32) -> F32:
    """Resample a glyph's darkness to a fixed height, keeping its aspect ratio."""
    h, w = patch.shape
    new_w = max(1, min(_SHAPE_W, round(w * _SHAPE_H / h)))
    rows = np.clip((np.arange(_SHAPE_H) + 0.5) * h / _SHAPE_H - 0.5, 0, h - 1)
    cols = np.clip((np.arange(new_w) + 0.5) * w / new_w - 0.5, 0, w - 1)
    grid = np.meshgrid(rows, cols, indexing="ij")
    out = np.zeros((_SHAPE_H, _SHAPE_W), dtype=np.float32)
    offset = (_SHAPE_W - new_w) // 2
    out[:, offset : offset + new_w] = ndimage.map_coordinates(patch, grid, order=1)
    norm = float(np.linalg.norm(out))
    return out / norm if norm else out


def _plausible(characters: str, boxes: list[tuple[int, tuple[slice, slice]]]) -> bool:
    """A sliver or a blob where a digit should be means the split went wrong."""
    for char, (_, (rows, cols)) in zip(characters, boxes, strict=True):
        aspect = (cols.stop - cols.start) / (rows.stop - rows.start)
        if char.isdigit() and char != "1" and not 0.38 <= aspect <= 1.15:
            return False
    return True


def _split_characters(
    dark: F32, threshold: float, characters: str, centre: tuple[float, float]
) -> tuple[npt.NDArray[np.int32], list[tuple[int, tuple[slice, slice]]]] | None:
    """Label one component per character, or ``None`` if that does not work out."""
    labels, count = ndimage.label(dark > threshold, structure=_EIGHT)
    parts: list[tuple[int, tuple[slice, slice], int]] = []  # id, box, pixels
    for index, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            continue
        # neighbouring text caught by a loose box is not part of this word
        if centre[0] <= (box[1].start + box[1].stop) / 2 <= centre[1]:
            parts.append((index, box, int((labels[box] == index).sum())))

    wanted = len(characters)
    while len(parts) > wanted:
        smallest = min(parts, key=lambda p: p[2])
        if smallest[2] > 3:
            return None
        parts.remove(smallest)

    # touching characters ("00") are cut at the thinnest column near the middle
    while len(parts) < wanted and parts:
        index, (rows, cols), _ = max(parts, key=lambda p: p[1][1].stop - p[1][1].start)
        width, height = cols.stop - cols.start, rows.stop - rows.start
        if width < 0.95 * height or width < 6:
            return None
        member = labels[rows, cols] == index
        ink = (dark[rows, cols] * member).sum(axis=0)
        pieces = max(2, round(width / (0.62 * height)))
        target = width / pieces
        low = max(1, int(target * 0.6))
        high = min(width - 1, int(target * 1.4) + 1)
        if high <= low:
            return None
        cut = low + int(np.argmin(ink[low:high]))
        count += 1
        right = labels[rows, cols][:, cut:]
        right[right == index] = count
        parts = [p for p in parts if p[0] != index]
        for new, offset in ((index, 0), (count, cut)):
            span = slice(cols.start + offset, cols.start + (cut if new == index else width))
            ys, xs = np.nonzero(labels[rows, span] == new)
            if ys.size == 0:
                return None
            box = (
                slice(rows.start + int(ys.min()), rows.start + int(ys.max()) + 1),
                slice(span.start + int(xs.min()), span.start + int(xs.max()) + 1),
            )
            parts.append((new, box, int(ys.size)))

    if len(parts) != wanted:
        return None
    boxes = sorted(((i, b) for i, b, _ in parts), key=lambda item: item[1][1].start)
    return (labels, boxes) if _plausible(characters, boxes) else None


def _segment(rgb: F32, word: Word) -> tuple[list[tuple[str, Glyph]], bool] | None:
    """Split a word into one glyph per character; ``None`` if that cannot be done."""
    x0, y0 = max(int(word.x0) - 4, 0), max(int(word.top) - 2, 0)
    x1, y1 = min(int(word.x1) + 5, rgb.shape[1]), min(int(word.bottom) + 3, rgb.shape[0])
    colour = rgb[y0:y1, x0:x1]
    if colour.shape[0] < _MIN_HEIGHT or colour.shape[1] < 3:
        return None
    gray = colour @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    paper = float(np.percentile(gray, 90))
    contrast = paper - float(gray.min())
    if contrast < 60:
        return None
    dark = np.clip((paper - gray) / contrast, 0.0, 1.0)
    characters = word.text.replace(" ", "")
    centre = (word.x0 - 2.5 - x0, word.x1 + 2.5 - x0)
    for threshold in _INK_THRESHOLDS:
        found = _split_characters(dark, threshold, characters, centre)
        if found is not None:
            labels, boxes = found
            break
    else:
        return None

    gradient = np.hypot(ndimage.sobel(dark, 0), ndimage.sobel(dark, 1)) / 4.0
    near_any_ink = ndimage.binary_dilation(dark > 0.08)
    fringe_map = np.abs(colour[..., 0] - colour[..., 2])
    for char, (_, box) in zip(characters, boxes, strict=True):
        height, width = box[0].stop - box[0].start, box[1].stop - box[1].start
        if char.isdigit() and char != "1" and not 0.4 <= width / height <= 1.15:
            return None
    out: list[tuple[str, Glyph]] = []
    for char, (index, box) in zip(characters, boxes, strict=True):
        rows, cols = box
        member = ndimage.binary_dilation(labels == index) & (dark > 0.08)
        core = labels == index
        if not core.any():
            return None
        column = dark[rows, cols] * core[rows, cols]
        profile = column.max(axis=1)
        # sub-pixel top and bottom: partly covered edge rows count partly
        top = y0 + rows.start + (1.0 - float(profile[0]))
        bottom = y0 + rows.stop - (1.0 - float(profile[-1]))
        outline = core & ~ndimage.binary_erosion(core)
        pad_r = slice(max(rows.start - 3, 0), rows.stop + 3)
        pad_c = slice(max(cols.start - 3, 0), cols.stop + 3)
        halo = gray[pad_r, pad_c][~near_any_ink[pad_r, pad_c]]
        out.append(
            (
                char,
                Glyph(
                    char=char,
                    x0=x0 + cols.start,
                    y0=y0 + rows.start,
                    x1=x0 + cols.stop,
                    y1=y0 + rows.stop,
                    height=bottom - top,
                    width=float(cols.stop - cols.start),
                    top=top,
                    bottom=bottom,
                    stroke=2.0 * float(core.sum()) / max(float(outline.sum()), 1.0),
                    core=float(dark[core].mean()),
                    soft=float(((dark[member] > 0.08) & (dark[member] < 0.75)).mean()),
                    sharp=float(np.percentile(gradient[member], 90)),
                    fringe=float(fringe_map[member].mean()),
                    halo=float(halo.mean()) if halo.size else paper,
                    halo_std=float(halo.std()) if halo.size else 0.0,
                    shape=_shape(column.astype(np.float32)),
                ),
            )
        )
    return out, True


def number_words(rgb: F32, words: Sequence[Word]) -> list[NumberWord]:
    """Every word containing digits that can be split into characters."""
    result: list[NumberWord] = []
    for word in words:
        if not any(c.isdigit() for c in word.text):
            continue
        segmented = _segment(rgb, word)
        if segmented is None:
            continue
        everything = [g for _, g in segmented[0]]
        digits, gaps = [], []
        for position, glyph in enumerate(everything):
            if not glyph.char.isdigit() or glyph.height < _MIN_HEIGHT:
                continue
            left = glyph.x0 - everything[position - 1].x1 if position else np.nan
            right = (
                everything[position + 1].x0 - glyph.x1 if position + 1 < len(everything) else np.nan
            )
            digits.append(glyph)
            gaps.append((left / glyph.height, right / glyph.height))
        if digits:
            result.append(
                NumberWord(
                    word,
                    digits,
                    gaps,
                    float(np.median([g.height for g in digits])),
                    float(np.median([g.stroke for g in digits])),
                )
            )
    return result


def _median(values: list[float], default: float) -> float:
    return float(np.median(values)) if values else default


def _shape_scores(glyph: Glyph, same: list[Glyph]) -> tuple[float, float, int]:
    """Best match with the same character elsewhere, and how well those match each other."""
    if not same:
        return 0.93, 0.0, 0
    stack = np.stack([g.shape.ravel() for g in same[:24]])
    best = float((stack @ glyph.shape.ravel()).max())
    typical = best
    if len(stack) >= 2:
        similar = stack @ stack.T
        np.fill_diagonal(similar, -1.0)
        typical = float(np.median(similar.max(axis=1)))
    return best, best - typical, min(len(same), 10)


FEATURE_COUNT = 30


def features(numbers: list[NumberWord], grain: bool) -> tuple[F32, list[tuple[int, int]]]:
    """One row per digit, and ``(word index, digit index)`` for each row."""
    rows: list[list[float]] = []
    keys: list[tuple[int, int]] = []
    page_height = _median([n.height for n in numbers], 1.0)
    flat = [(g, i) for i, n in enumerate(numbers) for g in n.glyphs]

    for w, number in enumerate(numbers):
        similar_size = [
            (g, i)
            for g, i in flat
            if abs(numbers[i].height - number.height) <= max(1.5, 0.12 * number.height)
        ]
        for d, glyph in enumerate(number.glyphs):
            mates = [g for k, g in enumerate(number.glyphs) if k != d]
            group = [g for g, i in similar_size if g is not glyph]
            # prefer peers of the same weight, so bold text is not compared with regular
            weighted = [g for g in group if 0.8 <= g.stroke / max(glyph.stroke, 0.1) <= 1.25]
            peers = weighted if len(weighted) >= 6 else group
            same_class = [g for g in peers if (g.char in _FLAT) == (glyph.char in _FLAT)]
            ref_h = _median([g.height for g in same_class], glyph.height)

            def rel(attr: str, pool: list[Glyph], g: Glyph = glyph) -> float:
                value = float(getattr(g, attr))
                return value - _median([float(getattr(o, attr)) for o in pool], value)

            same = [g for g in group if g.char == glyph.char]
            best, margin, count = _shape_scores(glyph, same)
            aspect = glyph.width / glyph.height
            same_aspect = _median([g.width / g.height for g in same], aspect)
            gap_values = [v for pair in number.gaps for v in pair if not np.isnan(v)]
            gap_ref = _median(gap_values, 0.1)
            left, right = number.gaps[d]
            rows.append(
                [
                    glyph.height - ref_h,
                    glyph.height / max(ref_h, 1.0),
                    rel("bottom", mates),
                    rel("top", mates),
                    glyph.stroke / max(_median([g.stroke for g in mates], glyph.stroke), 0.1),
                    glyph.stroke / max(_median([g.stroke for g in peers], glyph.stroke), 0.1),
                    rel("core", mates),
                    rel("core", peers),
                    rel("soft", mates),
                    rel("soft", peers),
                    glyph.sharp / max(_median([g.sharp for g in mates], glyph.sharp), 1e-3),
                    glyph.sharp / max(_median([g.sharp for g in peers], glyph.sharp), 1e-3),
                    rel("fringe", peers),
                    rel("halo", mates),
                    rel("halo", peers),
                    rel("halo_std", peers),
                    rel("halo_std", mates),
                    aspect / max(same_aspect, 0.05),
                    best,
                    margin,
                    float(count),
                    0.0 if np.isnan(left) else left - gap_ref,
                    0.0 if np.isnan(right) else right - gap_ref,
                    float(glyph.char in _FLAT),
                    number.height / page_height,
                    number.stroke / max(_median([g.stroke for g in peers], number.stroke), 0.1),
                    number.height,
                    float(len(mates)),
                    float(len(peers)),
                    float(grain),
                ]
            )
            keys.append((w, d))
    matrix = np.array(rows, dtype=np.float32).reshape(-1, FEATURE_COUNT)
    return matrix, keys
