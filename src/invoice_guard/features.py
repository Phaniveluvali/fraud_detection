"""Vectorised per-page measurements shared by the pixel-level checks.

Everything here works on a regular grid of ``block_size`` pixel squares. All full-resolution
arrays are cropped to the same ``rows * block_size`` by ``cols * block_size`` window so that
block indices line up across checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]
I16 = npt.NDArray[np.int16]


def grid_shape(height: int, width: int, block_size: int) -> tuple[int, int]:
    """Number of whole blocks that fit once one pixel is reserved for neighbour differences."""
    return max((height - 1) // block_size, 0), max((width - 1) // block_size, 0)


def block_mean(a: npt.NDArray[np.generic], rows: int, cols: int, bs: int) -> F32:
    window = a[: rows * bs, : cols * bs]
    reduced = window.reshape(rows, bs, cols, bs).mean(axis=(1, 3), dtype=np.float32)
    return np.asarray(reduced, dtype=np.float32)


def block_sum(a: npt.NDArray[np.generic], rows: int, cols: int, bs: int) -> npt.NDArray[np.int64]:
    window = a[: rows * bs, : cols * bs]
    reduced = window.reshape(rows, bs, cols, bs).sum(axis=(1, 3), dtype=np.int64)
    return np.asarray(reduced, dtype=np.int64)


@dataclass(frozen=True)
class PageFeatures:
    block_size: int
    rows: int
    cols: int
    luma: U8
    detail: F32  # mean gradient energy per block; high where text or lines live

    @property
    def usable(self) -> bool:
        return self.rows > 0 and self.cols > 0


def compute_features(image: Image.Image, block_size: int) -> PageFeatures:
    luma = np.asarray(image.convert("L"), dtype=np.uint8)
    rows, cols = grid_shape(luma.shape[0], luma.shape[1], block_size)
    if rows == 0 or cols == 0:
        return PageFeatures(block_size, 0, 0, luma, np.zeros((0, 0), np.float32))
    wide = luma.astype(np.int16)
    gx = np.abs(wide[:-1, 1:] - wide[:-1, :-1])
    gy = np.abs(wide[1:, :-1] - wide[:-1, :-1])
    detail = block_mean(gx + gy, rows, cols, block_size)
    return PageFeatures(block_size, rows, cols, luma, detail)


def error_level(image: Image.Image, quality: int) -> F32:
    """Mean absolute difference between the image and a JPEG re-save, per pixel.

    Regions that were saved at a different quality than their surroundings react
    differently to the re-save, which is the basis of error-level analysis.
    """
    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as resaved:
        b = np.asarray(resaved.convert("RGB"), dtype=np.int16)
    a = np.asarray(image, dtype=np.int16)
    return np.asarray(np.abs(a - b).sum(axis=2) / 3.0, dtype=np.float32)


def has_grain(
    feats: PageFeatures, paper_detail_max: float, min_median: float, min_blocks: int
) -> bool:
    """True for scans and photos; False for clean digital graphics."""
    noise = paper_noise(feats)
    values = noise[(feats.detail < paper_detail_max) & ~np.isnan(noise)]
    return bool(values.size >= min_blocks and np.median(values) >= min_median)


def paper_noise(feats: PageFeatures, min_fill: float = 0.5) -> F32:
    """Mean neighbour difference over near-white pixels, per block; NaN where not paper.

    Only blocks that are mostly blank paper take part. Scanned paper has a faint grain, while
    a filled or repainted patch is perfectly flat.
    """
    rows, cols, bs = feats.rows, feats.cols, feats.block_size
    wide = feats.luma.astype(np.int16)
    a = wide[:-1, :-1]
    b = wide[:-1, 1:]
    diff = np.abs(a - b)
    paper = (np.minimum(a, b) > 200) & (diff < 14)
    total = block_sum(diff * paper, rows, cols, bs).astype(np.float32)
    count = block_sum(paper, rows, cols, bs).astype(np.float32)
    out = np.full((rows, cols), np.nan, dtype=np.float32)
    valid = count >= min_fill * bs * bs
    out[valid] = total[valid] / count[valid]
    return out
