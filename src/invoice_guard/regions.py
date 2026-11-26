"""Turn block masks into rectangles and merge rectangles from different checks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from invoice_guard.models import Region

_EIGHT = np.ones((3, 3), dtype=bool)


def extract_regions(
    flag: npt.NDArray[np.bool_],
    strength: npt.NDArray[np.float32],
    *,
    block_size: int,
    page: int,
    signal: str,
    merge_radius: int,
    min_blocks: int,
) -> list[Region]:
    """Group flagged blocks that lie within ``merge_radius`` blocks of each other."""
    if not flag.any():
        return []
    grouped = ndimage.binary_dilation(flag, structure=_EIGHT, iterations=merge_radius)
    labels, _ = ndimage.label(grouped, structure=_EIGHT)
    flagged_labels = np.where(flag, labels, 0)
    regions: list[Region] = []
    for idx, box in enumerate(ndimage.find_objects(flagged_labels), start=1):
        if box is None:
            continue
        member = flagged_labels[box] == idx
        if int(member.sum()) < min_blocks:
            continue
        rows, cols = box
        regions.append(
            Region(
                page=page,
                x=cols.start * block_size,
                y=rows.start * block_size,
                w=(cols.stop - cols.start) * block_size,
                h=(rows.stop - rows.start) * block_size,
                signals=[signal],
                score=float(np.clip(strength[box][member].max(), 0.0, 1.0)),
            )
        )
    regions.sort(key=lambda r: (-r.w * r.h, r.y, r.x))
    return regions


def _touch(a: Region, b: Region, pad: int) -> bool:
    return (
        a.page == b.page
        and a.x - pad < b.x1
        and b.x - pad < a.x1
        and a.y - pad < b.y1
        and b.y - pad < a.y1
    )


def merge_regions(regions: list[Region], pad: int = 0) -> list[Region]:
    """Union boxes that overlap (within ``pad`` pixels) and pool their signals.

    A region reported by two independent checks is stronger evidence than either alone, so
    the merged region keeps the list of every signal that fired.
    """
    parent = list(range(len(regions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if _touch(regions[i], regions[j], pad):
                parent[find(j)] = find(i)

    groups: dict[int, list[Region]] = {}
    for i, region in enumerate(regions):
        groups.setdefault(find(i), []).append(region)

    merged: list[Region] = []
    for members in groups.values():
        x0 = min(m.x for m in members)
        y0 = min(m.y for m in members)
        x1 = max(m.x1 for m in members)
        y1 = max(m.y1 for m in members)
        signals = sorted({s for m in members for s in m.signals})
        notes = "; ".join(dict.fromkeys(m.note for m in members if m.note))
        merged.append(
            Region(
                page=members[0].page,
                x=x0,
                y=y0,
                w=x1 - x0,
                h=y1 - y0,
                signals=signals,
                score=max(m.score for m in members),
                note=notes,
            )
        )
    merged.sort(key=lambda r: (r.page, -len(r.signals), -r.score, r.y, r.x))
    for number, region in enumerate(merged, start=1):
        region.id = number
    return merged
