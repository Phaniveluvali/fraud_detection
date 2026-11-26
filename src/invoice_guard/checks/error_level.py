"""Error-level analysis: find areas saved at a different JPEG quality than the page."""

from __future__ import annotations

import numpy as np

from invoice_guard.checks.base import Check
from invoice_guard.document import Document
from invoice_guard.features import block_mean, error_level, has_grain
from invoice_guard.models import CheckResult, Region
from invoice_guard.regions import extract_regions

_HEATMAP_GAIN = 20.0
_STRENGTH_SCALE = 10.0


class ErrorLevelCheck(Check):
    name = "error_level"
    title = "Compression history"

    def run(self, doc: Document) -> CheckResult:
        s = self.settings
        regions: list[Region] = []
        analysed = 0
        per_page: list[dict[str, float | int]] = []

        for page in doc.pages:
            feats = page.features(s.block_size)
            if not feats.usable:
                continue
            ela = error_level(page.image, s.ela_quality)
            doc.artifacts[f"ela_p{page.index}"] = np.clip(ela * _HEATMAP_GAIN, 0, 255).astype(
                np.uint8
            )
            level = block_mean(ela, feats.rows, feats.cols, s.block_size)
            # Text edges always carry error, so compare error relative to the detail present.
            ratio = level / (feats.detail + 2.0)
            # Sharp edges on clean digital graphics light up regardless of editing.
            if not has_grain(feats, s.paper_detail_max, s.noise_min_median, s.min_paper_blocks):
                continue
            textual = feats.detail > s.detail_min
            sample = ratio[textual]
            if sample.size < s.min_detail_blocks:
                continue
            analysed += 1
            median = float(np.median(sample))
            spread = 1.4826 * float(np.median(np.abs(sample - median))) + 0.01
            z = (ratio - median) / spread
            flagged = textual & (z > s.ela_z_min) & (ratio > median * s.ela_ratio_min)
            found = extract_regions(
                flagged,
                np.clip(z / _STRENGTH_SCALE, 0.0, 1.0).astype(np.float32),
                block_size=s.block_size,
                page=page.index,
                signal=self.name,
                merge_radius=s.merge_radius,
                min_blocks=s.min_region_blocks,
            )
            regions.extend(found)
            per_page.append(
                {"page": page.index, "median_ratio": round(median, 4), "regions": len(found)}
            )

        evidence = {"pages": per_page, "resave_quality": s.ela_quality}
        if analysed == 0:
            return self.skipped(
                "Not a scan or photo with enough content, so compression is not compared."
            )
        if not regions:
            return self.passed("Compression is uniform across the page.", evidence)
        strongest = max(r.score for r in regions)
        count = len(regions)
        area = "1 area" if count == 1 else f"{count} areas"
        verb = "was" if count == 1 else "were"
        return self.hit(
            0.5 + 0.45 * strongest,
            f"{area} {verb} saved at a different quality than the rest of the page, "
            "which is typical of pasted or retyped content.",
            regions,
            evidence,
        )
