"""Noise consistency: scanned paper has grain, a filled or repainted patch does not."""

from __future__ import annotations

import numpy as np

from invoice_guard.checks.base import Check
from invoice_guard.document import Document
from invoice_guard.features import paper_noise
from invoice_guard.models import CheckResult, Region
from invoice_guard.regions import extract_regions


class NoiseConsistencyCheck(Check):
    name = "noise"
    title = "Background noise"

    def run(self, doc: Document) -> CheckResult:
        s = self.settings
        regions: list[Region] = []
        usable_pages = 0
        medians: list[float] = []

        for page in doc.pages:
            feats = page.features(s.block_size)
            if not feats.usable:
                continue
            noise = paper_noise(feats)
            noise[feats.detail >= s.paper_detail_max] = np.nan
            valid = ~np.isnan(noise)
            values = noise[valid]
            if values.size < s.min_paper_blocks:
                continue
            median = float(np.median(values))
            if median < s.noise_min_median:
                continue  # digital or heavily cleaned page: no grain to compare against
            usable_pages += 1
            medians.append(round(median, 3))
            too_clean = valid & (np.nan_to_num(noise, nan=np.inf) < median * s.noise_low_ratio)
            too_noisy = valid & (np.nan_to_num(noise, nan=0.0) > median * s.noise_high_ratio)
            strength = np.zeros_like(noise)
            strength[valid] = np.abs(1.0 - noise[valid] / median).clip(0.0, 1.0)
            regions.extend(
                extract_regions(
                    too_clean | too_noisy,
                    strength,
                    block_size=s.block_size,
                    page=page.index,
                    signal=self.name,
                    merge_radius=s.merge_radius,
                    min_blocks=s.min_region_blocks,
                )
            )

        evidence = {"median_noise": medians}
        if usable_pages == 0:
            return self.skipped(
                "The page background has no scan grain to compare, so it is skipped."
            )
        if not regions:
            return self.passed("Paper grain is consistent everywhere.", evidence)
        count = len(regions)
        area = "1 area" if count == 1 else f"{count} areas"
        has = "has" if count == 1 else "have"
        return self.hit(
            0.55 + 0.4 * max(r.score for r in regions),
            f"{area} {has} a noise pattern unlike the scanned paper around them, "
            "which suggests a filled or repainted patch.",
            regions,
            evidence,
        )
