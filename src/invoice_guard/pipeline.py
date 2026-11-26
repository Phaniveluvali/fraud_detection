"""Orchestrates one analysis: load, run checks concurrently, merge evidence, score."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PIL import Image

from invoice_guard._version import __version__
from invoice_guard.checks import Check, default_checks
from invoice_guard.config import Settings, get_settings
from invoice_guard.document import Document
from invoice_guard.models import CheckResult, Region, Report, Status
from invoice_guard.regions import merge_regions
from invoice_guard.scoring import coverage, risk_level, risk_score, summarise
from invoice_guard.store import Submission, SubmissionStore

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    report: Report
    page_images: list[Image.Image] = field(repr=False)
    ela_maps: dict[int, npt.NDArray[np.uint8]] = field(repr=False, default_factory=dict)


class AnalysisPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        checks: list[Check] | None = None,
        store: SubmissionStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.checks = checks if checks is not None else default_checks(self.settings, store)

    def analyze(self, data: bytes, filename: str, *, register: bool = False) -> AnalysisResult:
        started = time.perf_counter()
        safe_name = filename.replace("\\", "/").rsplit("/", 1)[-1][:255] or "upload"
        doc = Document.from_bytes(data, safe_name, self.settings)

        applicable = [c for c in self.checks if doc.kind in c.applies_to]
        results = self._run(doc, applicable)

        pad = self.settings.block_size
        regions = merge_regions([r for res in results for r in res.regions], pad=pad)
        self._add_region_text(doc, regions)
        score = risk_score(results, regions, self.settings)
        level = risk_level(score, self.settings)
        cov = coverage(results)

        report = Report(
            version=__version__,
            document_id=doc.sha256,
            filename=doc.filename,
            file_type=doc.kind,
            pages=len(doc.pages),
            risk_score=score,
            risk_level=level,
            summary=summarise(level, results, cov),
            checks=results,
            regions=regions,
            coverage=cov,
            warnings=doc.warnings,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        if register and self.store is not None:
            self._register(doc, score)

        ela = {
            int(key.removeprefix("ela_p")): value
            for key, value in doc.artifacts.items()
            if key.startswith("ela_p")
        }
        logger.info(
            "analysed %s: %s pages, score %s (%s) in %.0f ms",
            doc.sha256[:12],
            report.pages,
            score,
            level.value,
            report.duration_ms,
        )
        return AnalysisResult(report, [p.image for p in doc.pages], ela)

    # ------------------------------------------------------------------------- internals
    def _run(self, doc: Document, checks: list[Check]) -> list[CheckResult]:
        if not checks:
            return []
        workers = min(self.settings.workers, len(checks))
        if workers == 1:
            return [self._safe(check, doc) for check in checks]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="check") as pool:
            return list(pool.map(lambda c: self._safe(c, doc), checks))

    @staticmethod
    def _add_region_text(doc: Document, regions: list[Region]) -> None:
        if not regions:
            return
        try:
            words = doc.words
        except Exception:
            logger.warning("could not read text for regions", exc_info=True)
            return
        for r in regions:
            r.text = " ".join(
                w.text
                for w in words
                if w.page == r.page and w.x0 < r.x1 and w.x1 > r.x and r.y - 2 <= w.cy <= r.y1 + 2
            )[:200]

    @staticmethod
    def _safe(check: Check, doc: Document) -> CheckResult:
        """Run one check; a crash becomes an ERROR result instead of failing the request."""
        started = time.perf_counter()
        try:
            result = check.run(doc)
        except Exception as exc:
            logger.exception("check %s failed", check.name)
            result = CheckResult(
                name=check.name,
                title=check.title,
                status=Status.ERROR,
                weight=check.weight,
                detail=f"The check could not complete ({exc.__class__.__name__}).",
            )
        result.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    def _register(self, doc: Document, score: int) -> None:
        assert self.store is not None
        fingerprint = doc.artifacts.get("dhash")
        if fingerprint is None:
            from invoice_guard.store import dhash

            fingerprint = dhash(doc.pages[0].image)
        facts = doc.facts
        self.store.add(
            Submission(
                sha256=doc.sha256,
                dhash=fingerprint,
                filename=doc.filename,
                invoice_no=facts.invoice_number,
                total=str(facts.total) if facts.total is not None else None,
                risk_score=score,
                created_at=self.store.now(),
            )
        )


__all__ = ["AnalysisPipeline", "AnalysisResult", "Region"]
