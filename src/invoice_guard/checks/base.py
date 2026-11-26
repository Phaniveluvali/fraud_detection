"""Check interface and small helpers so each check only expresses its own logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from invoice_guard.config import Settings
from invoice_guard.document import Document
from invoice_guard.models import CheckResult, Region, Status

FLAG_SEVERITY = 0.6


class Check(ABC):
    """One independent line of evidence.

    Subclasses set ``name`` (the key used for weights), ``title`` (shown to reviewers) and,
    when they only make sense for one file type, ``applies_to``.
    """

    name: ClassVar[str]
    title: ClassVar[str]
    applies_to: ClassVar[frozenset[str]] = frozenset({"pdf", "image"})

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def weight(self) -> float:
        return self.settings.weights.get(self.name, 0.0)

    @abstractmethod
    def run(self, doc: Document) -> CheckResult:
        """Inspect ``doc`` and describe what was found. Must not mutate the document bytes."""

    # ------------------------------------------------------------------ result builders
    def _result(
        self,
        status: Status,
        detail: str,
        severity: float = 0.0,
        regions: list[Region] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> CheckResult:
        return CheckResult(
            name=self.name,
            title=self.title,
            status=status,
            severity=max(0.0, min(1.0, severity)),
            weight=self.weight,
            detail=detail,
            regions=regions or [],
            evidence=evidence or {},
        )

    def passed(self, detail: str, evidence: dict[str, Any] | None = None) -> CheckResult:
        return self._result(Status.PASS, detail, evidence=evidence)

    def skipped(self, detail: str) -> CheckResult:
        return self._result(Status.SKIPPED, detail)

    def hit(
        self,
        severity: float,
        detail: str,
        regions: list[Region] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> CheckResult:
        status = Status.FLAG if severity >= FLAG_SEVERITY else Status.REVIEW
        return self._result(status, detail, severity, regions, evidence)
