"""Registry of the checks that make up the default pipeline."""

from __future__ import annotations

from invoice_guard.checks.arithmetic import ArithmeticCheck
from invoice_guard.checks.base import Check
from invoice_guard.checks.digit_forensics import DigitForensicsCheck
from invoice_guard.checks.duplicate import DuplicateCheck
from invoice_guard.checks.error_level import ErrorLevelCheck
from invoice_guard.checks.font_consistency import FontConsistencyCheck
from invoice_guard.checks.image_metadata import ImageMetadataCheck
from invoice_guard.checks.noise import NoiseConsistencyCheck
from invoice_guard.checks.pdf_structure import PdfStructureCheck
from invoice_guard.checks.typography import TypographyCheck
from invoice_guard.config import Settings
from invoice_guard.store import SubmissionStore

__all__ = ["Check", "default_checks"]


def default_checks(settings: Settings, store: SubmissionStore | None = None) -> list[Check]:
    """Cheap deterministic checks first, then pixel forensics, then history lookups."""
    checks: list[Check] = [
        PdfStructureCheck(settings),
        ImageMetadataCheck(settings),
        ArithmeticCheck(settings),
        FontConsistencyCheck(settings),
        TypographyCheck(settings),
        DigitForensicsCheck(settings),
        ErrorLevelCheck(settings),
        NoiseConsistencyCheck(settings),
    ]
    if store is not None:
        checks.append(DuplicateCheck(settings, store))
    return checks
