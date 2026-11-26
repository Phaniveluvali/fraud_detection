"""Combine independent checks into one risk score and a short explanation.

Each triggered check contributes ``weight * severity`` as a probability-like number. The
contributions are combined as independent evidence (noisy-OR), so a second signal always
raises the score but nothing can exceed 100. Two different checks pointing at the same area
add a corroboration factor on top, because agreement between unrelated methods is the
strongest evidence available.
"""

from __future__ import annotations

from invoice_guard.config import Settings
from invoice_guard.models import CheckResult, Coverage, Region, RiskLevel, Status

# Checks that look at the same evidence count as one witness.
_FAMILY = {"digit_forensics": "typography", "font_consistency": "typography"}


def _independent(signals: list[str]) -> int:
    return len({_FAMILY.get(s, s) for s in signals})


def risk_score(checks: list[CheckResult], regions: list[Region], settings: Settings) -> int:
    remaining = 1.0
    for check in checks:
        if check.triggered:
            remaining *= 1.0 - min(1.0, check.weight * check.severity)
    if any(_independent(r.signals) >= 2 for r in regions):
        remaining *= 1.0 - settings.corroboration_weight
    return round(100 * (1.0 - remaining))


def risk_level(score: int, settings: Settings) -> RiskLevel:
    if score >= settings.high_threshold:
        return RiskLevel.HIGH
    if score >= settings.review_threshold:
        return RiskLevel.REVIEW
    return RiskLevel.LOW


def coverage(checks: list[CheckResult]) -> Coverage:
    return Coverage(
        ran=sum(c.status in (Status.PASS, Status.REVIEW, Status.FLAG) for c in checks),
        skipped=sum(c.status is Status.SKIPPED for c in checks),
        errored=sum(c.status is Status.ERROR for c in checks),
    )


def summarise(level: RiskLevel, checks: list[CheckResult], cov: Coverage) -> str:
    triggered = [c for c in checks if c.triggered]
    limited = " Coverage is limited, so a low score is weaker evidence." if cov.ran < 3 else ""
    if level is RiskLevel.LOW:
        return f"No indicators of tampering were found by the {cov.ran} checks that ran.{limited}"
    names = ", ".join(c.title.lower() for c in triggered)
    count = len({_FAMILY.get(c.name, c.name) for c in triggered})
    if level is RiskLevel.REVIEW:
        return (
            f"{count} indicator{'s merit' if count != 1 else ' merits'} manual review ({names}). "
            "Ask the sender to confirm the amounts before paying."
        )
    return (
        f"{count} independent indicator{'s point' if count != 1 else ' points'} to an edited "
        f"document ({names}). Hold payment and verify with the sender through a known contact."
    )
