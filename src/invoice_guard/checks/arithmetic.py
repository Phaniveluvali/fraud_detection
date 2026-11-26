"""Do the printed amounts agree with each other?

Tampering usually changes one number. Quantity times rate, line totals, subtotal, tax and the
grand total all constrain each other, so an edit to one of them tends to break an equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from invoice_guard.checks.base import Check
from invoice_guard.document import Document
from invoice_guard.models import CheckResult, Region
from invoice_guard.textlayout import Word, union_box


@dataclass(frozen=True)
class _Issue:
    kind: str
    message: str
    severity: float
    words: tuple[Word, ...]


def _fmt(value: Decimal) -> str:
    return f"{value:,.2f}" if value == value.quantize(Decimal("0.01")) else f"{value:,}"


class ArithmeticCheck(Check):
    name = "arithmetic"
    title = "Amounts add up"

    def run(self, doc: Document) -> CheckResult:
        if not doc.words:
            return self.skipped("No readable text, and OCR is not installed.")
        facts = doc.facts
        total = facts.last("total")
        if total is None:
            return self.skipped("No labelled total was found on the document.")

        tol = self.settings.amount_tolerance
        issues: list[_Issue] = []

        for row in facts.items:
            if row.qty is not None and row.rate is not None:
                expected_line = row.qty * row.rate
                if abs(expected_line - row.amount) > tol:
                    issues.append(
                        _Issue(
                            "line_item",
                            f"Line total {_fmt(row.amount)} but quantity x rate is "
                            f"{_fmt(expected_line)}.",
                            0.75,
                            row.words[-1:],
                        )
                    )

        items_sum = sum((r.amount for r in facts.items), Decimal(0)) if facts.items else None
        subtotal_line = facts.last("subtotal")
        if (
            subtotal_line is not None
            and items_sum is not None
            and abs(items_sum - subtotal_line.amount) > tol
        ):
            issues.append(
                _Issue(
                    "subtotal",
                    f"Line items add up to {_fmt(items_sum)} but the subtotal reads "
                    f"{_fmt(subtotal_line.amount)}.",
                    0.9,
                    subtotal_line.words[-1:],
                )
            )

        base = subtotal_line.amount if subtotal_line is not None else items_sum
        evidence: dict[str, object] = {"printed_total": str(total.amount)}
        if base is None:
            return self.skipped("A total was found but no subtotal or line items to compare to.")

        taxes = facts.all("tax")
        discounts = facts.all("discount")
        others = facts.all("other")
        tax_sum = sum((t.amount for t in taxes), Decimal(0))
        discount_sum = sum((abs(d.amount) for d in discounts), Decimal(0))
        other_sum = sum((o.amount for o in others), Decimal(0))
        expected_total = base - discount_sum + tax_sum + other_sum
        evidence.update(
            subtotal=str(base),
            items_sum=str(items_sum) if items_sum is not None else None,
            tax=str(tax_sum),
            expected_total=str(expected_total),
        )
        if abs(expected_total - total.amount) > tol:
            issues.append(
                _Issue(
                    "total",
                    f"Subtotal, tax and adjustments give {_fmt(expected_total)} but the "
                    f"printed total is {_fmt(total.amount)}.",
                    0.9,
                    total.words[-1:],
                )
            )

        # With a discount, tax may be charged before or after it; it must match one of them.
        bases = [base, base - discount_sum] if discounts else [base]
        for tax in taxes:
            if tax.rate is None:
                continue
            expected = [b * tax.rate / Decimal(100) for b in bases]
            if all(abs(value - tax.amount) > tol for value in expected):
                options = " or ".join(_fmt(v.quantize(Decimal("0.01"))) for v in expected)
                issues.append(
                    _Issue(
                        "tax_rate",
                        f"Tax of {_fmt(tax.amount)} does not match {tax.rate}% "
                        f"(it should be {options}).",
                        0.75,
                        tax.words[-1:],
                    )
                )

        if not issues:
            return self.passed(
                f"Items, tax and total agree: {_fmt(expected_total)} matches the printed total.",
                evidence,
            )

        evidence["issues"] = [{"kind": i.kind, "message": i.message} for i in issues]
        regions = [
            Region(
                page=i.words[0].page,
                x=x,
                y=y,
                w=w,
                h=h,
                signals=[self.name],
                score=i.severity,
                note=i.message,
            )
            for i in issues
            for x, y, w, h in [union_box(list(i.words))]
        ]
        return self.hit(
            max(i.severity for i in issues), " ".join(i.message for i in issues), regions, evidence
        )
