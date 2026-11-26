"""Read the money structure of an invoice from positioned text.

The parser is deliberately conservative. It only reports values it can tie to a label such as
"Subtotal" or "GST 18%", so a layout it does not understand yields no facts rather than wrong
ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from invoice_guard.amounts import is_money, parse_amount
from invoice_guard.textlayout import Word

_SUMMARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("discount", re.compile(r"\bdiscount\b")),
    (
        "other",
        re.compile(
            r"\b(?:shipping|freight|delivery|handling|round(?:ing)?(?:\s*off)?|adjustment"
            r"|surcharge|service\s+charge|fees?)\b"
        ),
    ),
    (
        "subtotal",
        re.compile(
            r"\b(?:sub[\s\-]?total|net\s+(?:amount|total)|taxable\s+(?:value|amount)"
            r"|total\s+before\s+tax)\b"
        ),
    ),
    ("tax", re.compile(r"(?:\b|sales)(?:vat|gst|cgst|sgst|igst|taxes|tax)(?![a-z])")),
    (
        "total",
        re.compile(
            r"grand\s*total|total\s*(?:due|payable|amount|invoice)|amount\s*(?:due|payable)"
            r"|balance\s*due|invoice\s*total|\btotal(?![a-z])"
        ),
    ),
)
_RATE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_HEADER_LEFT = re.compile(r"\b(?:qty|quantity|description|item|particulars)\b")
_HEADER_RIGHT = re.compile(r"\b(?:amount|price|total|rate|value)\b")
_INVOICE_NO = re.compile(
    r"invoice\s*(?:no\.?|number|num|#)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-/_.]{2,})", re.IGNORECASE
)


@dataclass(frozen=True)
class SummaryLine:
    kind: str
    label: str
    amount: Decimal
    rate: Decimal | None
    words: tuple[Word, ...]


@dataclass(frozen=True)
class ItemRow:
    amount: Decimal
    qty: Decimal | None
    rate: Decimal | None
    words: tuple[Word, ...]


@dataclass
class InvoiceFacts:
    summary: list[SummaryLine] = field(default_factory=list)
    items: list[ItemRow] = field(default_factory=list)
    invoice_number: str | None = None

    def last(self, kind: str) -> SummaryLine | None:
        matches = [s for s in self.summary if s.kind == kind]
        return matches[-1] if matches else None

    def all(self, kind: str) -> list[SummaryLine]:
        return [s for s in self.summary if s.kind == kind]

    @property
    def total(self) -> Decimal | None:
        line = self.last("total")
        return line.amount if line else None


def _classify(label: str) -> str | None:
    lowered = label.lower()
    for kind, pattern in _SUMMARY_PATTERNS:
        if pattern.search(lowered):
            return kind
    return None


def _amount_words(line: list[Word]) -> list[tuple[Word, Decimal]]:
    out: list[tuple[Word, Decimal]] = []
    for word in line:
        value = parse_amount(word.text)
        if value is not None:
            out.append((word, value))
    return out


def _summary_line(line: list[Word]) -> SummaryLine | None:
    amounts = _amount_words(line)
    if not amounts:
        return None
    last_word, value = amounts[-1]
    if last_word is not line[-1] or not is_money(last_word.text):
        return None
    label_words = [w for w in line if w is not last_word and parse_amount(w.text) is None]
    label = " ".join(w.text for w in label_words)
    kind = _classify(label)
    if kind is None:
        return None
    rate_match = _RATE.search(label)
    rate = None
    if rate_match:
        rate = parse_amount(rate_match.group(1))
    return SummaryLine(kind, label, value, rate, tuple(line))


def _item_row(line: list[Word]) -> ItemRow | None:
    amounts = _amount_words(line)
    if len(amounts) < 2:
        return None
    last_word, amount = amounts[-1]
    if last_word is not line[-1] or not is_money(last_word.text):
        return None
    qty = rate = None
    if len(amounts) >= 3:
        qty, rate = amounts[-3][1], amounts[-2][1]
    return ItemRow(amount, qty, rate, tuple(line))


def _line_text(line: list[Word]) -> str:
    return " ".join(w.text for w in line)


def parse_invoice(lines: list[list[Word]]) -> InvoiceFacts:
    facts = InvoiceFacts()
    header_idx: int | None = None
    first_summary_idx: int | None = None

    for idx, line in enumerate(lines):
        text = _line_text(line)
        if facts.invoice_number is None:
            match = _INVOICE_NO.search(text)
            if match:
                facts.invoice_number = match.group(1).upper().rstrip(".")
        lowered = text.lower()
        if header_idx is None and _HEADER_LEFT.search(lowered) and _HEADER_RIGHT.search(lowered):
            header_idx = idx
        summary = _summary_line(line)
        if summary is not None:
            facts.summary.append(summary)
            if first_summary_idx is None and (header_idx is None or idx > header_idx):
                first_summary_idx = idx

    if header_idx is not None and first_summary_idx is not None:
        for line in lines[header_idx + 1 : first_summary_idx]:
            row = _item_row(line)
            if row is not None:
                facts.items.append(row)
    return facts
