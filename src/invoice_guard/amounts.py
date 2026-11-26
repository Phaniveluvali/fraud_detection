"""Locale-tolerant money parsing.

Handles currency symbols, parentheses and trailing minus for negatives, Western, Indian and
European digit grouping, and three-decimal currencies such as the Omani rial.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"\d[\d.,]*")
_CURRENCY_PREFIX = "$€£₹¥"
_MINUS = ("-", chr(0x2212))


def parse_amount(text: str) -> Decimal | None:
    """Return the monetary value in ``text`` or ``None`` if it is not a plain amount."""
    s = text.strip().rstrip(":;")
    if not s or s.endswith("%"):
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    if s[:1] in _MINUS:
        negative, s = True, s[1:]
    if s[-1:] in _MINUS:
        negative, s = True, s[:-1]
    s = s.lstrip(_CURRENCY_PREFIX).strip()
    if not _NUMBER.fullmatch(s) or s[-1] in ".,":
        return None

    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        decimal_sep = "." if s.rfind(".") > s.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        s = s.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma:
        head, _, tail = s.rpartition(",")
        single = "," not in head
        s = s.replace(",", "") if (not single or len(tail) == 3) else f"{head}.{tail}"
    elif has_dot and s.count(".") > 1:
        s = s.replace(".", "")

    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    return -value if negative else value


def has_decimals(text: str) -> bool:
    """True when the token ends in a decimal part, which separates amounts from counts."""
    return re.search(r"[.,]\d{1,3}\)?$", text.strip()) is not None


def is_money(text: str) -> bool:
    """An amount as opposed to a bare count: it has decimals or a currency symbol."""
    stripped = text.strip().lstrip("(-" + chr(0x2212))
    return parse_amount(text) is not None and (
        has_decimals(text) or stripped.startswith(tuple(_CURRENCY_PREFIX))
    )
