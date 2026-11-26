"""Amounts set in a different typeface than the rest of the document.

A retyped figure is usually set in whatever font the editor had at hand. If most amounts share
the document's body font and a few do not, the odd ones deserve a look.
"""

from __future__ import annotations

import re
from collections import Counter

from invoice_guard.amounts import has_decimals, parse_amount
from invoice_guard.checks.base import Check
from invoice_guard.document import Document
from invoice_guard.models import CheckResult, Region
from invoice_guard.textlayout import Word

_SUBSET = re.compile(r"^([A-Z]{6})\+")
_SUFFIX = re.compile(r"(?:mt|ps|std|pro)+$")


def font_family(fontname: str) -> str:
    """Normalise ``ABCDEF+Arial-BoldMT`` to ``arial`` so weights and subsets compare equal."""
    base = _SUBSET.sub("", fontname).lower()
    base = re.split(r"[-,]", base)[0]
    return _SUFFIX.sub("", base) or base


def _is_amount(word: Word) -> bool:
    return parse_amount(word.text) is not None and (has_decimals(word.text) or "," in word.text)


class FontConsistencyCheck(Check):
    name = "font_consistency"
    title = "Font consistency"
    applies_to = frozenset({"pdf"})

    def run(self, doc: Document) -> CheckResult:
        words = [w for w in doc.words if w.fontname]
        if not words:
            return self.skipped("The PDF has no font information (image-only document).")

        weight_by_family: Counter[str] = Counter()
        for w in words:
            weight_by_family[font_family(w.fontname or "")] += len(w.text)
        body_family = weight_by_family.most_common(1)[0][0]

        amounts = [w for w in words if _is_amount(w)]
        odd = [w for w in amounts if font_family(w.fontname or "") != body_family]
        # If every amount differs the document simply uses a dedicated number font.
        if not amounts:
            return self.skipped("No amounts were found to compare.")

        subset_tags: dict[str, set[str]] = {}
        for w in words:
            m = _SUBSET.match(w.fontname or "")
            if m:
                subset_tags.setdefault(_SUBSET.sub("", w.fontname or ""), set()).add(m.group(1))
        duplicated = sorted(name for name, tags in subset_tags.items() if len(tags) > 1)

        evidence = {
            "body_font": body_family,
            "amount_fonts": dict(Counter(font_family(w.fontname or "") for w in amounts)),
            "duplicate_subsets": duplicated,
        }
        notable = odd and len(odd) < len(amounts)
        if not notable and not duplicated:
            return self.passed("All amounts share the document's typeface.", evidence)

        regions: list[Region] = []
        severity = 0.0
        parts: list[str] = []
        if notable:
            severity = 0.7
            for w in odd:
                regions.append(
                    Region(
                        page=w.page,
                        x=int(w.x0),
                        y=int(w.top),
                        w=max(int(w.x1 - w.x0), 1),
                        h=max(int(w.bottom - w.top), 1),
                        signals=[self.name],
                        score=0.7,
                        note=f'"{w.text}" is set in {font_family(w.fontname or "")}, '
                        f"the rest of the document in {body_family}.",
                    )
                )
            shown = ", ".join(f'"{w.text}"' for w in odd[:3])
            parts.append(
                f"{len(odd)} amount(s) ({shown}) use a different typeface "
                f"than the document's {body_family}."
            )
        if duplicated:
            severity = max(severity, 0.3)
            parts.append(
                "The same font is embedded more than once, which often happens when text is "
                f"added in an editor ({', '.join(duplicated[:3])})."
            )
        return self.hit(severity, " ".join(parts), regions, evidence)
