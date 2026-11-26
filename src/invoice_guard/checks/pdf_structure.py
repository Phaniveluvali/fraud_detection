"""PDF file history: revisions, producer and date consistency, and edited amounts."""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from typing import Any

import pikepdf

from invoice_guard.checks.base import Check
from invoice_guard.checks.pdf_revisions import NumericChange, numeric_changes, revision_ends
from invoice_guard.document import Document
from invoice_guard.models import CheckResult, Region

logger = logging.getLogger(__name__)

EDITOR_MARKERS = (
    "acrobat",
    "foxit",
    "pdf-xchange",
    "pdfxchange",
    "sejda",
    "ilovepdf",
    "smallpdf",
    "pdfescape",
    "nitro",
    "photoshop",
    "illustrator",
    "gimp",
    "canva",
    "inkscape",
    "pdf24",
    "pdfelement",
    "soda pdf",
)
_PDF_DATE = re.compile(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?")
_DAY = 86400.0


def parse_pdf_date(value: str) -> datetime | None:
    match = _PDF_DATE.match(value.strip())
    if not match:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    parts = [int(g) if g else d for g, d in zip(match.groups(), (0, 1, 1, 0, 0, 0), strict=True)]
    try:
        return datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
    except ValueError:
        return None


def _info_text(trailer_info: Any, key: str) -> str:
    if trailer_info is None:
        return ""
    value = trailer_info.get(key)
    return str(value) if value is not None else ""


def _read_info(pdf: pikepdf.Pdf) -> dict[str, str]:
    info = pdf.trailer.get("/Info")
    return {
        "producer": _info_text(info, "/Producer"),
        "creator": _info_text(info, "/Creator"),
        "created": _info_text(info, "/CreationDate"),
        "modified": _info_text(info, "/ModDate"),
    }


def _read_xmp(pdf: pikepdf.Pdf) -> dict[str, str]:
    try:
        with pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False) as meta:
            return {str(k): str(v) for k, v in meta.items()}
    except Exception:
        return {}


def _has_signature(pdf: pikepdf.Pdf) -> bool:
    try:
        form = pdf.Root.get("/AcroForm")
        if form is None:
            return False
        return any(f.get("/FT") == "/Sig" and "/V" in f for f in form.get("/Fields", []))
    except Exception:
        return False


def _is_editor(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in EDITOR_MARKERS)


def _revision_info(data: bytes, end: int) -> dict[str, str] | None:
    try:
        with pikepdf.open(io.BytesIO(data[:end])) as earlier:
            return _read_info(earlier)
    except pikepdf.PdfError:
        return None


class PdfStructureCheck(Check):
    name = "pdf_structure"
    title = "File history"
    applies_to = frozenset({"pdf"})

    def run(self, doc: Document) -> CheckResult:
        try:
            pdf = pikepdf.open(io.BytesIO(doc.data))
        except pikepdf.PdfError as exc:
            return self.skipped(f"The PDF structure could not be read ({exc.__class__.__name__}).")

        with pdf:
            signed = _has_signature(pdf)
            info = _read_info(pdf)
            xmp = _read_xmp(pdf)

        ends = revision_ends(doc.data)
        revisions = max(len(ends), 1)
        messages: list[tuple[float, str]] = []
        regions: list[Region] = []
        evidence: dict[str, Any] = {
            "revisions": revisions,
            "signed": signed,
            "producer": info["producer"],
            "creator": info["creator"],
        }

        if revisions > 1:
            note = " A signature is present, which also adds revisions." if signed else ""
            messages.append(
                (0.25 if signed else 0.55, f"The file was saved {revisions} times.{note}")
            )
            pt_to_px = {p.index: p.pt_to_px or 1.0 for p in doc.pages}
            changes = numeric_changes(doc.data, ends, pt_to_px, self.settings.max_pages)
            if changes:
                evidence["numeric_changes"] = [
                    {"page": c.page, "old": c.old, "new": c.new} for c in changes
                ]
                messages.append((0.95, _describe_changes(changes)))
                regions.extend(self._change_regions(changes))
            earlier = _revision_info(doc.data, ends[-2])
            if earlier and earlier["producer"] and earlier["producer"] != info["producer"]:
                messages.append(
                    (
                        0.45,
                        f'The producer changed from "{earlier["producer"]}" to '
                        f'"{info["producer"]}" between revisions.',
                    )
                )

        if (
            info["creator"]
            and info["producer"]
            and not _is_editor(info["creator"])
            and _is_editor(info["producer"])
        ):
            messages.append(
                (
                    0.5,
                    f'Created with "{info["creator"]}" but last saved with the editor '
                    f'"{info["producer"]}".',
                )
            )

        messages.extend(self._date_signals(info, xmp, evidence))

        xmp_producer = xmp.get("pdf:Producer", "")
        if xmp_producer and info["producer"] and xmp_producer.strip() != info["producer"].strip():
            messages.append((0.4, "The embedded XMP metadata disagrees with the document info."))

        if not messages:
            return self.passed("A single revision with consistent metadata.", evidence)

        severity = 1.0
        for sev, _ in messages:
            severity *= 1.0 - sev
        return self.hit(1.0 - severity, " ".join(m for _, m in messages), regions, evidence)

    @staticmethod
    def _date_signals(
        info: dict[str, str], xmp: dict[str, str], evidence: dict[str, Any]
    ) -> list[tuple[float, str]]:
        created = parse_pdf_date(info["created"] or xmp.get("xmp:CreateDate", ""))
        modified = parse_pdf_date(info["modified"] or xmp.get("xmp:ModifyDate", ""))
        if created is None or modified is None:
            return []
        lag = (modified - created).total_seconds()
        evidence["created"] = created.isoformat()
        evidence["modified"] = modified.isoformat()
        if lag < -60:
            return [(0.4, "The modification date is earlier than the creation date.")]
        if lag > _DAY:
            days = int(lag // _DAY)
            return [(0.35, f"The file was modified {days} days after it was created.")]
        return []

    @staticmethod
    def _change_regions(changes: list[NumericChange]) -> list[Region]:
        out: list[Region] = []
        for c in changes:
            x, y, w, h = c.box
            out.append(
                Region(
                    page=c.page,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    signals=["pdf_structure"],
                    score=0.95,
                    note=_describe_changes([c]),
                )
            )
        return out


def _describe_changes(changes: list[NumericChange]) -> str:
    parts = []
    for c in changes[:5]:
        if c.old and c.new:
            parts.append(f"{c.old} became {c.new}")
        elif c.new:
            parts.append(f"{c.new} was added")
        else:
            parts.append(f"{c.old} was removed")
    extra = f" (and {len(changes) - 5} more)" if len(changes) > 5 else ""
    return "Amounts changed between revisions: " + "; ".join(parts) + extra + "."
