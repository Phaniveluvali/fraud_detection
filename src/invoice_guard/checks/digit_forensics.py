"""Compare each digit of every amount with the digits around it.

Digits anywhere on the page serve as the reference, but only amounts are reported: dates,
phone numbers and reference numbers are out of scope for this version. A small model, trained
on synthetic edits (see ``scripts/train_digit_model.py``), decides how unusual a digit is.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from invoice_guard.amounts import is_money
from invoice_guard.checks.base import Check
from invoice_guard.document import Document, Page
from invoice_guard.features import has_grain
from invoice_guard.glyph_features import NumberWord, features, number_words
from invoice_guard.models import CheckResult, Region

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "digit_model.joblib"
_load_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load(path: Path) -> dict[str, Any] | None:
    # Only ever load the model shipped with this package: joblib files can execute code.
    if not path.exists():
        return None
    try:
        import joblib

        bundle: dict[str, Any] = joblib.load(path)
        return bundle
    except Exception:
        logger.warning("digit model could not be loaded", exc_info=True)
        return None


def load_model() -> dict[str, Any] | None:
    with _load_lock:
        return _load(MODEL_PATH)


class DigitForensicsCheck(Check):
    name = "digit_forensics"
    title = "Digits within amounts"

    def run(self, doc: Document) -> CheckResult:
        bundle = load_model()
        if bundle is None:
            return self.skipped("The digit model is not installed (run the training script).")
        if doc.kind == "pdf" and any(w.fontname for w in doc.words):
            return self.skipped("This PDF has real text; the font checks cover it exactly.")
        if not doc.words:
            return self.skipped("No readable text. Scans need the OCR extra installed.")

        threshold = self.settings.digit_threshold
        regions: list[Region] = []
        total = examined = digits = 0
        for page in doc.pages:
            words = [w for w in doc.words if w.page == page.index]
            total += sum(is_money(w.text) for w in words)
            numbers = number_words(np.asarray(page.image, np.float32), words)
            matrix, keys = features(numbers, self._grain(page))
            if not keys:
                continue
            amounts = [i for i, n in enumerate(numbers) if is_money(n.word.text)]
            examined += len(amounts)
            digits += sum(len(numbers[i].glyphs) for i in amounts)
            scores = bundle["model"].predict_proba(matrix)[:, 1]
            regions.extend(self._regions(page.index, numbers, keys, scores, threshold))

        evidence = {"amounts_found": total, "amounts_examined": examined, "digits": digits}
        if not digits:
            return self.skipped("No amounts could be separated into digits to compare.")
        coverage = f"{examined} of {total} amounts"
        if examined < total:
            coverage += "; the rest could not be split into characters"
        if not regions:
            return self.passed(f"No digit stands out. Examined {coverage}.", evidence)
        detail = " ".join(r.note for r in regions) + f" Examined {coverage}."
        return self.hit(max(r.score for r in regions), detail, regions, evidence)

    def _grain(self, page: Page) -> bool:
        s = self.settings
        return has_grain(
            page.features(s.block_size),
            s.paper_detail_max,
            s.noise_min_median,
            s.min_paper_blocks,
        )

    def _regions(
        self,
        page: int,
        numbers: list[NumberWord],
        keys: list[tuple[int, int]],
        scores: npt.NDArray[np.float64],
        threshold: float,
    ) -> list[Region]:
        flagged: dict[int, list[tuple[int, float]]] = {}
        for (word, digit), score in zip(keys, scores, strict=True):
            if score >= threshold and is_money(numbers[word].word.text):
                flagged.setdefault(word, []).append((digit, float(score)))

        regions = []
        for word, hits in flagged.items():
            number = numbers[word]
            chosen = [number.glyphs[d] for d, _ in hits]
            chars = ", ".join(f'"{g.char}"' for g in chosen)
            top = max(s for _, s in hits)
            strength = (top - threshold) / max(1.0 - threshold, 1e-6)
            x0, y0 = min(g.x0 for g in chosen), min(g.y0 for g in chosen)
            verb = "does" if len(chosen) == 1 else "do"
            regions.append(
                Region(
                    page=page,
                    x=x0,
                    y=y0,
                    w=max(g.x1 for g in chosen) - x0,
                    h=max(g.y1 for g in chosen) - y0,
                    signals=[self.name],
                    score=round(0.55 + 0.4 * min(max(strength, 0.0), 1.0), 3),
                    note=f'Digit {chars} in "{number.word.text}" {verb} not match the '
                    "lettering around it.",
                )
            )
        return regions
