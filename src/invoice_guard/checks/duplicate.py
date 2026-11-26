"""Has this invoice, or a close copy of it, been submitted before?"""

from __future__ import annotations

from invoice_guard.checks.base import Check
from invoice_guard.config import Settings
from invoice_guard.document import Document
from invoice_guard.models import CheckResult
from invoice_guard.store import SubmissionStore, dhash


class DuplicateCheck(Check):
    name = "duplicate"
    title = "Duplicate invoice"

    def __init__(self, settings: Settings, store: SubmissionStore) -> None:
        super().__init__(settings)
        self.store = store

    def run(self, doc: Document) -> CheckResult:
        fingerprint = dhash(doc.pages[0].image)
        doc.artifacts["dhash"] = fingerprint

        exact = self.store.by_sha(doc.sha256)
        if exact is not None:
            return self.hit(
                0.7,
                f'This exact file was already submitted as "{exact.filename}" on '
                f"{exact.created_at[:10]}.",
                evidence={"match": "sha256", "previous": exact.filename},
            )

        facts = doc.facts
        number = facts.invoice_number
        total = str(facts.total) if facts.total is not None else None
        if number:
            earlier = self.store.by_invoice_no(number)
            changed = [e for e in earlier if e.total and total and e.total != total]
            if changed:
                prev = changed[-1]
                return self.hit(
                    0.9,
                    f"Invoice {number} was submitted before with a different total "
                    f"({prev.total}, now {total}).",
                    evidence={"match": "invoice_number", "previous_total": prev.total},
                )
            if earlier:
                return self.hit(
                    0.5,
                    f"Invoice {number} was already submitted as "
                    f'"{earlier[-1].filename}" with the same total.',
                    evidence={"match": "invoice_number"},
                )
            return self.passed(f"Invoice {number} has not been seen before.")

        similar = self.store.near(fingerprint, self.settings.dhash_max_distance, doc.sha256)
        if similar:
            return self.hit(
                0.35,
                f'The page looks almost identical to "{similar[0].filename}", '
                "and no invoice number could be read to tell them apart.",
                evidence={"match": "visual", "previous": similar[0].filename},
            )
        return self.passed("No earlier submission matches this document.")
