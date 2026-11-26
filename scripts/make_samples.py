"""Write demo invoices (clean and tampered) to ``samples/``.

Usage: python scripts/make_samples.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import factories


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scan = factories.scan_invoice()
    good_pdf = factories.build_pdf(factories.invoice_items())
    edited = factories.edit_pdf(
        good_pdf, factories.invoice_items(total="14,071.00"), producer="Adobe Acrobat Pro 24"
    )
    files = {
        "invoice_scan_clean.jpg": scan,
        "invoice_scan_total_painted.jpg": factories.paint_over_total(scan),
        "invoice_scan_digit_changed.jpg": factories.change_first_digit(scan),
        "invoice_clean.pdf": good_pdf,
        "invoice_pdf_wrong_total.pdf": factories.build_pdf(
            factories.invoice_items(total="14,071.00")
        ),
        "invoice_pdf_edited.pdf": edited,
    }
    for name, data in files.items():
        (out / name).write_bytes(data)
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples"))
