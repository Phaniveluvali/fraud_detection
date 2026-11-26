"""Command line interface: ``invoice-guard analyze``, ``compare`` and ``serve``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from invoice_guard._version import __version__
from invoice_guard.compare import compare
from invoice_guard.config import Settings, get_settings
from invoice_guard.errors import InvoiceGuardError
from invoice_guard.logs import configure_logging
from invoice_guard.models import RiskLevel, Status
from invoice_guard.pipeline import AnalysisPipeline
from invoice_guard.render import level_label, overlay, write_artifacts
from invoice_guard.store import SubmissionStore

_MARK = {
    Status.PASS: "pass  ",
    Status.REVIEW: "REVIEW",
    Status.FLAG: "FLAG  ",
    Status.SKIPPED: "skip  ",
    Status.ERROR: "ERROR ",
}
_ORDER = {RiskLevel.LOW: 0, RiskLevel.REVIEW: 1, RiskLevel.HIGH: 2}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="invoice-guard", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyse one or more documents")
    analyze.add_argument("paths", nargs="+", type=Path, help="PDF or image files")
    analyze.add_argument("--out", type=Path, help="write report.json and images here")
    analyze.add_argument("--register", action="store_true", help="remember for duplicate checks")
    analyze.add_argument("--db", help="path of the duplicate-check database")
    analyze.add_argument(
        "--fail-on",
        choices=["review", "high"],
        help="exit with status 1 when any document reaches this level",
    )

    diff = sub.add_parser("compare", help="locate changes between an original and a suspect")
    diff.add_argument("original", type=Path)
    diff.add_argument("suspect", type=Path)
    diff.add_argument("--out", type=Path, help="write the suspect pages with changes boxed")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _print_report(path: Path, result_report) -> None:  # type: ignore[no-untyped-def]
    r = result_report
    print(
        f"\n{path.name}: {level_label(r.risk_level)}  score {r.risk_score}/100  "
        f"({r.duration_ms:.0f} ms)"
    )
    print(f"  {r.summary}")
    for check in r.checks:
        print(f"  [{_MARK[check.status]}] {check.title}: {check.detail}")
    for warning in r.warnings:
        print(f"  note: {warning}")
    if r.regions:
        print("  regions to review:")
        for region in r.regions:
            print(
                f"    {region.id}. page {region.page + 1} at ({region.x}, {region.y}) "
                f"{region.w}x{region.h} - {', '.join(region.signals)}"
            )


def _target_dir(out: Path, path: Path, used: set[str]) -> Path:
    """A distinct folder per input, so invoice.jpg and invoice.png do not overwrite."""
    base = f"{path.stem}_{path.suffix.lstrip('.').lower()}"
    name, number = base, 2
    while name in used:
        name, number = f"{base}_{number}", number + 1
    used.add(name)
    return out / name


def _compare(args: argparse.Namespace, settings: Settings) -> int:
    """Exit status 0 when nothing differs, 1 when something does, 2 on unreadable input."""
    try:
        result = compare(args.original.read_bytes(), args.suspect.read_bytes(), settings)
    except (InvoiceGuardError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 2
    for warning in result.warnings:
        print(f"note: {warning}")
    if result.identical:
        print("No visible differences.")
    for region in result.differences:
        print(
            f"{region.id}. page {region.page + 1} at ({region.x}, {region.y}) "
            f"{region.w}x{region.h} {region.note}"
        )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for index, page in enumerate(result.suspect_pages):
            boxes = [r for r in result.differences if r.page == index]
            overlay(page, boxes).save(args.out / f"page{index + 1}_changes.png")
    return 0 if result.identical else 1


def _analyze(args: argparse.Namespace, settings: Settings) -> int:
    store = None
    if settings.enable_duplicate_store and (args.register or args.db):
        store = SubmissionStore(args.db or settings.db_path)
    pipeline = AnalysisPipeline(settings, store=store)
    worst = RiskLevel.LOW
    status = 0
    used: set[str] = set()
    try:
        for path in args.paths:
            try:
                result = pipeline.analyze(path.read_bytes(), path.name, register=args.register)
            except (InvoiceGuardError, OSError) as exc:
                print(f"{path}: {exc}", file=sys.stderr)
                status = 2
                continue
            _print_report(path, result.report)
            if _ORDER[result.report.risk_level] > _ORDER[worst]:
                worst = result.report.risk_level
            if args.out:
                many = len(args.paths) > 1
                target = _target_dir(args.out, path, used) if many else args.out
                write_artifacts(result, target)
    finally:
        if store is not None:
            store.close()
    if status == 0 and args.fail_on and _ORDER[worst] >= _ORDER[RiskLevel(args.fail_on)]:
        return 1
    return status


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level if args.command == "serve" else "WARNING")
    if args.command == "serve":
        import uvicorn

        uvicorn.run("invoice_guard.api:create_app", factory=True, host=args.host, port=args.port)
        return 0
    if args.command == "compare":
        return _compare(args, settings)
    return _analyze(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
