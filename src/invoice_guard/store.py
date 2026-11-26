"""SQLite-backed history of submitted documents, used for duplicate and re-submission checks.

SQLite keeps the MVP dependency-free. The interface is small on purpose so it can move to
PostgreSQL (with a BK-tree or pgvector index for the perceptual hashes) without touching the
checks.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256      TEXT NOT NULL UNIQUE,
    dhash       TEXT NOT NULL,
    filename    TEXT NOT NULL,
    invoice_no  TEXT,
    total       TEXT,
    risk_score  INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submissions_invoice_no ON submissions (invoice_no);
"""

_HASH_SIZE = 16  # 16 x 16 differences -> 256 bit hash, far fewer template collisions than 64


def dhash(image: Image.Image) -> str:
    """256-bit difference hash as 64 hex characters."""
    small = image.convert("L").resize((_HASH_SIZE + 1, _HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):064x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


@dataclass(frozen=True)
class Submission:
    sha256: str
    dhash: str
    filename: str
    invoice_no: str | None
    total: str | None
    risk_score: int
    created_at: str


class SubmissionStore:
    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._tx() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._conn as conn:
            yield conn

    @staticmethod
    def _row(row: sqlite3.Row) -> Submission:
        return Submission(**{k: row[k] for k in Submission.__dataclass_fields__})

    def add(self, item: Submission) -> bool:
        """Insert a submission. Returns ``False`` if that exact file was already stored."""
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO submissions "
                "(sha256, dhash, filename, invoice_no, total, risk_score, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    item.sha256,
                    item.dhash,
                    item.filename,
                    item.invoice_no,
                    item.total,
                    item.risk_score,
                    item.created_at,
                ),
            )
            return cur.rowcount > 0

    def by_sha(self, sha256: str) -> Submission | None:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM submissions WHERE sha256 = ?", (sha256,)).fetchone()
        return self._row(row) if row else None

    def by_invoice_no(self, invoice_no: str) -> list[Submission]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM submissions WHERE invoice_no = ? ORDER BY id", (invoice_no,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def near(self, hash_hex: str, max_distance: int, exclude_sha: str) -> list[Submission]:
        """Linear scan over stored hashes; fine for tens of thousands of rows."""
        with self._tx() as conn:
            rows = conn.execute("SELECT * FROM submissions WHERE sha256 != ?", (exclude_sha,))
            candidates = rows.fetchall()
        return [self._row(r) for r in candidates if hamming(r["dhash"], hash_hex) <= max_distance]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
