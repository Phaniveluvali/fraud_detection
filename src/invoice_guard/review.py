"""A vision model's second opinion on the text our layout checks flagged.

Tests on a small vision model showed it cannot find an edited amount by itself: it answered
the same for an edited invoice and its original. It did better when asked to verify one
specific claim. It also tends to agree with whatever it is told, so every claim is repeated
about an untouched row; if the model agrees there too, its opinion is discarded.

The opinion is advisory and never changes the risk score.
"""

from __future__ import annotations

import json
import time

from PIL import Image

from invoice_guard.checks.typography import label_words
from invoice_guard.config import Settings
from invoice_guard.document import Document
from invoice_guard.json_spans import confidence, value_spans
from invoice_guard.llm import OllamaClient, encode_page
from invoice_guard.models import Region, Review, ReviewResponse, Verdict
from invoice_guard.textlayout import Word, union_box
from invoice_guard.typography import measure

MAX_REGIONS = 3

SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "verdict": {"type": "string", "enum": ["agree", "disagree", "unsure"]},
    },
    "required": ["observation", "verdict"],
}

Row = tuple[list[Word], Word]  # label words, value word


def claim(row: Row, direction: str) -> str:
    label = " ".join(w.text for w in row[0])
    return (
        f'An automatic check claims: the number "{row[1].text}" on the right is printed '
        f'noticeably {direction} and in a different typeface than the words "{label}" on the '
        "left. Look carefully. Is the claim true?"
    )


def _direction(doc: Document, row: Row) -> str:
    """Whether the value was measured smaller or larger than its label."""
    gray = doc.pages[row[1].page].features(doc.settings.block_size).luma
    label, value = measure(gray, row[0]), measure(gray, [row[1]])
    if label and value and value.height > label.height:
        return "larger"
    return "smaller"


def _rows(doc: Document) -> list[Row]:
    rows = []
    for summary in doc.facts.summary:
        label = label_words(summary.words)
        if label:
            rows.append((label, summary.words[-1]))
    return rows


def _inside(word: Word, region: Region) -> bool:
    return (
        word.page == region.page
        and word.x0 < region.x1
        and word.x1 > region.x
        and region.y - 2 <= word.cy <= region.y1 + 2
    )


def _crop(doc: Document, row: Row) -> Image.Image:
    page = doc.pages[row[1].page].image
    x, y, w, h = union_box([*row[0], row[1]])
    pad = int(0.6 * h)
    crop = page.crop(
        (
            max(x - pad, 0),
            max(y - pad, 0),
            min(x + w + pad, page.width),
            min(y + h + pad, page.height),
        )
    )
    scale = min(5.0, max(1.0, 1200 / crop.width))
    size = (round(crop.width * scale), round(crop.height * scale))
    return crop.resize(size, Image.Resampling.LANCZOS)


def _ask(
    client: OllamaClient, doc: Document, row: Row, direction: str, settings: Settings
) -> tuple[str, str, float | None, str]:
    """Return (verdict, observation, confidence, model)."""
    completion = client.chat(
        claim(row, direction), [encode_page(_crop(doc, row), settings.ollama_image_side)], SCHEMA
    )
    try:
        data = json.loads(completion.text)
        verdict = str(data["verdict"])
        span = value_spans(completion.text)[("verdict",)]
    except (ValueError, KeyError, TypeError, IndexError):
        return "unsure", "", None, completion.model
    score = confidence(completion.text, completion.tokens, span)
    return verdict, str(data.get("observation") or "")[:400], score, completion.model


def review(
    doc: Document,
    regions: list[Region],
    settings: Settings,
    client: OllamaClient | None = None,
) -> ReviewResponse:
    started = time.perf_counter()
    client = client or OllamaClient(settings)
    flagged = [r for r in regions if "typography" in r.signals and "the label" in r.note]
    flagged = flagged[:MAX_REGIONS]
    rows = _rows(doc)
    control = next(
        (row for row in rows if not any(_inside(row[1], r) for r in regions)),
        None,
    )

    reviews: list[Review] = []
    model = settings.ollama_model
    controls: dict[str, bool] = {}
    for region in flagged:
        row = next((row for row in rows if _inside(row[1], region)), None)
        if row is None:
            continue  # no label beside it, so there is nothing to compare it with
        direction = _direction(doc, row)
        answer, observation, score, model = _ask(client, doc, row, direction, settings)
        verdict = {"agree": Verdict.AGREES, "disagree": Verdict.DISAGREES}.get(
            answer, Verdict.UNSURE
        )
        if verdict is Verdict.AGREES and control is not None:
            if direction not in controls:
                controls[direction] = _ask(client, doc, control, direction, settings)[0] == "agree"
            if controls[direction]:
                verdict = Verdict.UNRELIABLE
        reviews.append(
            Review(
                region_id=region.id,
                text=row[1].text,
                verdict=verdict,
                confidence=score,
                observation=observation,
                control_checked=direction in controls,
            )
        )
    return ReviewResponse(
        model=model, reviews=reviews, duration_ms=round((time.perf_counter() - started) * 1000, 1)
    )
