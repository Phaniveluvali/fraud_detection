"""Where each value sits inside a JSON text, and how sure the model was while writing it."""

from __future__ import annotations

import json
import math
from json.decoder import scanstring  # type: ignore[attr-defined]

from invoice_guard.llm import Token

Path = tuple[str | int, ...]
_decoder = json.JSONDecoder()


def _skip(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def value_spans(text: str) -> dict[Path, tuple[int, int]]:
    """Character range of every scalar in ``text``; strings exclude their quotes.

    ``text`` must be valid JSON.
    """
    spans: dict[Path, tuple[int, int]] = {}

    def walk(i: int, path: Path) -> int:
        i = _skip(text, i)
        if text[i] in "{[":
            is_object = text[i] == "{"
            closing = "}" if is_object else "]"
            i = _skip(text, i + 1)
            index = 0
            while text[i] != closing:
                if is_object:
                    key, i = scanstring(text, i + 1)
                    i = walk(_skip(text, i) + 1, (*path, key))  # +1 skips the colon
                else:
                    i = walk(i, (*path, index))
                index += 1
                i = _skip(text, i)
                if text[i] == ",":
                    i = _skip(text, i + 1)
            return i + 1
        if text[i] == '"':
            _, end = scanstring(text, i + 1)
            spans[path] = (i + 1, end - 1)
            return int(end)
        _, end = _decoder.raw_decode(text, i)
        spans[path] = (i, end)
        return int(end)

    walk(0, ())
    return spans


def confidence(text: str, tokens: list[Token], span: tuple[int, int]) -> float | None:
    """Geometric mean probability of the tokens that produced ``text[start:end]``."""
    start = len(text[: span[0]].encode())
    end = len(text[: span[1]].encode())
    if end <= start:
        return None
    chosen: list[float] = []
    position = 0
    for token in tokens:
        following = position + len(token.data)
        if position < end and following > start:
            chosen.append(token.logprob)
        position = following
    if not chosen or position != len(text.encode()):
        return None  # tokens do not line up with the text, so no honest number exists
    return round(math.exp(sum(chosen) / len(chosen)), 4)
