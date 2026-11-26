"""Minimal client for an Ollama server, returning the text and its token probabilities."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image

from invoice_guard.config import Settings
from invoice_guard.errors import ExtractionUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Token:
    data: bytes
    logprob: float


@dataclass(frozen=True)
class Completion:
    model: str
    text: str
    tokens: list[Token]


def encode_page(image: Image.Image, max_side: int) -> str:
    """Base64 PNG, downscaled so that the image fits the model's context comfortably."""
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OllamaClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport

    def chat(self, prompt: str, images: list[str], schema: dict[str, Any]) -> Completion:
        """Ask the vision model for JSON matching ``schema``; try the fallback model next."""
        models = [self.settings.ollama_model, self.settings.ollama_fallback_model]
        failure = "no model is configured"
        with httpx.Client(
            base_url=self.settings.ollama_url,
            timeout=self.settings.ollama_timeout,
            transport=self._transport,
        ) as client:
            for model in dict.fromkeys(m for m in models if m):
                try:
                    return self._request(client, model, prompt, images, schema)
                except httpx.ConnectError as exc:
                    raise ExtractionUnavailableError(
                        "The Ollama server could not be reached"
                    ) from exc
                except httpx.TimeoutException as exc:
                    raise ExtractionUnavailableError("The model took too long to answer") from exc
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    failure = f"{model}: {exc.__class__.__name__}"
                    logger.warning("extraction with %s failed", model, exc_info=True)
        raise ExtractionUnavailableError(f"The model request failed ({failure})")

    def _request(
        self,
        client: httpx.Client,
        model: str,
        prompt: str,
        images: list[str],
        schema: dict[str, Any],
    ) -> Completion:
        response = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": images}],
                "format": schema,
                "stream": False,
                "logprobs": True,
                "keep_alive": "15m",
                "options": {
                    "temperature": 0,
                    "num_ctx": self.settings.ollama_context,
                    "num_predict": 3000,
                },
            },
        )
        response.raise_for_status()
        body = response.json()
        tokens = [
            Token(bytes(t.get("bytes") or str(t["token"]).encode()), float(t["logprob"]))
            for t in body.get("logprobs") or []
        ]
        return Completion(model, str(body["message"]["content"]), tokens)
