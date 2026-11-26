"""Visual evidence for reviewers: annotated pages and error-level heat maps."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from invoice_guard.models import Region, RiskLevel
from invoice_guard.pipeline import AnalysisResult

_BOX = (224, 55, 46)


def _lut() -> npt.NDArray[np.uint8]:
    v = np.linspace(0.0, 1.0, 256)
    r = 255 * np.minimum(1.0, v * 1.6)
    g = 255 * np.clip(v * 1.5 - 0.3, 0.0, 1.0)
    b = 70 + 185 * np.clip(v * 2.0 - 1.4, 0.0, 1.0)
    return (np.stack([r * 0.9, g * 0.8, b * 0.9], axis=1)).clip(0, 255).astype(np.uint8)


_LUT = _lut()


def heatmap(ela: npt.NDArray[np.uint8]) -> Image.Image:
    """Colourise an 8-bit error-level map."""
    return Image.fromarray(_LUT[ela], mode="RGB")


def overlay(page: Image.Image, regions: list[Region]) -> Image.Image:
    """Draw numbered boxes around every region on ``page``."""
    canvas = page.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width = max(2, page.width // 300)
    font_size = max(14, page.width // 60)
    font = ImageFont.load_default(size=font_size)
    for region in regions:
        pad = width * 2
        box = (region.x - pad, region.y - pad, region.x1 + pad, region.y1 + pad)
        draw.rectangle(box, fill=(*_BOX, 36), outline=_BOX, width=width)
        label = str(region.id)
        tag_w = int(font.getlength(label)) + 12
        tag_h = font_size + 8
        top = max(box[1] - tag_h, 0)
        draw.rectangle((box[0], top, box[0] + tag_w, top + tag_h), fill=_BOX)
        draw.text((box[0] + 6, top + 3), label, fill="white", font=font)
    return canvas


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_images(result: AnalysisResult) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    for index, page in enumerate(result.page_images):
        regions = [r for r in result.report.regions if r.page == index]
        images[f"page{index + 1}"] = page
        images[f"page{index + 1}_overlay"] = overlay(page, regions)
        ela = result.ela_maps.get(index)
        if ela is not None:
            images[f"page{index + 1}_ela"] = heatmap(ela)
    return images


def encode_image(image: Image.Image) -> str:
    return base64.b64encode(_png(image)).decode("ascii")


def encode_images(result: AnalysisResult) -> dict[str, str]:
    """Base64 PNGs, suitable for embedding in an API response."""
    return {
        name: base64.b64encode(_png(img)).decode("ascii")
        for name, img in build_images(result).items()
    }


def write_artifacts(result: AnalysisResult, out_dir: Path) -> list[Path]:
    """Write ``report.json`` and the annotated images into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(result.report.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    written.append(report_path)
    for name, image in build_images(result).items():
        path = out_dir / f"{name}.png"
        path.write_bytes(_png(image))
        written.append(path)
    return written


def level_label(level: RiskLevel) -> str:
    return {RiskLevel.LOW: "LOW", RiskLevel.REVIEW: "REVIEW", RiskLevel.HIGH: "HIGH"}[level]
