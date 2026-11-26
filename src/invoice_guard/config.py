"""Runtime configuration.

Every field can be overridden with an ``IG_`` prefixed environment variable or a ``.env``
file, for example ``IG_RENDER_DPI=200``. Thresholds live here, not in the checks, so they
can be calibrated per customer without code changes.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_WEIGHTS: dict[str, float] = {
    # Maximum contribution of each check to the final risk, as a probability-like number.
    "pdf_structure": 0.70,
    "arithmetic": 0.80,
    "error_level": 0.50,
    "noise": 0.30,
    "font_consistency": 0.50,
    "image_metadata": 0.40,
    "typography": 0.75,
    "digit_forensics": 0.65,
    "duplicate": 0.80,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IG_", env_file=".env", extra="ignore")

    # --- limits -----------------------------------------------------------------
    max_upload_mb: int = Field(25, ge=1, description="Largest accepted file, in MiB.")
    max_pages: int = Field(10, ge=1, description="Pages analysed per PDF; the rest are ignored.")
    max_image_pixels: int = Field(80_000_000, ge=1_000_000, description="Decompression-bomb cap.")
    max_image_side: int = Field(4000, ge=500, description="Longer side is downscaled to this.")
    render_dpi: int = Field(150, ge=72, le=400, description="Resolution for rendered PDF pages.")
    workers: int = Field(4, ge=1, le=32, description="Checks that may run concurrently.")

    # --- pixel forensics ----------------------------------------------------------
    block_size: int = Field(16, ge=8, le=64)
    ela_quality: int = Field(75, ge=50, le=95, description="JPEG quality of the ELA re-save.")
    detail_min: float = Field(9.0, ge=0.0, description="Gradient energy for a text-bearing block.")
    ela_z_min: float = Field(2.0, ge=0.0, description="Robust z-score a block must exceed.")
    ela_ratio_min: float = Field(1.8, ge=1.0, description="Multiple of the median ratio.")
    min_detail_blocks: int = Field(30, ge=1)
    paper_detail_max: float = Field(4.0, ge=0.0, description="Max gradient energy for paper.")
    noise_low_ratio: float = Field(0.3, gt=0.0, lt=1.0)
    noise_high_ratio: float = Field(3.0, gt=1.0)
    noise_min_median: float = Field(0.25, ge=0.0, description="Below this the page is too clean.")
    min_paper_blocks: int = Field(20, ge=1)
    merge_radius: int = Field(2, ge=0, description="Blocks within this distance form one region.")
    min_region_blocks: int = Field(2, ge=1)

    # --- size, font and resolution --------------------------------------------------
    type_size_ratio: float = Field(1.28, gt=1.0, description="Glyph height mismatch to report.")
    type_font_ratio: float = Field(1.22, gt=1.0, description="Both shape and weight differ.")
    type_font_ratio_strong: float = Field(1.45, gt=1.0, description="Either one alone.")
    type_sharpness_ratio: float = Field(0.7, gt=0.0, lt=1.0, description="Blur threshold.")
    type_align_px: int = Field(5, ge=1, description="Column misalignment to report.")
    type_min_height: float = Field(9.0, ge=1.0, description="Smaller text compares poorly.")
    digit_threshold: float = Field(
        0.9, gt=0.0, lt=1.0, description="Model score at which an amount digit is reported."
    )
    enable_ocr: bool = Field(True, description="Read scans and photos when OCR is installed.")

    # --- field extraction (Ollama vision model) --------------------------------------
    enable_extraction: bool = True
    ollama_url: str = Field("http://localhost:11434", description="Base URL of Ollama.")
    ollama_model: str = "qwen2.5vl:7b"
    ollama_fallback_model: str = Field("qwen2.5vl:3b", description="Used if the first fails.")
    ollama_timeout: float = Field(240.0, gt=0, description="Seconds; first call loads a model.")
    ollama_context: int = Field(8192, ge=2048)
    ollama_image_side: int = Field(1600, ge=500, description="Page images are shrunk to this.")
    ollama_max_pages: int = Field(3, ge=1)

    # --- text and arithmetic -------------------------------------------------------
    amount_tolerance: Decimal = Field(Decimal("0.02"), ge=Decimal(0))
    dhash_max_distance: int = Field(8, ge=0, le=64, description="Near-duplicate threshold, bits.")

    # --- scoring --------------------------------------------------------------------
    review_threshold: int = Field(25, ge=0, le=100)
    high_threshold: int = Field(60, ge=0, le=100)
    corroboration_weight: float = Field(0.35, ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # --- storage / ops ---------------------------------------------------------------
    db_path: str = "invoice_guard.db"
    enable_duplicate_store: bool = True
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_thresholds(self) -> Settings:
        if self.review_threshold >= self.high_threshold:
            raise ValueError("review_threshold must be lower than high_threshold")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
