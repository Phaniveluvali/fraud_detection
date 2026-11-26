"""Public data contracts: what a check returns and what the API serialises."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Status(str, Enum):
    PASS = "pass"  # noqa: S105 - a check outcome, not a credential
    REVIEW = "review"
    FLAG = "flag"
    SKIPPED = "skipped"
    ERROR = "error"


class RiskLevel(str, Enum):
    LOW = "low"
    REVIEW = "review"
    HIGH = "high"


class Region(BaseModel):
    """A rectangle on one page image, in that image's pixel coordinates."""

    id: int = 0
    page: int = Field(ge=0, description="Zero-based page index.")
    x: int
    y: int
    w: int
    h: int
    signals: list[str] = Field(default_factory=list, description="Check names that fired here.")
    score: float = Field(0.0, description="Strongest signal strength, 0 to 1.")
    note: str = ""
    text: str = Field("", description="Words printed inside the region, when readable.")

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h


class CheckResult(BaseModel):
    name: str
    title: str
    status: Status
    severity: float = Field(0.0, ge=0.0, le=1.0, description="How alarming a hit is, 0 to 1.")
    weight: float = Field(0.0, ge=0.0, le=1.0)
    detail: str
    regions: list[Region] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def triggered(self) -> bool:
        return self.status in (Status.REVIEW, Status.FLAG)


class Coverage(BaseModel):
    ran: int
    skipped: int
    errored: int


class Report(BaseModel):
    version: str
    document_id: str = Field(description="SHA-256 of the uploaded bytes.")
    filename: str
    file_type: str
    pages: int
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    summary: str
    checks: list[CheckResult]
    regions: list[Region]
    coverage: Coverage
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float


class ExtractedField(BaseModel):
    key: str
    label: str
    value: str
    confidence: float | None = Field(
        None, description="Model token probability, 0 to 1. Not a probability of being right."
    )
    page: int | None = None
    box: tuple[int, int, int, int] | None = Field(
        None, description="x, y, width, height on the page image, if the value was found there."
    )


class Extraction(BaseModel):
    model: str
    fields: list[ExtractedField]
    line_items: list[list[ExtractedField]]
    page_text_available: bool
    duration_ms: float


class Verdict(str, Enum):
    AGREES = "agrees"
    DISAGREES = "disagrees"
    UNSURE = "unsure"
    UNRELIABLE = "unreliable"  # it also agreed about an untouched row


class Review(BaseModel):
    region_id: int
    text: str
    verdict: Verdict
    confidence: float | None = None
    observation: str = ""
    control_checked: bool = False


class ReviewResponse(BaseModel):
    model: str
    reviews: list[Review]
    duration_ms: float


class CompareResponse(BaseModel):
    identical: bool
    differences: list[Region]
    warnings: list[str] = Field(default_factory=list)
    images: dict[str, str] | None = None


class AnalyzeResponse(BaseModel):
    report: Report
    images: dict[str, str] | None = Field(
        default=None, description="Base64 PNGs keyed by name, present when requested."
    )
