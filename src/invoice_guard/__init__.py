"""Invoice Guard: layered tamper detection for invoices and financial documents."""

from invoice_guard._version import __version__
from invoice_guard.config import Settings, get_settings
from invoice_guard.models import CheckResult, Region, Report, RiskLevel, Status
from invoice_guard.pipeline import AnalysisPipeline, AnalysisResult

__all__ = [
    "AnalysisPipeline",
    "AnalysisResult",
    "CheckResult",
    "Region",
    "Report",
    "RiskLevel",
    "Settings",
    "Status",
    "__version__",
    "get_settings",
]
