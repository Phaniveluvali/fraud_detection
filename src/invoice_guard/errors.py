"""Exceptions raised by the analysis pipeline.

The API layer maps these onto HTTP status codes, so callers should raise the most
specific subclass available instead of a bare ``InvoiceGuardError``.
"""

from __future__ import annotations


class InvoiceGuardError(Exception):
    """Base class for all errors raised on purpose by this package."""


class UnsupportedDocumentError(InvoiceGuardError):
    """The input is not a PDF or a supported raster image."""


class DocumentTooLargeError(InvoiceGuardError):
    """The input exceeds the configured upload or pixel limit."""


class ExtractionUnavailableError(InvoiceGuardError):
    """Field extraction is switched off, or the model server could not answer."""


class DocumentParseError(InvoiceGuardError):
    """The input looks like a supported type but could not be parsed."""
