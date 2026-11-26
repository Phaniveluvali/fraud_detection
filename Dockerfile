# syntax=docker/dockerfile:1

# ---- build wheels ------------------------------------------------------------
FROM python:3.12-slim AS build
WORKDIR /src
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY constraints.txt pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels --constraint constraints.txt ".[ocr]"

# ---- runtime -----------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IG_DB_PATH=/data/invoice_guard.db

# Needed by OpenCV (used by the OCR engine) and by ONNX Runtime / scikit-learn.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Wheels are mounted, not copied, so they do not add to the image size.
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-index --find-links=/wheels "invoice-guard[ocr]"

RUN useradd --create-home --uid 10001 guard && mkdir /data && chown guard /data
USER guard
WORKDIR /home/guard

# Fail the build, not the first request, if the OCR engine or digit model cannot load.
RUN python -c "from invoice_guard import ocr; from invoice_guard.checks.digit_forensics import load_model; assert ocr.is_available(), 'OCR unavailable'; assert load_model(), 'digit model unavailable'"

VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["uvicorn", "--factory", "invoice_guard.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
