# Invoice Guard

Tamper detection for invoices and financial documents. It finds edited amounts (painted over,
retyped, pasted or swapped digits) in PDFs and scanned images, explains why, and shows where.

No single method catches every forgery, so the tool layers independent checks and combines
them into a risk score for a human reviewer. It is a triage aid, not proof of fraud.

## What it checks

| Check | Applies to | Signal |
|---|---|---|
| `pdf_structure` | PDF | Incremental saves, **old and new amounts recovered from earlier revisions**, producer and date changes, editor software |
| `arithmetic` | PDF text layer, OCR | Quantity x rate, line totals, subtotal, tax rate and grand total must agree |
| `font_consistency` | PDF | Amounts set in a different typeface than the document, duplicate font subsets |
| `typography` | all (scans need OCR) | Each amount against its label and column: **size**, **font** shape and weight, **resolution** (blur, pasted pictures), alignment |
| `digit_forensics` | images, scans | Each digit of every **amount** against the digits around it, using a trained model |
| `error_level` | images, scanned PDFs | Regions saved at a different JPEG quality than the page |
| `noise` | images, scanned PDFs | Paper grain missing or different in a patch |
| `image_metadata` | images | Editing software in EXIF, inconsistent timestamps |
| `duplicate` | all (needs store) | Same file, same invoice number with a different total, or a near-identical page |

Scanned PDFs are analysed from their embedded image at native resolution, because re-rendering
would erase the compression trail.

Scoring: each triggered check adds `weight x severity` as independent evidence (noisy-OR).
Two checks that point at the same area add a corroboration bonus. Levels: below 25 low,
25 to 59 review, 60 and above high. All of these are settings.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"      # add ,ocr for scanned-invoice text checks
```

Python 3.10 or newer. OCR is optional (`pip install -e ".[ocr]"`, RapidOCR, Apache-2.0). Without
it, text-based checks are skipped for image-only documents.

## Use

```powershell
.\.venv\Scripts\python scripts\make_samples.py                 # demo invoices in .\samples
.\.venv\Scripts\invoice-guard analyze samples\invoice_pdf_edited.pdf --out out
.\.venv\Scripts\invoice-guard analyze samples\*.jpg --fail-on review   # exit code 1 in CI
.\.venv\Scripts\invoice-guard serve --port 8000
```

`--out` writes `report.json`, `page1_overlay.png` (numbered boxes) and `page1_ela.png`
(error-level heat map). Exit codes: 0 ok, 1 reached `--fail-on`, 2 unreadable input.

### Compare with an original

When a trusted copy exists, this finds every visible change, including ones no single-file
check can see:

```powershell
.\.venv\Scripts\invoice-guard compare original.png suspect.jpg --out out
```

Both files should have the same page size. Exit codes: 0 no differences, 1 differences, 2 error.

### Web screen

Start the server and open <http://127.0.0.1:8000/>. Upload an invoice to see the score, the
checks and the flagged areas, or switch to "Compare with original". `/docs` has the raw API.

### Field extraction

The web screen also reads the invoice fields (vendor, number, dates, amounts, line items)
with a vision model on an Ollama server, and highlights each one on the page. Set the server
in `.env`:

```
IG_OLLAMA_URL=http://localhost:11434
IG_OLLAMA_MODEL=qwen2.5vl:7b
IG_OLLAMA_FALLBACK_MODEL=qwen2.5vl:3b
```

- **Page images are sent to that server.** Use one you control.
- The confidence is the model's token probability while writing the value. It is not the
  chance that the value is correct: a model can be confidently wrong.
- Positions come from the PDF text or OCR, not from the model. A value that is not found
  there is marked, because the model may have misread or invented it.
- The model reads values; it does not decide whether the invoice is fraudulent.

### Second opinion on layout findings

When the size, font and resolution check flags a value, the vision model is shown that row
and asked to verify the claim. The same claim is made about an untouched row; if the model
agrees with that too, its answer is discarded.

**Do not rely on the model to find edits.** In testing, `qwen2.5vl:7b` gave the same answer
for an edited invoice and its original when asked to look for problems itself. Its opinion is
advisory, was tried on very few documents, and never changes the score.

### API

```
POST /v1/analyze?register=false&include_images=false     multipart field "file"
POST /v1/compare     multipart fields "original" and "suspect"
POST /v1/extract     multipart field "file"; needs the Ollama server
POST /v1/review      multipart field "file"; second opinion on layout findings
GET  /healthz
```

`register=true` stores a fingerprint for later duplicate checks. `include_images=true` adds
base64 PNGs. Errors: 413 too large, 415 unsupported type, 422 unreadable. Interactive docs at
`/docs`. In Docker:

```powershell
docker compose up -d --build
```

See [docs/DEPLOY.md](docs/DEPLOY.md) before moving it to a server.

### Library

```python
from invoice_guard import AnalysisPipeline

result = AnalysisPipeline().analyze(open("invoice.pdf", "rb").read(), "invoice.pdf")
print(result.report.risk_level, result.report.summary)
```

## Configuration

Environment variables with the `IG_` prefix or a `.env` file; see `.env.example` and
`src/invoice_guard/config.py` for every threshold. **Calibrate `ela_*` and `noise_*` on a few
dozen of your own invoices before relying on them**: public benchmarks show pixel forensics
transfer poorly between document sources, and a threshold tuned on about ten in-domain images
recovers much of the lost accuracy.

## Design

```
upload -> sniff type -> Document (pages, lazy text, shared block features)
       -> checks run concurrently, each isolated (a crash becomes an ERROR result)
       -> regions merged across checks -> risk score -> report + images
```

- `src/invoice_guard/checks/` one file per check; implement `Check.run` and register it in
  `checks/__init__.py` to add a new one.
- Pixel features are computed once per page and shared; everything is vectorised NumPy.
- Uploads stay in memory. Type is sniffed from magic bytes. Size, pixel count and page count
  are capped. Filenames are reduced to a basename.
- Licences were chosen to be commercially safe: PyPDFium2, pikepdf, pdfplumber and Pillow.
  PyMuPDF (AGPL) and the non-commercial research models (TruFor, DocTamper) are deliberately
  not dependencies.

## The digit model

`digit_forensics` is trained on synthetic invoices (`python scripts/train_digit_model.py`,
about 15 minutes). It reports amounts only (a decimal part or a currency symbol); dates,
quantities and reference numbers are out of scope for this version.

Measured on 720 generated pages in fonts it was not trained on, amounts only, threshold 0.9:

| Edit to an amount | Found |
|---|---|
| Digit retyped in another font | 98% |
| Whole number retyped | 98% |
| Digit retyped in the same font | 93% |
| Digit copied from elsewhere on the page | 82% |
| Untouched pages wrongly flagged | 2.5% |

Each edit row rests on about 40 to 60 pages. **Real invoices will do worse than generated
ones:** on the two real edited examples available it found one and missed the other. The
model file is a joblib pickle: only load the one built from this repository, and keep
scikit-learn at the pinned version.

## Limits

- `typography` compares text of at least a few pixels height. It was tuned on very few
  documents; a design that deliberately styles a value unlike its label will be flagged.
- A retyped figure in the same font, size and position leaves nothing for it to find.
- Single retyped digits are found by their height, sometimes by one pixel. This works only on
  crisp digital images, not scans, photos or resized copies, and a forger who matches the
  height will pass.
- Compression and noise analysis are skipped on clean digital graphics, which carry no grain.
- Screenshots, phone photos and re-scans remove most compression and metadata evidence.
- AI-inpainted edits are harder; error-level and noise signals weaken on fully regenerated
  images. Plan to add a trained localiser (for example a model fine-tuned on your own data).
- Clean digital pages have no grain, so `noise` is skipped there.
- Duplicate detection uses a linear scan over SQLite; move to PostgreSQL with an index for
  large volumes.
- Run the service in a sandbox with no network access: it parses untrusted PDFs.

## Develop

```powershell
.\.venv\Scripts\python -m pytest --cov=invoice_guard
.\.venv\Scripts\python -m ruff check . ; .\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m mypy src
```

`tests/factories.py` generates the synthetic invoices (scans, edits, incremental PDFs) used by
the tests and `scripts/make_samples.py`.
