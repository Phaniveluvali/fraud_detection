# Invoice tamper detection: what was done and where it stands

## The problem
Invoices can be altered with simple tools such as Paint: an amount is erased and retyped, or a
single digit is changed. The goal is a tool that flags the altered amount for a reviewer, using
only the submitted file (no original to compare with).

## Process

**1. Research**
- Surveyed open-source tools, research models, datasets and commercial vendors.
- Found no open-source model that works reliably on documents out of the box. Several of the
  strongest ones are restricted to non-commercial use.
- Decision: layer several explainable checks and keep a person making the final call.

**2. Quick demo**
- Built a browser demo to show the idea on sample invoices.

**3. Working application**
- Python service with a web screen and an API.
- Reviewer sees a risk score, the reason for each finding and the highlighted area.

**4. Testing on real edits, and fixing what failed**

| What failed | What was changed |
|---|---|
| A retyped total was missed | Added size, font, resolution and alignment checks |
| Normal bold labels were flagged as edits | Recognise it as the invoice's design |
| Single changed digits were missed | Added per-digit height and shape comparison |
| A wrong tax figure passed when there was a discount | Corrected the arithmetic check |
| Dates and plain text were flagged | Limited this version to amounts |

**5. AI models**
- Text recognition (OCR) reads the page.
- A vision model on our Ollama server extracts invoice fields, each with a confidence score.
- The same vision model was tested for spotting edits. It could not do so reliably, so its view
  is advisory only and never changes the score.
- A small in-house model was trained on generated invoices to compare the digits in amounts.

## What the tool checks today
- Do line items, tax and total add up?
- Does each amount match its surroundings in size, font, sharpness and alignment?
- Does any digit within an amount look different from the others?
- PDF edit history, image metadata, scan noise and compression, duplicate submissions.
- Optional comparison against an original, when one exists.

## Results
**Generated invoices** (720 pages, fonts the digit model had not seen, amounts only):

| Edit to an amount | Found by the digit model |
|---|---|
| Digit or number retyped in another font | 98% |
| Digit retyped in the same font | 93% |
| Digit copied from elsewhere on the page | 82% |
| Untouched pages wrongly flagged | 2.5% |

**Real examples:** two invoices, each with an original and an edited copy. Both edited copies are
flagged and both originals score zero. The tool was adjusted using these same examples, so this
shows it works on them, not how accurate it will be on new invoices.

139 automated tests pass.

## Limitations
- It will not catch every edit. A carefully matched or copied digit can pass, especially if the
  totals still add up.
- Screenshots, phone photos and resized images are harder and not properly measured.
- Only amounts are highlighted in this version. Dates and reference numbers are not covered.
- A match is a reason to review, not proof of fraud.
- Field extraction sends page images to the internal Ollama server.
- Not yet done: security review, load testing, user login, deployment.

## Next steps
1. Pilot on 200-500 real invoices, including known edits, to measure true accuracy and false alarms.
2. Calibrate and retrain on that data.
3. Add login, audit trail and deployment.
4. Extend checks to dates, bank details and invoice numbers.
