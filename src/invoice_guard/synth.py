"""Synthetic invoices and the edits a person might make to them on the spot.

Used to train and to measure the digit check. Pages vary in font, size, colour, paper and
file handling. Edits imitate a paint program: erase, then retype in another font or size, or
paste a digit copied from elsewhere.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from invoice_guard.textlayout import Word

FONT_DIR = "C:/Windows/Fonts/"
TRAIN_FONTS = (
    "arial",
    "calibri",
    "segoeui",
    "verdana",
    "tahoma",
    "times",
    "georgia",
    "trebuc",
    "consola",
    "cour",
)
TEST_FONTS = ("candara", "corbel", "constan", "framd", "pala", "bahnschrift", "cambria")
EDIT_KINDS = ("digit_same_font", "digit_other_font", "number_retyped", "digit_pasted")


@lru_cache(maxsize=2048)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for extension in ("ttf", "ttc"):
        try:
            return ImageFont.truetype(f"{FONT_DIR}{name}.{extension}", size)
        except OSError:
            continue
    raise OSError(f"font {name} is not installed")


@dataclass
class Page:
    image: Image.Image
    words: list[Word]
    body_font: str
    size: int
    ink: tuple[int, int, int]
    edited: list[tuple[int, int, int, int]] = field(default_factory=list)
    edit_kind: str = ""


def _money(rng: random.Random, value: float, decimals: bool, symbol: str) -> str:
    return f"{symbol}{value:,.2f}" if decimals else f"{symbol}{value:,.0f}"


def render(rng: random.Random, fonts: tuple[str, ...]) -> Page:
    """A clean invoice with exact word boxes."""
    body = rng.choice(fonts)
    size = rng.randint(14, 24)
    ink = rng.choice([(0, 0, 0), (25, 25, 25), (50, 50, 55), (28, 40, 80)])
    paper = rng.choice([(255, 255, 255), (255, 255, 255), (252, 252, 250), (246, 247, 244)])
    width, height = 900, 1000
    image = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(image)
    words: list[Word] = []

    def text(x: float, y: float, value: str, face: str, px: int, right: bool = False) -> None:
        f = font(face, px)
        if right:
            x -= draw.textlength(value, font=f)
        for part in value.split(" "):
            draw.text((x, y), part, font=f, fill=ink)
            left, top, r, bottom = draw.textbbox((x, y), part, font=f)
            words.append(Word(0, part, left, top, r, bottom))
            x += draw.textlength(part + " ", font=f)

    head = rng.choice(fonts)
    text(60, 40, "INVOICE", head, int(size * 1.8))
    line = int(size * 1.6)
    y = 100
    for value in (
        f"{rng.randint(10, 999)} Sierra Drive",
        f"Palo Alto CA {rng.randint(10000, 99999)}",
        f"{rng.randint(100, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
    ):
        text(60, y, value, body, size)
        y += line
    y = 100
    for label, value in (
        ("Invoice", f"{rng.randint(1000, 99999)}"),
        ("Date", f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/20{rng.randint(20, 29)}"),
        ("Due", f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/20{rng.randint(20, 29)}"),
    ):
        text(520, y, label, body, size)
        text(840, y, value, body, size, right=True)
        y += line

    decimals = rng.random() < 0.7
    symbol = rng.choice(["$", "$", ""])
    y += line * 2
    columns = (480, 660, 840)
    for x, name in zip(columns, ("Qty", "Rate", "Amount"), strict=True):
        text(x, y, name, body, size, right=True)
    text(60, y, "Description", body, size)
    y += int(line * 1.4)
    subtotal = 0.0
    for _ in range(rng.randint(2, 6)):
        quantity = rng.randint(1, 40)
        rate = rng.randint(2, 900) + (rng.choice([0, 0.5, 0.25]) if decimals else 0)
        amount = quantity * rate
        subtotal += amount
        text(60, y, rng.choice(["Stone work", "Brake cables", "Consulting", "Hosting"]), body, size)
        text(columns[0], y, str(quantity), body, size, right=True)
        text(columns[1], y, _money(rng, rate, decimals, symbol), body, size, right=True)
        text(columns[2], y, _money(rng, amount, decimals, symbol), body, size, right=True)
        y += line
    y += line
    tax = round(subtotal * 0.085, 2)
    bold_total = rng.random() < 0.5
    for label, figure in (("Subtotal", subtotal), ("Tax", tax), ("Total", subtotal + tax)):
        total = label == "Total"
        px = int(size * rng.choice([1.0, 1.0, 1.3])) if total else size
        face = body
        if total and bold_total:
            face = {
                "arial": "arialbd",
                "calibri": "calibrib",
                "segoeui": "segoeuib",
                "verdana": "verdanab",
                "times": "timesbd",
                "georgia": "georgiab",
                "candara": "candarab",
                "corbel": "corbelb",
                "pala": "palab",
            }.get(body, body)
        text(520, y, label, face, px)
        text(columns[2], y, _money(rng, figure, True, symbol), face, px, right=True)
        y += int(line * 1.3)
    cropped = image.crop((0, 0, width, min(y + 40, height)))
    return Page(cropped, words, body, size, ink)


def _reencode(image: Image.Image, fmt: str, quality: int = 90) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, fmt, quality=quality) if fmt == "JPEG" else image.save(buf, fmt)
    return Image.open(buf).convert("RGB")


def distribute(page: Page, rng: random.Random) -> Page:
    """What happens before anyone edits it: resized, screenshotted or saved as JPEG."""
    image = page.image
    if rng.random() < 0.35:
        scale = rng.uniform(0.8, 1.2)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
        page.words = [
            Word(0, w.text, w.x0 * scale, w.top * scale, w.x1 * scale, w.bottom * scale)
            for w in page.words
        ]
        page.size = round(page.size * scale)
    if rng.random() < 0.4:
        image = _reencode(image, "JPEG", rng.randint(80, 96))
    page.image = image
    # OCR boxes are loose; train with the same slack
    page.words = [
        Word(
            0,
            w.text,
            w.x0 - rng.uniform(0, 2),
            w.top - rng.uniform(0, 4),
            w.x1 + rng.uniform(0, 2),
            w.bottom + rng.uniform(0, 4),
        )
        for w in page.words
    ]
    return page


def _digit_boxes(page: Page, word: Word) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Approximate box of each digit, from the glyph splitter."""
    from invoice_guard.glyph_features import number_words

    found = number_words(np.asarray(page.image, np.float32), [word])
    if not found:
        return []
    positions = [i for i, c in enumerate(word.text) if c.isdigit()]
    return [
        (position, (g.x0, g.y0, g.x1, g.y1))
        for position, g in zip(positions, found[0].glyphs, strict=False)
    ]


def _paper(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    region = np.asarray(image.crop((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4)))
    flat = region.reshape(-1, 3)
    bright = flat[flat.sum(axis=1) >= np.percentile(flat.sum(axis=1), 60)]
    r, g, b = np.median(bright, axis=0)
    return int(r), int(g), int(b)


def _type(
    image: Image.Image,
    rng: random.Random,
    text: str,
    face: str,
    box: tuple[int, int, int, int],
    ink: tuple[int, int, int],
    height: int,
    fill: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Erase ``box`` and type ``text`` there with digits about ``height`` pixels tall."""
    px = max(8, round(height / 0.72))
    for _ in range(12):
        top, bottom = font(face, px).getbbox("0")[1::2]
        if bottom - top < height:
            px += 1
        elif bottom - top > height:
            px -= 1
        else:
            break
    f = font(face, px)
    draw = ImageDraw.Draw(image)
    draw.rectangle((box[0] - 1, box[1] - 1, box[2], box[3]), fill=fill)
    width = f.getlength(text)
    x = box[2] - width if len(text) > 2 else (box[0] + box[2] - width) / 2
    x += rng.choice([-1, 0, 0, 1])
    y = box[3] - f.getbbox("0")[3] + rng.choice([-1, 0, 0, 0, 1])
    mode = rng.choice(["smooth", "smooth", "aliased", "subpixel"])
    if mode == "subpixel":  # coloured fringes, as sub-pixel text rendering leaves
        w, h = int(width) + 4, int(f.size) * 2
        big = Image.new("L", (w, h), 0)
        ImageDraw.Draw(big).text((2, 2), text, font=f, fill=255)
        stretched = np.asarray(big.resize((w * 3, h), Image.Resampling.BILINEAR), np.float32)
        channels = [stretched[:, offset::3][:, :w] / 255.0 for offset in (0, 1, 2)]
        base = np.array(image, np.float32)
        ox, oy = int(x) - 2, int(y) - 2
        region = base[oy : oy + h, ox : ox + w]
        for c in range(3):
            alpha = channels[c][: region.shape[0], : region.shape[1]]
            region[..., c] = region[..., c] * (1 - alpha) + ink[c] * alpha
        image.paste(Image.fromarray(base.astype(np.uint8)))
    else:
        if mode == "aliased":
            draw.fontmode = "1"
        draw.text((x, y), text, font=f, fill=ink)
    left, top, right, bottom = draw.textbbox((x, y), text, font=f)
    return int(left), int(top), int(right), int(bottom)


def edit(page: Page, rng: random.Random, kind: str, fonts: tuple[str, ...]) -> bool:
    """Change one number in place. Returns False if this page offered nothing to edit."""
    candidates = [w for w in page.words if sum(c.isdigit() for c in w.text) >= 2]
    rng.shuffle(candidates)
    for word in candidates:
        digits = _digit_boxes(page, word)
        if len(digits) < 2:
            continue
        image = page.image.copy()
        height = int(np.median([b[3] - b[1] for _, b in digits]))
        other = rng.choice([f for f in fonts if f != page.body_font])
        position, box = rng.choice(digits)
        fill = _paper(image, box) if rng.random() < 0.7 else (255, 255, 255)
        ink = page.ink if rng.random() < 0.6 else (0, 0, 0)
        replacement = rng.choice([c for c in "0123456789" if c != word.text[position]])
        if kind == "digit_pasted":
            donors = [
                b
                for w in candidates
                if w is not word
                for p, b in _digit_boxes(page, w)
                if w.text[p] == replacement and abs((b[3] - b[1]) - height) <= 1
            ]
            if not donors:
                continue
            donor = rng.choice(donors)
            patch = page.image.crop((donor[0] - 1, donor[1] - 1, donor[2] + 1, donor[3] + 1))
            ImageDraw.Draw(image).rectangle((box[0] - 1, box[1] - 1, box[2], box[3]), fill=fill)
            x = (box[0] + box[2] - patch.width) // 2 + rng.choice([-1, 0, 1])
            image.paste(patch, (x, box[3] + 1 - patch.height + rng.choice([-1, 0, 0, 1])))
            page.edited = [box]
        elif kind in ("digit_same_font", "digit_other_font"):
            face = page.body_font if kind == "digit_same_font" else other
            size = height + rng.choice([-2, -1, 0, 0, 1, 2])
            page.edited = [_type(image, rng, replacement, face, box, ink, size, fill)]
        else:
            boxes = [b for _, b in digits]
            whole = (
                int(word.x0),
                min(b[1] for b in boxes),
                int(word.x1),
                max(b[3] for b in boxes),
            )
            value = "".join(rng.choice("123456789") if c.isdigit() else c for c in word.text)
            face = other if rng.random() < 0.8 else page.body_font
            size = height + rng.choice([-2, -1, 0, 0, 1])
            page.edited = [_type(image, rng, value, face, whole, ink, size, fill)]
        # saved again the way an editor would
        page.image = _reencode(image, "JPEG", rng.randint(88, 96)) if rng.random() < 0.3 else image
        page.edit_kind = kind
        return True
    return False


def sample(seed: int, fonts: tuple[str, ...], kind: str | None) -> Page | None:
    rng = random.Random(seed)  # noqa: S311 - reproducible test data
    page = distribute(render(rng, fonts), rng)
    if kind and not edit(page, rng, kind, fonts):
        return None
    return page


def hits(box: tuple[int, int, int, int], targets: list[tuple[int, int, int, int]]) -> bool:
    """True when the middle of ``box`` lies inside an edited area."""
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(t[0] - 1 <= x <= t[2] + 1 and t[1] - 2 <= y <= t[3] + 2 for t in targets)
