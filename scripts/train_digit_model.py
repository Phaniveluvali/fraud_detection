"""Train the per-digit tamper model on synthetic edits and report how well it does.

    python scripts/train_digit_model.py [--pages 6000]

Training pages use one set of fonts; the reported numbers come from a different set the model
never saw. The real invoice pairs in test-images are never used for training.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from invoice_guard import synth
from invoice_guard.config import Settings
from invoice_guard.features import compute_features, has_grain
from invoice_guard.glyph_features import features, number_words

ASSET = Path(__file__).resolve().parents[1] / "src" / "invoice_guard" / "assets"
SETTINGS = Settings()


def page_rows(task: tuple[int, tuple[str, ...], str | None]):  # type: ignore[no-untyped-def]
    seed, fonts, kind = task
    page = synth.sample(seed, fonts, kind)
    if page is None:
        return None
    s = SETTINGS
    grain = has_grain(
        compute_features(page.image, s.block_size),
        s.paper_detail_max,
        s.noise_min_median,
        s.min_paper_blocks,
    )
    numbers = number_words(np.asarray(page.image, np.float32), page.words)
    matrix, keys = features(numbers, grain)
    labels = []
    for word, digit in keys:
        g = numbers[word].glyphs[digit]
        labels.append(synth.hits((g.x0, g.y0, g.x1, g.y1), page.edited))
    return matrix, np.array(labels, bool), kind


def build(seeds: range, fonts: tuple[str, ...], pool: ProcessPoolExecutor):  # type: ignore[no-untyped-def]
    # half the pages stay clean so the model learns what normal variation looks like
    kinds = [None] * len(synth.EDIT_KINDS) + list(synth.EDIT_KINDS)
    tasks = [(seed, fonts, kinds[i % len(kinds)]) for i, seed in enumerate(seeds)]
    return [r for r in pool.map(page_rows, tasks, chunksize=16) if r is not None and len(r[0])]


def evaluate(model, pages, threshold: float) -> dict[str, float]:  # type: ignore[no-untyped-def]
    found: dict[str, list[bool]] = {}
    clean_flagged: list[bool] = []
    for matrix, labels, kind in pages:
        scores = model.predict_proba(matrix)[:, 1]
        if kind is None:
            clean_flagged.append(bool((scores >= threshold).any()))
        elif labels.any():
            found.setdefault(kind, []).append(bool((scores[labels] >= threshold).any()))
    result = {f"found_{k}": round(float(np.mean(v)), 3) for k, v in sorted(found.items())}
    result["clean_pages_falsely_flagged"] = round(float(np.mean(clean_flagged)), 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=6000)
    parser.add_argument("--false-alarm-rate", type=float, default=0.05)
    args = parser.parse_args()

    with ProcessPoolExecutor() as pool:
        train = build(range(args.pages), synth.TRAIN_FONTS, pool)
        calibrate = build(range(10**6, 10**6 + args.pages // 4), synth.TRAIN_FONTS, pool)
        test = build(range(2 * 10**6, 2 * 10**6 + args.pages // 3), synth.TEST_FONTS, pool)

    x = np.concatenate([m for m, _, _ in train])
    y = np.concatenate([labels for _, labels, _ in train])
    rng = np.random.default_rng(0)
    negatives = np.flatnonzero(~y)
    keep = np.concatenate(
        [np.flatnonzero(y), rng.choice(negatives, min(len(negatives), y.sum() * 15), False)]
    )
    print(f"digits: {len(y)} total, {int(y.sum())} edited, training on {len(keep)}")
    model = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=0,
    )
    model.fit(x[keep], y[keep])

    # threshold: at most this share of clean pages may have any digit flagged
    worst = [model.predict_proba(m)[:, 1].max() for m, _, k in calibrate if k is None]
    threshold = float(np.quantile(worst, 1.0 - args.false_alarm_rate))

    report = {
        "threshold": round(threshold, 4),
        "unseen_fonts": evaluate(model, test, threshold),
        "training_fonts": evaluate(model, calibrate, threshold),
        "train_pages": len(train),
        "test_pages": len(test),
    }
    ASSET.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, ASSET / "digit_model.joblib", compress=3)
    (ASSET / "digit_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
