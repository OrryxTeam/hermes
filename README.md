
# HERMES — Hybrid Exoplanet Recognition and Multi-planet Extraction System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.10-green.svg)](https://github.com/jax-ml/jax)
[![Flax NNX](https://img.shields.io/badge/Flax-NNX-orange.svg)](https://flax.readthedocs.io/)

**HERMES** is a differentiable, physics-informed pipeline for detecting and
vetting transiting exoplanets in stellar light curves. It combines a classical
transit search with a learned vetter under one JAX program: a transit-templated
periodogram proposes candidates, a differentiable transit model refines
them, and a dual-view CNN + transformer disposes of them with calibrated
probabilities — with the underlying physics (Kepler's laws, limb-darkened
transit geometry, stellar density) differentiable end to end.

---

## Why this design

Transit vetting is the task of deciding whether a periodic dip in a light curve
is a planet or a false positive (eclipsing binary, stellar variability,
instrumental systematics). HERMES makes that decision integrated and
physics-aware:

| Component | Approach |
|-----------|----------|
| **Search** | A matched-filter periodogram whose template is the *limb-darkened transit shape* itself (TLS-style), vmapped over a frequency-spaced period grid. |
| **Refinement** | Gradient descent through a differentiable Mandel–Agol transit model — the periodogram seed is sharpened by fitting epoch, depth, `a/R★` and impact parameter. |
| **Vetting** | A dual-view (global + local) 1D-CNN stem feeding a transformer over the global token sequence, fused with scalar candidate features. |
| **Physics** | A stellar-density (`ρ_circ`/`ρ★`) consistency term — used both as an input feature and as a differentiable training loss — and heteroscedastic outputs for calibrated parameter uncertainty. |
| **Calibration** | Temperature scaling on a held-out split, reported with reliability/ECE. |
| **Multi-planet** | Iterative transit subtraction, recovering successive candidates. |

`hermes/evaluators` scores the vetter against an AstroNet-style CNN and a
no-training BLS/SNR baseline on the same test set, reporting ROC-AUC, average
precision and ECE for each.

---

## Architecture

```
raw light curve
   │  preprocess: robust median detrend + transit-preserving (asymmetric) sigma-clip
   ▼
matched-filter periodogram  ──▶  candidate (period, epoch, duration)
   │  differentiable Mandel–Agol transit fit
   ▼  refined (period, epoch, depth, a/R★, b)
phase-fold ──▶ global view (2001) + local view (201)   [median/MAD normalised]
   │
TransitVetter:  CNN stems → transformer (global) ⊕ scalar features
   │            → classifier logit  +  heteroscedastic (depth, duration)
   ▼            + ρ_circ/ρ★ physics-consistency loss
subtract fitted transit, repeat  →  multi-planet candidates
temperature scaling  →  calibrated probabilities
```

The transit model is differentiable, so its gradients drive both the candidate
fit (`numerics/fit.py`) and the physics-consistency training loss — the search,
physics and model are one program (JAX numerics, Flax NNX network).

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data

HERMES reads everything from local files — there are no live downloads at train
time (per-target queries are slow and unreliable). Two inputs:

**1. Catalogue (labels + ephemerides)** — one CSV from the NASA Exoplanet
Archive (≈5 MB):

```bash
curl -o dataset/q1_q17_dr25_tce.csv \
  "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+q1_q17_dr25_tce&format=csv"
```

It provides `tce_period`, `tce_time0bk`, `tce_duration`, `tce_depth`, the
`av_training_set` label (`PC`/`AFP`/`NTP`) and stellar parameters. The KOI
cumulative table works too (`--config.data.catalog_format=koi`).

**2. Light curves** — the Kepler long-cadence FITS for the catalogue's stars.
Download them once from MAST (`python -m hermes.download` fetches only the
catalogue's KICs into `data.light_curve_dir`), then build with
`--config.data.source=fits`. A preprocessed one-row-per-star flux CSV is also
supported (`flux_csv`).

For smoke-testing the pipeline with no downloads, `--config.data.source=injection`
synthesises labelled light curves from the catalogue ephemerides (this is what
the tests use; it is not a substitute for real data).

Splits are made on the host star (`kepid`) to prevent leakage between partitions.

---

## Usage

```bash
# 1. Build the sharded view tensors (offline, one-off)
python -m hermes.build_dataset --config.data.source=fits

# 2. Train the vetter
python -m hermes.train

# 3. Evaluate + calibrate on the held-out test set
python -m hermes.evaluate --checkpoint=outputs/run_*/checkpoints/best
```

The default config (`hermes/configs/default.py`) loads automatically; override
any value on the command line (e.g. `--config.train.num_epochs=200`) or pass a
different config file with `--config=...`.

### Training output and model selection

Each epoch logs `train_loss`, `train_acc`, `val_loss`, `val_acc`, and `val_auc`
(validation ROC-AUC — the area under the ROC curve for planet-vs-false-positive
ranking). The **best** checkpoint, written to `outputs/run_*/checkpoints/best/`,
is the epoch with the **highest validation ROC-AUC** — chosen over `val_loss`
(a composite of classification, regression, and physics terms) and over
fixed-threshold accuracy because AUC is threshold-free and robust to the strong
class imbalance of TCE catalogues. Training stops early after
`train.early_stopping_patience` epochs without an AUC improvement.

---

## Performance

Measured on CPU with `python -m hermes.benchmark`:

- **Vetter** (0.82 M parameters): ≈2.4 ms per candidate when batched
  (≈410 candidates/s); single-candidate latency is ≈4.7 ms (fixed overhead with
  no batching to amortise it). Much faster on a GPU.
- **End-to-end pipeline** (search → differentiable fit → vetting): dominated by
  the periodogram search, which scales with the observational baseline — about
  1 s for a 90-day light curve, about 44 s for a full 1500-day Kepler light
  curve (≈180k trial periods over ≈73k points). Narrow the period range or use a
  GPU to reduce it; memory stays bounded regardless of grid size.


---

## Project structure

```
hermes/
  numerics/    transit · orbit · folding · periodogram · fit · physics
  data/        catalog · lightcurves · download · preprocess · splits · build_dataset · dataset
  models/      cnn · transformer · heads · vetter · baselines
  trainers/    losses · trainer · calibration · metrics
  evaluators/  harness · speed
  testing/     shared test fixtures
  configs/     default.py   (ml_collections config)
  pipeline.py  end-to-end detection + vetting
  train.py · evaluate.py · infer.py · build_dataset.py · download.py · benchmark.py
tests/         end-to-end tests (dataset · training · pipeline)
```

Unit tests are colocated with each module as `<module>_test.py` (e.g.
`numerics/transit_test.py`); the entry points run as `python -m hermes.<name>`.

---


