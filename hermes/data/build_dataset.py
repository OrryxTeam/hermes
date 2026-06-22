"""Offline builder turning a catalogue and light-curve store into view tensors.

For each TCE it cleans the light curve, folds it at the catalogue ephemeris into
normalised global and local views, and assembles scalar features (including the
stellar-density consistency diagnostic). Scalar features are standardised using
the training split; outputs are written as sharded .npz files.
"""

from __future__ import annotations

import json
import os

import numpy as np
from absl import logging

from hermes.numerics import physics
from hermes.configs.default import get_config
from hermes.data import catalog as catalog_lib
from hermes.data import lightcurves, splits
from hermes.numerics import folding, orbit
from hermes.data.preprocess import clean_light_curve

FEATURE_NAMES = (
    "log_period",
    "log_depth",
    "duration_over_period",
    "log_num_transits",
    "density_consistency",
    "stellar_teff_norm",
    "stellar_radius_solar",
    "stellar_logg",
)


def _positive(value: float, fallback: float) -> float:
  """Returns ``value`` if finite and positive, else ``fallback``."""
  value = float(value)
  return value if np.isfinite(value) and value > 0.0 else fallback


def build_store(config) -> lightcurves.LightCurveStore:
  """Constructs the light-curve store named by ``config.data.source``."""
  source = config.data.source
  if source == "injection":
    return lightcurves.InjectionStore(
        baseline_days=config.data.baseline_days,
        cadence_days=config.data.cadence_days,
        noise_ppm=config.data.noise_ppm,
        seed=config.seed,
    )
  if source == "fits":
    return lightcurves.FitsLightCurveStore(config.data.light_curve_dir)
  if source == "flux_csv":
    return lightcurves.FluxCsvStore(config.data.light_curve_dir)
  raise ValueError(f"Unknown data source: {source!r}")


def compute_scalar_features(
    period_days: float,
    depth_fraction: float,
    duration_days: float,
    stellar_density_cgs: float,
    stellar_teff_k: float,
    stellar_radius_solar: float,
    stellar_logg_cgs: float,
    baseline_days: float = 4.0 * 365.25,
) -> np.ndarray:
  """Builds the scalar feature vector from raw candidate quantities.

  Shared by the offline builder and the inference pipeline so the features are
  computed identically in both. Missing stellar values fall back to solar-like
  defaults. See `FEATURE_NAMES` for the order.

  Args:
    period_days: Orbital period in days.
    depth_fraction: Fractional transit depth.
    duration_days: Transit duration in days.
    stellar_density_cgs: Stellar mean density in g/cm^3 (<= 0 if unknown).
    stellar_teff_k: Stellar effective temperature in Kelvin.
    stellar_radius_solar: Stellar radius in solar radii.
    stellar_logg_cgs: Stellar surface gravity, ``log10(cm / s^2)``.
    baseline_days: Observational baseline (sets the transit count).

  Returns:
    A length-8 ``float32`` feature vector.
  """
  # Guard against missing catalogue values (a NaN here would poison the feature
  # standardisation and, in turn, the logits and loss).
  period_days = _positive(period_days, 1.0)
  depth = _positive(depth_fraction, 1e-6)
  duration_days = _positive(duration_days, 0.1)
  num_transits = max(baseline_days / period_days, 1.0)
  radius_ratio = np.sqrt(depth)
  a_over_rstar = (
      period_days / np.pi * (1.0 + radius_ratio) / max(duration_days, 1e-3)
  )
  if np.isfinite(stellar_density_cgs) and stellar_density_cgs > 0.0:
    consistency = float(
        orbit.density_consistency_logratio(
            period_days, a_over_rstar, stellar_density_cgs * 1000.0  # cgs -> SI
        )
    )
  else:
    consistency = 0.0

  teff = (
      stellar_teff_k
      if np.isfinite(stellar_teff_k)
      else physics.SOLAR_EFFECTIVE_TEMPERATURE_K
  )
  return np.array(
      [
          np.log10(period_days),
          np.log10(depth),
          duration_days / period_days,
          np.log10(num_transits),
          consistency,
          teff / physics.SOLAR_EFFECTIVE_TEMPERATURE_K,
          stellar_radius_solar if np.isfinite(stellar_radius_solar) else 1.0,
          stellar_logg_cgs if np.isfinite(stellar_logg_cgs) else 4.44,
      ],
      dtype=np.float32,
  )


def _scalar_features(cat: catalog_lib.TceCatalog, index: int) -> np.ndarray:
  """Computes the scalar feature vector for one catalogue row."""
  return compute_scalar_features(
      period_days=cat.period_days[index],
      depth_fraction=cat.depth_fraction[index],
      duration_days=cat.duration_days[index],
      stellar_density_cgs=cat.stellar_density_cgs[index],
      stellar_teff_k=cat.stellar_teff_k[index],
      stellar_radius_solar=cat.stellar_radius_solar[index],
      stellar_logg_cgs=cat.stellar_logg_cgs[index],
  )


def _build_rows(config, cat, store):
  """Builds views, features and labels for every usable catalogue row."""
  labels = tuple(config.data.labels)
  global_views, local_views, features = [], [], []
  binary_labels, class_labels, kepids = [], [], []
  period_days, depths, durations, stellar_density = [], [], [], []

  num_rows = len(cat)
  logging.info(
      "Building views from %d catalogue rows (source=%s)", num_rows,
      config.data.source,
  )
  for index in range(num_rows):
    if (index + 1) % 100 == 0:
      logging.info(
          "  processed %d/%d rows, kept %d examples",
          index + 1, num_rows, len(global_views),
      )
    light_curve = store.get(index, cat)
    if light_curve is None:
      continue
    clean = clean_light_curve(
        light_curve,
        cadence_days=config.data.cadence_days,
        detrend_window_days=config.data.detrend_window_days,
        sigma=config.data.sigma_clip,
    )
    if clean.time_days.size < config.data.local_bins:
      continue

    period = float(cat.period_days[index])
    epoch = float(cat.epoch_days[index])
    duration = float(cat.duration_days[index])
    if not (np.isfinite(period) and period > 0.0 and np.isfinite(epoch)):
      continue
    global_views.append(
        np.asarray(
            folding.normalize_view(
                folding.global_view(
                    clean.time_days, clean.flux, period, epoch,
                    config.data.global_bins,
                )
            ),
            dtype=np.float32,
        )
    )
    local_views.append(
        np.asarray(
            folding.normalize_view(
                folding.local_view(
                    clean.time_days, clean.flux, period, epoch, duration,
                    config.data.local_bins, config.data.local_durations,
                )
            ),
            dtype=np.float32,
        )
    )
    features.append(_scalar_features(cat, index))
    label = str(cat.label[index])
    binary_labels.append(int(label == config.data.positive_label))
    class_labels.append(labels.index(label) if label in labels else len(labels))
    kepids.append(int(cat.kepid[index]))
    period_days.append(period)
    depths.append(_positive(cat.depth_fraction[index], 1e-6))
    durations.append(_positive(duration, 0.1))
    density = cat.stellar_density_cgs[index]
    stellar_density.append(float(density) if np.isfinite(density) else 0.0)

  return {
      "global_view": np.stack(global_views),
      "local_view": np.stack(local_views),
      "scalar_features": np.stack(features),
      "label": np.asarray(binary_labels, dtype=np.int32),
      "label_class": np.asarray(class_labels, dtype=np.int32),
      "kepid": np.asarray(kepids, dtype=np.int64),
      "period_days": np.asarray(period_days, dtype=np.float32),
      "depth_fraction": np.asarray(depths, dtype=np.float32),
      "duration_days": np.asarray(durations, dtype=np.float32),
      "stellar_density_cgs": np.asarray(stellar_density, dtype=np.float32),
  }


def build(config=None) -> dict:
  """Builds and writes the full sharded dataset.

  Args:
    config: HERMES configuration; ``get_config()`` is used when ``None``.

  Returns:
    The metadata dictionary that is also written to ``meta.json``.
  """
  config = config or get_config()
  out_dir = config.data.processed_dir
  os.makedirs(out_dir, exist_ok=True)

  cat = catalog_lib.load_catalog(
      config.data.catalog_csv, config.data.catalog_format
  )
  cat = cat.with_labels(config.data.labels)
  rows = _build_rows(config, cat, build_store(config))

  manifest = splits.split_by_star(
      rows["kepid"],
      (config.data.split.train, config.data.split.validation,
       config.data.split.test),
      seed=config.seed,
  )
  manifest.save(os.path.join(out_dir, "split_manifest.json"))
  masks = splits.partition_masks(rows["kepid"], manifest)

  # Standardise scalar features using the training partition only. Replace any
  # residual non-finite value so a single bad row cannot poison a whole column.
  rows["scalar_features"] = np.nan_to_num(
      rows["scalar_features"], nan=0.0, posinf=0.0, neginf=0.0
  )
  train_features = rows["scalar_features"][masks["train"]]
  mean = train_features.mean(axis=0)
  std = train_features.std(axis=0) + 1e-6
  rows["scalar_features"] = (rows["scalar_features"] - mean) / std

  counts = {}
  for split_name, mask in masks.items():
    counts[split_name] = int(mask.sum())
    _write_shards(out_dir, split_name, rows, mask, config.data.shard_size)

  meta = {
      "feature_names": list(FEATURE_NAMES),
      "feature_mean": mean.tolist(),
      "feature_std": std.tolist(),
      "global_bins": config.data.global_bins,
      "local_bins": config.data.local_bins,
      "labels": list(config.data.labels),
      "counts": counts,
  }
  with open(os.path.join(out_dir, "meta.json"), "w") as handle:
    json.dump(meta, handle, indent=2)
  return meta


def _write_shards(out_dir, split_name, rows, mask, shard_size):
  """Writes one partition's rows to ``.npz`` shards."""
  indices = np.nonzero(mask)[0]
  for shard, start in enumerate(range(0, len(indices), shard_size)):
    chunk = indices[start : start + shard_size]
    path = os.path.join(out_dir, f"{split_name}_{shard:04d}.npz")
    np.savez(
        path, **{key: value[chunk] for key, value in rows.items()}
    )
