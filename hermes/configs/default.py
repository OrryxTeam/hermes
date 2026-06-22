"""Default HERMES configuration (ml_collections ConfigDict)."""

from __future__ import annotations

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns the default HERMES configuration.

  Returns:
    A nested `ml_collections.ConfigDict` with ``data``, ``search``,
    ``model`` and ``train`` sections.
  """
  config = ml_collections.ConfigDict()
  config.seed = 42

  data = config.data = ml_collections.ConfigDict()
  data.catalog_csv = "dataset/q1_q17_dr25_tce.csv"
  data.catalog_format = "tce"  # "tce" (DR25 TCE) or "koi" (KOI cumulative)
  data.light_curve_dir = "dataset/lightcurves"
  data.processed_dir = "dataset/processed"
  data.source = "injection"  # one of {"injection", "fits", "flux_csv"}
  data.labels = ("PC", "AFP", "NTP")
  data.positive_label = "PC"
  data.global_bins = 2001
  data.local_bins = 201
  data.local_durations = 4.0  # half-window of the local view, in durations
  data.shard_size = 2000
  # Injection / synthetic light-curve geometry.
  data.baseline_days = 1500.0
  data.cadence_days = 0.02043  # Kepler long-cadence sampling
  data.noise_ppm = 200.0
  # Detrending and cleaning.
  data.detrend_window_days = 2.0
  data.sigma_clip = 5.0
  # Star-level split fractions (split on ``kepid`` to avoid leakage).
  split = data.split = ml_collections.ConfigDict()
  split.train = 0.7
  split.validation = 0.15
  split.test = 0.15

  search = config.search = ml_collections.ConfigDict()
  search.period_min = 0.5
  search.period_max = 100.0
  search.oversample = 3.0
  search.num_bins = 256
  search.duration_fractions = (0.01, 0.02, 0.04, 0.08, 0.12)
  search.limb_darkening = (0.4, 0.3)

  model = config.model = ml_collections.ConfigDict()
  model.conv_channels = (16, 32, 64)
  model.kernel_size = 5
  model.transformer_layers = 2
  model.num_heads = 4
  model.embed_dim = 128
  model.mlp_dim = 256
  model.dropout_rate = 0.1
  model.num_scalar_features = 8

  train = config.train = ml_collections.ConfigDict()
  train.batch_size = 64
  train.num_epochs = 40
  train.early_stopping_patience = 10  # epochs without val gain; 0 disables
  train.learning_rate = 1e-3
  train.weight_decay = 1e-4
  train.warmup_steps = 500
  train.grad_clip_norm = 1.0
  train.regression_loss_weight = 0.1
  train.physics_loss_weight = 0.1
  train.positive_class_weight = 1.0  # set from data prevalence by the driver
  train.label_smoothing = 0.0

  return config
