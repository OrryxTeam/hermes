"""Synthetic catalogues, datasets and light curves for tests.

These helpers keep the test files small and consistent: a DR25-style catalogue
CSV writer, a small end-to-end dataset builder, and an injected single-transit
light curve.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from hermes.configs.default import get_config
from hermes.numerics import transit

_CSV_HEADER = (
    "kepid,tce_plnt_num,tce_period,tce_time0bk,tce_duration,tce_depth,"
    "av_training_set,tce_steff,tce_sradius,tce_slogg,tce_sdens"
)


def write_tce_csv(path, num_stars=12, label_set=("PC", "AFP", "NTP")) -> str:
  """Writes a small synthetic DR25-style TCE CSV and returns its path."""
  lines = [_CSV_HEADER]
  for star in range(1, num_stars + 1):
    label = label_set[star % len(label_set)]
    period = 5.0 + star % 11
    lines.append(f"{star},1,{period},1.0,3.5,2500.0,{label},5700,1.0,4.4,1.4")
  path.write_text("\n".join(lines) + "\n")
  return str(path)


def small_dataset_config(
    tmp_path, label_set=("PC", "AFP", "NTP"), num_stars=42
):
  """Writes a catalogue, configures small dims and builds the dataset."""
  from hermes.data import build_dataset

  config = get_config()
  config.data.catalog_csv = write_tce_csv(
      tmp_path / "tce.csv", num_stars, label_set
  )
  config.data.processed_dir = str(tmp_path / "processed")
  config.data.baseline_days = 60.0
  config.data.global_bins = 201
  config.data.local_bins = 51
  config.model.conv_channels = (8, 16)
  config.model.embed_dim = 32
  config.model.mlp_dim = 64
  config.model.transformer_layers = 1
  config.model.num_heads = 2
  build_dataset.build(config)
  return config


def injected_light_curve(
    period,
    epoch=1.0,
    depth_ratio=0.1,
    noise=1e-3,
    *,
    baseline=80.0,
    num=8000,
    seed=0,
):
  """Builds a noisy light curve with one injected transit."""
  time = np.linspace(0.0, baseline, num)
  clean = transit.transit_light_curve(
      jnp.asarray(time),
      period_days=period,
      epoch_days=epoch,
      radius_ratio=depth_ratio,
      a_over_rstar=11.0,
      impact_parameter=0.0,
  )
  rng = np.random.RandomState(seed)
  flux = np.asarray(clean) + rng.normal(0.0, noise, num)
  return jnp.asarray(time), jnp.asarray(flux)


def single_candidate_catalog(label="PC"):
  """A one-row catalogue for exercising a light-curve store."""
  from hermes.data.catalog import TceCatalog

  return TceCatalog(
      kepid=np.array([42]),
      planet_num=np.array([1]),
      period_days=np.array([6.0]),
      epoch_days=np.array([1.0]),
      duration_days=np.array([0.2]),
      depth_fraction=np.array([5e-3]),
      label=np.array([label]),
      stellar_teff_k=np.array([5700.0]),
      stellar_radius_solar=np.array([1.0]),
      stellar_logg_cgs=np.array([4.4]),
      stellar_density_cgs=np.array([1.4]),
  )
