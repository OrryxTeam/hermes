"""Detect and vet planets in one star's cached light curve.

Restores a trained vetter, runs the detection pipeline on the star's local FITS,
and prints the calibrated candidates as JSON.
"""

import dataclasses
import json

import jax
import numpy as np
import orbax.checkpoint as ocp
from absl import app, flags, logging
from flax import nnx
from ml_collections import config_flags

from hermes.data import dataset
from hermes.data.catalog import TceCatalog
from hermes.data.lightcurves import FitsLightCurveStore
from hermes.models.vetter import TransitVetter
from hermes.pipeline import VetterPipeline

_CONFIG = config_flags.DEFINE_config_file("config", "hermes/configs/default.py")
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint", None, "Checkpoint path to restore."
)
_KEPID = flags.DEFINE_integer(
    "kepid", None, "Kepler id of the target to analyse."
)
_MAX_PLANETS = flags.DEFINE_integer(
    "max_planets", 3, "Maximum candidates to extract."
)


def main(_) -> None:
  # Fold real Kepler BKJD timestamps (~1500 days) in double precision.
  jax.config.update("jax_enable_x64", True)
  logging.set_verbosity(logging.INFO)
  config = _CONFIG.value
  meta = dataset.load_meta(config.data.processed_dir)

  model = TransitVetter(
      config.model,
      meta["global_bins"],
      meta["local_bins"],
      rngs=nnx.Rngs(config.seed),
  )
  if _CHECKPOINT.value:
    model_state = ocp.StandardCheckpointer().restore(
        _CHECKPOINT.value, nnx.state(model, nnx.Param)
    )
    nnx.update(model, model_state)

  target = TceCatalog(
      kepid=np.array([_KEPID.value]),
      **{
          field: np.array([np.nan if field != "label" else "PC"])
          for field in (
              "planet_num", "period_days", "epoch_days", "duration_days",
              "depth_fraction", "label", "stellar_teff_k",
              "stellar_radius_solar", "stellar_logg_cgs", "stellar_density_cgs",
          )
      },
  )
  light_curve = FitsLightCurveStore(config.data.light_curve_dir).get(0, target)
  if light_curve is None:
    raise FileNotFoundError(f"No local FITS for KIC {_KEPID.value}")

  pipeline = VetterPipeline(model, meta, config.search, config.data)
  candidates = pipeline.detect(
      light_curve.time_days, light_curve.flux, light_curve.flux_err,
      max_planets=_MAX_PLANETS.value,
  )
  print(json.dumps([dataclasses.asdict(c) for c in candidates], indent=2))


if __name__ == "__main__":
  app.run(main)
