"""Benchmark inference speed of the vetter and the full detection pipeline.

Run with `python -m hermes.benchmark --config=hermes/configs/default.py`; pass
``--nopipeline`` to time only the vetter forward pass, or ``--output=PATH`` to
also write the results as JSON. Results are logged to stderr via absl.
"""

import json

import jax.numpy as jnp
import numpy as np
from absl import app, flags, logging
from flax import nnx
from ml_collections import config_flags

from hermes.evaluators import speed
from hermes.models.vetter import TransitVetter
from hermes.numerics import transit
from hermes.pipeline import VetterPipeline

_CONFIG = config_flags.DEFINE_config_file("config", "hermes/configs/default.py")
_PIPELINE = flags.DEFINE_bool(
    "pipeline", True, "Also benchmark the full detection pipeline."
)
_OUTPUT = flags.DEFINE_string(
    "output", None, "Write results JSON to this path."
)


def _synthetic_light_curve(config):
  """Builds a representative noisy single-transit light curve for timing."""
  num = int(config.data.baseline_days / config.data.cadence_days)
  time_days = np.arange(num) * config.data.cadence_days
  flux = transit.transit_light_curve(
      jnp.asarray(time_days),
      period_days=5.0,
      epoch_days=1.0,
      radius_ratio=0.1,
      a_over_rstar=12.0,
      impact_parameter=0.0,
  )
  noise = np.random.RandomState(0).normal(0.0, 2e-4, num)
  return time_days, np.asarray(flux) + noise


def main(_) -> None:
  logging.set_verbosity(logging.INFO)
  config = _CONFIG.value
  model = TransitVetter(
      config.model,
      config.data.global_bins,
      config.data.local_bins,
      rngs=nnx.Rngs(config.seed),
  )

  vetter = speed.benchmark_vetter(
      model,
      config.data.global_bins,
      config.data.local_bins,
      num_features=config.model.num_scalar_features,
  )
  logging.info(
      "vetter: %.2fM params on %s",
      vetter["num_params"] / 1e6, vetter["device"],
  )
  for row in vetter["timings"]:
    logging.info(
        "  batch=%4d  %.3f ms/candidate  %.0f candidates/s",
        row["batch_size"], row["ms_per_candidate"],
        row["candidates_per_second"],
    )

  report = {"vetter": vetter}
  if _PIPELINE.value:
    meta = {
        "feature_mean": [0.0] * config.model.num_scalar_features,
        "feature_std": [1.0] * config.model.num_scalar_features,
        "global_bins": config.data.global_bins,
        "local_bins": config.data.local_bins,
    }
    pipeline = VetterPipeline(model, meta, config.search, config.data)
    time_days, flux = _synthetic_light_curve(config)
    result = speed.benchmark_pipeline(pipeline, time_days, flux)
    report["pipeline"] = result
    logging.info(
        "pipeline: %.2f s / light curve (%d points; search+fit+vet)",
        result["seconds_per_light_curve"], result["num_points"],
    )

  if _OUTPUT.value:
    with open(_OUTPUT.value, "w") as handle:
      json.dump(report, handle, indent=2)
    logging.info("Wrote benchmark results to %s", _OUTPUT.value)


if __name__ == "__main__":
  app.run(main)
