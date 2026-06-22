"""End-to-end: the detection pipeline recovers an injected transit."""

import numpy as np
from flax import nnx

from hermes.data import dataset
from hermes.models.vetter import TransitVetter
from hermes.pipeline import StellarParameters, VetterPipeline
from hermes.testing import _fixtures


def test_pipeline_detects_injected_transit(tmp_path):
  """The pipeline recovers an injected transit and returns a valid candidate."""
  config = _fixtures.small_dataset_config(
      tmp_path, label_set=("PC", "NTP"), num_stars=20
  )
  # Narrow the search around the injected period to keep the e2e test quick.
  config.search.period_min = 4.0
  config.search.period_max = 10.0
  meta = dataset.load_meta(config.data.processed_dir)
  model = TransitVetter(
      config.model, meta["global_bins"], meta["local_bins"], rngs=nnx.Rngs(0)
  )
  pipeline = VetterPipeline(model, meta, config.search, config.data)

  time, flux = _fixtures.injected_light_curve(
      7.0, epoch=2.0, noise=3e-4, baseline=70.0, num=7000
  )
  candidates = pipeline.detect(
      np.asarray(time),
      np.asarray(flux),
      stellar=StellarParameters(),
      max_planets=1,
      power_threshold=0.0,
  )
  assert len(candidates) == 1
  candidate = candidates[0]
  assert abs(candidate.period_days - 7.0) < 0.1
  assert 0.0 <= candidate.probability <= 1.0
  assert candidate.planet_radius_earth > 0.0
  assert candidate.equilibrium_temperature_k > 0.0
