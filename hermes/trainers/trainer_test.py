"""Tests for checkpointing in hermes.trainers.trainer."""

import jax.numpy as jnp
from flax import nnx

from hermes.data import dataset
from hermes.models.vetter import TransitVetter
from hermes.testing import _fixtures
from hermes.trainers.trainer import Trainer


def test_checkpoint_round_trip(tmp_path):
  """Saving overwrites best/ and restoring returns the latest predictions."""
  config = _fixtures.small_dataset_config(tmp_path, num_stars=18)
  train = dataset.HermesDataset(config.data.processed_dir, "train")
  model = TransitVetter(
      config.model, config.data.global_bins, config.data.local_bins,
      rngs=nnx.Rngs(0),
  )
  trainer = Trainer(
      model, config.train, steps_per_epoch=2,
      checkpoint_dir=str(tmp_path / "ckpt"),
  )
  trainer.train_epoch(train.iterate(8, shuffle=True, seed=0))
  trainer.save()  # an early "best"

  # A later improvement overwrites best/; restore must return this newer model.
  trainer.train_epoch(train.iterate(8, shuffle=True, seed=1))
  batch = next(train.iterate(8, shuffle=False))
  model.eval()
  before = model(
      batch["global_view"], batch["local_view"], batch["scalar_features"]
  )
  trainer.save()
  # Corrupt the parameters, then restore and confirm predictions return.
  nnx.update(model, nnx.state(TransitVetter(
      config.model, config.data.global_bins, config.data.local_bins,
      rngs=nnx.Rngs(99)), nnx.Param))
  trainer.restore()
  model.eval()
  after = model(
      batch["global_view"], batch["local_view"], batch["scalar_features"]
  )
  assert jnp.allclose(before["logit"], after["logit"], atol=1e-5)
