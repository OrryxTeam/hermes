"""Restore a trained vetter, calibrate on validation, and report test ROC-AUC,
average precision and expected calibration error (before and after calibration).
"""

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl import app, flags, logging
from flax import nnx
from ml_collections import config_flags

from hermes.data import dataset
from hermes.evaluators import harness
from hermes.models.vetter import TransitVetter
from hermes.trainers import calibration

_CONFIG = config_flags.DEFINE_config_file("config", "hermes/configs/default.py")
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint", None, "Checkpoint path to restore."
)


def _collect_logits(model, data, batch_size):
  """Returns concatenated logits and labels over a dataset split."""
  model.eval()
  logits, labels = [], []
  for batch in data.iterate(batch_size, shuffle=False):
    outputs = model(
        jnp.asarray(batch["global_view"]),
        jnp.asarray(batch["local_view"]),
        jnp.asarray(batch["scalar_features"]),
    )
    logits.append(np.asarray(outputs["logit"]))
    labels.append(np.asarray(batch["label"]))
  return np.concatenate(logits), np.concatenate(labels)


def main(_) -> None:
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
    restored = ocp.StandardCheckpointer().restore(
        _CHECKPOINT.value, nnx.state(model, nnx.Param)
    )
    nnx.update(model, restored)

  val = dataset.HermesDataset(config.data.processed_dir, "validation")
  test = dataset.HermesDataset(config.data.processed_dir, "test")
  batch_size = config.train.batch_size

  val_logits, val_labels = _collect_logits(model, val, batch_size)
  temperature = calibration.fit_temperature(val_logits, val_labels)
  logging.info("Fitted calibration temperature: %.3f", temperature)

  test_logits, test_labels = _collect_logits(model, test, batch_size)
  uncalibrated = np.asarray(calibration.apply_temperature(test_logits, 1.0))
  calibrated = np.asarray(
      calibration.apply_temperature(test_logits, temperature)
  )

  metrics = harness.score_metrics(calibrated, test_labels)
  logging.info("Test ROC-AUC: %.4f", metrics["roc_auc"])
  logging.info("Test average precision: %.4f", metrics["average_precision"])
  logging.info(
      "Test ECE: %.4f -> %.4f (after calibration)",
      calibration.expected_calibration_error(uncalibrated, test_labels),
      calibration.expected_calibration_error(calibrated, test_labels),
  )


if __name__ == "__main__":
  app.run(main)
