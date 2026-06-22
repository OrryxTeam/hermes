"""Train the transit vetter on the sharded dataset.

Run with `python -m hermes.train --config=hermes/configs/default.py`;
hyperparameters live in the config and can be overridden on the command line.
"""

import os
from datetime import datetime

import jax.numpy as jnp
import numpy as np
from absl import app, logging
from flax import nnx
from ml_collections import config_flags

from hermes.data import dataset
from hermes.models.vetter import TransitVetter
from hermes.trainers import metrics
from hermes.trainers.trainer import Trainer

_CONFIG = config_flags.DEFINE_config_file("config", "hermes/configs/default.py")


def _validation_auc(model, val_data, batch_size) -> float:
  """Computes the validation ROC-AUC from the model's logits."""
  model.eval()
  scores, labels = [], []
  for batch in val_data.iterate(batch_size, shuffle=False):
    logits = model(
        jnp.asarray(batch["global_view"]),
        jnp.asarray(batch["local_view"]),
        jnp.asarray(batch["scalar_features"]),
    )["logit"]
    scores.append(np.asarray(logits))
    labels.append(np.asarray(batch["label"]))
  return metrics.roc_auc(np.concatenate(scores), np.concatenate(labels))


def main(_) -> None:
  logging.set_verbosity(logging.INFO)
  config = _CONFIG.value

  meta = dataset.load_meta(config.data.processed_dir)
  train_data = dataset.HermesDataset(config.data.processed_dir, "train")
  val_data = dataset.HermesDataset(config.data.processed_dir, "validation")

  # Up-weight the positive class by its inverse prevalence in the training set.
  prevalence = max(float(train_data.arrays["label"].mean()), 1e-6)
  config.train.positive_class_weight = (1.0 - prevalence) / prevalence

  run_dir = os.path.join(
      "outputs", datetime.now().strftime("run_%Y%m%d_%H%M%S")
  )
  model = TransitVetter(
      config.model,
      meta["global_bins"],
      meta["local_bins"],
      rngs=nnx.Rngs(config.seed),
  )
  steps_per_epoch = max(len(train_data) // config.train.batch_size, 1)
  trainer = Trainer(
      model, config.train, steps_per_epoch, os.path.join(run_dir, "checkpoints")
  )

  # The best checkpoint is the epoch with the highest validation ROC-AUC -- the
  # area under the ROC curve for planet-vs-false-positive ranking. It is
  # threshold-free and robust to class imbalance, so it tracks discrimination
  # better than the composite loss or fixed-threshold accuracy.
  best_auc = -1.0
  best_epoch = 0
  best_train_acc = 0.0
  best_val_acc = 0.0
  epochs_without_improvement = 0
  patience = config.train.early_stopping_patience
  logging.info("Selecting best checkpoint by validation ROC-AUC")
  for epoch in range(config.train.num_epochs):
    train_metrics = trainer.train_epoch(
        train_data.iterate(config.train.batch_size, shuffle=True, seed=epoch)
    )
    val_metrics = trainer.evaluate(
        val_data.iterate(config.train.batch_size, shuffle=False)
    )
    val_auc = _validation_auc(model, val_data, config.train.batch_size)
    logging.info(
        "epoch %d  train_loss=%.4f train_acc=%.3f  val_loss=%.4f"
        " val_acc=%.3f val_auc=%.3f",
        epoch + 1, train_metrics["loss"], train_metrics["accuracy"],
        val_metrics["loss"], val_metrics["accuracy"], val_auc,
    )
    if val_auc > best_auc:
      best_auc = val_auc
      best_epoch = epoch + 1
      best_train_acc = train_metrics["accuracy"]
      best_val_acc = val_metrics["accuracy"]
      epochs_without_improvement = 0
      trainer.save("best")
    else:
      epochs_without_improvement += 1
      if patience and epochs_without_improvement >= patience:
        logging.info("Early stopping; no val AUC gain in %d epochs", patience)
        break

  logging.info(
      "Done; best epoch %d: val_auc=%.4f val_acc=%.3f train_acc=%.3f"
      " (saved to best/)",
      best_epoch, best_auc, best_val_acc, best_train_acc,
  )


if __name__ == "__main__":
  app.run(main)
