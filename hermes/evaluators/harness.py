"""Benchmark harness comparing the vetter against the AstroNet and view-SNR
baselines with ROC-AUC, average precision and ECE.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from hermes.models import baselines
from hermes.trainers import calibration, metrics


def _collect(dataset, batch_size: int, score_fn: Callable) -> tuple:
  """Collects scores and labels over a dataset split."""
  scores, labels = [], []
  for batch in dataset.iterate(batch_size, shuffle=False):
    scores.append(np.asarray(score_fn(batch)))
    labels.append(np.asarray(batch["label"]))
  return np.concatenate(scores), np.concatenate(labels)


def score_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
  """Returns ROC-AUC and average precision for a set of scores."""
  return {
      "roc_auc": metrics.roc_auc(scores, labels),
      "average_precision": metrics.average_precision(scores, labels),
  }


def vetter_scores(
    model, dataset, batch_size: int = 64, temperature: float = 1.0
):
  """Collects calibrated HERMES probabilities and labels."""
  model.eval()

  def score_fn(batch):
    outputs = model(
        jnp.asarray(batch["global_view"]),
        jnp.asarray(batch["local_view"]),
        jnp.asarray(batch["scalar_features"]),
    )
    return calibration.apply_temperature(outputs["logit"], temperature)

  return _collect(dataset, batch_size, score_fn)


def astronet_scores(model, dataset, batch_size: int = 64):
  """Collects AstroNet probabilities and labels."""
  model.eval()

  def score_fn(batch):
    logits = model(
        jnp.asarray(batch["global_view"]), jnp.asarray(batch["local_view"])
    )
    return jax.nn.sigmoid(logits)

  return _collect(dataset, batch_size, score_fn)


def compare(
    vetter,
    astronet,
    dataset,
    *,
    temperature: float = 1.0,
    batch_size: int = 64,
) -> dict[str, dict[str, float]]:
  """Evaluates all methods on ``dataset`` and returns their metrics.

  Args:
    vetter: A trained `hermes.models.vetter.TransitVetter`.
    astronet: A trained `hermes.models.baselines.AstroNetClassifier`.
    dataset: The held-out `hermes.data.dataset.HermesDataset`.
    temperature: Calibration temperature for the vetter probabilities.
    batch_size: Evaluation batch size.

  Returns:
    A dict keyed by method (``"hermes"``, ``"astronet"``, ``"bls_snr"``) of
    metric dicts; the HERMES entry also includes its expected calibration error.
  """
  results: dict[str, dict[str, float]] = {}

  hermes_scores, labels = vetter_scores(
      vetter, dataset, batch_size, temperature
  )
  results["hermes"] = score_metrics(hermes_scores, labels)
  results["hermes"]["ece"] = calibration.expected_calibration_error(
      hermes_scores, labels
  )

  astro_scores, _ = astronet_scores(astronet, dataset, batch_size)
  results["astronet"] = score_metrics(astro_scores, labels)

  bls = baselines.view_snr_scores(dataset.arrays["local_view"])
  results["bls_snr"] = score_metrics(bls, dataset.arrays["label"])
  return results
