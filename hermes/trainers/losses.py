"""Composite vetter loss.

Classification cross-entropy, a heteroscedastic Gaussian regression on transit
depth and duration, and a stellar-density physics-consistency penalty.
"""

from __future__ import annotations

import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

from hermes.numerics import orbit


def _density_log_ratio(
    period_days: Array,
    pred_log_depth: Array,
    pred_log_duration: Array,
    stellar_density_cgs: Array,
) -> Array:
  """Stellar-density log-ratio implied by the predicted transit parameters."""
  depth = jnp.exp(pred_log_depth)
  duration = jnp.clip(jnp.exp(pred_log_duration), 1e-3, None)
  radius_ratio = jnp.sqrt(jnp.clip(depth, 1e-8, 0.5))
  a_over_rstar = period_days / jnp.pi * (1.0 + radius_ratio) / duration
  density_si = jnp.clip(stellar_density_cgs * 1000.0, 1e-3, None)
  return orbit.density_consistency_logratio(
      period_days, a_over_rstar, density_si
  )


def vetter_loss(
    model: nnx.Module,
    batch: dict[str, Array],
    regression_weight: float,
    physics_weight: float,
    positive_weight: float = 1.0,
) -> tuple[Array, dict[str, Array]]:
  """Computes the composite vetter loss and auxiliary metrics.

  Args:
    model: The `hermes.nn.vetter.TransitVetter` instance.
    batch: A batch dict with views, scalar features, labels, and the raw
      ``period_days``, ``depth_fraction``, ``duration_days`` and
      ``stellar_density_cgs`` arrays.
    regression_weight: Weight of the heteroscedastic regression term.
    physics_weight: Weight of the physics-consistency term.
    positive_weight: Up-weighting of the positive (planet-candidate) class in
      the classification loss, to counter the class imbalance of TCE catalogues.

  Returns:
    A tuple ``(loss, aux)`` where ``aux`` holds the component losses, the logits
    and the labels (for metrics).
  """
  outputs = model(
      batch["global_view"], batch["local_view"], batch["scalar_features"]
  )
  labels = batch["label"].astype(jnp.float32)
  logits = outputs["logit"]
  per_example = optax.sigmoid_binary_cross_entropy(logits, labels)
  sample_weight = jnp.where(labels > 0.5, positive_weight, 1.0)
  classification = jnp.sum(per_example * sample_weight) / jnp.sum(sample_weight)

  targets = jnp.stack(
      [jnp.log(batch["depth_fraction"]), jnp.log(batch["duration_days"])],
      axis=-1,
  )
  mean = outputs["mean"]
  log_variance = outputs["log_variance"]
  gaussian_nll = jnp.sum(
      0.5 * jnp.exp(-log_variance) * (mean - targets) ** 2 + 0.5 * log_variance,
      axis=-1,
  )
  positive = labels
  positive_count = jnp.maximum(jnp.sum(positive), 1.0)
  regression = jnp.sum(gaussian_nll * positive) / positive_count

  log_ratio = _density_log_ratio(
      batch["period_days"], mean[:, 0], mean[:, 1], batch["stellar_density_cgs"]
  )
  has_density = (batch["stellar_density_cgs"] > 0.0).astype(jnp.float32)
  consistent = positive * has_density
  physics = jnp.sum(log_ratio**2 * consistent) / jnp.maximum(
      jnp.sum(consistent), 1.0
  )

  total = (
      classification + regression_weight * regression + physics_weight * physics
  )
  aux = {
      "classification": classification,
      "regression": regression,
      "physics": physics,
      "accuracy": jnp.mean((logits > 0.0) == (labels > 0.5)),
  }
  return total, aux
