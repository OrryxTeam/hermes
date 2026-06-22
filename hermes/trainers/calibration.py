"""Probability calibration by temperature scaling (reliability curve, ECE)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array


def fit_temperature(
    logits: Array,
    labels: Array,
    *,
    steps: int = 300,
    learning_rate: float = 0.05,
) -> float:
  """Fits the temperature that minimises validation cross-entropy.

  Args:
    logits: Uncalibrated logits, shape ``(n,)``.
    labels: Binary labels, shape ``(n,)``.
    steps: Number of optimisation steps.
    learning_rate: Adam learning rate.

  Returns:
    The fitted temperature ``T`` (a positive scalar).
  """
  logits = jnp.asarray(logits)
  labels = jnp.asarray(labels, dtype=logits.dtype)

  def loss(log_temperature: Array) -> Array:
    scaled = logits * jnp.exp(-log_temperature)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scaled, labels))

  optimizer = optax.adam(learning_rate)

  def step(carry, _):
    value, state = carry
    grad = jax.grad(loss)(value)
    updates, state = optimizer.update(grad, state)
    return (optax.apply_updates(value, updates), state), None

  log_temperature = jnp.asarray(0.0, dtype=logits.dtype)
  initial = (log_temperature, optimizer.init(log_temperature))
  (log_temperature, _), _ = jax.lax.scan(step, initial, None, length=steps)
  return float(jnp.exp(log_temperature))


def apply_temperature(logits: Array, temperature: float) -> Array:
  """Returns calibrated probabilities ``sigmoid(logits / temperature)``."""
  return jax.nn.sigmoid(jnp.asarray(logits) / temperature)


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes a reliability curve over equal-width probability bins.

  Args:
    probabilities: Predicted probabilities in ``[0, 1]``.
    labels: Binary labels.
    num_bins: Number of probability bins.

  Returns:
    A tuple ``(mean_confidence, empirical_accuracy, count)`` per bin.
  """
  probabilities = np.asarray(probabilities)
  labels = np.asarray(labels)
  edges = np.linspace(0.0, 1.0, num_bins + 1)
  bin_index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, num_bins - 1)

  confidence = np.zeros(num_bins)
  accuracy = np.zeros(num_bins)
  count = np.zeros(num_bins)
  for b in range(num_bins):
    mask = bin_index == b
    count[b] = mask.sum()
    if count[b] > 0:
      confidence[b] = probabilities[mask].mean()
      accuracy[b] = labels[mask].mean()
  return confidence, accuracy, count


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> float:
  """Computes the expected calibration error (ECE).

  Args:
    probabilities: Predicted probabilities in ``[0, 1]``.
    labels: Binary labels.
    num_bins: Number of probability bins.

  Returns:
    The ECE: the count-weighted mean ``|confidence - accuracy|`` over bins.
  """
  confidence, accuracy, count = reliability_curve(
      probabilities, labels, num_bins
  )
  weights = count / max(count.sum(), 1.0)
  return float(np.sum(weights * np.abs(confidence - accuracy)))
