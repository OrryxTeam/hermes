"""Tests for probability calibration in hermes.trainers.calibration."""

import numpy as np

from hermes.trainers import calibration


def _overconfident_logits(seed=0, n=4000, scale=2.5):
  """Builds labels and deliberately overconfident logits for them."""
  rng = np.random.RandomState(seed)
  true_prob = rng.uniform(0.05, 0.95, n)
  labels = (rng.uniform(size=n) < true_prob).astype(np.float32)
  true_logit = np.log(true_prob / (1.0 - true_prob))
  return scale * true_logit, labels


def test_fit_temperature_recovers_scale():
  """Temperature scaling recovers the over-confidence factor."""
  logits, labels = _overconfident_logits(scale=2.5)
  assert 2.0 < calibration.fit_temperature(logits, labels) < 3.0


def test_temperature_scaling_reduces_ece():
  """Calibration lowers the expected calibration error."""
  logits, labels = _overconfident_logits(scale=2.5)
  before = calibration.expected_calibration_error(
      1.0 / (1.0 + np.exp(-logits)), labels
  )
  temperature = calibration.fit_temperature(logits, labels)
  after = calibration.expected_calibration_error(
      np.asarray(calibration.apply_temperature(logits, temperature)), labels
  )
  assert after < before


def test_reliability_curve_shapes():
  """The reliability curve returns one entry per bin and totals all points."""
  logits, labels = _overconfident_logits(n=1000)
  probs = 1.0 / (1.0 + np.exp(-logits))
  confidence, accuracy, count = calibration.reliability_curve(
      probs, labels, num_bins=10
  )
  assert confidence.shape == accuracy.shape == count.shape == (10,)
  assert count.sum() == 1000
