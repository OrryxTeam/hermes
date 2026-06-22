"""Tests for detection metrics in hermes.trainers.metrics."""

import numpy as np

from hermes.trainers import metrics


def test_roc_auc_perfect_and_random():
  """ROC-AUC is 1 for perfect ranking and ~0.5 for random scores."""
  labels = np.array([0, 0, 1, 1])
  assert metrics.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
  assert metrics.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
  rng = np.random.RandomState(0)
  scores = rng.uniform(size=2000)
  random_labels = rng.randint(0, 2, size=2000)
  assert abs(metrics.roc_auc(scores, random_labels) - 0.5) < 0.05


def test_average_precision_perfect():
  """Average precision is 1 when all positives rank first."""
  scores = np.array([0.9, 0.8, 0.2, 0.1])
  labels = np.array([1, 1, 0, 0])
  assert metrics.average_precision(scores, labels) == 1.0


def test_precision_recall_at_threshold():
  """Precision and recall match a hand-computed example."""
  scores = np.array([0.9, 0.6, 0.4, 0.1])
  labels = np.array([1, 0, 1, 0])
  precision, recall = metrics.precision_recall(scores, labels, threshold=0.5)
  assert precision == 0.5 and recall == 0.5
