"""Detection metrics for imbalanced binary vetting.

ROC-AUC, average precision and precision/recall, in NumPy (no scikit-learn).
"""

from __future__ import annotations

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
  """Area under the ROC curve via the rank-based Mann-Whitney statistic.

  Args:
    scores: Predicted scores (higher means more likely positive).
    labels: Binary labels (1 positive, 0 negative).

  Returns:
    The ROC AUC in ``[0, 1]``; ``0.5`` when one class is absent.
  """
  scores = np.asarray(scores, dtype=np.float64)
  labels = np.asarray(labels)
  num_pos = int(np.sum(labels == 1))
  num_neg = int(np.sum(labels == 0))
  if num_pos == 0 or num_neg == 0:
    return 0.5
  order = np.argsort(scores, kind="mergesort")
  ranks = np.empty_like(order, dtype=np.float64)
  ranks[order] = _average_ranks(scores[order])
  rank_sum_pos = np.sum(ranks[labels == 1])
  return (rank_sum_pos - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
  """Area under the precision-recall curve (average precision).

  Args:
    scores: Predicted scores (higher means more likely positive).
    labels: Binary labels (1 positive, 0 negative).

  Returns:
    The average precision in ``[0, 1]``; ``0`` when no positives are present.
  """
  scores = np.asarray(scores, dtype=np.float64)
  labels = np.asarray(labels)
  num_pos = int(np.sum(labels == 1))
  if num_pos == 0:
    return 0.0
  order = np.argsort(scores, kind="mergesort")[::-1]
  sorted_labels = labels[order]
  true_positive = np.cumsum(sorted_labels == 1)
  precision = true_positive / (np.arange(len(sorted_labels)) + 1)
  recall = true_positive / num_pos
  recall_delta = np.diff(recall, prepend=0.0)
  return float(np.sum(precision * recall_delta))


def precision_recall(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
  """Precision and recall at a probability threshold.

  Args:
    scores: Predicted probabilities or scores.
    labels: Binary labels.
    threshold: Decision threshold.

  Returns:
    A tuple ``(precision, recall)``.
  """
  scores = np.asarray(scores)
  labels = np.asarray(labels)
  predicted = scores >= threshold
  true_positive = int(np.sum(predicted & (labels == 1)))
  predicted_positive = int(np.sum(predicted))
  actual_positive = int(np.sum(labels == 1))
  precision = true_positive / predicted_positive if predicted_positive else 0.0
  recall = true_positive / actual_positive if actual_positive else 0.0
  return precision, recall


def _average_ranks(sorted_scores: np.ndarray) -> np.ndarray:
  """Returns 1-based ranks with ties assigned their average rank."""
  n = len(sorted_scores)
  ranks = np.arange(1, n + 1, dtype=np.float64)
  start = 0
  for i in range(1, n + 1):
    if i == n or sorted_scores[i] != sorted_scores[start]:
      ranks[start:i] = (start + 1 + i) / 2.0
      start = i
  return ranks
