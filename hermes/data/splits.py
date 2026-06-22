"""Leakage-free train/validation/test splits on the host star (kepid)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np


@dataclasses.dataclass(frozen=True)
class SplitManifest:
  """Star identifiers assigned to each partition.

  Attributes:
    train: Host-star ``kepid`` values in the training partition.
    validation: Host-star ``kepid`` values in the validation partition.
    test: Host-star ``kepid`` values in the test partition.
  """

  train: np.ndarray
  validation: np.ndarray
  test: np.ndarray

  def save(self, path: str) -> None:
    """Writes the manifest to ``path`` as JSON."""
    payload = {name: getattr(self, name).tolist() for name in _PARTITIONS}
    with open(path, "w") as handle:
      json.dump(payload, handle)

  @classmethod
  def load(cls, path: str) -> "SplitManifest":
    """Reads a manifest written by `save`."""
    with open(path) as handle:
      payload = json.load(handle)
    return cls(**{name: np.asarray(payload[name]) for name in _PARTITIONS})


_PARTITIONS = ("train", "validation", "test")


def split_by_star(
    kepids: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int = 42,
) -> SplitManifest:
  """Partitions unique host stars into train/validation/test sets.

  Args:
    kepids: Host-star id for every TCE (duplicates allowed).
    fractions: ``(train, validation, test)`` fractions; they should sum to one.
    seed: Random seed controlling the shuffle.

  Returns:
    A `SplitManifest` of disjoint ``kepid`` arrays.
  """
  unique = np.unique(kepids)
  rng = np.random.RandomState(seed)
  rng.shuffle(unique)

  n = len(unique)
  train_end = int(round(fractions[0] * n))
  val_end = train_end + int(round(fractions[1] * n))
  return SplitManifest(
      train=unique[:train_end],
      validation=unique[train_end:val_end],
      test=unique[val_end:],
  )


def partition_masks(
    kepids: np.ndarray, manifest: SplitManifest
) -> dict[str, np.ndarray]:
  """Maps each partition to a boolean mask over the TCE rows.

  Args:
    kepids: Host-star id for every TCE row.
    manifest: Star-level split assignment.

  Returns:
    A dict from partition name to a boolean mask selecting its TCE rows.
  """
  return {
      name: np.isin(kepids, getattr(manifest, name)) for name in _PARTITIONS
  }
