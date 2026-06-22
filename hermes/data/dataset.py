"""Reader and batch iterator for the sharded view-tensor dataset."""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator

import numpy as np

_ARRAY_KEYS = (
    "global_view",
    "local_view",
    "scalar_features",
    "label",
    "label_class",
    "kepid",
    "period_days",
    "depth_fraction",
    "duration_days",
    "stellar_density_cgs",
)


def load_meta(processed_dir: str) -> dict:
  """Loads the dataset metadata written during construction."""
  with open(os.path.join(processed_dir, "meta.json")) as handle:
    return json.load(handle)


class HermesDataset:
  """In-memory view of one dataset partition.

  Attributes:
    arrays: Mapping from array name to the concatenated partition arrays.
  """

  def __init__(self, processed_dir: str, split: str):
    """Loads all shards for ``split`` from ``processed_dir``.

    Args:
      processed_dir: Directory containing the ``.npz`` shards.
      split: Partition name (``train``, ``validation`` or ``test``).

    Raises:
      FileNotFoundError: If no shards exist for the requested split.
    """
    paths = sorted(glob.glob(os.path.join(processed_dir, f"{split}_*.npz")))
    if not paths:
      raise FileNotFoundError(
          f"No shards for split {split!r} in {processed_dir}"
      )
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in _ARRAY_KEYS}
    for path in paths:
      with np.load(path) as shard:
        for key in _ARRAY_KEYS:
          chunks[key].append(shard[key])
    self.arrays = {
        key: np.concatenate(values) for key, values in chunks.items()
    }

  def __len__(self) -> int:
    return len(self.arrays["label"])

  def iterate(
      self,
      batch_size: int,
      shuffle: bool,
      seed: int = 0,
      drop_remainder: bool = False,
  ) -> Iterator[dict[str, np.ndarray]]:
    """Yields batches of the partition.

    Args:
      batch_size: Number of examples per batch.
      shuffle: Whether to shuffle the order each pass.
      seed: Random seed used when shuffling.
      drop_remainder: Whether to drop a trailing partial batch.

    Yields:
      A dict of batched arrays (views, scalar features and labels).
    """
    order = np.arange(len(self))
    if shuffle:
      np.random.RandomState(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
      batch_index = order[start : start + batch_size]
      if drop_remainder and len(batch_index) < batch_size:
        break
      yield {key: value[batch_index] for key, value in self.arrays.items()}
