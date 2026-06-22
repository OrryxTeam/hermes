"""Tests for star-level splits in hermes.data.splits."""

import numpy as np

from hermes.data import splits


def test_split_by_star_is_disjoint():
  """Stars are partitioned without overlap and every star is assigned."""
  kepids = np.repeat(np.arange(100), 2)
  manifest = splits.split_by_star(kepids, (0.7, 0.15, 0.15), seed=1)
  all_stars = np.concatenate(
      [manifest.train, manifest.validation, manifest.test]
  )
  assert len(np.unique(all_stars)) == 100
  assert set(manifest.train) & set(manifest.test) == set()


def test_partition_masks_cover_every_row():
  """Every TCE row falls into exactly one partition mask."""
  kepids = np.repeat(np.arange(40), 2)
  manifest = splits.split_by_star(kepids, (0.7, 0.15, 0.15), seed=2)
  masks = splits.partition_masks(kepids, manifest)
  total = sum(int(mask.sum()) for mask in masks.values())
  assert total == len(kepids)
