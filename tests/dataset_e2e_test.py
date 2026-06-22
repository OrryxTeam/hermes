"""End-to-end: build the sharded dataset and read it back."""

import os

import numpy as np

from hermes.data import dataset, splits
from hermes.testing import _fixtures


def test_build_and_read_roundtrip(tmp_path):
  """The offline builder writes shards that the reader can batch."""
  config = _fixtures.small_dataset_config(tmp_path, num_stars=12)
  meta = dataset.load_meta(config.data.processed_dir)
  assert sum(meta["counts"].values()) == 12
  assert len(meta["feature_names"]) == 8

  manifest = splits.SplitManifest.load(
      os.path.join(config.data.processed_dir, "split_manifest.json")
  )
  assert set(manifest.train) & set(manifest.validation) == set()

  train = dataset.HermesDataset(config.data.processed_dir, "train")
  batch = next(train.iterate(batch_size=4, shuffle=True, seed=0))
  assert batch["global_view"].shape == (4, 201)
  assert batch["local_view"].shape == (4, 51)
  assert batch["scalar_features"].shape == (4, 8)
  assert set(np.unique(batch["label"])) <= {0, 1}
