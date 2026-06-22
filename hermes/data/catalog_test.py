"""Tests for the TCE/KOI catalogue reader in hermes.data.catalog."""

import numpy as np

from hermes.data import catalog as catalog_lib
from hermes.testing import _fixtures


def test_catalog_units_and_labels(tmp_path):
  """Durations convert hours->days, depths ppm->fraction, labels parse."""
  csv_path = _fixtures.write_tce_csv(tmp_path / "tce.csv", num_stars=3)
  cat = catalog_lib.load_tce_catalog(csv_path)
  assert len(cat) == 3
  assert np.allclose(cat.duration_days[0], 3.5 / 24.0)
  assert np.allclose(cat.depth_fraction[0], 2500.0 * 1e-6)
  assert set(cat.label) <= {"PC", "AFP", "NTP"}


def test_catalog_label_filter(tmp_path):
  """Filtering keeps only the requested labels."""
  csv_path = _fixtures.write_tce_csv(tmp_path / "tce.csv", num_stars=9)
  cat = catalog_lib.load_tce_catalog(csv_path).with_labels(("PC",))
  assert set(cat.label) == {"PC"}
