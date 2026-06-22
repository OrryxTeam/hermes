"""Tests for the light-curve sources in hermes.data.lightcurves."""

import numpy as np

from hermes.data import lightcurves
from hermes.numerics import folding
from hermes.testing import _fixtures


def test_injection_pc_produces_transit():
  """An injected planet candidate leaves a dip in the folded global view."""
  catalog = _fixtures.single_candidate_catalog("PC")
  store = lightcurves.InjectionStore(baseline_days=60.0, noise_ppm=50.0)
  light_curve = store.get(0, catalog)
  view = np.asarray(
      folding.global_view(
          light_curve.time_days, light_curve.flux, 6.0, 1.0, 201
      )
  )
  assert view[100] < np.median(view) - 1e-3


def test_injection_ntp_has_no_coherent_transit():
  """A non-transiting injection has no dip at the catalogue ephemeris."""
  catalog = _fixtures.single_candidate_catalog("NTP")
  store = lightcurves.InjectionStore(baseline_days=60.0, noise_ppm=50.0)
  light_curve = store.get(0, catalog)
  view = np.asarray(
      folding.global_view(
          light_curve.time_days, light_curve.flux, 6.0, 1.0, 201
      )
  )
  assert view[100] > np.median(view) - 1e-3
