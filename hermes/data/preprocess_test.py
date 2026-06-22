"""Tests for light-curve cleaning in hermes.data.preprocess."""

import numpy as np

from hermes.data import preprocess
from hermes.data.lightcurves import LightCurve


def test_preprocess_detrends_and_clips():
  """Detrending flattens a trend and sigma-clipping removes an outlier."""
  flux = 1.0 + 0.01 * np.linspace(0.0, 1.0, 500)
  flux[250] = 5.0  # upward outlier
  detrended = preprocess.median_detrend(flux, window_points=51)
  assert abs(np.median(detrended) - 1.0) < 1e-3
  assert not preprocess.sigma_clip_mask(detrended, sigma=5.0)[250]


def test_sigma_clip_preserves_transits():
  """Asymmetric clipping keeps a deep downward dip (a transit)."""
  flux = 1.0 + np.random.RandomState(0).normal(0.0, 1e-4, 500)
  flux[250] = 0.99  # deep dip, many sigma below the noise
  assert preprocess.sigma_clip_mask(flux, sigma=5.0)[250]


def test_clean_light_curve_unit_baseline():
  """A cleaned light curve has a unit median baseline and finite flux."""
  time = np.arange(2000) * 0.02
  flux = 1.05 + 0.001 * np.random.RandomState(0).normal(size=2000)
  cleaned = preprocess.clean_light_curve(
      LightCurve(time, flux, np.full_like(flux, 1e-3)), cadence_days=0.02
  )
  assert abs(np.median(cleaned.flux) - 1.0) < 1e-2
  assert np.all(np.isfinite(cleaned.flux))
