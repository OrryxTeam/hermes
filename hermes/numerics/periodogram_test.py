"""Tests for the transit periodogram in hermes.numerics.periodogram."""

import jax
import jax.numpy as jnp
import numpy as np

from hermes.numerics import periodogram
from hermes.testing import _fixtures


def test_frequency_grid_is_uniform_in_frequency():
  """The period grid has constant spacing in frequency."""
  periods = periodogram.frequency_grid(1.0, 20.0, baseline_days=80.0)
  spacing = jnp.diff(1.0 / periods)
  assert jnp.allclose(spacing, spacing[0], rtol=1e-6)
  assert periods.min() >= 1.0 - 1e-9 and periods.max() <= 20.0 + 1e-6


def test_recovers_injected_period():
  """The search recovers an injected transit period to grid resolution."""
  time, flux = _fixtures.injected_light_curve(5.3, epoch=1.0)
  result = periodogram.transit_periodogram(
      time, flux, period_min=4.0, period_max=7.0
  )
  assert jnp.abs(result.best_period - 5.3) < 0.05
  assert result.best_depth > 0.0


def test_recovers_epoch_and_depth_at_exact_period():
  """At the exact period the epoch, duration and depth are accurate.

  A single-period grid removes folding smear, isolating the lag-to-epoch
  mapping and the fitted depth from grid resolution.
  """
  time, flux = _fixtures.injected_light_curve(5.3, epoch=1.0, noise=0.0)
  result = periodogram.transit_periodogram(
      time, flux, periods=jnp.array([5.3])
  )
  assert jnp.abs((result.best_epoch % 5.3) - 1.0) < 5.3 / 256
  assert jnp.abs(result.best_duration - 0.17) < 0.06  # T14 ~ 0.169 d
  assert jnp.abs(result.best_depth - 0.01) < 0.004  # depth ~ k^2 = 0.01


def test_recovers_period_under_jit():
  """The search is jittable and recovers the period when compiled."""
  time, flux = _fixtures.injected_light_curve(3.21, epoch=0.5, depth_ratio=0.12)
  grid = periodogram.frequency_grid(2.5, 4.0, baseline_days=80.0)
  search = jax.jit(
      lambda t, f: periodogram.transit_periodogram(
          t, f, periods=grid
      ).best_period
  )
  assert jnp.abs(search(time, flux) - 3.21) < 0.05


def test_power_exceeds_noise_only_baseline():
  """An injected transit yields far higher power than a pure-noise curve."""
  time, flux = _fixtures.injected_light_curve(7.0, epoch=2.0)
  with_transit = periodogram.transit_periodogram(
      time, flux, period_min=5.0, period_max=9.0
  )
  rng = np.random.RandomState(1)
  noise = 1.0 + jnp.asarray(rng.normal(0.0, 1e-3, time.shape))
  noise_only = periodogram.transit_periodogram(
      time, noise, period_min=5.0, period_max=9.0
  )
  assert with_transit.best_power > 5.0 * noise_only.best_power
