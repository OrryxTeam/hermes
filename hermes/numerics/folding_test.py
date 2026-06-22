"""Tests for phase folding and view binning in hermes.numerics.folding."""

import jax.numpy as jnp

from hermes.numerics import folding, transit


def test_fold_phase_centres_transit_at_zero():
  """Times at the epoch fold to phase zero; half a period folds to +/-0.5."""
  time = jnp.array([3.0, 3.0 + 2.5, 3.0 + 0.6])
  phase = folding.fold_phase(time, period_days=5.0, epoch_days=3.0)
  assert jnp.allclose(phase[0], 0.0)
  assert jnp.abs(phase[1]) == 0.5
  assert jnp.allclose(phase[2], 0.6 / 5.0)


def test_bin_mean_averages_and_fills_empty():
  """Bin means average their members and empty bins take the fill value."""
  coordinate = jnp.array([0.05, 0.06, 0.95])
  values = jnp.array([2.0, 4.0, 8.0])
  binned, counts = folding.bin_mean(
      coordinate, values, num_bins=10, lower=0.0, upper=1.0, empty_fill=-1.0
  )
  assert jnp.allclose(binned[0], 3.0)  # mean of 2 and 4
  assert jnp.allclose(binned[9], 8.0)
  assert jnp.allclose(binned[5], -1.0)  # empty
  assert counts[0] == 2.0


def test_global_view_shows_transit_dip():
  """A folded transit produces a dip at the centre of the global view."""
  time = jnp.linspace(0.0, 60.0, 6000)
  flux = transit.transit_light_curve(
      time,
      period_days=4.0,
      epoch_days=1.0,
      radius_ratio=0.1,
      a_over_rstar=12.0,
      impact_parameter=0.0,
  )
  view = folding.global_view(
      time, flux, period_days=4.0, epoch_days=1.0, num_bins=201
  )
  edges = jnp.concatenate([view[:20], view[-20:]])
  assert view[100] < jnp.median(edges) - 1e-3


def test_normalize_view_bottoms_at_minus_one():
  """Normalisation bounds the view and fixes the transit minimum at -1."""
  view = jnp.ones(51).at[25].set(0.98)
  normalised = folding.normalize_view(view)
  assert jnp.isclose(jnp.min(normalised), -1.0)
  assert jnp.max(jnp.abs(normalised)) <= 1.0 + 1e-6
