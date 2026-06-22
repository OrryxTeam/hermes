"""Tests for Keplerian geometry in hermes.numerics.orbit."""

import chex
import jax
import jax.numpy as jnp

from hermes.numerics import orbit, physics


def test_density_round_trip():
  """``a/Rstar`` and stellar density are inverse maps for any period."""
  period = 12.3
  density = 2.5 * physics.SOLAR_MEAN_DENSITY_SI
  a_over_rstar = orbit.a_over_rstar_from_density(period, density)
  recovered = orbit.stellar_density_from_transit(period, a_over_rstar)
  assert jnp.allclose(recovered, density, rtol=1e-10)


def test_earth_sun_scaled_semimajor_axis():
  """A solar-density star with a one-year orbit gives ``a/Rstar ~ 215``."""
  a_over_rstar = orbit.a_over_rstar_from_density(
      physics.DAYS_PER_YEAR, physics.SOLAR_MEAN_DENSITY_SI
  )
  assert jnp.abs(a_over_rstar - physics.SOLAR_RADII_PER_AU) < 1.0


def test_density_consistency_zero_when_self_consistent():
  """The consistency log-ratio vanishes for a self-consistent transit."""
  period, density = 8.0, 1.2 * physics.SOLAR_MEAN_DENSITY_SI
  a_over_rstar = orbit.a_over_rstar_from_density(period, density)
  logratio = orbit.density_consistency_logratio(period, a_over_rstar, density)
  assert jnp.allclose(logratio, 0.0, atol=1e-10)


def test_projected_separation_center_equals_impact_parameter():
  """At transit centre the separation equals the impact parameter."""
  separation, in_front = orbit.projected_separation(
      time_days=jnp.array(5.0),
      period_days=10.0,
      epoch_days=5.0,
      a_over_rstar=20.0,
      impact_parameter=0.3,
  )
  assert jnp.allclose(separation, 0.3, atol=1e-9)
  assert in_front > 0.0


def test_projected_separation_masks_secondary_eclipse():
  """Half a period after transit the planet is behind the star."""
  _, in_front = orbit.projected_separation(
      time_days=jnp.array(10.0),
      period_days=10.0,
      epoch_days=5.0,
      a_over_rstar=20.0,
      impact_parameter=0.0,
  )
  assert in_front < 0.0


def test_earth_sun_transit_duration():
  """The central Earth-Sun transit lasts ~13 hours."""
  duration_days = orbit.transit_duration_days(
      period_days=physics.DAYS_PER_YEAR,
      a_over_rstar=physics.SOLAR_RADII_PER_AU,
      radius_ratio=1.0 / physics.SOLAR_RADII_PER_EARTH_RADIUS,
      impact_parameter=0.0,
  )
  assert jnp.abs(duration_days * 24.0 - 13.0) < 1.0


def test_grazing_duration_is_finite_and_zero():
  """A non-transiting geometry yields a finite, zero duration."""
  duration = orbit.transit_duration_days(
      period_days=10.0,
      a_over_rstar=20.0,
      radius_ratio=0.1,
      impact_parameter=2.0,
  )
  assert jnp.isfinite(duration) and jnp.allclose(duration, 0.0)


def test_duration_gradient_is_finite():
  """The duration is differentiable in the impact parameter."""
  grad = jax.grad(
      lambda b: orbit.transit_duration_days(10.0, 20.0, 0.1, b)
  )(0.5)
  chex.assert_tree_all_finite(grad)
