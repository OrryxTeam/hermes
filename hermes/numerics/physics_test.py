"""Tests for intrinsic astrophysical conversions in hermes.numerics.physics."""

import jax.numpy as jnp

from hermes.numerics import physics


def test_planet_radius_recovers_earth():
  """An Earth-over-Sun depth should map back to one Earth radius."""
  depth = (1.0 / physics.SOLAR_RADII_PER_EARTH_RADIUS) ** 2
  radius = physics.planet_radius_earth(depth, stellar_radius_solar=1.0)
  assert jnp.allclose(radius, 1.0, rtol=1e-6)


def test_planet_radius_clips_negative_depth():
  """Negative depths are clipped, never producing NaNs."""
  radius = physics.planet_radius_earth(-1e-3)
  assert jnp.isfinite(radius) and radius == 0.0


def test_semi_major_axis_recovers_one_au():
  """A one-year orbit around a solar-mass star is one astronomical unit."""
  axis = physics.semi_major_axis_au(
      physics.DAYS_PER_YEAR, stellar_mass_solar=1.0
  )
  assert jnp.allclose(axis, 1.0, rtol=1e-6)


def test_equilibrium_temperature_of_earth():
  """Earth's equilibrium temperature is ~255 K for a Bond albedo of 0.3."""
  temperature = physics.equilibrium_temperature(
      semi_major_axis_au=1.0, bond_albedo=0.3
  )
  assert jnp.abs(temperature - 255.0) < 5.0


def test_solar_mean_density_value():
  """The tabulated mean solar density is ~1408 kg/m^3."""
  assert jnp.abs(physics.SOLAR_MEAN_DENSITY_SI - 1408.0) < 5.0
