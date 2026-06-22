"""Physical constants and intrinsic astrophysical conversions (SI units)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

# -- Fundamental constants (SI) ----------------------------------------------

GRAVITATIONAL_CONSTANT_SI: float = 6.674_30e-11  # m^3 kg^-1 s^-2
STEFAN_BOLTZMANN_SI: float = 5.670_374_419e-8  # W m^-2 K^-4

# -- Solar / terrestrial reference values (SI) -------------------------------

SOLAR_MASS_KG: float = 1.988_409_87e30
SOLAR_RADIUS_M: float = 6.957e8
EARTH_RADIUS_M: float = 6.371_0e6
JUPITER_RADIUS_M: float = 7.149_2e7
SOLAR_EFFECTIVE_TEMPERATURE_K: float = 5772.0

# -- Unit conversions --------------------------------------------------------

ASTRONOMICAL_UNIT_M: float = 1.495_978_707e11
SECONDS_PER_DAY: float = 86_400.0
DAYS_PER_YEAR: float = 365.25

#: Solar radii per Earth radius (``SOLAR_RADIUS_M / EARTH_RADIUS_M``).
SOLAR_RADII_PER_EARTH_RADIUS: float = SOLAR_RADIUS_M / EARTH_RADIUS_M
#: Solar radii per astronomical unit (``ASTRONOMICAL_UNIT_M / SOLAR_RADIUS_M``).
SOLAR_RADII_PER_AU: float = ASTRONOMICAL_UNIT_M / SOLAR_RADIUS_M

#: Mean solar density ``3 M_sun / (4 pi R_sun^3)`` in kg m^-3 (~1408).
SOLAR_MEAN_DENSITY_SI: float = (
    3.0 * SOLAR_MASS_KG / (4.0 * jnp.pi * SOLAR_RADIUS_M**3)
)


def planet_radius_earth(
    transit_depth: Array, stellar_radius_solar: Array | float = 1.0
) -> Array:
  """Converts a transit depth into a planet radius.

  The fractional transit depth of an opaque planet is ``delta = (Rp / Rstar)^2``
  (the depth attributable to limb darkening and grazing geometry is handled by
  the forward model, not here), so ``Rp = Rstar * sqrt(delta)``.

  Args:
    transit_depth: Fractional flux decrement during transit. Negative values are
      clipped to zero before taking the square root.
    stellar_radius_solar: Stellar radius in solar radii.

  Returns:
    Planet radius in Earth radii.
  """
  radius_ratio = jnp.sqrt(jnp.maximum(transit_depth, 0.0))
  return radius_ratio * stellar_radius_solar * SOLAR_RADII_PER_EARTH_RADIUS


def equilibrium_temperature(
    semi_major_axis_au: Array,
    stellar_radius_solar: Array | float = 1.0,
    stellar_temperature_k: Array | float = SOLAR_EFFECTIVE_TEMPERATURE_K,
    bond_albedo: Array | float = 0.3,
    heat_redistribution: Array | float = 1.0,
) -> Array:
  """Computes the planetary equilibrium temperature from energy balance.

  ``T_eq = T_star * sqrt(f * R_star / (2 a)) * (1 - A)^(1/4)``. ``f`` is the
  heat-redistribution factor (``1`` for uniform redistribution, ``2`` for
  instant re-radiation of the dayside) and ``A`` is the Bond albedo.

  Args:
    semi_major_axis_au: Orbital semi-major axis in astronomical units.
    stellar_radius_solar: Stellar radius in solar radii.
    stellar_temperature_k: Stellar effective temperature in Kelvin.
    bond_albedo: Bond albedo in ``[0, 1)``.
    heat_redistribution: Redistribution factor ``f`` (see above).

  Returns:
    Equilibrium temperature in Kelvin.
  """
  radius_over_distance = stellar_radius_solar / (
      semi_major_axis_au * SOLAR_RADII_PER_AU
  )
  return (
      stellar_temperature_k
      * jnp.sqrt(heat_redistribution * radius_over_distance / 2.0)
      * (1.0 - bond_albedo) ** 0.25
  )


def semi_major_axis_au(
    period_days: Array, stellar_mass_solar: Array | float = 1.0
) -> Array:
  """Computes the orbital semi-major axis from Kepler's third law.

  ``a^3 / P^2 = G M_star / (4 pi^2)``. In solar units this reduces to
  ``a [AU] = (M_star [Msun] * P [yr]^2)^(1/3)``.

  Args:
    period_days: Orbital period in days.
    stellar_mass_solar: Stellar mass in solar masses.

  Returns:
    Semi-major axis in astronomical units.
  """
  period_years = period_days / DAYS_PER_YEAR
  return (stellar_mass_solar * period_years**2) ** (1.0 / 3.0)
