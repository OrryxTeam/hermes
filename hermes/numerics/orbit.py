"""Keplerian orbit geometry and stellar-density relations.

The projected separation assumes a circular orbit (the standard transit-search
assumption); eccentricity enters only the duration. Angles are radians, periods
days, densities SI (kg/m^3).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from hermes.numerics import physics


def a_over_rstar_from_density(
    period_days: Array, stellar_density_si: Array
) -> Array:
  """Scaled semi-major axis ``a / Rstar`` from period and stellar density.

  For a circular orbit Kepler's third law gives
  ``rho_star = (3 pi / (G P^2)) (a / Rstar)^3``, hence
  ``a / Rstar = (G rho_star P^2 / (3 pi))^(1/3)``.

  Args:
    period_days: Orbital period in days.
    stellar_density_si: Stellar mean density in kg m^-3.

  Returns:
    Dimensionless scaled semi-major axis ``a / Rstar``.
  """
  period_s = period_days * physics.SECONDS_PER_DAY
  return (
      physics.GRAVITATIONAL_CONSTANT_SI
      * stellar_density_si
      * period_s**2
      / (3.0 * jnp.pi)
  ) ** (1.0 / 3.0)


def stellar_density_from_transit(
    period_days: Array, a_over_rstar: Array
) -> Array:
  """Stellar mean density implied by an observed transit (circular orbit).

  Inverts `a_over_rstar_from_density`:
  ``rho_circ = (3 pi / (G P^2)) (a / Rstar)^3``.

  Args:
    period_days: Orbital period in days.
    a_over_rstar: Scaled semi-major axis ``a / Rstar``.

  Returns:
    Inferred stellar mean density in kg m^-3.
  """
  period_s = period_days * physics.SECONDS_PER_DAY
  return (
      3.0
      * jnp.pi
      * a_over_rstar**3
      / (physics.GRAVITATIONAL_CONSTANT_SI * period_s**2)
  )


def density_consistency_logratio(
    period_days: Array,
    a_over_rstar: Array,
    catalog_density_si: Array,
) -> Array:
  """Log-ratio between the transit-inferred and catalogued stellar density.

  ``log(rho_circ / rho_star)``. It is zero for a circular orbit around a star
  whose catalogued density is correct, positive/negative for eccentric orbits
  depending on the argument of periastron, and large in magnitude for blended
  eclipsing binaries. It is used both as an input feature and as a
  physics-consistency penalty during training.

  Args:
    period_days: Orbital period in days.
    a_over_rstar: Scaled semi-major axis ``a / Rstar`` from the transit fit.
    catalog_density_si: Catalogued stellar mean density in kg m^-3.

  Returns:
    Natural logarithm of ``rho_circ / rho_star``.
  """
  rho_circ = stellar_density_from_transit(period_days, a_over_rstar)
  return jnp.log(rho_circ) - jnp.log(catalog_density_si)


def cos_inclination(a_over_rstar: Array, impact_parameter: Array) -> Array:
  """Cosine of the orbital inclination for a circular orbit.

  ``b = (a / Rstar) cos i``, so ``cos i = b / (a / Rstar)``.

  Args:
    a_over_rstar: Scaled semi-major axis ``a / Rstar``.
    impact_parameter: Transit impact parameter ``b``.

  Returns:
    ``cos i``.
  """
  return impact_parameter / a_over_rstar


def projected_separation(
    time_days: Array,
    period_days: Array,
    epoch_days: Array,
    a_over_rstar: Array,
    impact_parameter: Array,
) -> tuple[Array, Array]:
  """Sky-projected star-planet separation for a circular orbit.

  With orbital phase angle ``theta = 2 pi (t - t0) / P`` measured from transit
  centre, the separation in units of the stellar radius is
  ``z = sqrt((a/Rstar)^2 sin^2(theta) + b^2 cos^2(theta))`` and the planet is in
  front of the star (primary transit, rather than behind it for the secondary
  eclipse) when ``cos(theta) > 0``.

  Args:
    time_days: Observation times in days.
    period_days: Orbital period in days.
    epoch_days: Reference transit centre time ``t0`` in days.
    a_over_rstar: Scaled semi-major axis ``a / Rstar``.
    impact_parameter: Transit impact parameter ``b``.

  Returns:
    A tuple ``(z, in_front)`` where ``z`` is the projected separation in stellar
    radii and ``in_front`` is ``cos(theta)`` (positive during primary transit).
  """
  theta = 2.0 * jnp.pi * (time_days - epoch_days) / period_days
  sin_theta = jnp.sin(theta)
  cos_theta = jnp.cos(theta)
  separation = jnp.sqrt(
      (a_over_rstar * sin_theta) ** 2 + (impact_parameter * cos_theta) ** 2
  )
  return separation, cos_theta


def transit_duration_days(
    period_days: Array,
    a_over_rstar: Array,
    radius_ratio: Array,
    impact_parameter: Array,
    eccentricity: Array | float = 0.0,
    omega_rad: Array | float = 0.0,
) -> Array:
  """Total (first-to-fourth contact) transit duration ``T14``.

  Following Winn (2010), eq. 14,
  ``T14 = (P / pi) arcsin[ sqrt((1 + k)^2 - b^2) / ((a/Rstar) sin i) ]
          * sqrt(1 - e^2) / (1 + e sin omega)``,
  with ``k`` the planet-to-star radius ratio and ``b`` the impact parameter.
  Grazing and non-transiting geometries (``(1 + k)^2 <= b^2``) yield a duration
  of zero rather than a NaN.

  Args:
    period_days: Orbital period in days.
    a_over_rstar: Scaled semi-major axis ``a / Rstar``.
    radius_ratio: Planet-to-star radius ratio ``k = Rp / Rstar``.
    impact_parameter: Transit impact parameter ``b``.
    eccentricity: Orbital eccentricity.
    omega_rad: Argument of periastron in radians.

  Returns:
    Total transit duration ``T14`` in days.
  """
  cos_i = impact_parameter / a_over_rstar
  sin_i = jnp.sqrt(jnp.maximum(1.0 - cos_i**2, 0.0))
  chord = jnp.sqrt(
      jnp.maximum((1.0 + radius_ratio) ** 2 - impact_parameter**2, 0.0)
  )
  # Clamp the arcsin argument to its valid range so close-in or grazing
  # geometries stay finite and differentiable.
  arg = jnp.clip(chord / (a_over_rstar * sin_i), -1.0, 1.0)
  eccentricity_factor = jnp.sqrt(jnp.maximum(1.0 - eccentricity**2, 0.0)) / (
      1.0 + eccentricity * jnp.sin(omega_rad)
  )
  return (period_days / jnp.pi) * jnp.arcsin(arg) * eccentricity_factor
