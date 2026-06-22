"""Differentiable limb-darkened transit light-curve models (Mandel & Agol 2002).

Distances are in stellar radii, flux is normalised to one out of transit, and
every function is pure and safe under jit, vmap and grad.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
from jax import Array

from hermes.numerics import orbit


def uniform_disk_obscuration(separation: Array, radius_ratio: Array) -> Array:
  """Occulted flux fraction for an opaque planet on a uniform-brightness star.

  This is the ``lambda_e`` function of Mandel & Agol (2002): the fraction of the
  stellar flux blocked when a disk of radius ``radius_ratio`` (in stellar radii)
  sits at sky-projected centre-to-centre separation ``separation`` (also in
  stellar radii). It equals the overlap area of the two circles divided by the
  stellar disk area.

  Args:
    separation: Sky-projected separation ``z`` in units of the stellar radius.
    radius_ratio: Planet-to-star radius ratio ``k = Rp / Rstar``.

  Returns:
    Occulted flux fraction in ``[0, 1]``.
  """
  z = jnp.abs(separation)
  p = radius_ratio

  # Partial-overlap formula. Both the ``arccos`` branch points (argument equal
  # to +/-1) and the ``sqrt`` zero occur only at the region boundaries, where
  # their gradients are singular. To keep ``jax.grad`` finite we feed those
  # singular operations arguments that stay strictly interior outside the
  # partial-overlap region, then discard the result with ``jnp.where``.
  is_partial = (z < 1.0 + p) & (z > jnp.abs(1.0 - p))
  denom_planet = jnp.where(is_partial, 2.0 * p * z, 1.0)
  denom_star = jnp.where(is_partial, 2.0 * z, 1.0)
  arg_planet = jnp.where(
      is_partial, jnp.clip((p * p + z * z - 1.0) / denom_planet, -1.0, 1.0), 0.0
  )
  arg_star = jnp.where(
      is_partial, jnp.clip((1.0 - p * p + z * z) / denom_star, -1.0, 1.0), 0.0
  )
  triangle_squared = jnp.where(
      is_partial,
      jnp.maximum(4.0 * z * z - (1.0 + z * z - p * p) ** 2, 0.0),
      1.0,
  )
  lens_triangle = 0.5 * jnp.sqrt(triangle_squared)
  partial = (
      p * p * jnp.arccos(arg_planet) + jnp.arccos(arg_star) - lens_triangle
  ) / jnp.pi

  # Region selection: no overlap, planet fully inside the disk (or, if the
  # planet is larger than the star, fully covering it), or partial overlap.
  fully_overlapping = jnp.where(p > 1.0, jnp.ones_like(z), p * p)
  obscuration = jnp.where(
      z >= 1.0 + p,
      jnp.zeros_like(z),
      jnp.where(z <= jnp.abs(1.0 - p), fully_overlapping, partial),
  )
  return obscuration


def uniform_disk_flux(separation: Array, radius_ratio: Array) -> Array:
  """Normalised flux for a uniform-brightness star during transit.

  Args:
    separation: Sky-projected separation ``z`` in units of the stellar radius.
    radius_ratio: Planet-to-star radius ratio ``k = Rp / Rstar``.

  Returns:
    Flux normalised to one out of transit.
  """
  return 1.0 - uniform_disk_obscuration(separation, radius_ratio)


@functools.lru_cache(maxsize=None)
def _legendre_nodes_unit(num_radii: int) -> tuple[np.ndarray, np.ndarray]:
  """Gauss-Legendre nodes and weights on the unit interval ``[0, 1]``.

  Results are cached because ``num_radii`` is a static quadrature resolution.
  They are returned as host (NumPy) constants so that, when the transit model is
  used inside a ``jax`` transformation, they fold in as constants rather than
  leaking as cached tracers across traces.

  Args:
    num_radii: Number of quadrature nodes.

  Returns:
    A tuple ``(radii, weights)`` of NumPy arrays on ``[0, 1]``.
  """
  nodes, weights = np.polynomial.legendre.leggauss(num_radii)
  radii = 0.5 * (nodes + 1.0)
  return radii, 0.5 * weights


def quadratic_limb_darkened_flux(
    separation: Array,
    radius_ratio: Array,
    limb_u1: Array | float = 0.0,
    limb_u2: Array | float = 0.0,
    *,
    num_radii: int = 256,
) -> Array:
  """Normalised transit flux for a quadratically limb-darkened star.

  The surface brightness follows the quadratic limb-darkening law in
  ``mu = sqrt(1 - r^2)``, the cosine of the emergent angle at normalised radius
  ``r``. The occulted flux is integrated over concentric annuli, with the
  radial integral split at the transit contact radii into a fully-covered piece
  ``[0, p - z]`` (each annulus entirely inside the planet) and a partially
  covered piece ``[|z - p|, min(z + p, 1)]``. Each piece is integrated by
  Gauss-Legendre quadrature whose nodes are strictly interior, so the integrand
  is smooth there and the ``arccos`` arguments never reach their singular branch
  points -- giving both spectral accuracy and finite gradients. With
  ``u1 = u2 = 0`` the result equals `uniform_disk_flux`.

  Args:
    separation: Sky-projected separation ``z`` in stellar radii. May be an array
      of any shape.
    radius_ratio: Planet-to-star radius ratio ``k`` (scalar).
    limb_u1: First quadratic limb-darkening coefficient.
    limb_u2: Second quadratic limb-darkening coefficient.
    num_radii: Number of radial quadrature nodes per piece (static).

  Returns:
    Flux normalised to one out of transit, with the shape of ``separation``.
  """
  z = jnp.abs(separation)
  p = radius_ratio
  nodes, weights = _legendre_nodes_unit(num_radii)

  def disk_integral(lower: Array, upper: Array, annulus_fraction):
    """Integrates ``I(r) * fraction(r) * 2 r`` over ``[lower, upper]``."""
    upper = jnp.maximum(upper, lower)
    width = upper - lower
    radii = lower[..., jnp.newaxis] + width[..., jnp.newaxis] * nodes
    # Out-of-transit separations push an empty (zero-width) interval's nodes to
    # ``radii > 1``; replace the foliation depth there with a constant so the
    # ``sqrt`` gradient stays finite (the contribution is already zeroed by the
    # zero width, but ``0 * inf`` would otherwise poison the gradient).
    foliation = jnp.where(radii < 1.0, 1.0 - radii**2, 1.0)
    mu = jnp.sqrt(foliation)
    intensity = 1.0 - limb_u1 * (1.0 - mu) - limb_u2 * (1.0 - mu) ** 2
    integrand = intensity * annulus_fraction(radii) * 2.0 * radii
    return width * jnp.sum(weights * integrand, axis=-1)

  zeros = jnp.zeros_like(z)
  ones = jnp.ones_like(z)
  total_flux = disk_integral(zeros, ones, lambda r: 1.0)

  # Annuli entirely inside the planet disk contribute their full intensity.
  covered_flux = disk_integral(
      zeros, jnp.clip(p - z, 0.0, 1.0), lambda r: 1.0
  )

  # Annuli straddling the planet limb contribute the arc fraction inside it.
  lower = jnp.abs(z - p)
  upper = jnp.minimum(z + p, 1.0)

  def arc_fraction(radii: Array) -> Array:
    z_axis = z[..., jnp.newaxis]
    # The partial region is empty (zero integration width) when ``z`` vanishes
    # or for the degenerate intervals produced out of transit, which can drive
    # ``radii`` above one and the arccos argument outside ``[-1, 1]``. Guard the
    # denominator and feed arccos only interior arguments, assigning the
    # saturated annuli their exact fractions (1 inside the planet, 0 outside),
    # so both the forward pass and its gradient stay finite.
    denom = 2.0 * radii * z_axis
    safe_denom = jnp.where(denom > 0.0, denom, 1.0)
    argument = (radii**2 + z_axis**2 - p * p) / safe_denom
    fully_inside = argument <= -1.0
    fully_outside = argument >= 1.0
    interior = jnp.logical_not(fully_inside | fully_outside)
    arc = jnp.arccos(jnp.where(interior, argument, 0.0)) / jnp.pi
    return jnp.where(fully_inside, 1.0, jnp.where(fully_outside, 0.0, arc))

  partial_flux = disk_integral(lower, upper, arc_fraction)

  occulted = covered_flux + partial_flux
  return 1.0 - occulted / total_flux


def transit_light_curve(
    time_days: Array,
    *,
    period_days: Array,
    epoch_days: Array,
    radius_ratio: Array,
    a_over_rstar: Array,
    impact_parameter: Array,
    limb_darkening: tuple[Array | float, Array | float] = (0.0, 0.0),
    num_radii: int = 256,
) -> Array:
  """Limb-darkened transit light curve for a circular orbit.

  The Keplerian projected separation ``z(t)`` is evaluated by
  `hermes.numerics.orbit.projected_separation`, and the secondary eclipse
  (planet behind the star) is masked so that only the primary transit produces a
  flux decrement.

  Args:
    time_days: Observation times in days.
    period_days: Orbital period in days.
    epoch_days: Reference transit centre time ``t0`` in days.
    radius_ratio: Planet-to-star radius ratio ``k``.
    a_over_rstar: Scaled semi-major axis ``a / Rstar``.
    impact_parameter: Transit impact parameter ``b``.
    limb_darkening: Quadratic limb-darkening coefficients ``(u1, u2)``.
    num_radii: Number of radial quadrature nodes (static).

  Returns:
    Flux normalised to one out of transit, with the shape of ``time_days``.
  """
  separation, in_front = orbit.projected_separation(
      time_days, period_days, epoch_days, a_over_rstar, impact_parameter
  )
  limb_u1, limb_u2 = limb_darkening
  flux = quadratic_limb_darkened_flux(
      separation, radius_ratio, limb_u1, limb_u2, num_radii=num_radii
  )
  return jnp.where(in_front > 0.0, flux, 1.0)
