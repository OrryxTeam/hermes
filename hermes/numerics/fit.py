"""Gradient refinement of a transit candidate through the differentiable model.

Holds the period fixed and optimises epoch, depth, scaled semi-major axis and
impact parameter to minimise the chi-square against the data.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from hermes.numerics import orbit, transit


class TransitFit(NamedTuple):
  """Refined transit parameters and fit quality.

  Attributes:
    period_days: Orbital period in days (held fixed during refinement).
    epoch_days: Refined transit epoch in days.
    depth: Refined fractional transit depth.
    duration_days: Transit duration implied by the refined geometry.
    a_over_rstar: Refined scaled semi-major axis.
    impact_parameter: Refined impact parameter.
    chi_square: Final weighted chi-square of the fit.
  """

  period_days: Array
  epoch_days: Array
  depth: Array
  duration_days: Array
  a_over_rstar: Array
  impact_parameter: Array
  chi_square: Array


_LIMB_DARKENING = (0.4, 0.3)


def _model_flux(params: Array, time_days: Array, period_days: Array) -> Array:
  """Transit model for the unconstrained parameter vector ``params``."""
  epoch, log_depth, log_a, impact_raw = params
  depth = jnp.exp(log_depth)
  radius_ratio = jnp.sqrt(jnp.clip(depth, 1e-8, 0.5))
  a_over_rstar = jnp.exp(log_a)
  impact_parameter = 1.2 * jax.nn.sigmoid(impact_raw)
  return transit.transit_light_curve(
      time_days,
      period_days=period_days,
      epoch_days=epoch,
      radius_ratio=radius_ratio,
      a_over_rstar=a_over_rstar,
      impact_parameter=impact_parameter,
      limb_darkening=_LIMB_DARKENING,
      num_radii=64,
  )


def refine_transit(
    time_days: Array,
    flux: Array,
    flux_err: Array,
    period_days: float,
    init_epoch: float,
    init_depth: float,
    init_duration: float,
    *,
    steps: int = 300,
    learning_rate: float = 0.05,
) -> TransitFit:
  """Refines a transit candidate by gradient descent through the model.

  Args:
    time_days: Observation times in days.
    flux: Normalised flux.
    flux_err: Per-point flux uncertainties.
    period_days: Orbital period in days (held fixed).
    init_epoch: Initial transit epoch in days.
    init_depth: Initial fractional transit depth.
    init_duration: Initial transit duration in days.
    steps: Number of optimisation steps.
    learning_rate: Adam learning rate.

  Returns:
    A `TransitFit` with the refined parameters and final chi-square.
  """
  time_days = jnp.asarray(time_days)
  flux = jnp.asarray(flux)
  weights = 1.0 / jnp.asarray(flux_err) ** 2
  period = jnp.asarray(period_days, dtype=flux.dtype)

  init_depth = max(float(init_depth), 1e-6)
  radius_ratio = jnp.sqrt(init_depth)
  init_a = (
      period_days / jnp.pi * (1.0 + radius_ratio) / max(init_duration, 1e-3)
  )
  params = jnp.array(
      [init_epoch, jnp.log(init_depth), jnp.log(init_a), 0.0], dtype=flux.dtype
  )

  optimizer = optax.adam(learning_rate)

  def chi_square(params: Array) -> Array:
    residual = flux - _model_flux(params, time_days, period)
    return jnp.sum(weights * residual**2)

  def step(carry, _):
    params, opt_state = carry
    loss, grads = jax.value_and_grad(chi_square)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    return (optax.apply_updates(params, updates), opt_state), loss

  (params, _), _ = jax.lax.scan(
      step, (params, optimizer.init(params)), None, length=steps
  )

  epoch, log_depth, log_a, impact_raw = params
  depth = jnp.exp(log_depth)
  a_over_rstar = jnp.exp(log_a)
  impact_parameter = 1.2 * jax.nn.sigmoid(impact_raw)
  duration = orbit.transit_duration_days(
      period,
      a_over_rstar,
      jnp.sqrt(jnp.clip(depth, 1e-8, 0.5)),
      impact_parameter,
  )
  return TransitFit(
      period_days=period,
      epoch_days=jnp.mod(epoch, period),
      depth=depth,
      duration_days=duration,
      a_over_rstar=a_over_rstar,
      impact_parameter=impact_parameter,
      chi_square=chi_square(params),
  )
