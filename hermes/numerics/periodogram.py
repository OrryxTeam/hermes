"""Transit-templated matched-filter periodogram (Transit Least Squares).

For each trial period the folded flux is correlated with a limb-darkened transit
template via the FFT; the statistic is the weighted-least-squares depth
significance, maximised over phase lag and duration.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hermes.numerics import folding, transit


class Periodogram(NamedTuple):
  """Result of a transit periodogram search.

  Attributes:
    periods: Trial periods in days, shape ``(num_periods,)``.
    power: Detection statistic at each trial period, shape ``(num_periods,)``.
    best_period: Period in days maximising the power.
    best_epoch: Transit centre time in days at the best period.
    best_duration: Transit duration in days at the best period.
    best_depth: Fitted fractional transit depth at the best period.
    best_power: Maximum detection statistic.
  """

  periods: Array
  power: Array
  best_period: Array
  best_epoch: Array
  best_duration: Array
  best_depth: Array
  best_power: Array


def frequency_grid(
    period_min: float,
    period_max: float,
    baseline_days: float,
    oversample: float = 3.0,
    min_duration_fraction: float = 0.05,
) -> Array:
  """Builds a transit-duration-aware period grid uniform in frequency.

  Unlike the generic ``1 / baseline`` spacing of a Lomb-Scargle periodogram, a
  transit search must resolve periods finely enough that, over the full
  baseline, a period error does not smear the transit across more than a
  fraction of its own duration (otherwise the true period is diluted and a
  harmonic can win). Following Ofir (2014) the frequency step is therefore
  ``df = min_duration_fraction / (oversample * baseline)`` -- finer than the
  Lomb-Scargle grid by roughly ``1 / duration_fraction``. This makes a long
  baseline much more expensive to search, which is inherent to transit
  detection.

  Args:
    period_min: Shortest period to search, in days.
    period_max: Longest period to search, in days.
    baseline_days: Total time span of the observations, in days.
    oversample: Frequency oversampling factor.
    min_duration_fraction: Smallest transit duration as a fraction of the period
      (the shortest expected duty cycle), which sets the required resolution.

  Returns:
    Trial periods in days, in decreasing-frequency (increasing-period) order.
  """
  freq_max = 1.0 / period_min
  freq_min = 1.0 / period_max
  df = min_duration_fraction / (oversample * baseline_days)
  num = int(jnp.ceil((freq_max - freq_min) / df)) + 1
  frequencies = freq_max - df * jnp.arange(num)
  frequencies = jnp.clip(frequencies, freq_min, freq_max)
  return 1.0 / frequencies


@functools.lru_cache(maxsize=None)
def _transit_templates(
    num_bins: int,
    duration_fractions: tuple[float, ...],
    limb_u1: float,
    limb_u2: float,
    num_radii: int,
) -> tuple[Array, Array]:
  """Builds unit-depth transit templates, one per fractional duration.

  Each template is a limb-darkened, central (``b = 0``) transit centred on phase
  zero whose total duration equals ``q`` of the period; the implied scaled
  semi-major axis is ``a / Rstar = 1 / (pi q)``. The templates do not depend on
  the trial period, so they are cached and reused across the whole search.

  Args:
    num_bins: Number of phase bins.
    duration_fractions: Trial durations as fractions of the period.
    limb_u1: First quadratic limb-darkening coefficient.
    limb_u2: Second quadratic limb-darkening coefficient.
    num_radii: Radial quadrature resolution for the transit model.

  Returns:
    A tuple ``(templates, templates_squared)`` each of shape
    ``(num_durations, num_bins)``, normalised to unit peak depth.
  """
  phase = (jnp.arange(num_bins) + 0.5) / num_bins - 0.5

  def one_template(q: float) -> Array:
    a_over_rstar = 1.0 / (jnp.pi * q)
    flux = transit.transit_light_curve(
        phase,  # time in units of the period (t0 = 0, period = 1)
        period_days=1.0,
        epoch_days=0.0,
        radius_ratio=0.1,
        a_over_rstar=a_over_rstar,
        impact_parameter=0.0,
        limb_darkening=(limb_u1, limb_u2),
        num_radii=num_radii,
    )
    depth = 1.0 - flux
    peak = jnp.max(depth)
    return depth / jnp.where(peak > 0.0, peak, 1.0)

  templates = jnp.stack([one_template(q) for q in duration_fractions])
  return templates, templates**2


def _period_power(
    period: Array,
    time_days: Array,
    flux: Array,
    weights: Array,
    fft_template: Array,
    fft_template_squared: Array,
    num_bins: int,
) -> tuple[Array, Array, Array, Array]:
  """Maximum matched-filter power over duration and phase at one period.

  Args:
    period: Trial period in days.
    time_days: Observation times in days.
    flux: Mean-subtracted flux (zero out of transit, negative in transit).
    weights: Per-point inverse-variance weights.
    fft_template: FFT of the unit-depth templates, shape ``(num_q, num_bins)``.
    fft_template_squared: FFT of the squared templates, same shape.
    num_bins: Number of phase bins.

  Returns:
    A tuple ``(power, duration_index, lag, depth)`` at the best (duration, lag).
  """
  phase = folding.fold_phase(time_days, period, 0.0)
  index = jnp.floor((phase + 0.5) * num_bins).astype(jnp.int32)
  index = jnp.clip(index, 0, num_bins - 1)

  weight_bins = jnp.zeros(num_bins, flux.dtype).at[index].add(weights)
  flux_bins = jnp.zeros(num_bins, flux.dtype).at[index].add(weights * flux)

  total_weight = jnp.sum(weight_bins)
  total_flux = jnp.sum(flux_bins)

  # Circular cross-correlations of the binned data with every template.
  fft_weight = jnp.fft.rfft(weight_bins)
  fft_flux = jnp.fft.rfft(flux_bins)
  s_ws = jnp.fft.irfft(fft_weight[None, :] * jnp.conj(fft_template), num_bins)
  s_wss = jnp.fft.irfft(
      fft_weight[None, :] * jnp.conj(fft_template_squared), num_bins
  )
  s_fs = jnp.fft.irfft(fft_flux[None, :] * jnp.conj(fft_template), num_bins)

  variance = s_wss - s_ws**2 / total_weight
  numerator = s_fs - total_flux * s_ws / total_weight
  safe_variance = jnp.where(variance > 0.0, variance, jnp.inf)
  power = numerator**2 / safe_variance
  depth = -numerator / safe_variance  # transit dims the flux, so depth > 0

  flat_index = jnp.argmax(power)
  duration_index, lag = jnp.unravel_index(flat_index, power.shape)
  best_power = power[duration_index, lag]
  best_depth = depth[duration_index, lag]
  return best_power, duration_index, lag, best_depth


def transit_periodogram(
    time_days: Array,
    flux: Array,
    flux_err: Array | None = None,
    *,
    period_min: float = 0.5,
    period_max: float = 100.0,
    oversample: float = 3.0,
    periods: Array | None = None,
    min_duration_fraction: float = 0.05,
    num_bins: int = 256,
    duration_fractions: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.12),
    limb_darkening: tuple[float, float] = (0.4, 0.3),
    num_radii: int = 128,
    period_chunk_size: int = 256,
) -> Periodogram:
  """Searches a light curve for periodic transits with a matched filter.

  Args:
    time_days: Observation times in days.
    flux: Normalised flux (one out of transit).
    flux_err: Per-point flux uncertainties; uniform weighting if ``None``.
    period_min: Shortest period to search, in days.
    period_max: Longest period to search, in days.
    oversample: Frequency oversampling factor for the period grid.
    periods: Explicit trial periods in days. When ``None`` they are built with
      `frequency_grid` from the data baseline; pass an explicit grid to use
      the search under ``jax.jit`` (the grid length must then be static).
    min_duration_fraction: Shortest transit duty cycle, which sets the period
      grid resolution (smaller is finer and more expensive).
    num_bins: Number of phase bins (static).
    duration_fractions: Trial transit durations as fractions of the period.
    limb_darkening: Quadratic limb-darkening coefficients for the template.
    num_radii: Radial quadrature resolution for the template (static).
    period_chunk_size: Number of trial periods evaluated per vectorised chunk;
      smaller uses less memory on long-baseline grids.

  Returns:
    A `Periodogram` with the power spectrum and the best candidate.
  """
  time_days = jnp.asarray(time_days)
  flux = jnp.asarray(flux)
  if flux_err is None:
    weights = jnp.ones_like(flux)
  else:
    weights = 1.0 / jnp.asarray(flux_err) ** 2

  # Work with a zero-baseline flux so the matched filter measures the dip.
  flux_centered = flux - jnp.average(flux, weights=weights)

  if periods is None:
    baseline = float(time_days[-1] - time_days[0])
    periods = frequency_grid(
        period_min, period_max, baseline, oversample,
        min_duration_fraction=min_duration_fraction,
    )

  templates, templates_squared = _transit_templates(
      num_bins, tuple(duration_fractions), *limb_darkening, num_radii
  )
  fft_template = jnp.fft.rfft(templates, axis=-1)
  fft_template_squared = jnp.fft.rfft(templates_squared, axis=-1)

  # Evaluate the grid in chunks (vectorised within a chunk, looped across them)
  # so memory stays bounded: a single vmap over a long-baseline grid of ~10^5
  # periods would materialise (num_periods x num_points) intermediates and run
  # out of memory.
  def search(period):
    return _period_power(
        period,
        time_days,
        flux_centered,
        weights,
        fft_template,
        fft_template_squared,
        num_bins,
    )

  power, duration_index, lag, depth = jax.lax.map(
      search, periods, batch_size=period_chunk_size
  )

  best = jnp.argmax(power)
  best_period = periods[best]
  best_fraction = jnp.asarray(duration_fractions)[duration_index[best]]
  # The template is centred on phase zero, so the lag is the transit phase.
  best_epoch = (lag[best] + 0.5) / num_bins * best_period
  return Periodogram(
      periods=periods,
      power=power,
      best_period=best_period,
      best_epoch=best_epoch,
      best_duration=best_fraction * best_period,
      best_depth=depth[best],
      best_power=power[best],
  )
