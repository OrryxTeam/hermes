"""Light-curve cleaning: running-median detrend, transit-preserving (asymmetric)
sigma-clipping and unit-baseline normalisation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from hermes.data.lightcurves import LightCurve


def median_detrend(
    flux: np.ndarray, window_points: int
) -> np.ndarray:
  """Removes slow trends by dividing out a running median.

  Args:
    flux: Flux samples.
    window_points: Running-median window in samples (forced odd, >= 3).

  Returns:
    Detrended flux normalised to a unit baseline.
  """
  window_points = max(3, window_points | 1)
  trend = ndimage.median_filter(flux, size=window_points, mode="nearest")
  trend = np.where(np.abs(trend) > 0.0, trend, 1.0)
  return flux / trend


def sigma_clip_mask(
    flux: np.ndarray, sigma: float, lower_sigma: float = np.inf, iters: int = 5
) -> np.ndarray:
  """Computes a boolean mask of points within robust deviations of the median.

  Clipping is asymmetric by default: upward excursions (e.g. cosmic rays) are
  rejected at ``sigma`` deviations; downward excursions only at ``lower_sigma``
  (infinite by default). This preserves transits, which are deep downward dips
  that a symmetric clip would discard.

  Args:
    flux: Flux samples.
    sigma: Upper clipping threshold in robust standard deviations.
    lower_sigma: Lower clipping threshold; ``inf`` keeps all downward points.
    iters: Maximum number of clipping iterations.

  Returns:
    Boolean mask that is true for retained points.
  """
  mask = np.isfinite(flux)
  for _ in range(iters):
    centre = np.median(flux[mask])
    scatter = 1.4826 * np.median(np.abs(flux[mask] - centre))
    if scatter == 0.0:
      # The median absolute deviation degenerates when most points are
      # identical (e.g. a perfectly flat detrended baseline); fall back to the
      # standard deviation so isolated outliers are still rejected.
      scatter = float(np.std(flux[mask]))
    if scatter == 0.0:
      break
    deviation = flux - centre
    new_mask = (
        mask
        & (deviation <= sigma * scatter)
        & (deviation >= -lower_sigma * scatter)
    )
    if new_mask.sum() == mask.sum():
      mask = new_mask
      break
    mask = new_mask
  return mask


def clean_light_curve(
    light_curve: LightCurve,
    cadence_days: float,
    detrend_window_days: float = 2.0,
    sigma: float = 5.0,
) -> LightCurve:
  """Detrends, sigma-clips and renormalises a light curve.

  Args:
    light_curve: Raw light curve.
    cadence_days: Sampling interval, used to size the detrend window.
    detrend_window_days: Running-median window in days.
    sigma: Sigma-clipping threshold.

  Returns:
    A cleaned `LightCurve` with finite, detrended, unit-baseline flux.
  """
  finite = (
      np.isfinite(light_curve.time_days)
      & np.isfinite(light_curve.flux)
  )
  time = light_curve.time_days[finite]
  flux = light_curve.flux[finite]
  flux_err = light_curve.flux_err[finite]

  window_points = int(round(detrend_window_days / cadence_days))
  flux = median_detrend(flux, window_points)

  keep = sigma_clip_mask(flux, sigma)
  baseline = np.median(flux[keep]) if keep.any() else 1.0
  baseline = baseline if baseline != 0.0 else 1.0
  return LightCurve(
      time[keep], flux[keep] / baseline, flux_err[keep] / abs(baseline)
  )
