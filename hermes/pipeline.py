"""End-to-end inference.

Search, differentiable refinement, calibrated vetting and iterative subtraction
for multi-planet systems.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from flax import nnx

from hermes.numerics import physics
from hermes.data.build_dataset import compute_scalar_features
from hermes.data.lightcurves import LightCurve
from hermes.numerics import fit, folding, periodogram, transit
from hermes.data.preprocess import clean_light_curve
from hermes.trainers.calibration import apply_temperature


@dataclasses.dataclass(frozen=True)
class StellarParameters:
  """Host-star parameters used to derive physical candidate quantities."""

  radius_solar: float = 1.0
  mass_solar: float = 1.0
  teff_k: float = physics.SOLAR_EFFECTIVE_TEMPERATURE_K
  logg_cgs: float = 4.44
  density_cgs: float = 1.41


@dataclasses.dataclass(frozen=True)
class Candidate:
  """A detected and vetted transit candidate.

  Attributes:
    period_days: Refined orbital period.
    epoch_days: Refined transit epoch.
    depth: Refined fractional transit depth.
    duration_days: Refined transit duration.
    impact_parameter: Refined impact parameter.
    probability: Calibrated planet probability from the vetter.
    planet_radius_earth: Planet radius in Earth radii.
    semi_major_axis_au: Orbital semi-major axis in AU.
    equilibrium_temperature_k: Planetary equilibrium temperature in Kelvin.
    periodogram_power: Detection power at the candidate period.
  """

  period_days: float
  epoch_days: float
  depth: float
  duration_days: float
  impact_parameter: float
  probability: float
  planet_radius_earth: float
  semi_major_axis_au: float
  equilibrium_temperature_k: float
  periodogram_power: float


class VetterPipeline:
  """Runs detection, refinement and calibrated vetting on a light curve."""

  def __init__(
      self,
      model: nnx.Module,
      meta: dict,
      search_config,
      data_config,
      temperature: float = 1.0,
  ):
    """Initialises the pipeline.

    Args:
      model: A trained `hermes.nn.vetter.TransitVetter`.
      meta: Dataset metadata (``feature_mean``/``feature_std``/view sizes).
      search_config: The ``search`` configuration section.
      data_config: The ``data`` configuration section (cleaning, view sizes).
      temperature: Calibration temperature for the probability.
    """
    self.model = model
    self.feature_mean = np.asarray(meta["feature_mean"], dtype=np.float32)
    self.feature_std = np.asarray(meta["feature_std"], dtype=np.float32)
    self.global_bins = meta["global_bins"]
    self.local_bins = meta["local_bins"]
    self.search = search_config
    self.data = data_config
    self.temperature = temperature

  def detect(
      self,
      time_days: np.ndarray,
      flux: np.ndarray,
      flux_err: np.ndarray | None = None,
      stellar: StellarParameters = StellarParameters(),
      max_planets: int = 3,
      power_threshold: float = 0.0,
  ) -> list[Candidate]:
    """Detects and vets up to ``max_planets`` transits in a light curve.

    Args:
      time_days: Observation times in days.
      flux: Raw flux.
      flux_err: Optional per-point uncertainties.
      stellar: Host-star parameters for physical derivations.
      max_planets: Maximum number of candidates to extract.
      power_threshold: Stop once the periodogram power falls below this.

    Returns:
      A list of `Candidate` in detection order.
    """
    if flux_err is None:
      flux_err = np.full_like(flux, np.std(flux))
    clean = clean_light_curve(
        LightCurve(time_days, flux, flux_err),
        cadence_days=self.data.cadence_days,
        detrend_window_days=self.data.detrend_window_days,
        sigma=self.data.sigma_clip,
    )

    residual = np.asarray(clean.flux)
    candidates: list[Candidate] = []
    self.model.eval()
    for _ in range(max_planets):
      result = periodogram.transit_periodogram(
          clean.time_days,
          residual,
          period_min=self.search.period_min,
          period_max=self.search.period_max,
          oversample=self.search.oversample,
          num_bins=self.search.num_bins,
          duration_fractions=tuple(self.search.duration_fractions),
          limb_darkening=tuple(self.search.limb_darkening),
      )
      if float(result.best_power) < power_threshold:
        break
      candidate = self._vet_candidate(
          clean.time_days, residual, clean.flux_err, result, stellar
      )
      candidates.append(candidate)
      residual = self._subtract(clean.time_days, residual, candidate)
    return candidates

  def _vet_candidate(self, time_days, residual, flux_err, result, stellar):
    """Refines, folds, scores and characterises a single candidate."""
    refined = fit.refine_transit(
        time_days, residual, flux_err,
        period_days=float(result.best_period),
        init_epoch=float(result.best_epoch),
        init_depth=float(result.best_depth),
        init_duration=float(result.best_duration),
    )
    period = float(refined.period_days)
    epoch = float(refined.epoch_days)
    depth = float(refined.depth)
    duration = float(refined.duration_days)

    global_view = folding.normalize_view(
        folding.global_view(
            time_days, residual, period, epoch, self.global_bins
        )
    )
    local_view = folding.normalize_view(
        folding.local_view(
            time_days, residual, period, epoch, duration,
            self.local_bins, self.data.local_durations,
        )
    )
    features = compute_scalar_features(
        period, depth, duration, stellar.density_cgs, stellar.teff_k,
        stellar.radius_solar, stellar.logg_cgs,
    )
    features = (features - self.feature_mean) / self.feature_std

    outputs = self.model(
        global_view[None].astype(jnp.float32),
        local_view[None].astype(jnp.float32),
        jnp.asarray(features[None]),
    )
    probability = float(
        apply_temperature(outputs["logit"], self.temperature)[0]
    )

    semi_major_axis = float(
        physics.semi_major_axis_au(period, stellar.mass_solar)
    )
    return Candidate(
        period_days=period,
        epoch_days=epoch,
        depth=depth,
        duration_days=duration,
        impact_parameter=float(refined.impact_parameter),
        probability=probability,
        planet_radius_earth=float(
            physics.planet_radius_earth(depth, stellar.radius_solar)
        ),
        semi_major_axis_au=semi_major_axis,
        equilibrium_temperature_k=float(
            physics.equilibrium_temperature(
                semi_major_axis, stellar.radius_solar, stellar.teff_k
            )
        ),
        periodogram_power=float(result.best_power),
    )

  def _subtract(self, time_days, residual, candidate):
    """Divides out the fitted transit model to expose further planets."""
    model_flux = transit.transit_light_curve(
        time_days,
        period_days=candidate.period_days,
        epoch_days=candidate.epoch_days,
        radius_ratio=float(np.sqrt(max(candidate.depth, 1e-8))),
        a_over_rstar=candidate.period_days / np.pi
        * (1.0 + np.sqrt(max(candidate.depth, 1e-8)))
        / max(candidate.duration_days, 1e-3),
        impact_parameter=candidate.impact_parameter,
    )
    return np.asarray(residual) / np.asarray(model_flux)
