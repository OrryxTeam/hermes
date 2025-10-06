"""Signal analysis for light curve transit detection.

This module provides BLS (Box Least Squares) period search and transit
characterization following standard exoplanet detection methodology.
"""

import jax.numpy as jnp
import numpy as np
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TransitSignature:
  """Container for detected transit properties."""
  period: float
  epoch: float
  depth: float
  duration: float
  snr: float
  

def run_bls_period_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    period_min: float = 0.5,
    period_max: float = 100.0,
    num_periods: int = 5000,
    num_durations: int = 15
) -> TransitSignature:
  """Runs Box Least Squares period search on light curve.
  
  This is the standard method for detecting periodic transit signals.
  Following your JAX BLS example pattern with astropy.
  
  Args:
    time: Time array in days.
    flux: Normalized flux array.
    flux_err: Flux uncertainties (optional).
    period_min: Minimum period to search (days).
    period_max: Maximum period to search (days).
    num_periods: Number of test periods.
    num_durations: Number of test durations.
    
  Returns:
    TransitSignature with detected properties.
  """
  # Handle flux errors
  if flux_err is None:
    flux_err = np.full_like(flux, np.std(flux))
  
  # Create BLS model
  bls = BoxLeastSquares(time * u.day, flux, flux_err)
  
  # Generate period grid
  periods = np.linspace(period_min, period_max, num_periods)
  
  # Generate duration grid (2-20% of minimum period)
  min_duration = 0.02 * period_min
  max_duration = 0.2 * period_min
  durations = np.linspace(min_duration, max_duration, num_durations) * u.day
  
  # Run BLS
  periodogram = bls.power(periods * u.day, durations, objective='snr')
  
  # Extract best parameters
  best_idx = np.argmax(periodogram.power)
  
  return TransitSignature(
      period=float(periodogram.period[best_idx].value),
      epoch=float(periodogram.transit_time[best_idx].value),
      depth=float(periodogram.depth[best_idx]),
      duration=float(periodogram.duration[best_idx].value),
      snr=float(periodogram.power[best_idx])
  )


def detect_multiple_planets(
    time: np.ndarray,
    flux: np.ndarray,
    max_planets: int = 5,
    snr_threshold: float = 7.0
) -> List[TransitSignature]:
  """Detects multiple planets by iterative BLS and signal subtraction.
  
  Args:
    time: Time array in days.
    flux: Flux array.
    max_planets: Maximum number of planets to search for.
    snr_threshold: Minimum SNR to accept a detection.
    
  Returns:
    List of detected TransitSignatures.
  """
  detected_planets = []
  residual_flux = flux.copy()
  
  for _ in range(max_planets):
    # Run BLS on residual
    transit = run_bls_period_search(time, residual_flux)
    
    if transit.snr < snr_threshold:
      break
    
    detected_planets.append(transit)
    
    # Subtract detected transit from residual
    residual_flux = subtract_transit_model(
        time, residual_flux, transit.period, transit.epoch, 
        transit.depth, transit.duration
    )
  
  return detected_planets


def subtract_transit_model(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    depth: float,
    duration: float
) -> np.ndarray:
  """Subtracts a transit model from the light curve.
  
  Args:
    time: Time array.
    flux: Flux array.
    period: Period of transit.
    epoch: Time of first transit.
    depth: Transit depth.
    duration: Transit duration.
    
  Returns:
    Residual flux with transit removed.
  """
  # Simple box model for transit
  phase = np.mod(time - epoch, period) / period
  in_transit = (phase < duration / (2 * period)) | (phase > 1 - duration / (2 * period))
  
  model_flux = np.ones_like(flux)
  model_flux[in_transit] = 1.0 - depth
  
  # Divide out the model
  residual = flux / model_flux
  
  return residual


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
  """Phase-folds light curve at given period.
  
  Args:
    time: Time array.
    flux: Flux array.
    period: Folding period.
    epoch: Reference epoch.
    
  Returns:
    Tuple of (phase, flux) sorted by phase.
  """
  phase = np.mod(time - epoch, period) / period
  sort_idx = np.argsort(phase)
  return phase[sort_idx], flux[sort_idx]

