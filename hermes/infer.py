"""Predictor combining neural network and physics-based analysis.

This module combines:
1. Neural network predictions (planet count, existence)
2. BLS period search on actual light curves  
3. Physics-based parameter computation
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from hermes.model import MultiPlanetDetector
from hermes.signal_analysis import run_bls_period_search, detect_multiple_planets
from hermes import physics


class ExoplanetPredictor:
  """Combines ML predictions with physics-based analysis."""
  
  def __init__(self, model: MultiPlanetDetector, stellar_radius_solar: float = 1.0,
               stellar_mass_solar: float = 1.0):
    """Initializes predictor.
    
    Args:
      model: Trained MultiPlanetDetector model.
      stellar_radius_solar: Stellar radius in solar radii.
      stellar_mass_solar: Stellar mass in solar masses.
    """
    self.model = model
    self.stellar_radius_solar = stellar_radius_solar
    self.stellar_mass_solar = stellar_mass_solar
  
  def predict(
      self,
      time: np.ndarray,
      flux: np.ndarray,
      flux_err: Optional[np.ndarray] = None,
      target_id: str = 'unknown'
  ) -> Dict[str, Any]:
    """Makes full prediction combining ML and physics.
    
    Pipeline:
    1. NN predicts planet count and existence probabilities
    2. BLS detects actual transit signals in light curve
    3. Physics computes radius (from depth) and distance (from period)
    
    Args:
      time: Time array in days.
      flux: Normalized flux array.
      flux_err: Flux uncertainties (optional).
      target_id: Target identifier.
      
    Returns:
      Complete prediction dictionary.
    """
    start_time = datetime.now()
    
    # Step 1: Neural network prediction on processed light curve
    # (This gives us planet count and existence probabilities)
    light_curve_features = self._prepare_features(flux)
    nn_outputs = self.model(light_curve_features)
    
    predicted_count = int(jnp.argmax(nn_outputs['planet_count_probs'][0]))
    count_confidence = float(jnp.max(nn_outputs['planet_count_probs'][0]))
    
    # Step 2: BLS analysis on actual light curve to find periods
    detected_transits = detect_multiple_planets(
        time, flux, max_planets=predicted_count, snr_threshold=7.0
    )
    
    # Step 3: Combine NN existence probs with BLS detections
    planets = []
    for i, transit in enumerate(detected_transits):
      if i >= predicted_count:
        break
      
      # Get existence probability from NN
      existence_prob = float(nn_outputs['existence_probs'][i][0, 0])
      
      # Compute physical parameters from BLS results
      period = transit.period
      depth = transit.depth
      
      # Use physics equations
      radius = float(physics.compute_planet_radius(
          jnp.array(depth), self.stellar_radius_solar
      ))
      
      distance = float(physics.compute_semi_major_axis(
          jnp.array(period), self.stellar_mass_solar
      ))
      
      temperature = float(physics.compute_equilibrium_temperature(
          jnp.array(distance)
      ))
      
      planets.append({
          'planet_id': i + 1,
          'existence_probability': existence_prob,
          'orbital_period_days': period,
          'period_uncertainty': 0.0,  # Could compute from BLS periodogram
          'planet_radius_earth': radius,
          'radius_uncertainty': 0.0,  # Could compute from depth uncertainty
          'orbital_distance_au': distance,
          'distance_uncertainty': 0.0,
          'equilibrium_temperature_k': temperature,
          'confidence_score': existence_prob,
          'transit_depth': depth,
          'transit_duration_hours': transit.duration * 24,
          'signal_to_noise': transit.snr
      })
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return {
        'target_id': target_id,
        'predicted_planet_count': predicted_count,
        'planet_count_confidence': count_confidence,
        'planets': planets,
        'processing_time_ms': processing_time,
        'inference_timestamp': datetime.now().isoformat(),
        'method': 'hybrid_nn_physics',
        'notes': (
            'Planet count and existence from neural network. '
            'Period from BLS. Radius from transit depth. '
            'Distance from Kepler\'s Third Law.'
        )
    }
  
  def _prepare_features(self, flux: np.ndarray) -> jnp.ndarray:
    """Prepares light curve for neural network.
    
    Args:
      flux: Flux array.
      
    Returns:
      Processed features [1, 2048].
    """
    # Pad or truncate to 2048
    if len(flux) < 2048:
      flux_padded = np.pad(flux, (0, 2048 - len(flux)))
    else:
      flux_padded = flux[:2048]
    
    # Normalize
    flux_padded = (flux_padded - np.mean(flux_padded)) / (np.std(flux_padded) + 1e-6)
    
    return jnp.expand_dims(jnp.array(flux_padded, dtype=jnp.float32), axis=0)

