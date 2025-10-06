"""Physical calculations for exoplanet parameters.

This module computes planet properties from observed light curve features
using established astrophysical relationships.
"""

import jax.numpy as jnp

def compute_semi_major_axis(period_days: jnp.ndarray, 
                            stellar_mass_solar: float = 1.0) -> jnp.ndarray:
  """Computes orbital semi-major axis using Kepler's Third Law.
  
  Kepler's Third Law: a^3 / P^2 = G * M_star / (4 * pi^2)
  
  For solar-mass stars: a [AU] = (P [years])^(2/3)
  
  Args:
    period_days: Orbital period in days.
    stellar_mass_solar: Stellar mass in solar masses.
    
  Returns:
    Semi-major axis in AU.
  """
  period_years = period_days / 365.25
  semi_major_axis = (period_years ** 2 * stellar_mass_solar) ** (1.0 / 3.0)
  return semi_major_axis


def compute_planet_radius(transit_depth: jnp.ndarray, 
                          stellar_radius_solar: float = 1.0) -> jnp.ndarray:
  """Computes planet radius from transit depth.
  
  Transit depth δ = (R_p / R_star)^2
  Therefore: R_p = R_star * sqrt(δ)
  
  Args:
    transit_depth: Fractional flux decrease during transit.
    stellar_radius_solar: Stellar radius in solar radii.
    
  Returns:
    Planet radius in Earth radii (1 R_sun ≈ 109.2 R_earth).
  """
  SOLAR_TO_EARTH_RADII = 109.2
  radius_ratio = jnp.sqrt(jnp.maximum(transit_depth, 0.0))
  planet_radius_earth = radius_ratio * stellar_radius_solar * SOLAR_TO_EARTH_RADII
  return planet_radius_earth


def compute_transit_duration(period_days: jnp.ndarray, 
                             semi_major_axis_au: jnp.ndarray,
                             stellar_radius_solar: float = 1.0,
                             impact_parameter: float = 0.0) -> jnp.ndarray:
  """Computes transit duration from orbital geometry.
  
  Args:
    period_days: Orbital period in days.
    semi_major_axis_au: Semi-major axis in AU.
    stellar_radius_solar: Stellar radius in solar radii.
    impact_parameter: Impact parameter (0 = central transit).
    
  Returns:
    Transit duration in hours.
  """
  AU_TO_SOLAR_RADII = 215.03  # 1 AU ≈ 215 solar radii
  a_in_stellar_radii = semi_major_axis_au * AU_TO_SOLAR_RADII
  
  # Duration ≈ (P / pi) * arcsin(R_star / a)
  # Simplified for small angles and central transits
  duration_days = (period_days / jnp.pi) * jnp.arcsin(
      stellar_radius_solar / (a_in_stellar_radii + 1e-10)
  )
  
  return duration_days * 24.0  # Convert to hours


def compute_equilibrium_temperature(semi_major_axis_au: jnp.ndarray,
                                    stellar_temperature_k: float = 5778.0,
                                    albedo: float = 0.3) -> jnp.ndarray:
  """Computes planet equilibrium temperature.
  
  T_eq = T_star * (R_star / (2 * a))^0.5 * (1 - A)^0.25
  
  Args:
    semi_major_axis_au: Semi-major axis in AU.
    stellar_temperature_k: Stellar temperature in Kelvin.
    albedo: Bond albedo (0-1).
    
  Returns:
    Equilibrium temperature in Kelvin.
  """
  STELLAR_RADIUS_AU = 0.00465  # 1 R_sun ≈ 0.00465 AU
  
  temperature = stellar_temperature_k * jnp.sqrt(
      STELLAR_RADIUS_AU / (2.0 * semi_major_axis_au)
  )
  temperature *= (1.0 - albedo) ** 0.25
  
  return temperature

