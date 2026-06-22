"""Tests for differentiable transit refinement in hermes.numerics.fit."""

import chex
import jax.numpy as jnp

from hermes.numerics import fit, transit
from hermes.testing import _fixtures


def test_refine_recovers_injected_parameters():
  """Gradient refinement recovers depth and epoch from a noisy transit."""
  time, flux = _fixtures.injected_light_curve(
      6.0, epoch=1.3, depth_ratio=0.1, noise=5e-4, baseline=40.0, num=4000
  )
  flux_err = jnp.full(time.shape, 5e-4)
  result = fit.refine_transit(
      time, flux, flux_err,
      period_days=6.0, init_epoch=1.4, init_depth=0.007, init_duration=0.25,
      steps=200,
  )
  chex.assert_tree_all_finite(result)
  assert jnp.abs(result.depth - 0.01) < 0.003  # injected k^2 = 0.01
  assert jnp.abs(result.epoch_days - 1.3) < 0.02
  assert result.duration_days > 0.0


def test_refine_reduces_chi_square():
  """Refinement lowers the chi-square relative to the initial guess."""
  time, flux = _fixtures.injected_light_curve(
      4.5, epoch=0.7, depth_ratio=0.12, noise=5e-4, baseline=30.0, num=3000
  )
  flux_err = jnp.full(time.shape, 5e-4)
  result = fit.refine_transit(
      time, flux, flux_err,
      period_days=4.5, init_epoch=0.9, init_depth=0.005, init_duration=0.3,
      steps=200,
  )
  initial_model = transit.transit_light_curve(
      time,
      period_days=4.5,
      epoch_days=0.9,
      radius_ratio=jnp.sqrt(0.005),
      a_over_rstar=4.5 / jnp.pi * 1.07 / 0.3,
      impact_parameter=0.6,
  )
  initial_chi = jnp.sum(((flux - initial_model) / flux_err) ** 2)
  assert result.chi_square < initial_chi
