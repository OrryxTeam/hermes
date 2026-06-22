"""Tests for the transit forward models in hermes.numerics.transit."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from hermes.numerics import transit


def _overlap_fraction_reference(z: float, p: float) -> float:
  """Brute-force grid estimate of the two-circle overlap fraction.

  Integrates the indicator of "inside the star and inside the planet" over a
  fine Cartesian grid, an independent reference for the analytic overlap.
  """
  grid = np.linspace(-1.0, 1.0, 2001)
  x, y = np.meshgrid(grid, grid)
  cell_area = (grid[1] - grid[0]) ** 2
  in_star = x**2 + y**2 <= 1.0
  in_planet = (x - z) ** 2 + y**2 <= p**2
  return float(np.sum(in_star & in_planet) * cell_area / np.pi)


def test_obscuration_no_overlap():
  """No flux is blocked when the disks do not overlap."""
  obscuration = transit.uniform_disk_obscuration(jnp.array(1.5), 0.1)
  assert jnp.allclose(obscuration, 0.0)


def test_obscuration_fully_inside_equals_depth():
  """A planet fully on the disk blocks exactly ``k^2`` of the flux."""
  obscuration = transit.uniform_disk_obscuration(jnp.array(0.0), 0.1)
  assert jnp.allclose(obscuration, 0.01, atol=1e-12)


def test_obscuration_matches_grid_reference():
  """The analytic overlap matches a brute-force grid integration."""
  for z in [0.0, 0.5, 0.85, 1.0, 1.1]:
    analytic = float(transit.uniform_disk_obscuration(jnp.array(z), 0.2))
    assert abs(analytic - _overlap_fraction_reference(z, 0.2)) < 2e-3


def test_obscuration_symmetric_in_separation():
  """Obscuration depends only on the magnitude of the separation."""
  pos = transit.uniform_disk_obscuration(jnp.array(0.4), 0.15)
  neg = transit.uniform_disk_obscuration(jnp.array(-0.4), 0.15)
  assert jnp.allclose(pos, neg)


def test_limb_darkened_reduces_to_uniform_disk():
  """With zero coefficients the LD model equals the uniform-disk flux."""
  z = jnp.linspace(0.0, 1.3, 60)
  ld = transit.quadratic_limb_darkened_flux(z, 0.1, 0.0, 0.0, num_radii=512)
  assert jnp.max(jnp.abs(ld - transit.uniform_disk_flux(z, 0.1))) < 1e-4


def test_limb_darkening_deepens_central_transit():
  """Limb darkening makes a central transit deeper than the uniform case."""
  z = jnp.array(0.0)
  uniform = transit.quadratic_limb_darkened_flux(z, 0.1, 0.0, 0.0)
  darkened = transit.quadratic_limb_darkened_flux(z, 0.1, 0.4, 0.3)
  assert darkened < uniform


def test_light_curve_out_of_transit_is_unity():
  """Far from transit the normalised flux is one."""
  flux = transit.transit_light_curve(
      jnp.array([2.5]),
      period_days=10.0,
      epoch_days=0.0,
      radius_ratio=0.1,
      a_over_rstar=20.0,
      impact_parameter=0.0,
  )
  assert jnp.allclose(flux, 1.0, atol=1e-6)


def test_light_curve_masks_secondary_eclipse():
  """No dip occurs at the secondary-eclipse phase."""
  flux = transit.transit_light_curve(
      jnp.array([5.0]),
      period_days=10.0,
      epoch_days=0.0,
      radius_ratio=0.1,
      a_over_rstar=20.0,
      impact_parameter=0.0,
  )
  assert jnp.allclose(flux, 1.0)


def test_light_curve_central_depth_matches_radius_ratio():
  """A small central transit has depth approximately ``k^2``."""
  flux = transit.transit_light_curve(
      jnp.array([0.0]),
      period_days=10.0,
      epoch_days=0.0,
      radius_ratio=0.05,
      a_over_rstar=20.0,
      impact_parameter=0.0,
      limb_darkening=(0.0, 0.0),
  )
  assert jnp.allclose(flux[0], 1.0 - 0.05**2, atol=1e-4)


def test_light_curve_gradients_are_finite():
  """The light curve is differentiable in all transit parameters."""
  time = jnp.linspace(-0.2, 0.2, 64)

  def min_flux(params):
    period, epoch, radius_ratio, a_over_rstar, impact = params
    flux = transit.transit_light_curve(
        time,
        period_days=period,
        epoch_days=epoch,
        radius_ratio=radius_ratio,
        a_over_rstar=a_over_rstar,
        impact_parameter=impact,
        limb_darkening=(0.4, 0.3),
        num_radii=128,
    )
    return jnp.min(flux)

  grads = jax.grad(min_flux)(jnp.array([10.0, 0.0, 0.1, 20.0, 0.3]))
  chex.assert_tree_all_finite(grads)


def test_light_curve_vmap_over_radius_ratio():
  """The model vectorises over a batch of radius ratios."""
  time = jnp.linspace(-0.1, 0.1, 32)

  def depth(radius_ratio):
    flux = transit.transit_light_curve(
        time,
        period_days=10.0,
        epoch_days=0.0,
        radius_ratio=radius_ratio,
        a_over_rstar=20.0,
        impact_parameter=0.0,
        num_radii=128,
    )
    return 1.0 - jnp.min(flux)

  depths = jax.vmap(depth)(jnp.array([0.05, 0.1, 0.15]))
  assert depths.shape == (3,)
  assert jnp.all(jnp.diff(depths) > 0.0)  # deeper for larger planets


class TransitVariantsTest(chex.TestCase):
  """The light curve behaves identically under jit and eager execution."""

  @chex.variants(with_jit=True, without_jit=True)
  def test_light_curve_jit_and_eager_agree(self):
    time = jnp.linspace(-0.2, 0.2, 64)

    def light_curve(radius_ratio):
      return transit.transit_light_curve(
          time,
          period_days=10.0,
          epoch_days=0.0,
          radius_ratio=radius_ratio,
          a_over_rstar=20.0,
          impact_parameter=0.1,
          limb_darkening=(0.4, 0.3),
          num_radii=128,
      )

    flux = self.variant(light_curve)(0.1)
    chex.assert_tree_all_finite(flux)
    chex.assert_shape(flux, (64,))
    assert float(jnp.min(flux)) < 1.0
