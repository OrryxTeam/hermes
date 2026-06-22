"""Phase folding and global/local view binning (Shallue & Vanderburg 2018).

Binning uses hard assignment, so it is differentiable in the flux values and
runs under jit and vmap.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def fold_phase(
    time_days: Array, period_days: Array, epoch_days: Array
) -> Array:
  """Folds observation times onto orbital phase in ``[-0.5, 0.5)``.

  Phase zero corresponds to the transit centre ``epoch_days``.

  Args:
    time_days: Observation times in days.
    period_days: Folding period in days.
    epoch_days: Reference transit centre time in days.

  Returns:
    Orbital phase in ``[-0.5, 0.5)`` with the same shape as ``time_days``.
  """
  phase = (time_days - epoch_days) / period_days
  return phase - jnp.round(phase)


def bin_mean(
    coordinate: Array,
    values: Array,
    num_bins: int,
    lower: float,
    upper: float,
    empty_fill: float = 1.0,
) -> tuple[Array, Array]:
  """Bins ``values`` by ``coordinate`` into equal-width bins and averages them.

  Points outside ``[lower, upper)`` are ignored. Empty bins are set to
  ``empty_fill`` (the out-of-transit flux level for normalised light curves).

  Args:
    coordinate: Bin coordinate per point (for example, orbital phase).
    values: Values to average per bin (for example, normalised flux).
    num_bins: Number of equal-width bins (static).
    lower: Lower edge of the binned range (inclusive).
    upper: Upper edge of the binned range (exclusive).
    empty_fill: Value assigned to bins that receive no points.

  Returns:
    A tuple ``(binned_values, counts)`` each of shape ``(num_bins,)``.
  """
  scaled = (coordinate - lower) / (upper - lower) * num_bins
  index = jnp.floor(scaled).astype(jnp.int32)
  in_range = (index >= 0) & (index < num_bins)
  index = jnp.where(in_range, index, 0)
  weight = in_range.astype(values.dtype)

  totals = jnp.zeros(num_bins, values.dtype).at[index].add(values * weight)
  counts = jnp.zeros(num_bins, values.dtype).at[index].add(weight)
  safe_counts = jnp.where(counts > 0, counts, 1.0)
  averaged = jnp.where(counts > 0, totals / safe_counts, empty_fill)
  return averaged, counts


def normalize_view(view: Array) -> Array:
  """Normalises a view to median zero and minimum -1 (AstroNet style).

  Subtracting the median removes the out-of-transit baseline; dividing by the
  transit depth (the magnitude of the most negative bin) fixes the transit
  bottom at -1. This is bounded and scale-stable, unlike dividing by a robust
  scatter, which explodes for sparse or near-flat folded views whose scatter is
  almost zero. The same transform is used offline and at inference.

  Args:
    view: Binned flux view.

  Returns:
    The normalised view, with the same shape.
  """
  centered = view - jnp.median(view)
  depth = jnp.maximum(-jnp.min(centered), 1e-6)
  return centered / depth


def global_view(
    time_days: Array,
    flux: Array,
    period_days: Array,
    epoch_days: Array,
    num_bins: int = 2001,
) -> Array:
  """Phase-folded global view across the full orbital period.

  Args:
    time_days: Observation times in days.
    flux: Normalised flux (one out of transit).
    period_days: Folding period in days.
    epoch_days: Reference transit centre time in days.
    num_bins: Number of phase bins (static).

  Returns:
    Binned flux of shape ``(num_bins,)`` spanning phase ``[-0.5, 0.5)``.
  """
  phase = fold_phase(time_days, period_days, epoch_days)
  view, _ = bin_mean(phase, flux, num_bins, -0.5, 0.5)
  return view


def local_view(
    time_days: Array,
    flux: Array,
    period_days: Array,
    epoch_days: Array,
    duration_days: Array,
    num_bins: int = 201,
    num_durations: float = 4.0,
) -> Array:
  """Phase-folded local view zoomed onto the transit.

  The view spans ``+/- num_durations`` transit durations around phase zero,
  expressed as a fraction of the orbital period.

  Args:
    time_days: Observation times in days.
    flux: Normalised flux (one out of transit).
    period_days: Folding period in days.
    epoch_days: Reference transit centre time in days.
    duration_days: Transit duration in days.
    num_bins: Number of phase bins (static).
    num_durations: Half-width of the window in transit durations.

  Returns:
    Binned flux of shape ``(num_bins,)`` zoomed onto the transit.
  """
  phase = fold_phase(time_days, period_days, epoch_days)
  half_window = num_durations * duration_days / period_days
  view, _ = bin_mean(phase, flux, num_bins, -half_window, half_window)
  return view
