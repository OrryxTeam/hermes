"""Inference-speed benchmarks for the vetter and the full pipeline."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def _count_params(model: nnx.Module) -> int:
  """Returns the total number of parameters in ``model``."""
  leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
  return int(sum(leaf.size for leaf in leaves))


def benchmark_vetter(
    model: nnx.Module,
    global_bins: int,
    local_bins: int,
    *,
    num_features: int = 8,
    batch_sizes: tuple[int, ...] = (1, 64, 512),
    num_iters: int = 50,
) -> dict:
  """Times the vetter forward pass after JIT warmup.

  The first call per batch size is discarded so compilation is excluded, and
  each timed call blocks on its result so asynchronous dispatch is not mistimed.

  Args:
    model: The vetter to benchmark.
    global_bins: Length of the global view.
    local_bins: Length of the local view.
    num_features: Number of scalar features.
    batch_sizes: Batch sizes to time.
    num_iters: Timed iterations per batch size.

  Returns:
    A dict with the parameter count, device, and a per-batch list of timings.
  """
  model.eval()

  @nnx.jit
  def forward(model, global_view, local_view, features):
    return model(global_view, local_view, features)["logit"]

  timings = []
  for batch in batch_sizes:
    global_view = jnp.ones((batch, global_bins))
    local_view = jnp.ones((batch, local_bins))
    features = jnp.ones((batch, num_features))
    forward(model, global_view, local_view, features).block_until_ready()
    start = time.perf_counter()
    for _ in range(num_iters):
      forward(model, global_view, local_view, features).block_until_ready()
    seconds = (time.perf_counter() - start) / num_iters
    timings.append({
        "batch_size": batch,
        "ms_per_batch": seconds * 1e3,
        "ms_per_candidate": seconds * 1e3 / batch,
        "candidates_per_second": batch / seconds,
    })
  return {
      "num_params": _count_params(model),
      "device": jax.devices()[0].platform,
      "timings": timings,
  }


def benchmark_pipeline(
    pipeline,
    time_days: np.ndarray,
    flux: np.ndarray,
    *,
    max_planets: int = 1,
    num_repeats: int = 3,
) -> dict:
  """Times the end-to-end detection pipeline on one light curve.

  The first call is discarded so JIT compilation is excluded. The result is
  dominated by the periodogram search, which scales with the observational
  baseline, not by the (millisecond) vetter forward pass.

  Args:
    pipeline: A `hermes.pipeline.VetterPipeline`.
    time_days: Observation times in days.
    flux: Flux array.
    max_planets: Candidates to extract per call.
    num_repeats: Timed repeats after warmup.

  Returns:
    A dict with seconds per light curve and the point count.
  """
  pipeline.detect(time_days, flux, max_planets=max_planets)  # warm up
  start = time.perf_counter()
  for _ in range(num_repeats):
    pipeline.detect(time_days, flux, max_planets=max_planets)
  seconds = (time.perf_counter() - start) / num_repeats
  return {
      "seconds_per_light_curve": seconds,
      "num_points": int(len(time_days)),
      "max_planets": max_planets,
  }
