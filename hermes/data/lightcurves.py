"""Light-curve sources.

A physics-based injection source (always available) plus optional local-file
adapters for Kepler FITS and a preprocessed flux CSV.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np
from absl import logging

from hermes.data.catalog import TceCatalog
from hermes.numerics import transit


@dataclasses.dataclass(frozen=True)
class LightCurve:
  """A single light curve.

  Attributes:
    time_days: Observation times in days.
    flux: Normalised flux (approximately one out of transit).
    flux_err: Per-point flux uncertainties.
  """

  time_days: np.ndarray
  flux: np.ndarray
  flux_err: np.ndarray


class LightCurveStore(Protocol):
  """Maps a catalogue index to a light curve, or ``None`` if unavailable."""

  def get(self, index: int, catalog: TceCatalog) -> LightCurve | None:
    ...


def _transit_signal(
    time: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    depth: float,
    impact_parameter: float = 0.0,
) -> np.ndarray:
  """Evaluates a limb-darkened transit on ``time`` from catalogue quantities.

  The scaled semi-major axis is inferred from the catalogue duration via the
  small-angle duration relation ``T14 ~ (P/pi)*sqrt((1+k)^2-b^2)/(a/R*)``.
  """
  radius_ratio = float(np.sqrt(max(depth, 1e-8)))
  chord = np.sqrt(max((1.0 + radius_ratio) ** 2 - impact_parameter**2, 1e-6))
  a_over_rstar = period / np.pi * chord / max(duration, 1e-3)
  flux = transit.transit_light_curve(
      time,
      period_days=period,
      epoch_days=epoch,
      radius_ratio=radius_ratio,
      a_over_rstar=max(a_over_rstar, 1.5),
      impact_parameter=impact_parameter,
      limb_darkening=(0.4, 0.3),
      num_radii=64,
  )
  return np.asarray(flux)


@dataclasses.dataclass(frozen=True)
class InjectionStore:
  """Synthesises labelled light curves from the catalogue ephemeris.

  Attributes:
    baseline_days: Time span of each synthesised light curve.
    cadence_days: Sampling interval between points.
    noise_ppm: White-noise level in parts per million.
    seed: Base random seed (combined with the catalogue index for determinism).
  """

  baseline_days: float = 90.0
  cadence_days: float = 0.02043
  noise_ppm: float = 200.0
  seed: int = 0

  def get(self, index: int, catalog: TceCatalog) -> LightCurve:
    """Synthesises the light curve for catalogue row ``index``."""
    rng = np.random.RandomState(self.seed + int(index))
    num = int(self.baseline_days / self.cadence_days)
    time = np.arange(num) * self.cadence_days
    noise = self.noise_ppm * 1e-6
    flux = 1.0 + rng.normal(0.0, noise, num)

    period = _finite(catalog.period_days[index], rng.uniform(3.0, 30.0))
    epoch = _finite(catalog.epoch_days[index], rng.uniform(0.0, period))
    epoch = float(np.mod(epoch, period))
    duration = _finite(catalog.duration_days[index], 0.1 * period**(1.0 / 3.0))
    depth = _finite(catalog.depth_fraction[index], rng.uniform(1e-4, 1e-2))
    label = str(catalog.label[index])

    if label == "PC":
      flux *= _transit_signal(time, period, epoch, duration, depth)
    elif label == "AFP":
      flux *= self._false_positive(rng, time, period, epoch, duration, depth)
    else:  # NTP / UNK: variability and systematics, no coherent transit.
      flux *= self._stellar_variability(rng, time)

    flux_err = np.full(num, noise, dtype=np.float64)
    return LightCurve(time, flux.astype(np.float64), flux_err)

  def _false_positive(self, rng, time, period, epoch, duration, depth):
    """Injects an astrophysical false positive (eclipsing-binary-like)."""
    mode = rng.choice(("secondary", "odd_even", "grazing"))
    if mode == "secondary":
      primary = _transit_signal(time, period, epoch, duration, depth)
      secondary = _transit_signal(
          time, period, epoch + 0.5 * period, duration, 0.4 * depth
      )
      return primary * secondary
    if mode == "odd_even":
      odd = _transit_signal(time, 2.0 * period, epoch, duration, depth)
      even = _transit_signal(
          time, 2.0 * period, epoch + period, duration, 0.6 * depth
      )
      return odd * even
    return _transit_signal(  # grazing, V-shaped
        time, period, epoch, duration, 4.0 * depth, impact_parameter=0.98
    )

  def _stellar_variability(self, rng, time):
    """Injects correlated stellar variability with no transit."""
    amplitude = rng.uniform(1e-4, 2e-3)
    frequency = rng.uniform(0.05, 1.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return 1.0 + amplitude * np.sin(2.0 * np.pi * frequency * time + phase)


@dataclasses.dataclass(frozen=True)
class FitsLightCurveStore:
  """Reads locally stored Kepler long-cadence FITS files (optional astropy).

  Attributes:
    directory: Root directory containing ``kic_<id>`` light-curve FITS files.
  """

  directory: str

  def get(self, index: int, catalog: TceCatalog) -> LightCurve | None:
    """Loads and stitches the local FITS quarters for the row's host star."""
    try:
      from astropy.io import fits  # noqa: PLC0415 (optional dependency)
    except ImportError as error:  # pragma: no cover - exercised only with deps
      raise ImportError(
          "FitsLightCurveStore requires astropy; install the optional data "
          "dependencies or use InjectionStore."
      ) from error

    import glob
    import os

    kepid = int(catalog.kepid[index])
    # Files may live in a per-target subdirectory (as written by the downloader)
    # or flat in ``directory``; search both layouts.
    pattern_sub = os.path.join(self.directory, f"{kepid:09d}", "*_llc.fits")
    pattern_flat = os.path.join(self.directory, f"*{kepid:09d}*.fits")
    paths = sorted(set(glob.glob(pattern_sub)) | set(glob.glob(pattern_flat)))
    times, fluxes = [], []
    for path in paths:
      try:  # pragma: no cover - needs real FITS
        with fits.open(path, memmap=False) as hdul:
          data = hdul[1].data
          times.append(np.asarray(data["TIME"], dtype=np.float64))
          fluxes.append(np.asarray(data["PDCSAP_FLUX"], dtype=np.float64))
      except Exception as error:  # noqa: BLE001 - skip unreadable/corrupt files
        logging.warning("Skipping unreadable FITS %s: %s", path, error)
    if not times:
      return None
    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    flux = flux / np.median(flux)
    return LightCurve(time, flux, np.full_like(flux, np.std(flux)))


@dataclasses.dataclass(frozen=True)
class FluxCsvStore:
  """Reads a preprocessed one-row-per-star flux CSV (numpy only).

  The CSV is expected to hold one light curve per row of equally spaced flux
  samples; the optional first column may be a label and is ignored here.

  Attributes:
    csv_path: Path to the flux CSV file.
    cadence_days: Sampling interval assigned to the columns.
    has_label_column: Whether the first column is a label to drop.
  """

  csv_path: str
  cadence_days: float = 0.02043
  has_label_column: bool = True

  def __post_init__(self):
    table = np.loadtxt(self.csv_path, delimiter=",", skiprows=1)
    flux = table[:, 1:] if self.has_label_column else table
    object.__setattr__(self, "_flux", flux)

  def get(self, index: int, catalog: TceCatalog) -> LightCurve | None:
    """Returns the stored flux row matching the catalogue index."""
    flux_table = self._flux  # type: ignore[attr-defined]
    if index >= len(flux_table):
      return None
    flux = np.asarray(flux_table[index], dtype=np.float64)
    flux = flux / np.nanmedian(flux)
    time = np.arange(len(flux)) * self.cadence_days
    return LightCurve(time, flux, np.full_like(flux, np.nanstd(flux)))


def _finite(value: float, fallback: float) -> float:
  """Returns ``value`` if finite and positive, otherwise ``fallback``."""
  value = float(value)
  return value if np.isfinite(value) and value > 0.0 else float(fallback)
