"""Reader for the Kepler DR25 TCE and KOI cumulative catalogues.

Parses the NASA Exoplanet Archive CSV (standard library only) into a
struct-of-arrays in days / fractional depth / solar units.
"""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Sequence

import numpy as np

# Source-column names in the DR25 TCE CSV.
_KEPID = "kepid"
_PLANET_NUM = "tce_plnt_num"
_PERIOD = "tce_period"  # days
_EPOCH = "tce_time0bk"  # days (BKJD = BJD - 2,454,833)
_DURATION = "tce_duration"  # hours
_DEPTH = "tce_depth"  # parts per million
_LABEL = "av_training_set"
_TEFF = "tce_steff"  # Kelvin
_SRADIUS = "tce_sradius"  # solar radii
_SLOGG = "tce_slogg"  # log10(cm / s^2)
_SDENS = "tce_sdens"  # g / cm^3


@dataclasses.dataclass(frozen=True)
class TceCatalog:
  """Struct-of-arrays view of a TCE catalogue.

  All arrays share the same length (one entry per TCE). Units are days for time
  quantities, a dimensionless fraction for depth, and solar/SI units for the
  stellar parameters. Missing numeric values are represented as ``NaN``.

  Attributes:
    kepid: Kepler identification number of the host star.
    planet_num: TCE planet number within the star.
    period_days: Orbital period in days.
    epoch_days: Transit epoch in days (BKJD).
    duration_days: Transit duration in days.
    depth_fraction: Fractional transit depth.
    label: Training-set label (``PC``/``AFP``/``NTP``/``UNK``).
    stellar_teff_k: Stellar effective temperature in Kelvin.
    stellar_radius_solar: Stellar radius in solar radii.
    stellar_logg_cgs: Stellar surface gravity, ``log10(cm / s^2)``.
    stellar_density_cgs: Stellar mean density in g / cm^3.
  """

  kepid: np.ndarray
  planet_num: np.ndarray
  period_days: np.ndarray
  epoch_days: np.ndarray
  duration_days: np.ndarray
  depth_fraction: np.ndarray
  label: np.ndarray
  stellar_teff_k: np.ndarray
  stellar_radius_solar: np.ndarray
  stellar_logg_cgs: np.ndarray
  stellar_density_cgs: np.ndarray

  def __len__(self) -> int:
    return len(self.kepid)

  def filter(self, mask: np.ndarray) -> "TceCatalog":
    """Returns a new catalogue keeping only the rows where ``mask`` is true."""
    return TceCatalog(
        **{
            f.name: getattr(self, f.name)[mask]
            for f in dataclasses.fields(self)
        }
    )

  def with_labels(self, labels: Sequence[str]) -> "TceCatalog":
    """Returns the subset whose label is in ``labels``."""
    keep = np.isin(self.label, np.asarray(labels))
    return self.filter(keep)


def _to_float(value: str) -> float:
  """Parses a CSV cell to ``float``, mapping blanks to ``NaN``."""
  if value is None or value.strip() == "":
    return float("nan")
  try:
    return float(value)
  except ValueError:
    return float("nan")


def load_tce_catalog(csv_path: str) -> TceCatalog:
  """Loads and unit-converts a DR25 TCE catalogue CSV.

  Args:
    csv_path: Path to the local DR25 TCE CSV file.

  Returns:
    A `TceCatalog` with one entry per TCE row.

  Raises:
    KeyError: If a required column is absent from the CSV header.
  """
  kepid, planet_num, label = [], [], []
  period, epoch, duration, depth = [], [], [], []
  teff, sradius, slogg, sdens = [], [], [], []

  with open(csv_path, newline="") as handle:
    reader = csv.DictReader(_skip_comment_lines(handle))
    required = {_KEPID, _PERIOD, _EPOCH, _DURATION, _LABEL}
    missing = required - set(reader.fieldnames or [])
    if missing:
      raise KeyError(f"TCE catalogue missing columns: {sorted(missing)}")

    for row in reader:
      kepid.append(int(_to_float(row[_KEPID])))
      planet_num.append(int(_to_float(row.get(_PLANET_NUM, "1")) or 1))
      label.append((row.get(_LABEL) or "UNK").strip() or "UNK")
      period.append(_to_float(row[_PERIOD]))
      epoch.append(_to_float(row[_EPOCH]))
      duration.append(_to_float(row[_DURATION]) / 24.0)  # hours -> days
      depth.append(_to_float(row.get(_DEPTH, "")) * 1e-6)  # ppm -> fraction
      teff.append(_to_float(row.get(_TEFF, "")))
      sradius.append(_to_float(row.get(_SRADIUS, "")))
      slogg.append(_to_float(row.get(_SLOGG, "")))
      sdens.append(_to_float(row.get(_SDENS, "")))

  return TceCatalog(
      kepid=np.asarray(kepid, dtype=np.int64),
      planet_num=np.asarray(planet_num, dtype=np.int32),
      period_days=np.asarray(period, dtype=np.float64),
      epoch_days=np.asarray(epoch, dtype=np.float64),
      duration_days=np.asarray(duration, dtype=np.float64),
      depth_fraction=np.asarray(depth, dtype=np.float64),
      label=np.asarray(label, dtype=object).astype(str),
      stellar_teff_k=np.asarray(teff, dtype=np.float64),
      stellar_radius_solar=np.asarray(sradius, dtype=np.float64),
      stellar_logg_cgs=np.asarray(slogg, dtype=np.float64),
      stellar_density_cgs=np.asarray(sdens, dtype=np.float64),
  )


#: KOI disposition -> training label. CONFIRMED/CANDIDATE are positives (PC);
#: FALSE POSITIVE maps to an astrophysical false positive (AFP).
_KOI_DISPOSITION_TO_LABEL = {
    "CONFIRMED": "PC",
    "CANDIDATE": "PC",
    "FALSE POSITIVE": "AFP",
}


def load_koi_catalog(csv_path: str) -> TceCatalog:
  """Loads a Kepler Objects of Interest (KOI) cumulative CSV.

  The KOI table is an alternative label/ephemeris source to the DR25 TCE table
  (same Kepler data, different NASA Exoplanet Archive table). Disposition is
  mapped to the same ``PC``/``AFP`` labels HERMES uses, so the resulting
  `TceCatalog` is a drop-in for `load_tce_catalog`. Download it with::

      curl -o dataset/koi.csv "https://exoplanetarchive.ipac.caltech.edu/TAP/\\
      sync?query=select+*+from+cumulative&format=csv"

  Args:
    csv_path: Path to the local KOI cumulative CSV file.

  Returns:
    A `TceCatalog`, one entry per KOI.

  Raises:
    KeyError: If a required column is absent from the CSV header.
  """
  rows: dict[str, list] = {key: [] for key in (
      "kepid", "plnt", "label", "period", "epoch", "duration", "depth",
      "teff", "sradius", "slogg", "sdens")}

  with open(csv_path, newline="") as handle:
    reader = csv.DictReader(_skip_comment_lines(handle))
    required = {"kepid", "koi_disposition", "koi_period", "koi_time0bk"}
    missing = required - set(reader.fieldnames or [])
    if missing:
      raise KeyError(f"KOI catalogue missing columns: {sorted(missing)}")

    for row in reader:
      disposition = (row.get("koi_disposition") or "").strip().upper()
      rows["label"].append(_KOI_DISPOSITION_TO_LABEL.get(disposition, "UNK"))
      rows["kepid"].append(int(_to_float(row["kepid"])))
      rows["plnt"].append(1)
      rows["period"].append(_to_float(row["koi_period"]))
      rows["epoch"].append(_to_float(row["koi_time0bk"]))
      rows["duration"].append(_to_float(row.get("koi_duration", "")) / 24.0)
      rows["depth"].append(_to_float(row.get("koi_depth", "")) * 1e-6)
      rows["teff"].append(_to_float(row.get("koi_steff", "")))
      rows["sradius"].append(_to_float(row.get("koi_srad", "")))
      rows["slogg"].append(_to_float(row.get("koi_slogg", "")))
      rows["sdens"].append(_to_float(row.get("koi_srho", "")))

  return TceCatalog(
      kepid=np.asarray(rows["kepid"], dtype=np.int64),
      planet_num=np.asarray(rows["plnt"], dtype=np.int32),
      period_days=np.asarray(rows["period"], dtype=np.float64),
      epoch_days=np.asarray(rows["epoch"], dtype=np.float64),
      duration_days=np.asarray(rows["duration"], dtype=np.float64),
      depth_fraction=np.asarray(rows["depth"], dtype=np.float64),
      label=np.asarray(rows["label"], dtype=object).astype(str),
      stellar_teff_k=np.asarray(rows["teff"], dtype=np.float64),
      stellar_radius_solar=np.asarray(rows["sradius"], dtype=np.float64),
      stellar_logg_cgs=np.asarray(rows["slogg"], dtype=np.float64),
      stellar_density_cgs=np.asarray(rows["sdens"], dtype=np.float64),
  )


def load_catalog(csv_path: str, catalog_format: str = "tce") -> TceCatalog:
  """Loads a catalogue in either the ``tce`` or ``koi`` format.

  Args:
    csv_path: Path to the local catalogue CSV.
    catalog_format: ``"tce"`` for the DR25 TCE table, ``"koi"`` for the KOI
      cumulative table.

  Returns:
    A `TceCatalog`.

  Raises:
    ValueError: If ``catalog_format`` is not recognised.
  """
  if catalog_format == "tce":
    return load_tce_catalog(csv_path)
  if catalog_format == "koi":
    return load_koi_catalog(csv_path)
  raise ValueError(f"Unknown catalog_format: {catalog_format!r}")


def _skip_comment_lines(handle):
  """Yields CSV lines, dropping the leading ``#`` comment block if present."""
  for line in handle:
    if not line.startswith("#"):
      yield line
