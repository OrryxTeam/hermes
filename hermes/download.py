"""Bulk-download the Kepler light curves referenced by a catalogue.

Downloads each host star's long-cadence FITS from the MAST archive into
``data.light_curve_dir`` (resumable; cached files are skipped). Run with
`python -m hermes.download --config=hermes/configs/default.py`.
"""

from absl import app, flags, logging
from ml_collections import config_flags

from hermes.data import catalog as catalog_lib
from hermes.data import download

_CONFIG = config_flags.DEFINE_config_file("config", "hermes/configs/default.py")
_MAX_STARS = flags.DEFINE_integer(
    "max_stars", None, "Limit the number of stars."
)
_WORKERS = flags.DEFINE_integer("workers", 4, "Concurrent download workers.")


def main(_) -> None:
  logging.set_verbosity(logging.INFO)
  config = _CONFIG.value
  catalog = catalog_lib.load_catalog(
      config.data.catalog_csv, config.data.catalog_format
  ).with_labels(config.data.labels)

  kepids = list(dict.fromkeys(int(k) for k in catalog.kepid))
  if _MAX_STARS.value is not None:
    kepids = kepids[: _MAX_STARS.value]

  logging.info("Downloading light curves for %d stars -> %s",
               len(kepids), config.data.light_curve_dir)
  results = download.download_targets(
      kepids, config.data.light_curve_dir, max_workers=_WORKERS.value
  )
  obtained = sum(1 for paths in results.values() if paths)
  logging.info("Obtained FITS for %d / %d stars", obtained, len(kepids))


if __name__ == "__main__":
  app.run(main)
