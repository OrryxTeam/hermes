"""Build the sharded view-tensor dataset from a catalogue and light-curve store.

Run with `python -m hermes.build_dataset --config=hermes/configs/default.py`.
"""

import jax

# The offline build folds large Kepler BKJD timestamps (~1500 days); enable
# double precision so the phase computation is not truncated to float32.
jax.config.update("jax_enable_x64", True)

from absl import app, logging  # noqa: E402
from ml_collections import config_flags  # noqa: E402

from hermes.data import build_dataset  # noqa: E402

_CONFIG = config_flags.DEFINE_config_file("config", "hermes/configs/default.py")


def main(_) -> None:
  logging.set_verbosity(logging.INFO)
  meta = build_dataset.build(_CONFIG.value)
  logging.info("Built dataset: %s", meta["counts"])


if __name__ == "__main__":
  app.run(main)
