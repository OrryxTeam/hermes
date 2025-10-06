"""Inference script - following t5x pattern.

Root-level infer.py for making predictions.
"""

import json
from pathlib import Path

from absl import app, flags, logging
import pandas as pd
from flax import nnx

from hermes.model import MultiPlanetDetector
from hermes.predictor import ExoplanetPredictor


FLAGS = flags.FLAGS
flags.DEFINE_string('input_csv', None, 'Input CSV with light curve', required=True)
flags.DEFINE_string('output', 'predictions.json', 'Output file')
flags.DEFINE_string('checkpoint', None, 'Checkpoint path')


def main(argv):
  """Main inference function."""
  del argv
  
  logging.set_verbosity(logging.INFO)
  
  # Load light curve
  df = pd.read_csv(FLAGS.input_csv)
  time = df['time'].values
  flux = df['flux'].values
  flux_err = df.get('flux_err', pd.Series()).values if 'flux_err' in df else None
  
  logging.info(f'Loaded {len(flux)} data points from {FLAGS.input_csv}')
  
  # Initialize model
  rngs = nnx.Rngs(42)
  model = MultiPlanetDetector(rngs=rngs)
  
  if FLAGS.checkpoint:
    logging.info(f'Loading checkpoint: {FLAGS.checkpoint}')
    # TODO: Implement checkpoint loading
  else:
    logging.warning('No checkpoint - using random weights')
  
  # Create predictor
  predictor = ExoplanetPredictor(model)
  
  # Run prediction (combines NN + BLS + physics)
  logging.info('Running prediction...')
  results = predictor.predict(time, flux, flux_err, Path(FLAGS.input_csv).stem)
  
  # Save
  with open(FLAGS.output, 'w') as f:
    json.dump(results, f, indent=2)
  
  # Display
  logging.info(f'\n{"="*50}')
  logging.info(f'Predicted {results["predicted_planet_count"]} planet(s)')
  logging.info(f'Confidence: {results["planet_count_confidence"]:.3f}')
  
  for p in results['planets']:
    logging.info(f'\nPlanet {p["planet_id"]}:')
    logging.info(f'  Period: {p["orbital_period_days"]:.2f} days (from BLS)')
    logging.info(f'  Radius: {p["planet_radius_earth"]:.2f} R_earth (from depth)')
    logging.info(f'  Distance: {p["orbital_distance_au"]:.3f} AU (Kepler\'s 3rd law)')
    logging.info(f'  Temp: {p["equilibrium_temperature_k"]:.0f} K')
    logging.info(f'  SNR: {p["signal_to_noise"]:.1f}')
  
  logging.info(f'\nSaved to {FLAGS.output}')


if __name__ == '__main__':
  app.run(main)
