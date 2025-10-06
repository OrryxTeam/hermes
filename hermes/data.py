"""Data loading with TensorFlow Datasets and numpy fallback for macOS."""

import numpy as np
import gin
from pathlib import Path
from typing import Dict, Iterator
from absl import logging

# Try to import TensorFlow, fallback to numpy if threading issues
try:
  import os
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
  import tensorflow as tf
  tf.config.set_visible_devices([], 'GPU')
  TF_AVAILABLE = True
except Exception as e:
  logging.warning(f'TensorFlow not available: {e}')
  TF_AVAILABLE = False


def load_npz_data(npz_path: str) -> Dict:
  """Loads NPZ file into dict."""
  data = np.load(npz_path)
  return {
      'light_curve': data['global_views'],
      'planet_count': data['num_planets'].astype(np.int32),
      'planet_exists': data['planet_exists']
  }


class NumpyDataLoader:
  """Simple numpy-based data loader (fallback for TF issues)."""
  
  def __init__(self, data_dict: Dict, batch_size: int, shuffle: bool, seed: int, drop_remainder: bool = True):
    self.data = data_dict
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_remainder = drop_remainder
    self.rng = np.random.RandomState(seed)
    self.n_samples = len(data_dict['light_curve'])
    
    # Log configuration for debugging
    if not drop_remainder and self.n_samples < batch_size:
      logging.info(f'  Small dataset detected: {self.n_samples} examples < {batch_size} batch size')
      logging.info(f'  Will create 1 batch with {self.n_samples} examples (drop_remainder={drop_remainder})')
  
  def num_batches(self):
    """Returns the number of batches that will be yielded."""
    if self.drop_remainder:
      return self.n_samples // self.batch_size
    else:
      return (self.n_samples + self.batch_size - 1) // self.batch_size  # Ceiling division
  
  def __iter__(self):
    indices = np.arange(self.n_samples)
    if self.shuffle:
      self.rng.shuffle(indices)
    
    batches_yielded = 0
    for i in range(0, self.n_samples, self.batch_size):
      batch_indices = indices[i:i + self.batch_size]
      
      # Handle incomplete batches
      if len(batch_indices) < self.batch_size:
        if self.drop_remainder:
          continue  # Drop incomplete batch
        # Otherwise, yield the incomplete batch
      
      batches_yielded += 1
      yield {
          'light_curve': self.data['light_curve'][batch_indices],
          'planet_count': self.data['planet_count'][batch_indices],
          'planet_exists': self.data['planet_exists'][batch_indices]
      }
    
    # Warn if no batches were yielded when they should have been
    if batches_yielded == 0 and not self.drop_remainder and self.n_samples > 0:
      logging.error(f'ERROR: No batches yielded despite drop_remainder=False and {self.n_samples} samples!')


@gin.configurable
class ExoplanetTFDataPipeline:
  """Data pipeline using TensorFlow Datasets or numpy fallback."""
  
  def __init__(self, batch_size=32, shuffle_buffer_size=10000,
               prefetch_buffer_size=-1, cache=True, seed=42, max_planets=5):
    self.batch_size = batch_size
    self.shuffle_buffer_size = shuffle_buffer_size
    self.prefetch_buffer_size = prefetch_buffer_size
    self.cache = cache
    self.seed = seed
    self.max_planets = max_planets
    
    if TF_AVAILABLE:
      tf.random.set_seed(seed)
  
  def create_dataset_from_npz(self, npz_path: str, shuffle: bool, drop_remainder: bool = True) -> Iterator:
    """Creates dataset from NPZ file.
    
    Uses numpy-based loader to avoid TensorFlow threading issues on macOS.
    
    Args:
      npz_path: Path to NPZ file.
      shuffle: Whether to shuffle.
      drop_remainder: Whether to drop incomplete batches.
      
    Returns:
      Data iterator.
    """
    data = load_npz_data(npz_path)
    return NumpyDataLoader(data, self.batch_size, shuffle, self.seed, drop_remainder)
  
  def create_dataset(self, split: str, shuffle: bool, drop_remainder: bool = True) -> Iterator:
    """Creates synthetic dataset.
    
    Args:
      split: 'train', 'validation', or 'test'.
      shuffle: Whether to shuffle.
      drop_remainder: Whether to drop incomplete batches.
      
    Returns:
      Data iterator.
    """
    # Generate synthetic data
    rng = np.random.RandomState(self.seed + hash(split) % 1000)
    n = {'train': 8000, 'validation': 1000, 'test': 1000}[split]
    
    light_curves = []
    planet_counts = []
    planet_exists_list = []
    
    for _ in range(n):
      flux = np.ones(2048) + rng.normal(0, 0.001, 2048)
      n_planets = rng.randint(0, 6)
      
      # Add transits
      for i in range(n_planets):
        period = rng.uniform(5, 50)
        depth = rng.uniform(0.001, 0.02)
        phase = (np.arange(2048) % int(period * 100)) / (period * 100)
        in_transit = phase < 0.05
        flux[in_transit] *= (1 - depth)
      
      planet_exists = np.zeros(self.max_planets, dtype=np.float32)
      planet_exists[:n_planets] = 1.0
      
      light_curves.append(flux)
      planet_counts.append(n_planets)
      planet_exists_list.append(planet_exists)
    
    data = {
        'light_curve': np.array(light_curves, dtype=np.float32),
        'planet_count': np.array(planet_counts, dtype=np.int32),
        'planet_exists': np.array(planet_exists_list, dtype=np.float32)
    }
    
    return NumpyDataLoader(data, self.batch_size, shuffle, self.seed, drop_remainder)


@gin.configurable
def load_data(data_source='synthetic', batch_size=32):
  """Loads train/val/test datasets.
  
  Uses numpy-based loading to avoid TensorFlow threading issues.
  Still configurable via gin-config.
  
  Args:
    data_source: 'synthetic', 'koi', or path to dataset directory.
    batch_size: Batch size.
    
  Returns:
    Dict with 'train', 'validation', 'test' iterators.
  """
  # Check if we should use KOI CSV + NASA API approach
  if data_source == 'koi':
    koi_csv = Path('dataset/koi.csv')
    if koi_csv.exists():
      logging.info('Using KOI CSV + NASA API for real Kepler data')
      try:
        from hermes.koi_data_loader import load_koi_data
        return load_koi_data(
            koi_csv_path=str(koi_csv),
            batch_size=batch_size,
            disposition_filter=['CONFIRMED']  # Only use confirmed planets for training
        )
      except ImportError as e:
        logging.warning(f'Could not import KOI data loader: {e}')
        logging.info('Falling back to synthetic data')
        data_source = 'synthetic'
    else:
      logging.warning('KOI CSV not found, falling back to synthetic data')
      data_source = 'synthetic'
  
  pipeline = ExoplanetTFDataPipeline(batch_size=batch_size)
  
  if data_source != 'synthetic':
    data_path = Path(data_source)
    if data_path.exists():
      train_npz = data_path / 'train' / 'train_data.npz'
      val_npz = data_path / 'validation' / 'validation_data.npz'
      test_npz = data_path / 'test' / 'test_data.npz'
      
      if train_npz.exists():
        logging.info(f'Loading dataset from {data_source}')
        return {
            'train': pipeline.create_dataset_from_npz(str(train_npz), shuffle=True, drop_remainder=True),
            'validation': pipeline.create_dataset_from_npz(str(val_npz), shuffle=False, drop_remainder=False),
            'test': pipeline.create_dataset_from_npz(str(test_npz), shuffle=False, drop_remainder=False)
        }
  
  logging.info('Using synthetic dataset')
  return {
      'train': pipeline.create_dataset('train', shuffle=True, drop_remainder=True),
      'validation': pipeline.create_dataset('validation', shuffle=False, drop_remainder=False),
      'test': pipeline.create_dataset('test', shuffle=False, drop_remainder=False)
  }


def create_data_loader(dataset_or_iterator):
  """Returns iterator (passthrough for numpy loaders).
  
  Compatible with both TensorFlow Datasets and numpy iterators.
  """
  if hasattr(dataset_or_iterator, 'as_numpy_iterator'):
    return dataset_or_iterator.as_numpy_iterator()
  return dataset_or_iterator