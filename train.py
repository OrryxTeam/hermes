"""Training script - following t5x pattern.

Root-level train.py with gin-config for all parameters.
"""

import os
import sys
import platform
import time

# Disable pyarrow BEFORE any other imports on macOS
if platform.system() == 'Darwin':
    import disable_pyarrow
from datetime import datetime
from pathlib import Path

# Configure threading for macOS to prevent mutex lock errors
# These must be set before importing JAX
if platform.system() == 'Darwin':  # macOS-specific configuration
    # CRITICAL: Prevent pyarrow from being imported (causes mutex lock on macOS)
    # This must be set BEFORE any imports that might trigger pandas/pyarrow
    os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'  # Prevents pyarrow import in pandas
    os.environ['PANDAS_TESTING_MODE'] = '0'  # Additional safety
    
    # Set threading limits to prevent mutex conflicts
    os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP conflicts
    os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL conflicts  
    os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent OpenBLAS conflicts
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # Prevent Accelerate framework conflicts
    os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Prevent NumExpr conflicts
    
    # JAX-specific settings for macOS
    os.environ['JAX_PLATFORMS'] = 'cpu'  # Use CPU backend on macOS
    os.environ['JAX_ENABLE_X64'] = 'True'  # Enable 64-bit precision
    os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1'
    
    # Disable fork safety check on macOS
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    
    # Use spawn instead of fork for multiprocessing
    # Commenting out to test if this causes mutex lock
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)

# Note: TensorFlow is imported inside hermes.data with proper configuration
# to avoid macOS threading issues

from absl import app, flags, logging
import gin
import jax
import numpy as np
from flax import nnx

from hermes.model import MultiPlanetDetector
from hermes.trainer import Trainer
from hermes.data import load_data, create_data_loader


FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'configs/default.gin', 'Gin config file')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory')


@gin.configurable('train')
def train(num_epochs=100, batch_size=32, eval_every_steps=1,
          checkpoint_every_steps=500, log_every_steps=10, seed=42):
  """Main training loop.
  
  Args:
    num_epochs: Number of epochs.
    batch_size: Batch size.
    eval_every_steps: Eval frequency.
    checkpoint_every_steps: Checkpoint frequency.
    log_every_steps: Logging frequency.
    seed: Random seed.
  """
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  run_dir = Path('outputs') / f'run_{timestamp}'
  run_dir.mkdir(parents=True, exist_ok=True)
  
  with open(run_dir / 'config.gin', 'w') as f:
    f.write(gin.operative_config_str())
  
  logging.info(f'Run: {run_dir}')
  logging.info(f'JAX devices: {jax.devices()}')
  
  # Model
  rngs = nnx.Rngs(seed)
  model = MultiPlanetDetector(rngs=rngs)
  
  # Trainer
  trainer = Trainer(model=model, run_dir=str(run_dir))
  
  # Data
  data_source = FLAGS.dataset_dir or 'synthetic'
  datasets = load_data(data_source=data_source, batch_size=batch_size)
  
  # Calculate dataset sizes and batch counts
  train_size = len(datasets['train'].data['light_curve']) if hasattr(datasets['train'], 'data') else 'Unknown'
  val_size = len(datasets['validation'].data['light_curve']) if hasattr(datasets['validation'], 'data') else 'Unknown'
  test_size = len(datasets['test'].data['light_curve']) if hasattr(datasets['test'], 'data') else 'Unknown'
  
  # Get actual batch counts from the data loaders
  train_batches = datasets['train'].num_batches() if hasattr(datasets['train'], 'num_batches') else 'Unknown'
  val_batches = datasets['validation'].num_batches() if hasattr(datasets['validation'], 'num_batches') else 'Unknown'
  test_batches = datasets['test'].num_batches() if hasattr(datasets['test'], 'num_batches') else 'Unknown'
  
  # Calculate dropped examples
  if isinstance(train_size, int) and isinstance(train_batches, int):
    train_dropped = train_size - (train_batches * batch_size)
    train_dropped = max(0, train_dropped)  # Can't be negative
  else:
    train_dropped = 0
  
  logging.info(f'\n{"="*60}')
  logging.info(f'Dataset Information')
  logging.info(f'{"="*60}')
  logging.info(f'Training:   {train_size:>5} examples → {train_batches:>3} batches (drop_remainder=True)')
  logging.info(f'Validation: {val_size:>5} examples → {val_batches:>3} batches (drop_remainder=False)')
  logging.info(f'Test:       {test_size:>5} examples → {test_batches:>3} batches (drop_remainder=False)')
  logging.info(f'Batch size: {batch_size}')
  
  # Train
  global_step = 0
  best_val_loss = float('inf')
  
  for epoch in range(num_epochs):
    epoch_start_time = time.time()
    logging.info(f'\n{"="*60}')
    logging.info(f'Epoch {epoch+1}/{num_epochs}')
    logging.info(f'{"="*60}')
    
    # Training phase
    trainer.train_metrics.reset()
    train_iter = create_data_loader(datasets['train'])
    epoch_train_steps = 0
    
    for batch_idx, batch in enumerate(train_iter):
      loss = trainer.train_step_fn(batch)
      global_step += 1
      epoch_train_steps += 1
      
      # Log training metrics every N steps
      if global_step % log_every_steps == 0:
        m = trainer.train_metrics.compute()
        logging.info(f'  Step {global_step} (batch {batch_idx+1}): '
                    f'train_loss={m["loss"]:.4f}, train_acc={m["accuracy"]:.3f}')
    
    # Compute final training metrics for the epoch
    train_metrics = trainer.train_metrics.compute()
    train_loss = train_metrics["loss"]
    train_acc = train_metrics["accuracy"]
    
    # Validation phase - run on entire validation set
    logging.info('  Running validation...')
    trainer.eval_metrics.reset()
    val_iter = create_data_loader(datasets['validation'])
    val_steps = 0
    val_examples_processed = 0
    
    for val_batch in val_iter:
      trainer.eval_step_fn(val_batch)
      val_steps += 1
      val_examples_processed += len(val_batch['light_curve'])
    
    # Compute validation metrics
    if val_steps > 0:
      val_metrics = trainer.eval_metrics.compute()
      val_loss = val_metrics["loss"]
      val_acc = val_metrics["accuracy"]
      logging.info(f'  Validated on {val_examples_processed} examples in {val_steps} batches')
    else:
      logging.warning('  WARNING: No validation batches were processed!')
      logging.warning('  This should not happen - check data loader configuration.')
      val_loss = float('nan')
      val_acc = float('nan')
    
    # Calculate epoch duration
    epoch_duration = time.time() - epoch_start_time
    
    # Calculate total examples processed
    train_examples = epoch_train_steps * batch_size
    val_examples = val_steps * batch_size
    
    # Log epoch summary
    logging.info(f'\n  Epoch {epoch+1} Summary:')
    logging.info(f'  {"─"*40}')
    logging.info(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3%}')
    logging.info(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.3%}')
    logging.info(f'  Time: {epoch_duration:.2f}s ({train_examples/epoch_duration:.0f} examples/sec)')
    
    # Save checkpoint if validation improved (skip if NaN)
    if not np.isnan(val_loss) and val_loss < best_val_loss:
      best_val_loss = val_loss
      logging.info(f'  ✓ New best validation loss! Saving checkpoint...')
      trainer.save_checkpoint()
    
    # Also save regular checkpoints at intervals
    elif global_step % checkpoint_every_steps == 0:
      trainer.save_checkpoint()
  
  # Save final checkpoint at end of training
  logging.info(f'\n{"="*60}')
  logging.info(f'Training Complete!')
  logging.info(f'Best validation loss: {best_val_loss:.4f}')
  logging.info(f'Saving final checkpoint...')
  trainer.save_checkpoint()
  logging.info(f'{"="*60}')
  
  trainer.close()


def main(argv):
  del argv
  logging.set_verbosity(logging.INFO)
  
  # Import all gin-configurable functions before parsing config
  # This ensures gin can find them
  gin.parse_config_file(FLAGS.config, skip_unknown=False)
  
  # Call the train function with gin-configured parameters
  train()


if __name__ == '__main__':
  app.run(main)