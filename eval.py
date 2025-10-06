"""Evaluation script - following t5x pattern.

Root-level eval.py for model evaluation.
"""

import os
import sys
import platform
import time

# Disable pyarrow and configure threading on macOS to prevent mutex lock
if platform.system() == "Darwin":
    sys.path.insert(0, os.path.dirname(__file__))
    import disable_pyarrow

    # Set threading environment variables
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
    )
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import numpy as np
import jax.numpy as jnp

from absl import app, flags, logging
import gin
from flax import nnx

from hermes.model import MultiPlanetDetector
from hermes.trainer import Trainer
from hermes.data import load_data, create_data_loader


FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/default.gin", "Gin config file")
flags.DEFINE_string("checkpoint", None, "Checkpoint to evaluate", required=True)
flags.DEFINE_string("dataset_dir", "data/kepler_full", "Dataset directory")


def main(argv):
    """Main evaluation function."""
    del argv

    logging.set_verbosity(logging.INFO)
    # Import train to register gin-configurable, but skip unknown parameters
    # since eval doesn't use all training parameters
    gin.parse_config_file(FLAGS.config, skip_unknown=True)

    logging.info("=" * 60)
    logging.info("Model Evaluation")
    logging.info("=" * 60)

    # Model
    rngs = nnx.Rngs(42)
    model = MultiPlanetDetector(rngs=rngs)

    # Create trainer with a dummy run_dir (we won't save anything)
    trainer = Trainer(model=model, run_dir="eval_output")
    
    # Recreate checkpoint manager with the correct path
    import os
    import orbax.checkpoint as ocp
    trainer.checkpoint_dir = os.path.abspath(FLAGS.checkpoint)
    
    # Check if checkpoint directory exists
    if not os.path.exists(trainer.checkpoint_dir):
        logging.error(f"Checkpoint directory does not exist: {trainer.checkpoint_dir}")
        return
        
    trainer.checkpoint_manager = ocp.CheckpointManager(
        trainer.checkpoint_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=False)  # create=False for reading
    )
    
    # Log available checkpoints
    available_steps = trainer.checkpoint_manager.all_steps()
    logging.info(f"Looking for checkpoints in: {trainer.checkpoint_dir}")
    logging.info(f"Available checkpoint steps: {available_steps}")

    # Restore checkpoint
    restored = trainer.restore_checkpoint()

    if restored is None:
        logging.error(f"Failed to restore checkpoint from {trainer.checkpoint_dir}")
        if not available_steps:
            logging.error("No checkpoints found in the directory!")
        return

    logging.info(f"Successfully loaded checkpoint from step {restored}")

    # Data
    data_source = FLAGS.dataset_dir or "synthetic"
    datasets = load_data(data_source=data_source, batch_size=32)

    # ========== Full Test Set Evaluation ==========
    logging.info("\n" + "=" * 60)
    logging.info("Full Test Set Evaluation")
    logging.info("=" * 60)

    trainer.eval_metrics.reset()
    num_batches = 0
    start_time = time.time()

    for batch in create_data_loader(datasets["test"]):
        trainer.eval_step_fn(batch)
        num_batches += 1

    eval_time = time.time() - start_time
    metrics = trainer.eval_metrics.compute()

    logging.info(f"Test Loss: {metrics['loss']:.4f}")
    logging.info(f"Test Accuracy: {metrics['accuracy']:.3%}")
    logging.info(f"Evaluated {num_batches} batches in {eval_time:.2f}s")
    logging.info(f"Average time per batch: {eval_time / num_batches:.3f}s")

    # ========== Single Prediction Timing ==========
    logging.info("\n" + "=" * 60)
    logging.info("Single Light Curve Prediction Timing")
    logging.info("=" * 60)

    # Get a single test example
    test_batch = next(iter(create_data_loader(datasets["test"])))
    single_light_curve = test_batch["light_curve"][0:1]  # Take first example

    # Warm-up run (JIT compilation)
    _ = model(single_light_curve)

    # Time multiple runs for accurate measurement
    num_runs = 100
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        predictions = model(single_light_curve)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)

    logging.info(f"Single prediction timing (over {num_runs} runs):")
    logging.info(f"  Mean:   {mean_time:.2f} ms")
    logging.info(f"  Median: {median_time:.2f} ms")
    logging.info(f"  Std:    {std_time:.2f} ms")
    logging.info(f"  Min:    {min_time:.2f} ms")
    logging.info(f"  Max:    {max_time:.2f} ms")

    # Show prediction details for the single example
    planet_count_probs = predictions["planet_count_probs"][0]
    predicted_count = jnp.argmax(planet_count_probs)
    confidence = float(planet_count_probs[predicted_count])

    logging.info(f"\nPrediction for single light curve:")
    logging.info(f"  Predicted planet count: {predicted_count}")
    logging.info(f"  Confidence: {confidence:.2%}")
    logging.info(f"  Planet count probabilities:")
    for i, prob in enumerate(planet_count_probs):
        logging.info(f"    {i} planets: {float(prob):.2%}")

    logging.info("\n" + "=" * 60)
    logging.info("Evaluation Complete")
    logging.info("=" * 60)

    trainer.close()


if __name__ == "__main__":
    app.run(main)
