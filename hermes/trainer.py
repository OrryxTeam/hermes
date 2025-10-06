"""Trainer for exoplanet detection model."""

import os
import jax
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import args as ocp_args
from flax import nnx
from tensorboardX import SummaryWriter
from absl import logging
import gin


@gin.configurable
def create_optimizer(learning_rate: float = 0.001, weight_decay: float = 1e-5,
                    gradient_clip_norm: float = 1.0) -> optax.GradientTransformation:
  """Creates optimizer with warmup and clipping."""
  return optax.chain(
      optax.clip_by_global_norm(gradient_clip_norm),
      optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
  )


def compute_loss(model, batch, loss_weights):
  """Computes loss for planet detection.
  
  Args:
    model: Model instance.
    batch: Batch dict with 'light_curve', 'planet_count', 'planet_exists'.
    loss_weights: Loss weight dict.
    
  Returns:
    Tuple of (total_loss, aux_info).
  """
  outputs = model(batch['light_curve'])
  
  # Planet count loss
  count_loss = optax.softmax_cross_entropy_with_integer_labels(
      outputs['planet_count_logits'], batch['planet_count']
  ).mean()
  
  # Existence losses
  existence_losses = []
  if 'planet_exists' in batch:
    for i, existence_pred in enumerate(outputs['existence_probs']):
      target = batch['planet_exists'][:, i:i+1]
      ex_loss = optax.sigmoid_binary_cross_entropy(existence_pred, target).mean()
      existence_losses.append(ex_loss)
  
  total_loss = (
      count_loss * loss_weights.get('planet_count', 1.5) +
      sum(existence_losses) * loss_weights.get('planet_existence', 2.0) / max(len(existence_losses), 1)
  )
  
  aux = {
      'planet_count_logits': outputs['planet_count_logits'],
      'planet_count_labels': batch['planet_count']
  }
  
  return total_loss, aux


# Temporarily disable JIT on macOS to prevent mutex issues
import platform
if platform.system() == 'Darwin':
    def train_step(model, optimizer, metrics, batch, loss_weights):
      """Single training step (without JIT on macOS)."""
      (loss, aux), grads = nnx.value_and_grad(compute_loss, has_aux=True)(
          model, batch, loss_weights
      )
      optimizer.update(model=model, grads=grads)  # Flax 0.11.0+ API
      metrics.update(loss=loss, logits=aux['planet_count_logits'], 
                     labels=aux['planet_count_labels'])
      return loss, grads
    
    def eval_step(model, metrics, batch, loss_weights):
      """Single eval step (without JIT on macOS)."""
      loss, aux = compute_loss(model, batch, loss_weights)
      metrics.update(loss=loss, logits=aux['planet_count_logits'], 
                     labels=aux['planet_count_labels'])
      return loss
else:
    @nnx.jit
    def train_step(model, optimizer, metrics, batch, loss_weights):
      """Single training step (JIT-compiled pure function)."""
      (loss, aux), grads = nnx.value_and_grad(compute_loss, has_aux=True)(
          model, batch, loss_weights
      )
      optimizer.update(model=model, grads=grads)  # Flax 0.11.0+ API
      metrics.update(loss=loss, logits=aux['planet_count_logits'], 
                     labels=aux['planet_count_labels'])
      return loss, grads
    
    @nnx.jit
    def eval_step(model, metrics, batch, loss_weights):
      """Single eval step (JIT-compiled pure function)."""
      loss, aux = compute_loss(model, batch, loss_weights)
      metrics.update(loss=loss, logits=aux['planet_count_logits'], 
                     labels=aux['planet_count_labels'])
      return loss


@gin.configurable
class Trainer:
  """Manages training loop, checkpointing, and logging."""
  
  def __init__(self, model, run_dir, learning_rate=0.001, weight_decay=1e-5,
               gradient_clip_norm=1.0, loss_weight_planet_count=1.5, 
               loss_weight_planet_existence=2.0):
    self.model = model
    self.step = 0
    
    # Loss weights
    self.loss_weights = {
        'planet_count': loss_weight_planet_count,
        'planet_existence': loss_weight_planet_existence
    }
    
    # Metrics
    self.train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        accuracy=nnx.metrics.Accuracy()
    )
    self.eval_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        accuracy=nnx.metrics.Accuracy()
    )
    
    # Optimizer (Flax 0.11.0+ requires wrt argument)
    self.optimizer = nnx.Optimizer(
        model, 
        create_optimizer(learning_rate, weight_decay, gradient_clip_norm),
        wrt=nnx.Param
    )
    
    # Checkpointing - use absolute path for orbax
    self.checkpoint_dir = os.path.abspath(os.path.join(run_dir, 'checkpoints'))
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    self.checkpoint_manager = ocp.CheckpointManager(
        self.checkpoint_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    )
    
    # TensorBoard
    self.log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(self.log_dir, exist_ok=True)
    self.writer = SummaryWriter(logdir=self.log_dir)
    logging.info(f'TensorBoard: {self.log_dir}')
  
  def train_step_fn(self, batch):
    """Executes training step with logging."""
    loss, _ = train_step(self.model, self.optimizer, self.train_metrics,
                             batch, self.loss_weights)
    self.step += 1
    self.writer.add_scalar('train/loss', float(loss), self.step)
    return loss
  
  def eval_step_fn(self, batch):
    """Executes eval step."""
    loss = eval_step(self.model, self.eval_metrics, batch, self.loss_weights)
    return loss
  
  def save_checkpoint(self):
    """Saves checkpoint."""
    params = nnx.state(self.model, nnx.Param)
    self.checkpoint_manager.save(
        self.step,
        args=ocp_args.Composite(
            model=ocp_args.StandardSave(params),
            step=ocp_args.JsonSave(self.step)
        )
    )
    logging.info(f'Saved checkpoint at step {self.step}')
  
  def restore_checkpoint(self):
    """Restores latest checkpoint."""
    step = self.checkpoint_manager.latest_step()
    if step is None:
      return None
    
    abstract_params = jax.tree.map(
        ocp.utils.to_shape_dtype_struct, nnx.state(self.model, nnx.Param)
    )
    restored = self.checkpoint_manager.restore(
        step,
        args=ocp_args.Composite(
            model=ocp_args.StandardRestore(abstract_params),
            step=ocp_args.JsonRestore()
        )
    )
    nnx.update(self.model, restored['model'])
    self.step = restored['step']
    logging.info(f'Restored checkpoint from step {step}')
    return step
  
  def close(self):
    """Closes resources."""
    self.writer.close()
    self.checkpoint_manager.wait_until_finished()
