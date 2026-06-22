"""Flax NNX training loop: optimizer schedule, metrics and checkpointing."""

from __future__ import annotations

import os

import ml_collections
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array

from hermes.trainers.losses import vetter_loss


def create_optimizer(
    config: ml_collections.ConfigDict, steps_per_epoch: int
) -> optax.GradientTransformation:
  """Builds an AdamW optimizer with a warmup-cosine schedule and clipping.

  Args:
    config: The ``train`` configuration section.
    steps_per_epoch: Number of optimizer steps per epoch (sets the schedule).

  Returns:
    The composed optax gradient transformation.
  """
  total_steps = max(config.num_epochs * steps_per_epoch, 2)
  # The cosine phase spans ``decay_steps - warmup_steps``, so warmup must stay
  # strictly below the total (it can exceed it for very short runs).
  warmup_steps = min(config.warmup_steps, total_steps - 1)
  schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=config.learning_rate,
      warmup_steps=warmup_steps,
      decay_steps=total_steps,
  )
  return optax.chain(
      optax.clip_by_global_norm(config.grad_clip_norm),
      optax.adamw(schedule, weight_decay=config.weight_decay),
  )


@nnx.jit(
    static_argnames=("regression_weight", "physics_weight", "positive_weight")
)
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: dict[str, Array],
    regression_weight: float,
    physics_weight: float,
    positive_weight: float,
) -> Array:
  """Runs one gradient step and updates the metrics in place."""

  def loss_fn(model: nnx.Module):
    return vetter_loss(
        model, batch, regression_weight, physics_weight, positive_weight
    )

  (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
  optimizer.update(model, grads)
  metrics.update(loss=loss, accuracy=aux["accuracy"])
  return loss


@nnx.jit(
    static_argnames=("regression_weight", "physics_weight", "positive_weight")
)
def eval_step(
    model: nnx.Module,
    metrics: nnx.MultiMetric,
    batch: dict[str, Array],
    regression_weight: float,
    physics_weight: float,
    positive_weight: float,
) -> Array:
  """Evaluates one batch and updates the metrics in place."""
  loss, aux = vetter_loss(
      model, batch, regression_weight, physics_weight, positive_weight
  )
  metrics.update(loss=loss, accuracy=aux["accuracy"])
  return loss


class Trainer:
  """Owns the optimizer, metrics and checkpointing for a training run."""

  def __init__(
      self,
      model: nnx.Module,
      config: ml_collections.ConfigDict,
      steps_per_epoch: int,
      checkpoint_dir: str | None = None,
  ):
    """Initialises the trainer.

    Args:
      model: The model to train.
      config: The ``train`` configuration section.
      steps_per_epoch: Optimizer steps per epoch (for the LR schedule).
      checkpoint_dir: Directory for checkpoints; checkpointing is disabled when
        ``None``.
    """
    self.model = model
    self.optimizer = nnx.Optimizer(
        model, create_optimizer(config, steps_per_epoch), wrt=nnx.Param
    )
    self.regression_weight = config.regression_loss_weight
    self.physics_weight = config.physics_loss_weight
    self.positive_weight = config.positive_class_weight
    self.train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        accuracy=nnx.metrics.Average("accuracy"),
    )
    self.eval_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        accuracy=nnx.metrics.Average("accuracy"),
    )
    self._checkpointer = ocp.StandardCheckpointer() if checkpoint_dir else None
    self.checkpoint_dir = (
        os.path.abspath(checkpoint_dir) if checkpoint_dir else None
    )

  def train_epoch(self, batches) -> dict[str, float]:
    """Trains on an iterable of batches and returns the epoch metrics."""
    self.model.train()
    self.train_metrics.reset()
    for batch in batches:
      train_step(
          self.model,
          self.optimizer,
          self.train_metrics,
          batch,
          self.regression_weight,
          self.physics_weight,
          self.positive_weight,
      )
    return {k: float(v) for k, v in self.train_metrics.compute().items()}

  def evaluate(self, batches) -> dict[str, float]:
    """Evaluates on an iterable of batches and returns the metrics."""
    self.model.eval()
    self.eval_metrics.reset()
    for batch in batches:
      eval_step(
          self.model,
          self.eval_metrics,
          batch,
          self.regression_weight,
          self.physics_weight,
          self.positive_weight,
      )
    return {k: float(v) for k, v in self.eval_metrics.compute().items()}

  def save(self, name: str = "best") -> None:
    """Saves the model parameters under ``checkpoint_dir/name``, overwriting.

    Callers save on validation improvement, so the ``best`` checkpoint always
    holds the best-by-validation model rather than the last epoch.
    """
    if self._checkpointer is None:
      return
    path = os.path.join(self.checkpoint_dir, name)
    self._checkpointer.save(path, nnx.state(self.model, nnx.Param), force=True)

  def restore(self, name: str = "best") -> None:
    """Restores parameters saved under ``checkpoint_dir/name``."""
    if self._checkpointer is None:
      return
    path = os.path.join(self.checkpoint_dir, name)
    abstract = nnx.state(self.model, nnx.Param)
    restored = self._checkpointer.restore(path, abstract)
    nnx.update(self.model, restored)
