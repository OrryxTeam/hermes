"""End-to-end: training learns, and the benchmark harness compares methods."""

from flax import nnx

from hermes.data import dataset
from hermes.evaluators import harness
from hermes.models import baselines
from hermes.models.vetter import TransitVetter
from hermes.testing import _fixtures
from hermes.trainers.trainer import Trainer


def _positive_weight(train):
  prevalence = max(float(train.arrays["label"].mean()), 1e-6)
  return (1.0 - prevalence) / prevalence


def test_training_improves_accuracy(tmp_path):
  """On the separable PC-vs-NTP task, training raises accuracy."""
  config = _fixtures.small_dataset_config(tmp_path, label_set=("PC", "NTP"))
  config.train.num_epochs = 40
  config.train.learning_rate = 3e-3
  config.train.warmup_steps = 10
  train = dataset.HermesDataset(config.data.processed_dir, "train")
  config.train.positive_class_weight = _positive_weight(train)

  model = TransitVetter(
      config.model, config.data.global_bins, config.data.local_bins,
      rngs=nnx.Rngs(0),
  )
  trainer = Trainer(model, config.train, max(len(train) // 16, 1))
  before = trainer.evaluate(train.iterate(16, shuffle=False))
  for epoch in range(config.train.num_epochs):
    trainer.train_epoch(train.iterate(16, shuffle=True, seed=epoch))
  after = trainer.evaluate(train.iterate(16, shuffle=False))

  assert after["accuracy"] > 0.8  # above the majority-class baseline
  assert after["accuracy"] > before["accuracy"]


def test_harness_compares_all_methods(tmp_path):
  """The harness trains the baselines and reports valid metrics for each."""
  config = _fixtures.small_dataset_config(
      tmp_path, label_set=("PC", "NTP"), num_stars=30
  )
  train = dataset.HermesDataset(config.data.processed_dir, "train")
  positive_weight = _positive_weight(train)
  config.train.positive_class_weight = positive_weight

  vetter = TransitVetter(
      config.model, config.data.global_bins, config.data.local_bins,
      rngs=nnx.Rngs(0),
  )
  trainer = Trainer(vetter, config.train, steps_per_epoch=2)
  for epoch in range(10):
    trainer.train_epoch(train.iterate(16, shuffle=True, seed=epoch))

  astronet = baselines.AstroNetClassifier(
      config.model, config.data.global_bins, config.data.local_bins,
      rngs=nnx.Rngs(1),
  )
  baselines.train_astronet(
      astronet, train, num_epochs=10, batch_size=16,
      learning_rate=3e-3, positive_weight=positive_weight,
  )

  results = harness.compare(vetter, astronet, train)
  assert set(results) == {"hermes", "astronet", "bls_snr"}
  for values in results.values():
    assert 0.0 <= values["roc_auc"] <= 1.0
    assert 0.0 <= values["average_precision"] <= 1.0
  assert "ece" in results["hermes"]
