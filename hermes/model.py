"""Exoplanet detection model.

This model predicts planet count and existence probabilities,
then uses BLS and physics to compute actual planet parameters.
"""

import jax.numpy as jnp
from flax import nnx
from typing import Dict, Any
import gin


class MultiHeadAttention(nnx.Module):
  """Multi-head self-attention layer."""
  
  def __init__(self, hidden_dim: int = 512, num_heads: int = 8, 
               dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.head_dim = hidden_dim // num_heads
    
    self.qkv = nnx.Linear(hidden_dim, 3 * hidden_dim, rngs=rngs)
    self.output_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
    self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    self.layer_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
  
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    batch_size, seq_len, _ = x.shape
    residual = x
    
    qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    q, k, v = jnp.transpose(qkv, (2, 0, 3, 1, 4))
    
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.head_dim)
    attn = nnx.softmax(scores, axis=-1)
    attn = self.dropout(attn)
    
    out = jnp.matmul(attn, v)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.hidden_dim)
    out = self.dropout(self.output_proj(out))
    
    return self.layer_norm(out + residual)


class FeatureEncoder(nnx.Module):
  """Encodes light curve into features."""
  
  def __init__(self, input_dim: int = 2048, hidden_dim: int = 512,
               dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
    self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
    self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
    self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
  
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = nnx.relu(self.ln1(self.fc1(x)))
    x = self.dropout(x)
    x = nnx.relu(self.ln2(self.fc2(x)))
    return jnp.expand_dims(x, axis=1)


class PlanetExistenceHead(nnx.Module):
  """Predicts planet existence probability."""
  
  def __init__(self, hidden_dim: int = 512, *, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(hidden_dim, 64, rngs=rngs)
    self.fc2 = nnx.Linear(64, 1, rngs=rngs)
  
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = nnx.relu(self.fc1(x))
    return nnx.sigmoid(self.fc2(x))


@gin.configurable
class MultiPlanetDetector(nnx.Module):
  """Exoplanet detection model.
  
  Predicts:
    - Planet count (0-5)
    - Existence probability for each planet slot
  
  Physical parameters (period, radius, distance) are COMPUTED using:
    - BLS period search on the actual light curve
    - Transit depth measurement
    - Kepler's Third Law
  """
  
  def __init__(self, input_features_dim: int = 2048, 
               max_detectable_planets: int = 5,
               hidden_dim: int = 512, num_attention_heads: int = 8,
               dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
    self.max_planets = max_detectable_planets
    
    # Feature encoding
    self.encoder = FeatureEncoder(input_features_dim, hidden_dim, dropout_rate, rngs=rngs)
    
    # Planet slot positional encoding
    self.position_encoding = nnx.Param(
        nnx.initializers.normal(stddev=0.02)(
            rngs.params(), (1, max_detectable_planets, hidden_dim)
        )
    )
    
    # Attention
    self.attention = MultiHeadAttention(hidden_dim, num_attention_heads, dropout_rate, rngs=rngs)
    
    # Planet count head
    self.count_fc1 = nnx.Linear(hidden_dim, 256, rngs=rngs)
    self.count_dropout = nnx.Dropout(dropout_rate * 2, rngs=rngs)
    self.count_fc2 = nnx.Linear(256, 6, rngs=rngs)
    
    # Planet existence heads
    self.existence_heads = nnx.List([
        PlanetExistenceHead(hidden_dim, rngs=rngs)
        for _ in range(max_detectable_planets)
    ])
  
  def __call__(self, x: jnp.ndarray) -> Dict[str, Any]:
    """Forward pass.
    
    Args:
      x: Light curves [batch, input_features_dim].
      
    Returns:
      Dictionary with:
        - planet_count_logits: Logits for 0-5 planets [batch, 6]
        - planet_count_probs: Probabilities [batch, 6]
        - existence_probs: List of existence probabilities for each slot
    """
    # Normalize
    x_mean = jnp.mean(x, axis=-1, keepdims=True)
    x_std = jnp.std(x, axis=-1, keepdims=True) + 1e-6
    x = (x - x_mean) / x_std
    
    # Encode
    features = self.encoder(x)
    
    # Create planet slots
    planet_slots = jnp.tile(features, (1, self.max_planets, 1))
    planet_slots = planet_slots + self.position_encoding.value
    
    # Attention
    attended = self.attention(planet_slots)
    
    # Planet count prediction
    global_features = jnp.mean(attended, axis=1)
    count_logits = self.count_fc2(self.count_dropout(nnx.relu(self.count_fc1(global_features))))
    count_probs = nnx.softmax(count_logits)
    
    # Existence predictions for each slot
    existence_probs = [
        head(attended[:, i, :]) 
        for i, head in enumerate(self.existence_heads)
    ]
    
    return {
        'planet_count_logits': count_logits,
        'planet_count_probs': count_probs,
        'existence_probs': existence_probs,
        'features': attended
    }

