"""Shared pytest configuration.

Double precision is enabled across the suite so the numerics tests can assert
tight agreement with analytic limits and finite-difference gradients.
"""

import jax

jax.config.update("jax_enable_x64", True)
