"""
siren_model.py
--------------
Definition of the SIREN (Sinusoidal Representation Network) network using Flax,
as used to represent the absorption field in the detector.

Architecture:
    - Input: normalized (r, cos θ, sin θ, z) → 4 features
    - Hidden layers: SineLayer with activations sin(ω₀ · Wx + b)
    - Output: scalar (absorption correction)

Usage:
    from siren_model import SIREN, SineLayer, build_model, apply_model
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# SineLayer
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    """Dense layer with sinusoidal activation.

    Args:
        features   : number of output neurons
        is_first   : True for the first layer (different initialization)
        omega_0    : frequency of the sinusoidal activation
    """
    features: int
    is_first: bool = False
    omega_0: float = 30.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        input_dim = inputs.shape[-1]

        if self.is_first:
            # Init paper SIREN : U(-1/n_in, 1/n_in)
            weight_init = nn.initializers.uniform(scale=1.0 / input_dim)
        else:
            # Init hidden layers : U(-√(6/n_in)/ω₀, √(6/n_in)/ω₀)
            scale = np.sqrt(6.0 / input_dim) / self.omega_0
            weight_init = nn.initializers.uniform(scale=scale)

        x = nn.Dense(
            features=self.features,
            kernel_init=weight_init,
            bias_init=nn.initializers.uniform(scale=1.0),
        )(inputs)

        return jnp.sin(self.omega_0 * x)


# ---------------------------------------------------------------------------
# SIREN
# ---------------------------------------------------------------------------

class SIREN(nn.Module):
    """Complete SIREN network.

    Args:
        hidden_features  : width of hidden layers (e.g., 256)
        hidden_layers    : number of hidden layers (e.g., 3)
        out_features     : output dimension (1 for a scalar)
        outermost_linear : if True, the outermost layer is linear (without sin)
        w0               : frequency ω₀ for all layers

    Forward returns:
        (output, inputs) — the inputs tuple is retained for compatibility
        with loss functions that require it.
    """
    hidden_features: int
    hidden_layers: int
    out_features: int
    outermost_linear: bool = False
    w0: float = 30.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = SineLayer(
            features=self.hidden_features,
            is_first=True,
            omega_0=self.w0,
            name="SineLayer_0",
        )(inputs)

        for i in range(self.hidden_layers):
            x = SineLayer(
                features=self.hidden_features,
                is_first=False,
                omega_0=self.w0,
                name=f"SineLayer_{i + 1}",
            )(x)

        if self.outermost_linear:
            scale = np.sqrt(6.0 / self.hidden_features) / self.w0
            init = nn.initializers.uniform(scale=scale)
            x = nn.Dense(
                features=self.out_features,
                kernel_init=init,
                bias_init=nn.initializers.uniform(scale=1.0),
                name="Dense_0",
            )(x)
        else:
            x = SineLayer(
                features=self.out_features,
                is_first=False,
                omega_0=self.w0,
                name="SineLayer_final",
            )(x)

        return x, inputs



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(
    hidden_features: int = 256,
    hidden_layers: int = 3,
    out_features: int = 1,
    outermost_linear: bool = True,
    w0: float = 6.0,
) -> SIREN:
    """Instantiate the SIREN model using the notebook's hyperparameters."""
    return SIREN(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        outermost_linear=outermost_linear,
        w0=w0,
    )


def init_model(model: SIREN, key: jax.Array, input_dim: int = 4) -> dict:
    """Initializes the model's Flax parameters.

    Args:
        model     : SIREN instance
        key       : JAX PRNGKey
        input_dim : input dimensions (default: 4: r, cos θ, sin θ, z)

    Returns:
        Flax parameter dictionary (structure {“SineLayer_0”: {...}, ...})
    """
    dummy = jax.random.uniform(key, (1, input_dim), minval=-1.0, maxval=1.0)
    return model.init(key, dummy)["params"]


def apply_model(
    model: SIREN,
    params: dict,
    x: jnp.ndarray,
    output_scale: float = 5.0,
    output_shift: float = 1.0,
) -> jnp.ndarray:
    """Evaluates SIREN and returns the physical absorption correction.

    The raw output of the network is in latent space; the following is applied:
        absorption_correction = output / output_scale + output_shift

    Args:
        model        : SIREN instance
        params       : Flax parameters
        x            : (N, 4) — (r/R, cos θ, sin θ, 2z/H)
        output_scale : scaling factor (5.0 in the notebook)
        output_shift : shift (1.0 → correction centered at 1)

    Returns:
        (N,) multiplicative correction of the absorption length
    """
    out, _ = model.apply({"params": params}, x)
    return out.squeeze(-1)/ output_scale  + output_shift 
