"""
siren_loader.py
---------------
Loading a pre-trained SIREN model from a parameter file (.npz).

Provides:
    - load_siren_params  : loads weights from a .npz file
    - setup_siren        : reconstructs the model with ready-to-use parameters
    - make_absorption_fn : returns an xyz → absorption correction function
    - encode_cylindrical : converts Cartesian coordinates (x, y, z)
                           to normalized input (r, cos θ, sin θ, z_norm)

Typical usage:
    from siren_loader import setup_siren, make_absorption_fn

    model, params = setup_siren(“checkpoints/siren_v1.npz”)
    absorption_fn = make_absorption_fn(model, params, R=16.9, H=36.0)

    correction = absorption_fn(xyz_points)  # (N,) corrections
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Callable, Tuple

from lucid.AbsorptionSIREN.siren_model import SIREN, build_model, apply_model


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_siren_params(params: dict, path: str) -> None:
    """Saves the Flax parameters to an .npz file.

    The Flax structure is serialized using NumPy via pickle (allow_pickle=True),
    which preserves the layer names and hierarchy.

    Args:
        params : Flax dictionary returned by model.init(...)[“params”]
        path   : output path (e.g., “checkpoints/siren_v1.npz”)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # We convert the JAX leaves to NumPy before saving
    params_np = jax.tree_util.tree_map(np.array, params)
    np.savez(path, params=np.array([params_np], dtype=object))
    print(f"[siren_loader] Parameters saved → {path}")


def load_siren_params(path: str) -> dict:
    """Loads settings from an .npz file.

    Args:
        path: path to the .npz file (e.g., “checkpoints/siren_v1.npz”)

    Returns:
        Flax settings dictionary (leaves as a jnp.ndarray)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found : {path}")

    data = np.load(path, allow_pickle=True)
    params_np = data["params"][0]
    # Convert NumPy arrays to JNP
    params = jax.tree_util.tree_map(jnp.array, params_np)
    print(f"[siren_loader] Parameters loaded from {path}")
    return params


# ---------------------------------------------------------------------------
# Quick setup
# ---------------------------------------------------------------------------

def setup_siren(
    params_path: str,
    hidden_features: int = 256,
    hidden_layers: int = 3,
    out_features: int = 1,
    outermost_linear: bool = True,
    w0: float = 6.0,
) -> Tuple[SIREN, dict]:
    """Reconstructs the SIREN model and loads its parameters from a file.

    The hyperparameters must match those used during
    training (default values = notebook configuration).

    Args:
        params_path      : path to the .npz file containing the parameters
        hidden_features  : width of the hidden layers
        hidden_layers    : number of hidden layers
        out_features     : output dimension (1 = scalar)
        outermost_linear : last linear or sinusoidal layer
        w0               : frequency ω₀

    Returns:
        (model, params) — ready to be passed to apply_model()
    """
    model = build_model(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        outermost_linear=outermost_linear,
        w0=w0,
    )
    params = load_siren_params(params_path)
    return model, params


# ---------------------------------------------------------------------------
# Geometrical encoding
# ---------------------------------------------------------------------------

def encode_cylindrical(
    xyz: jnp.ndarray,
    R: float = 16.9,
    H: float = 38.0,
) -> jnp.ndarray:
    """Converts Cartesian coordinates (N, 3) to SIREN input (N, 4).

    Encoding:
        r_norm  = √(x²+y²) / R          ∈ [0, 1]
        cos_θ   = x / (r + ε)           ∈ [-1, 1]
        sin_θ   = y / (r + ε)           ∈ [-1, 1]
        z_norm  = 2z / H                ∈ [-1, 1]

    Args:
        xyz : (N, 3) Cartesian coordinates in meters
        R   : cylinder radius (m)
        H   : cylinder height (m)

    Returns:
        (N, 4) — normalized input for the SIREN
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = jnp.sqrt(x ** 2 + y ** 2)
    return jnp.stack(
        [
            r / R,
            x / (r + 1e-12),
            y / (r + 1e-12),
            2.0 * z / H,
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Absorption function
# ---------------------------------------------------------------------------

def make_absorption_fn(
    model: SIREN,
    #params: dict,
    R: float = 16.9,
    H: float = 38.0,
    base_absorption: float = 1.0,
    output_scale: float = 5.0,
    output_shift: float = 1.0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Returns a function (N, 3) → (N,) that returns the local absorption length.

    The SIREN multiplicative correction is applied to a base value:
        absorption(xyz) = base_absorption × SIREN_correction(xyz)

    Args:
        model           : SIREN instance
        params          : loaded Flax parameters
        R               : cylinder radius (m)
        H               : cylinder height (m)
        base_absorption : reference absorption length (m)
        output_scale    : decoding factor for raw output
        output_shift    : shift (1.0 → correction centered at 1)

    Returns:
        Callable : xyz (N, 3) → absorption_length (N,) in meters
    """
    @jax.jit
    def absorption_fn(xyz: jnp.ndarray, params) -> jnp.ndarray:
        x_enc = encode_cylindrical(xyz, R=R, H=H)
        correction = apply_model(
            model, params, x_enc,
            output_scale=output_scale,
            output_shift=output_shift,
        )
        return base_absorption * correction

    return absorption_fn
