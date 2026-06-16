"""
siren_train.py
--------------
Replicates the simulation from the SIREN_Absorption.ipynb notebook:
    - Generates photon tracks in the detector cylinder
    - Trains SIREN on the true absorption function
    - Saves the final parameters in .npz format

Usage:
    python siren_train.py
    python siren_train.py --steps 1000 --output checkpoints/siren_v2.npz
"""

import argparse
import functools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from .siren_model import build_model, init_model
from .siren_loader import save_siren_params


# ---------------------------------------------------------------------------
# Geometry of the detector
# ---------------------------------------------------------------------------

R = 16.9   # radius
H = 38.0   # height


# ---------------------------------------------------------------------------
# Sampling dans le cylindre
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=1)
def sample_uniform_in_cylinder(key: jax.Array, n: int):
    """Sample n points uniformly distributed within the cylinder."""
    k1, k2, k3 = jax.random.split(key, 3)
    r     = R * jnp.sqrt(jax.random.uniform(k1, (n,)))
    theta = jax.random.uniform(k2, (n,), minval=-jnp.pi, maxval=jnp.pi)
    z     = jax.random.uniform(k3, (n,), minval=-H / 2, maxval=H / 2)
    return r * jnp.cos(theta), r * jnp.sin(theta), z


@functools.partial(jax.jit, static_argnums=1)
def sample_uniform_direction(key: jax.Array, n: int):
    """Samples n uniform directions on the unit sphere."""
    k1, k2   = jax.random.split(key)
    phi      = jax.random.uniform(k1, (n,), minval=-jnp.pi, maxval=jnp.pi)
    costheta = jax.random.uniform(k2, (n,), minval=-1.0, maxval=1.0)
    sintheta = jnp.sqrt(1.0 - costheta ** 2)
    return sintheta * jnp.cos(phi), sintheta * jnp.sin(phi), costheta


@functools.partial(jax.jit, static_argnums=(1, 2))
def generate_line_samples(
    key: jax.Array,
    n_lines: int,
    n_pts: int,
) -> jnp.ndarray:
    """Generates photon rays and returns points in normalized cylindrical coordinates.

    Each line starts at a random point on the cylinder in a random direction
    and extends to the wall. Points are sampled uniformly
    along each line.

    Returns:
        (n_lines, n_pts, 4) — (r/R, cos θ, sin θ, 2z/H)
    """
    k1, k2 = jax.random.split(key)
    T = jnp.linspace(0, 1, n_pts + 2)[1:-1][None, :]  # (1, n_pts)

    x0, y0, z0 = sample_uniform_in_cylinder(k1, n_lines)
    dx, dy, dz = sample_uniform_direction(k2, n_lines)

    # Calculating the exit time t_exit for each line
    a    = dx ** 2 + dy ** 2
    b    = 2 * (x0 * dx + y0 * dy)
    c    = x0 ** 2 + y0 ** 2 - R ** 2
    disc = jnp.sqrt(jnp.maximum(b ** 2 - 4 * a * c, 0.0))

    t1 = (-b + disc) / (2 * a + 1e-12)
    t2 = (-b - disc) / (2 * a + 1e-12)

    t_wall = jnp.where(t1 > 0, t1, jnp.inf)
    t_wall = jnp.where((t2 > 0) & (t2 < t_wall), t2, t_wall)
    t_top  = jnp.where((H / 2 - z0) / (dz + 1e-30) > 0, (H / 2 - z0) / (dz + 1e-30), jnp.inf)
    t_bot  = jnp.where((-H / 2 - z0) / (dz + 1e-30) > 0, (-H / 2 - z0) / (dz + 1e-30), jnp.inf)
    t_exit = jnp.minimum(t_wall, jnp.minimum(t_top, t_bot))  # (n_lines,)

    # Cartesian points along lines
    t_pts = T * t_exit[:, None]
    x = x0[:, None] + t_pts * dx[:, None]
    y = y0[:, None] + t_pts * dy[:, None]
    z = z0[:, None] + t_pts * dz[:, None]

    # Standard cylindrical encoding
    r = jnp.sqrt(x ** 2 + y ** 2)
    return jnp.stack(
        [r / R, x / (r + 1e-12), y / (r + 1e-12), 2 * z / H],
        axis=-1,
    )  # (n_lines, n_pts, 4)


# ---------------------------------------------------------------------------
# True absorption function (ground truth)
# ---------------------------------------------------------------------------

@jax.jit
def z_dependence(z: jnp.ndarray) -> jnp.ndarray:
    c, d, a_tba   = -0.163e-3, -3.676e-3, -4.91
    z_threshold   = -22.0 / H
    z_phys        = z * H / 2
    b_t           = c * a_tba ** 2 + d * a_tba
    z_eff         = jnp.maximum(z_phys, z_threshold)
    return 1.0 + z_eff * b_t


@jax.jit
def r_dependence(r: jnp.ndarray) -> jnp.ndarray:
    scale    = 0.05
    exponent = 1.0
    r_phys   = r * R
    r2       = (r_phys / R) ** 2
    return 1.0 + scale * r2 ** (exponent / 2)


@jax.jit
def sinusoidal_azimuth(r: jnp.ndarray, cos_t: jnp.ndarray, sin_t: jnp.ndarray) -> jnp.ndarray:
    amplitude  = 0.05
    frequency  = 8
    r_turnon   = 10.0
    sharpness  = 4.0
    r2         = r ** 2
    theta      = jnp.arctan2(sin_t, cos_t)
    envelope   = (1.0 - jnp.exp(-r2 * R ** 2 / r_turnon ** 2)) ** sharpness
    return 1.0 + amplitude * envelope * jnp.sin(frequency * theta)


@jax.jit
def target_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Returns the true absorption correction in latent space.

    Input: (N, 4) — (r/R, cos θ, sin θ, 2z/H)
    Output: (N, 1) — value in latent space (correction × 5 - 5)
    """
    r, cos_t, sin_t, z = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return ((z_dependence(z) * r_dependence(r) * sinusoidal_azimuth(r, cos_t, sin_t) - 1) * 5).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def build_optimizer(lr: float = 5e-4, n_steps: int = 500):
    """Builds the Adam optimizer with an exponential scheduler."""
    scheduler = optax.exponential_decay(
        init_value=lr,
        transition_steps=100,
        decay_rate=0.97,
        transition_begin=500,
        end_value=5e-8,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
    )


def make_step_fn(model):
    """Returns the JIT-compiled step function."""

    @jax.jit
    def step(params, opt_state, x, y):
        def loss_fn(p):
            def process_line(xline):
                temp, _ = model.apply({"params": p}, xline)
                return jnp.mean(temp)
            pred = jax.vmap(process_line)(x)
            return jnp.mean((pred - y) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return step


def make_loss_L2(model):
    @jax.jit
    def loss_L2(params, x, y):
        pred, _ = model.apply({"params": params}, x)
        return jnp.mean((pred - y) ** 2)
    return loss_L2


def train(
    n_steps: int = 500,
    n_lines: int = 1000,
    n_pts_per_line: int = 20,
    lr: float = 5e-4,
    seed: int = 0,
    output_path: str = "checkpoints/siren_final.npz",
    plot: bool = True,
) -> dict:
    """Trains the SIREN model and saves the final parameters.

    Args:
        n_steps        : number of iterations
        n_lines        : lines of photons per batch
        n_pts_per_line : points per line
        lr             : initial learning rate
        seed           : random seed
        output_path    : save path (.npz)
        plot           : displays loss curves and slices at the end

    Returns:
        dict of trained Flax parameters
    """
    global opt

    key   = jax.random.PRNGKey(seed)
    model = build_model()
    loss_L2 = make_loss_L2(model)
    opt   = build_optimizer(lr=lr, n_steps=n_steps)

    # Init
    key, init_key = jax.random.split(key)
    params    = init_model(model, init_key)
    opt_state = opt.init(params)

    step = make_step_fn(model)

    # Compilation (warmup)
    print("JAX compilation (first step)...")
    t0 = time.time()
    xs_dummy = generate_line_samples(jax.random.PRNGKey(99), n_lines, n_pts_per_line)
    params, opt_state, _ = step(params, opt_state, xs_dummy, jnp.zeros(n_lines))
    # Reboot after warmup
    params    = init_model(model, init_key)
    opt_state = opt.init(params)
    print(f"  Compilation in {time.time() - t0:.1f}s\n")

    # Training loop
    train_losses, train_losses_L2, val_losses_L2 = [], [], []
    times = []

    print(f"{'Step':>6}  {'Train':>12}  {'Train L2':>12}  {'Val L2':>12}  {'Time (s)':>10}")
    print("─" * 60)

    for i in range(n_steps):
        key = jax.random.PRNGKey(2 * i + seed)
        k1, k2, k3 = jax.random.split(key, 3)

        xs_train = generate_line_samples(jax.random.PRNGKey(2 * i + 1 + seed), n_lines, n_pts_per_line)
        ys_train = jnp.array([jnp.mean(target_fn(xs_line)) for xs_line in xs_train])

        t_start = time.time()
        params, opt_state, train_loss = step(params, opt_state, xs_train, ys_train)
        train_loss.block_until_ready()
        times.append(time.time() - t_start)

        # Validation
        n_val = n_lines * n_pts_per_line // 5
        r = jax.random.uniform(k1, (n_val,))
        theta = jax.random.uniform(k2, (n_val,), minval=-1.0, maxval=1.0)
        z = jax.random.uniform(k3, (n_val,), minval=-1.0, maxval=1.0)
        xs_val = jnp.stack([
            r,
            jnp.cos(theta),
            jnp.sin(theta),
            z
        ], axis=1)
        ys_val = target_fn(xs_val)

        val_loss_L2   = loss_L2(params, xs_val, ys_val)
        xs_train_flat = xs_train.reshape(-1, 4)
        train_loss_L2 = loss_L2(params, xs_train_flat, target_fn(xs_train_flat))

        train_losses.append(float(jnp.log(train_loss + 1e-12)))
        train_losses_L2.append(float(jnp.log(train_loss_L2 + 1e-12)))
        val_losses_L2.append(float(jnp.log(val_loss_L2 + 1e-12)))

        if i % 100 == 0 or i == n_steps - 1:
            print(
                f"{i:>6}  {float(train_loss):>12.7f}  "
                f"{float(train_loss_L2):>12.7f}  "
                f"{float(val_loss_L2):>12.7f}  "
                f"{times[-1]:>10.3f}s"
            )

    times_arr = jnp.array(times)
    print(f"\nTemps moyen par step : {float(jnp.mean(times_arr)):.3f}s ± {float(jnp.std(times_arr)):.3f}s")

    # Sauvegarde
    save_siren_params(params, output_path)

    # Visualisation
    if plot:
        _plot_results(model, params, train_losses, train_losses_L2, val_losses_L2)

    return params


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_results(model, params, train_losses, train_losses_L2, val_losses_L2):
    """Reproduces the figures from the notebook: slice z=0 + loss curves."""

    # --- Slice z=0 ---
    grid   = jnp.linspace(-R, R, 200)
    X, Y   = jnp.meshgrid(grid, grid)
    mask   = X ** 2 + Y ** 2 <= R ** 2
    r      = jnp.sqrt(X ** 2 + Y ** 2) / R
    theta  = jnp.arctan2(Y, X)
    points = jnp.stack([r.ravel(), jnp.cos(theta).ravel(), jnp.sin(theta).ravel(), jnp.zeros_like(r).ravel()], axis=-1)

    pred_raw, _ = model.apply({"params": params}, points)
    pred = (pred_raw.reshape(200, 200) / 5 + 1)
    true = (target_fn(points) / 5 + 1).reshape(200, 200)

    pred_m = np.where(mask, pred, np.nan)
    true_m = np.where(mask, true, np.nan)
    err    = np.where(mask, np.abs(pred - true) / (np.abs(true) + 1e-6), np.nan)

    vmin = float(jnp.minimum(true.min(), pred.min()))
    vmax = float(jnp.maximum(true.max(), pred.max()))

    fig, axs = plt.subplots(1, 5, figsize=(18, 3))
    plt.subplots_adjust(wspace=0.7)

    im0 = axs[0].imshow(true_m, extent=[-R, R, -R, R], cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title("True (z=0)"); fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred_m, extent=[-R, R, -R, R], cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("SIREN"); fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(err, extent=[-R, R, -R, R], cmap="magma")
    axs[2].set_title("Relative error"); fig.colorbar(im2, ax=axs[2])

    axs[3].plot(train_losses, linewidth=0.5)
    axs[3].set_title("Loss (average over lines)")
    axs[3].set_xlabel("Step")

    axs[4].plot(train_losses_L2, label="train", linewidth=0.5)
    axs[4].plot(val_losses_L2, label="val", linewidth=0.5)
    axs[4].legend(); axs[4].set_title("L2 Loss")
    axs[4].set_xlabel("Step")

    for ax in axs[:3]:
        ax.set_aspect("equal")

    plt.suptitle("SIREN — Training Results", fontsize=12)
    plt.tight_layout()
    #plt.savefig("siren_training_results.png", dpi=150, bbox_inches="tight")
    #print("Figure sauvegardée → siren_training_results.png")
    plt.show()
