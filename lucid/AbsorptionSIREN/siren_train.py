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
from matplotlib.colors import TwoSlopeNorm

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
    err = np.where(
        mask,
        (pred - true) / (np.abs(true) + 1e-6),
        np.nan
    )

    vmin = float(jnp.minimum(true.min(), pred.min()))
    vmax = float(jnp.maximum(true.max(), pred.max()))

    fig, axs = plt.subplots(1, 5, figsize=(18, 3))
    plt.subplots_adjust(wspace=0.7)

    im0 = axs[0].imshow(true_m, extent=[-R, R, -R, R], cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title("True (z=0)"); fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred_m, extent=[-R, R, -R, R], cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("SIREN"); fig.colorbar(im1, ax=axs[1])

    vmax_err = np.nanmax(np.abs(err))

    norm = TwoSlopeNorm(
        vmin=-vmax_err,
        vcenter=0.0,
        vmax=vmax_err
    )
    
    im2 = axs[2].imshow(
        err,
        extent=[-R, R, -R, R],
        cmap="RdBu_r",   # bleu -> blanc -> rouge
        norm=norm
    )
    
    axs[2].set_title("Signed relative error")
    fig.colorbar(im2, ax=axs[2])

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

# ── Injector geometry (global constants) ────────────────────────────────────

# 4 top injectors (cap z = +H/2)
r_top      = R * 0.3
angles_top = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
tilt       = 0.0
top_origins    = np.array([[r_top*np.cos(a), r_top*np.sin(a),  H/2] for a in angles_top])
top_directions = np.array([[-tilt*np.cos(a), -tilt*np.sin(a), -1.0] for a in angles_top])
top_directions /= np.linalg.norm(top_directions, axis=1, keepdims=True)

def _make_lateral_group(theta, z_side, deltas):
    """Build origins and inward directions for one lateral injector group + its symmetric."""
    origins    = np.array([[R*np.cos(theta), R*np.sin(theta), z] for z in z_side])
    directions = np.stack([np.cos(theta + np.pi + deltas),
                           np.sin(theta + np.pi + deltas),
                           np.zeros(len(deltas))], axis=1)
    sym_origins    = origins    * np.array([-1, -1, 1])
    sym_directions = directions * np.array([-1, -1, 1])
    return origins, directions, sym_origins, sym_directions

z_side = np.linspace(-H/2*0.8, H/2*0.8, 5)
deltas = np.linspace(0,0,5)#np.linspace(-np.pi/9, np.pi/9, 5)

side_origins1, side_directions1, sym_origins1, sym_directions1 = _make_lateral_group(0.0,     z_side, deltas)
side_origins2, side_directions2, sym_origins2, sym_directions2 = _make_lateral_group(np.pi/2, z_side, deltas)

# Final assembly → (24, 3)
all_origins    = np.concatenate([top_origins,    side_origins1,    sym_origins1,    side_origins2,    sym_origins2])
all_directions = np.concatenate([top_directions, side_directions1, sym_directions1, side_directions2, sym_directions2])
all_directions /= np.linalg.norm(all_directions, axis=1, keepdims=True)


# ── Ray / cylinder intersection ──────────────────────────────────────────────

@functools.partial(jax.jit, static_argnums=())
def ray_cylinder_exit_batch(x0, y0, z0, dx, dy, dz):
    """Return the exit time t for each ray (x0+t·dx, y0+t·dy, z0+t·dz) hitting the cylinder."""
    a    = dx**2 + dy**2
    b    = 2.0 * (x0*dx + y0*dy)
    c    = x0**2 + y0**2 - R**2
    sq   = jnp.sqrt(jnp.maximum(b**2 - 4*a*c, 0.0))
    t1   = (-b + sq) / (2*a + 1e-12)
    t2   = (-b - sq) / (2*a + 1e-12)
    t_wall = jnp.where(t1 > 1e-8, t1, jnp.inf)
    t_wall = jnp.where((t2 > 1e-8) & (t2 < t_wall), t2, t_wall)
    t_top  = jnp.where((t_top := ( H/2 - z0) / (dz + 1e-30)) > 1e-8, t_top, jnp.inf)
    t_bot  = jnp.where((t_bot := (-H/2 - z0) / (dz + 1e-30)) > 1e-8, t_bot, jnp.inf)
    return jnp.minimum(t_wall, jnp.minimum(t_top, t_bot))


# ── Injector sampling ────────────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnums=(1,))
def generate_injector_samples(key, n_pts_per_line, origins=None, directions=None):
    """Sample n_pts_per_line random points along each injector ray.

    Returns:
        Array of shape (n_injectors * n_pts_per_line, 4) in normalised cylindrical coords.
    """
    origins    = jnp.array(all_origins    if origins    is None else origins)
    directions = jnp.array(all_directions if directions is None else directions)
    x0, y0, z0 = origins[:,0],    origins[:,1],    origins[:,2]
    dx, dy, dz  = directions[:,0], directions[:,1], directions[:,2]

    t_exit = ray_cylinder_exit_batch(x0, y0, z0, dx, dy, dz)
    t      = jax.random.uniform(key, (origins.shape[0], n_pts_per_line)) * t_exit[:, None]
    x = x0[:, None] + t * dx[:, None]
    y = y0[:, None] + t * dy[:, None]
    z = z0[:, None] + t * dz[:, None]

    r      = jnp.sqrt(x**2 + y**2)
    return jnp.stack([r / R, x / (r + 1e-12), y / (r + 1e-12), 2.0 * z / H], axis=-1).reshape(-1, 4)


# ── Training ─────────────────────────────────────────────────────────────────

def train_injector(
    n_steps: int = 500,
    n_pts_per_line: int = 20,
    lr: float = 5e-4,
    seed: int = 0,
    output_path: str = "siren_injector_final.npz",
    plot: bool = True,
) -> dict:
    """Train the SIREN by sampling along injector rays.

    Unlike train(), the number of lines is fixed by the injector geometry
    (len(all_origins)). The ray loss integrates n_pts_per_line points per
    injector and compares against the physical measurement of each injector.

    Args:
        n_steps        : number of training iterations
        n_pts_per_line : points sampled per injector ray
        lr             : initial learning rate
        seed           : random seed
        output_path    : save path (.npz)
        plot           : plot loss curves and slices at the end
    Returns:
        Trained Flax parameter dict
    """
    global opt  # required for the JIT closure in make_step_fn

    n_injectors = len(all_origins)
    key         = jax.random.PRNGKey(seed)
    model       = build_model()
    loss_L2     = make_loss_L2(model)
    opt         = build_optimizer(lr=lr, n_steps=n_steps)

    # Init
    key, init_key = jax.random.split(key)
    params    = init_model(model, init_key)
    opt_state = opt.init(params)
    step      = make_step_fn(model)

    # JIT warmup
    print("JAX compilation (first step)...")
    t0       = time.time()
    xs_dummy = generate_injector_samples(jax.random.PRNGKey(99), n_pts_per_line)
    params, opt_state, _ = step(params, opt_state,
                                xs_dummy.reshape(n_injectors, n_pts_per_line, 4),
                                jnp.zeros(n_injectors))
    params    = init_model(model, init_key)   # reset after warmup
    opt_state = opt.init(params)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # Training loop
    train_losses, train_losses_L2, val_losses_L2 = [], [], []
    times = []

    print(f"{'Step':>6}  {'Train':>12}  {'Train L2':>12}  {'Val L2':>12}  {'Time':>8}")
    print("─" * 58)

    for i in range(n_steps):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2*i + seed), 3)

        # Sample along injector rays
        xs_flat  = generate_injector_samples(jax.random.PRNGKey(2*i + 1 + seed), n_pts_per_line)
        xs_train = xs_flat.reshape(n_injectors, n_pts_per_line, 4)
        ys_train = jnp.array([jnp.mean(target_fn(xs_train[j])) for j in range(n_injectors)])

        # Gradient step
        t_start = time.time()
        params, opt_state, train_loss = step(params, opt_state, xs_train, ys_train)
        train_loss.block_until_ready()
        times.append(time.time() - t_start)

        # Validation on random points inside the cylinder
        n_val  = n_injectors * n_pts_per_line // 5
        r      = jax.random.uniform(k1, (n_val,))
        theta  = jax.random.uniform(k2, (n_val,), minval=-1.0, maxval=1.0)
        z      = jax.random.uniform(k3, (n_val,), minval=-1.0, maxval=1.0)
        xs_val = jnp.stack([r, jnp.cos(theta), jnp.sin(theta), z], axis=1)
        val_loss_L2   = loss_L2(params, xs_val,  target_fn(xs_val))
        train_loss_L2 = loss_L2(params, xs_flat, target_fn(xs_flat))

        train_losses.append(float(jnp.log(train_loss + 1e-12)))
        train_losses_L2.append(float(jnp.log(train_loss_L2 + 1e-12)))
        val_losses_L2.append(float(jnp.log(val_loss_L2 + 1e-12)))

        if i % 100 == 0 or i == n_steps - 1:
            print(f"{i:>6}  {float(train_loss):>12.7f}  {float(train_loss_L2):>12.7f}"
                  f"  {float(val_loss_L2):>12.7f}  {times[-1]:>7.3f}s")

    times_arr = jnp.array(times)
    print(f"\nAverage step time: {float(jnp.mean(times_arr)):.3f}s ± {float(jnp.std(times_arr)):.3f}s")

    save_siren_params(params, output_path)

    if not plot:
        return params

    # Loss curves + slices
    _plot_results(model, params, train_losses, train_losses_L2, val_losses_L2)

    # 3D injector visualisation (Plotly)
    _plot_injectors_3d()

    return params


def _plot_injectors_3d(
    groups: list | None = None,
    title: str = "Injectors geometry",
    RAY_FRAC: float = 0.8,
):
    """Render injector rays and cylinder in an interactive Plotly 3D figure.

    Args:
        groups   : list of (origins, directions, label, color) to display.
                   Defaults to the three fixed injector groups (top + 2 lateral).
        title    : figure title
        RAY_FRAC : fraction of the ray length to draw (0 < RAY_FRAC ≤ 1)
    """
    import plotly.graph_objects as go

    if groups is None:
        groups = [
            (top_origins,                                  top_directions,                                  "Top",       "teal"),
            (np.vstack([side_origins1, sym_origins1]),     np.vstack([side_directions1, sym_directions1]), "Lateral 1", "royalblue"),
            (np.vstack([side_origins2, sym_origins2]),     np.vstack([side_directions2, sym_directions2]), "Lateral 2", "orange"),
        ]

    def _exit_t(ox, oy, oz, dx, dy, dz):
        """Exit time for a ray hitting the cylinder wall or caps."""
        a, b = dx**2 + dy**2, 2*(ox*dx + oy*dy)
        sq   = np.sqrt(max(b**2 - 4*a*(ox**2 + oy**2 - R**2), 0.0))
        ts   = [(-b + sq) / (2*a + 1e-12), (-b - sq) / (2*a + 1e-12)]
        t_wall = min((t for t in ts if t > 1e-8), default=np.inf)
        t_caps = [(H/2 - oz) / (dz + 1e-30), (-H/2 - oz) / (dz + 1e-30)]
        return min(t_wall, *(t for t in t_caps if t > 1e-8), np.inf)

    def _add_group(origins, directions, label, color):
        """Add ray segments and origin markers for one injector group."""
        dirs = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
        xs, ys, zs = [], [], []
        for o, d in zip(origins, dirs):
            end = o + d * _exit_t(*o, *d) * RAY_FRAC
            xs += [o[0], end[0], None]
            ys += [o[1], end[1], None]
            zs += [o[2], end[2], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=dict(color=color, width=4), name=label))
        fig.add_trace(go.Scatter3d(x=origins[:,0], y=origins[:,1], z=origins[:,2],
                                   mode="markers", marker=dict(size=5, color=color),
                                   showlegend=False))

    fig = go.Figure()

    for origs, dirs, label, color in groups:
        _add_group(np.array(origs), np.array(dirs), label, color)

    # Cylinder surface
    T, Z = np.meshgrid(np.linspace(0, 2*np.pi, 60), np.linspace(-H/2, H/2, 2))
    fig.add_trace(go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z, opacity=0.18,
                             colorscale=[[0, "lightblue"], [1, "lightblue"]],
                             showscale=False, name="Cylinder"))

    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=40, b=0))
    fig.show()

# ── Injector geometry ────────────────────────────────────────────────────────

def _make_injector_origins_directions():
    """Fixed injectors: 4 top + 20 lateral (2 groups × 5 positions × symmetry)."""
    r_top, tilt = R * 0.3, 0.0
    angles_top     = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    top_origins    = np.array([[r_top*np.cos(a), r_top*np.sin(a),  H/2] for a in angles_top])
    top_directions = np.array([[-tilt*np.cos(a), -tilt*np.sin(a), -1.0] for a in angles_top])
    top_directions /= np.linalg.norm(top_directions, axis=1, keepdims=True)

    z_side = np.linspace(-H/2*0.8, H/2*0.8, 5)
    deltas = np.linspace(-np.pi/9,  np.pi/9, 5)

    def _lateral(theta):
        origs = np.array([[R*np.cos(theta), R*np.sin(theta), z] for z in z_side])
        dirs  = np.stack([np.cos(theta + np.pi + deltas),
                          np.sin(theta + np.pi + deltas),
                          np.zeros(5)], axis=1)
        return origs, dirs, origs * [-1, -1, 1], dirs * [-1, -1, 1]

    groups = [_lateral(0.0), _lateral(np.pi/2)]
    origins    = np.concatenate([top_origins]    + [g[i] for g in groups for i in (0, 2)])
    directions = np.concatenate([top_directions] + [g[i] for g in groups for i in (1, 3)])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions


def _sample_wall_rays(n_lines: int, rng: np.random.Generator):
    """Sample n_lines rays from the cylinder wall with a 30° inward cone."""
    HALF_ANGLE = 30 * np.pi / 180
    area_side  = 2 * np.pi * R * H
    area_cap   = np.pi * R**2
    area_total = area_side + 2 * area_cap

    u          = rng.uniform(size=n_lines)
    is_side    = u < area_side / area_total
    is_top_cap = (u >= area_side / area_total) & (u < (area_side + area_cap) / area_total)
    is_bot_cap = ~(is_side | is_top_cap)

    origins    = np.zeros((n_lines, 3))
    directions = np.zeros((n_lines, 3))

    def _cone(n, inward, e1, e2):
        """Uniform direction inside a cone of half-angle HALF_ANGLE around inward."""
        phi   = rng.uniform(0, 2*np.pi, n)
        cos_d = rng.uniform(np.cos(HALF_ANGLE), 1.0, n)
        sin_d = np.sqrt(1 - cos_d**2)
        return (cos_d[:, None] * inward
                + sin_d[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))

    # Lateral wall
    n_s = is_side.sum()
    if n_s:
        theta = rng.uniform(0, 2*np.pi, n_s)
        z     = rng.uniform(-H/2, H/2, n_s)
        origins[is_side]    = np.stack([R*np.cos(theta), R*np.sin(theta), z], axis=1)
        e_r   = np.stack([ np.cos(theta),  np.sin(theta), np.zeros(n_s)], axis=1)
        e_phi = np.stack([-np.sin(theta),  np.cos(theta), np.zeros(n_s)], axis=1)
        directions[is_side] = _cone(n_s, -e_r, e_phi, np.tile([0, 0, 1], (n_s, 1)))

    # Top cap (z = +H/2)
    n_t = is_top_cap.sum()
    if n_t:
        r_t, th_t = R * np.sqrt(rng.uniform(0, 1, n_t)), rng.uniform(0, 2*np.pi, n_t)
        origins[is_top_cap]    = np.stack([r_t*np.cos(th_t), r_t*np.sin(th_t), np.full(n_t,  H/2)], axis=1)
        directions[is_top_cap] = _cone(n_t, np.tile([0, 0, -1], (n_t, 1)).astype(float),
                                             np.tile([1, 0,  0], (n_t, 1)).astype(float),
                                             np.tile([0, 1,  0], (n_t, 1)).astype(float))

    # Bottom cap (z = -H/2)
    n_b = is_bot_cap.sum()
    if n_b:
        r_b, th_b = R * np.sqrt(rng.uniform(0, 1, n_b)), rng.uniform(0, 2*np.pi, n_b)
        origins[is_bot_cap]    = np.stack([r_b*np.cos(th_b), r_b*np.sin(th_b), np.full(n_b, -H/2)], axis=1)
        directions[is_bot_cap] = _cone(n_b, np.tile([0, 0, 1], (n_b, 1)).astype(float),
                                            np.tile([1, 0, 0], (n_b, 1)).astype(float),
                                            np.tile([0, 1, 0], (n_b, 1)).astype(float))

    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions


# ── Training ─────────────────────────────────────────────────────────────────

def train_injector_random(
    n_steps: int = 500,
    n_pts_per_line: int = 20,
    n_wall_lines: int = 30,
    lr: float = 5e-4,
    seed: int = 0,
    output_path: str = "siren_injector2_final.npz",
    plot: bool = True,
) -> dict:
    """Like train_injector but augmented with rays sampled randomly on the wall
    (positions + directions drawn once before training and kept fixed).
    """
    global opt

    # Combine fixed injectors with random wall rays (sampled once)
    wall_origins, wall_directions = _sample_wall_rays(n_wall_lines, np.random.default_rng(seed))
    combined_origins    = np.concatenate([all_origins,    wall_origins])
    combined_directions = np.concatenate([all_directions, wall_directions])
    n_lines = len(combined_origins)

    model   = build_model()
    loss_L2 = make_loss_L2(model)
    opt     = build_optimizer(lr=lr, n_steps=n_steps)

    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    params    = init_model(model, init_key)
    opt_state = opt.init(params)
    step      = make_step_fn(model)

    # JIT warmup
    print("JAX compilation (first step)...")
    t0 = time.time()
    xs_dummy = generate_injector_samples(jax.random.PRNGKey(99), n_pts_per_line,
                                         origins=combined_origins, directions=combined_directions)
    params, opt_state, _ = step(params, opt_state,
                                xs_dummy.reshape(n_lines, n_pts_per_line, 4),
                                jnp.zeros(n_lines))
    params    = init_model(model, init_key)   # reset after warmup
    opt_state = opt.init(params)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # Training loop
    train_losses, train_losses_L2, val_losses_L2, times = [], [], [], []

    print(f"{'Step':>6}  {'Train':>12}  {'Train L2':>12}  {'Val L2':>12}  {'Time':>8}")
    print("─" * 58)

    for i in range(n_steps):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2*i + seed), 3)

        xs_flat  = generate_injector_samples(jax.random.PRNGKey(2*i + 1 + seed), n_pts_per_line,
                                             origins=combined_origins, directions=combined_directions)
        xs_train = xs_flat.reshape(n_lines, n_pts_per_line, 4)
        ys_train = jnp.array([jnp.mean(target_fn(xs_train[j])) for j in range(n_lines)])

        t_start = time.time()
        params, opt_state, train_loss = step(params, opt_state, xs_train, ys_train)
        train_loss.block_until_ready()
        times.append(time.time() - t_start)

        # Validation on random points inside the cylinder
        n_val  = n_lines * n_pts_per_line // 5
        r      = jax.random.uniform(k1, (n_val,))
        theta  = jax.random.uniform(k2, (n_val,), minval=-1.0, maxval=1.0)
        z      = jax.random.uniform(k3, (n_val,), minval=-1.0, maxval=1.0)
        xs_val = jnp.stack([r, jnp.cos(theta), jnp.sin(theta), z], axis=1)
        val_loss_L2   = loss_L2(params, xs_val,  target_fn(xs_val))
        train_loss_L2 = loss_L2(params, xs_flat, target_fn(xs_flat))

        train_losses.append(float(jnp.log(train_loss + 1e-12)))
        train_losses_L2.append(float(jnp.log(train_loss_L2 + 1e-12)))
        val_losses_L2.append(float(jnp.log(val_loss_L2 + 1e-12)))

        if i % 100 == 0 or i == n_steps - 1:
            print(f"{i:>6}  {float(train_loss):>12.7f}  {float(train_loss_L2):>12.7f}"
                  f"  {float(val_loss_L2):>12.7f}  {times[-1]:>7.3f}s")

    times_arr = jnp.array(times)
    print(f"\nAverage step time: {float(jnp.mean(times_arr)):.3f}s ± {float(jnp.std(times_arr)):.3f}s")

    save_siren_params(params, output_path)

    if plot:
        _plot_results(model, params, train_losses, train_losses_L2, val_losses_L2)
        _plot_injectors_3d(
            groups=[(wall_origins, wall_directions, "Wall (random, 30° cone)", "crimson")],
            title="Injectors geometry (random wall rays)",
        )

    return params