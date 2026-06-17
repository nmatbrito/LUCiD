"""
visualize_3d.py
---------------
Visualization of a 3D field (e.g., SIREN absorption correction) within the
cylindrical detector, using plot_interactive: interactive dashboard with a slider (Plotly)
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from typing import Callable, Optional, Sequence


def _make_xy_grid(R: float, res: int):
    """A 2D Cartesian grid in the disk of radius R."""
    g = np.linspace(-R, R, res)
    X, Y = np.meshgrid(g, g)
    mask = X ** 2 + Y ** 2 <= R ** 2
    return X, Y, mask
    
# ---------------------------------------------------------------------------
# Interactive Plotly Dashboard (Z-axis slider)
# ---------------------------------------------------------------------------


def plot_interactive_plotly(
    fn: Callable[[np.ndarray], np.ndarray],
    R: float = 16.9,
    H: float = 38.0,
    n_slices: int = 20,
    res: int = 150,
    cmap: str = "Viridis",
    title: str = "Interactive 3D representation",
):
    """Figure Plotly interactive avec slider pour faire défiler les slices z.
 
    Args:
        fn       : fonction (N, 3) → (N,) en coordonnées cartésiennes (m)
        n_slices : nombre de niveaux z dans le slider
        res      : résolution de chaque slice
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly non installé. Lance : pip install plotly")
        return
 
    z_levels = np.linspace(-H / 2 * 0.95, H / 2 * 0.95, n_slices)
    X, Y, mask = _make_xy_grid(R, res)
 
    # Pré-calcul de toutes les slices
    all_grids, all_vals = [], []
    for z0 in z_levels:
        Z   = np.full_like(X, z0)
        xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        g   = np.array(fn(xyz)).reshape(res, res)
        g_m = np.where(mask, g, np.nan)
        all_grids.append(g_m)
        all_vals.append(g[mask])
 
    vmin = float(np.nanmin(np.concatenate(all_vals)))
    vmax = float(np.nanmax(np.concatenate(all_vals)))
 
    # Construction des frames
    frames = [
        go.Frame(
            data=[go.Heatmap(z=g, x=np.linspace(-R, R, res), y=np.linspace(-R, R, res),
                             colorscale=cmap, zmin=vmin, zmax=vmax, showscale=True, hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<br>correction: %{z:.4f}<extra></extra>")],
            name=f"{z0:.1f}",
        )
        for g, z0 in zip(all_grids, z_levels)
    ]
 
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            width=700,
            height=700,
            xaxis_title="x (m)", yaxis_title="y (m)",
            yaxis_scaleanchor="x",
            sliders=[{
                "steps": [
                    {"args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                     "label": f"{z:.1f} m", "method": "animate"}
                    for f, z in zip(frames, z_levels)
                ],
                "currentvalue": {"prefix": "z = ", "suffix": " m"},
                "pad": {"t": 50},
            }],
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Play",  "method": "animate",
                     "args": [None, {"frame": {"duration": 120}, "fromcurrent": True}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                ],
                "x": 0.15, "y": 1.0,
            }],
        ),
        frames=frames,
    )
    fig.show()

def plot_interactive_true(
    target_fn: Callable[[np.ndarray], np.ndarray],
    R: float = 16.9,
    H: float = 38.0,
    base_absorption: float = 1.0,
    n_slices: int = 30,
    res: int = 250,
    cmap: str = "Viridis",
    title: str = "Ground truth absorption field",
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly non installé.")
        return

    z_levels = np.linspace(-H / 2 * 0.95, H / 2 * 0.95, n_slices)
    X, Y, mask = _make_xy_grid(R, res)
    xy = np.linspace(-R, R, res)

    all_grids, all_vals = [], []
    for z0 in z_levels:
        x_flat = X.ravel()
        y_flat = Y.ravel()
        r_flat = np.sqrt(x_flat**2 + y_flat**2)
        pts_cyl = jnp.stack([
            r_flat / R,
            x_flat / (r_flat + 1e-12),
            y_flat / (r_flat + 1e-12),
            2.0 * z0 / H * np.ones_like(r_flat),
        ], axis=-1)

        g = (np.array(target_fn(pts_cyl)).squeeze() / 5 + 1).reshape(res, res) * base_absorption
        g_m = np.where(mask, g, np.nan)
        all_grids.append(g_m)
        all_vals.append(g[mask])

    vmin = float(np.nanmin(np.concatenate(all_vals)))
    vmax = float(np.nanmax(np.concatenate(all_vals)))

    heatmap_kwargs = dict(
        x=xy, y=xy, colorscale=cmap, zmin=vmin, zmax=vmax,
        showscale=True,
        hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<br>absorption: %{z:.4f} m<extra></extra>",
    )

    frames = [
        go.Frame(
            data=[go.Heatmap(z=all_grids[i], **heatmap_kwargs)],
            name=f"{z0:.1f}",
        )
        for i, z0 in enumerate(z_levels)
    ]

    fig = go.Figure(
        data=[go.Heatmap(z=all_grids[0], **heatmap_kwargs)],
        layout=go.Layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            width=700,
            height=650,
            margin=dict(l=20, r=20, t=80, b=20),
            xaxis=dict(title="x (m)", constrain="domain"),
            yaxis=dict(title="y (m)", scaleanchor="x"),
            sliders=[{
                "steps": [
                    {"args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                     "label": f"{z:.1f} m", "method": "animate"}
                    for f, z in zip(frames, z_levels)
                ],
                "currentvalue": {"prefix": "z = ", "suffix": " m"},
                "pad": {"t": 50},
            }],
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 120}, "fromcurrent": True}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                ],
                "x": 0.15, "y": 1.0,
            }],
        ),
        frames=frames,
    )
    fig.show()

def plot_interactive_comparison(
    fn: Callable[[np.ndarray], np.ndarray],
    target_fn: Callable[[np.ndarray], np.ndarray],
    R: float = 16.9,
    H: float = 38.0,
    base_absorption: float = 1.0,
    n_slices: int = 20,
    res: int = 150,
    cmap: str = "Viridis",
    title: str = "SIREN prediction vs Ground truth",
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly non installé.")
        return

    z_levels = np.linspace(-H / 2 * 0.95, H / 2 * 0.95, n_slices)
    X, Y, mask = _make_xy_grid(R, res)
    xy = np.linspace(-R, R, res)

    all_pred, all_true, all_vals = [], [], []
    for z0 in z_levels:
        # SIREN : coordonnées cartésiennes
        xyz = np.stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), z0)], axis=-1)
        g_pred = np.array(fn(xyz)).reshape(res, res)

        # target_fn : coordonnées cylindriques normalisées
        x_flat, y_flat = X.ravel(), Y.ravel()
        r_flat = np.sqrt(x_flat**2 + y_flat**2)
        pts_cyl = jnp.stack([
            r_flat / R,
            x_flat / (r_flat + 1e-12),
            y_flat / (r_flat + 1e-12),
            2.0 * z0 / H * np.ones_like(r_flat),
        ], axis=-1)
        g_true = (np.array(target_fn(pts_cyl)).squeeze() / 5 + 1).reshape(res, res) * base_absorption

        all_pred.append(np.where(mask, g_pred, np.nan))
        all_true.append(np.where(mask, g_true, np.nan))
        all_vals.extend([g_pred[mask], g_true[mask]])

    vmin = float(np.nanmin(np.concatenate(all_vals)))
    vmax = float(np.nanmax(np.concatenate(all_vals)))

    heatmap_kwargs = dict(
        x=xy, y=xy, colorscale=cmap, zmin=vmin, zmax=vmax,
        showscale=True,
        hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<br>correction: %{z:.4f}<extra></extra>",
    )

    frames = [
        go.Frame(
            data=[
                go.Heatmap(z=all_pred[i], **heatmap_kwargs),
                go.Heatmap(z=all_true[i], **heatmap_kwargs),
            ],
            name=f"{z0:.1f}",
        )
        for i, z0 in enumerate(z_levels)
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("SIREN prediction", "Ground truth"),
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Heatmap(z=all_pred[0], **heatmap_kwargs), row=1, col=1)
    fig.add_trace(go.Heatmap(z=all_true[0], **heatmap_kwargs), row=1, col=2)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        width=1200,
        height=700,
        xaxis=dict(title="x (m)", constrain="domain"),
        yaxis=dict(title="y (m)", scaleanchor="x"),
        xaxis2=dict(title="x (m)", constrain="domain"),
        yaxis2=dict(title="y (m)", scaleanchor="x2"),
        sliders=[{
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                 "label": f"{z:.1f} m", "method": "animate"}
                for f, z in zip(frames, z_levels)
            ],
            "currentvalue": {"prefix": "z = ", "suffix": " m"},
            "pad": {"t": 50},
        }],
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 120}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
            "x": 0.05, "y": 1.0,
        }],
    )
    fig.frames = frames
    
    fig.show()
    fig.write_html("absorption_interactive.html")
    return

def plot_rz_heatmap(
    model,
    params,
    target_fn: Callable,
    R: float = 16.9,
    H: float = 38.0,
    theta_plot: float = np.pi / 2,
    Nr: int = 200,
    Nz: int = 200,
):
    """Heatmap r-z pour un angle theta fixé, comparant SIREN vs vérité terrain.

    Args:
        model      : modèle SIREN (Flax)
        params     : paramètres du modèle
        target_fn  : fonction cible (N, 4) → (N,) en coordonnées cylindriques normalisées
        R          : rayon du détecteur (m)
        H          : hauteur du détecteur (m)
        theta_plot : angle azimutal fixé (rad)
        Nr, Nz     : résolution de la grille r et z
    """
    r_vals = np.linspace(0, R, Nr)
    z_vals = np.linspace(-H / 2, H / 2, Nz)
    R_grid, Z_grid = np.meshgrid(r_vals, z_vals)  # (Nz, Nr)

    points = jnp.stack([
        R_grid.ravel() / R,
        jnp.full(Nr * Nz, jnp.cos(theta_plot)),
        jnp.full(Nr * Nz, jnp.sin(theta_plot)),
        2 * Z_grid.ravel() / H,
    ], axis=1)  # (Nr*Nz, 4)

    true = (target_fn(points) / 5 + 1).reshape(Nz, Nr)
    pred, _ = model.apply({"params": params}, points)
    pred = (pred / 5 + 1).reshape(Nz, Nr)
    err = (pred - true) / jnp.abs(true)

    vmin = float(jnp.minimum(true.min(), pred.min()))
    vmax = float(jnp.maximum(true.max(), pred.max()))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"r-z heatmap at θ = {theta_plot:.2f} rad", y=1.02)

    im0 = axs[0].pcolormesh(r_vals, z_vals, true, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title("True")
    axs[0].set_xlabel("r (m)")
    axs[0].set_ylabel("z (m)")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].pcolormesh(r_vals, z_vals, pred, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("SIREN")
    axs[1].set_xlabel("r (m)")
    plt.colorbar(im1, ax=axs[1])
    
    vmax = np.max(np.abs(err))
    
    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax
    )
    
    im2 = axs[2].pcolormesh(
        r_vals,
        z_vals,
        err,
        cmap="RdBu_r",  # ou "seismic", "bwr"
        norm=norm,
        shading="auto"
    )
    
    axs[2].set_title("Relative error")
    axs[2].set_xlabel("r (m)")
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()

def compute_chi_square(
    fn: Callable[[np.ndarray], np.ndarray],
    target_fn: Callable[[np.ndarray], np.ndarray],
    R: float = 16.9,
    H: float = 38.0,
    base_absorption: float = 1.0,
    n_slices: int = 20,
    res: int = 150,
) -> dict:
    """Compute the chi-square between the SIREN prediction and the ground truth
    over a 3D grid inside the cylindrical detector volume.

    The chi-square is defined as:
        χ² = Σ (pred_i - true_i)² / true_i

    and the reduced chi-square as:
        χ²_red = χ² / N_dof

    Args:
        fn              : SIREN prediction function (N, 3) → (N,) in Cartesian coordinates
        target_fn       : ground truth function (N, 4) → (N,) in normalised cylindrical coordinates
        R               : detector radius (m)
        H               : detector height (m)
        base_absorption : base value for renormalising target_fn
        n_slices        : number of z levels sampled
        res             : XY grid resolution per slice

    Returns:
        dict with keys:
            "chi2"         : total chi-square (float)
            "chi2_reduced" : reduced chi-square χ²/N (float)
            "n_points"     : number of valid points used (int)
            "rmse"         : RMSE between prediction and ground truth (float)
            "mean_rel_err" : mean relative error |pred - true| / |true| (float)
    """
    z_levels = np.linspace(-H / 2 * 0.95, H / 2 * 0.95, n_slices)
    X, Y, mask = _make_xy_grid(R, res)

    chi2_total = 0.0
    n_points = 0
    sq_errors = []
    rel_errors = []

    for z0 in z_levels:
        x_flat = X.ravel()
        y_flat = Y.ravel()

        # SIREN prediction in Cartesian coordinates
        xyz = np.stack([x_flat, y_flat, np.full_like(x_flat, z0)], axis=-1)
        g_pred = np.array(fn(xyz)).reshape(res, res)

        # Ground truth in normalised cylindrical coordinates
        r_flat = np.sqrt(x_flat**2 + y_flat**2)
        pts_cyl = jnp.stack([
            r_flat / R,
            x_flat / (r_flat + 1e-12),
            y_flat / (r_flat + 1e-12),
            2.0 * z0 / H * np.ones_like(r_flat),
        ], axis=-1)
        g_true = (np.array(target_fn(pts_cyl)).squeeze() / 5 + 1).reshape(res, res) * base_absorption

        # Keep only points inside the cylinder
        pred_valid = g_pred[mask]
        true_valid = g_true[mask]

        # Avoid division by zero
        nonzero = np.abs(true_valid) > 1e-12

        chi2_total += float(np.sum(
            (pred_valid[nonzero] - true_valid[nonzero]) ** 2 / true_valid[nonzero]
        ))
        sq_errors.append((pred_valid - true_valid) ** 2)
        rel_errors.append(np.abs(pred_valid[nonzero] - true_valid[nonzero]) / np.abs(true_valid[nonzero]))
        n_points += int(np.sum(nonzero))

    rmse = float(np.sqrt(np.mean(np.concatenate(sq_errors))))
    mean_rel_err = float(np.mean(np.concatenate(rel_errors)))

    results = {
        "chi2":         chi2_total,
        "chi2_reduced": chi2_total / n_points if n_points > 0 else np.nan,
        "n_points":     n_points,
        "rmse":         rmse,
        "mean_rel_err": mean_rel_err,
    }

    print(f"χ²                = {results['chi2']:.4f}")
    print(f"χ²/N (reduced)    = {results['chi2_reduced']:.6f}")
    print(f"N points          = {results['n_points']}")
    print(f"RMSE              = {results['rmse']:.6f}")
    print(f"Mean relative err = {results['mean_rel_err']:.4%}")

    return results