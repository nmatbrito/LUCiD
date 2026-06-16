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
from matplotlib.colors import Normalize
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