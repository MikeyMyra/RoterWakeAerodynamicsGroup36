"""Interactive Plotly rendering of the rotor trailing-wake vortex surface.

Uses the same helical geometry as plot_wake_surface.py (via build_wake_nodes),
but renders with Plotly's WebGL surfaces. Unlike matplotlib's painter's
algorithm, Plotly has a real depth buffer, so the intertwined blade sheets
occlude each other correctly even where they interpenetrate.

Opens an interactive HTML view in the browser (drag to rotate, scroll to zoom).
"""

import numpy as np
import plotly.graph_objects as go

from plot_wake_surface import build_wake_nodes
from Lifting_line import BEM


# distinct, high-contrast hue per blade
BLADE_COLORS = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8',
                '#f58231', '#42d4f4', '#911eb4', '#a9a9a9']


def wake_figure(bem, nodes):
    """Return a Plotly Figure of the wake sheets, one Surface per blade."""
    fig = go.Figure()

    for b in range(bem.n_blades):
        X = nodes[b, :, :, 0]
        Y = nodes[b, :, :, 1]
        Z = nodes[b, :, :, 2]
        color = BLADE_COLORS[b % len(BLADE_COLORS)]

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=np.full_like(Z, b),      # constant -> flat blade color
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=1.0,
            name=f'blade {b + 1}',
            showlegend=True,
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.2,
                          roughness=0.6, fresnel=0.1),
            lightposition=dict(x=-1000, y=-1000, z=2000),
            contours=dict(  # thin isolines convey the helical winding
                x=dict(highlight=False),
                y=dict(highlight=False),
                z=dict(highlight=False),
            ),
        ))

    # rotor disk edge for reference
    circ = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scatter3d(
        x=np.zeros_like(circ), y=bem.radius * np.cos(circ), z=bem.radius * np.sin(circ),
        mode='lines', line=dict(color='black', width=4),
        name='rotor disk',
    ))

    # equal-ish aspect: axial axis is much longer than the disk
    pts = nodes.reshape(-1, 3)
    x_range = pts[:, 0].max() - pts[:, 0].min()
    yz_range = max(np.ptp(pts[:, 1]), np.ptp(pts[:, 2]))
    fig.update_layout(
        title=f'Wake vortex surface ({bem.n_blades} blades)',
        scene=dict(
            xaxis_title='x (axial, m)',
            yaxis_title='y (m)',
            zaxis_title='z (m)',
            aspectmode='manual',
            aspectratio=dict(x=x_range / yz_range, y=1, z=1),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


if __name__ == "__main__":
    bem = BEM(J=1.6, radius=0.7, n_blades=6, U_inf=60)
    bem.tlst = np.arange(0, 0.2, 0.0015)

    nodes = build_wake_nodes(bem, resolution=80, a_ind_wake=-0.2, spacing='cosine')

    fig = wake_figure(bem, nodes)
    fig.show()   # opens interactive view in the default browser
