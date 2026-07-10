"""Plot the trailing wake vortex surface of the rotor.

Reconstructs the same helical wake geometry that make_the_rotor() builds
inside BEM.Lifting_line(), then renders each blade's wake as a 3D surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Lifting_line import BEM


def build_wake_nodes(bem, resolution=80, a_ind_wake=-0.2, spacing='cosine'):
    """Return blade-wake node arrays with shape (n_blades, n_stations, n_theta, 3)."""

    omega = (2 * np.pi * bem.rpm) / 60

    # --- radial stations (identical to Lifting_line) ---
    if spacing == 'cosine':
        theta = np.linspace(0, np.pi, resolution + 1)
        r_norm_tmp = 0.5 * (1 - np.cos(theta))
        r_stations_norm = bem.blade_start_fraction + (1 - bem.blade_start_fraction) * r_norm_tmp
    else:
        r_stations_norm = np.linspace(bem.blade_start_fraction, 1, resolution + 1)

    r_stations_norm = np.insert(r_stations_norm, 0, 0)
    r_stations_abs = r_stations_norm * bem.radius

    # blade properties at each station node
    twist = np.where(r_stations_norm >= bem.blade_start_fraction,
                     -50 * r_stations_norm + 35 + bem.collective_blade_pitch
                     + bem.collective_blade_pitch_location * 50 - 35, 0.0)
    chord = np.where(r_stations_norm >= bem.blade_start_fraction,
                     (0.18 - 0.06 * r_stations_norm) * bem.radius, 0.0)

    # only stations on the blade shed a wake
    on_blade = r_stations_norm >= bem.blade_start_fraction

    theta_array = omega * bem.tlst          # wake azimuth per shed step
    n_theta = len(theta_array)

    def rot_yz(vec, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([vec[0], vec[1] * c - vec[2] * s, vec[1] * s + vec[2] * c])

    stations = np.where(on_blade)[0]
    nodes = np.zeros((bem.n_blades, len(stations), n_theta, 3))

    for b in range(bem.n_blades):
        angle_rotation = 2 * np.pi / bem.n_blades * b
        for si, i in enumerate(stations):
            r = r_stations_abs[i]
            twist_rad = np.radians(twist[i])
            # wake starts a quarter chord behind the TE (1.25 * chord from the LE)
            x_start = 1.25 * chord[i] * np.sin(-twist_rad)
            z_start = -1.25 * chord[i] * np.cos(twist_rad)

            # cumulative helix (matches the incremental construction in the solver)
            x = x_start + theta_array / (np.pi / (bem.radius * bem.J)) * (1 + a_ind_wake)
            y = r * np.cos(theta_array)
            z = z_start - r * np.sin(theta_array)

            pts = np.vstack([x, y, z])            # (3, n_theta)
            pts = rot_yz(pts, angle_rotation)     # rotate into blade frame
            nodes[b, si] = pts.T

    return nodes


if __name__ == "__main__":
    bem = BEM(J=1.6, radius=0.7, n_blades=6, U_inf=60)
    bem.tlst = np.arange(0, 0.2, 0.0015)   # finer azimuth -> smooth helices

    nodes = build_wake_nodes(bem, resolution=80, a_ind_wake=-0.2, spacing='cosine')

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(projection='3d')
    # perspective (not ortho) gives real foreshortening -> readable depth
    ax.set_proj_type('persp', focal_length=0.5)

    base_colors = plt.cm.turbo(np.linspace(0.05, 0.95, bem.n_blades))
    light = np.array([-0.3, -0.4, 0.85])
    light /= np.linalg.norm(light)

    # Build EVERY blade's quads into ONE Poly3DCollection so matplotlib
    # depth-sorts all faces together -> correct interleaving between sheets.
    r_stride, t_stride = 2, 2
    verts, facecolors = [], []
    for b in range(bem.n_blades):
        P = nodes[b]                        # (n_stations, n_theta, 3)
        ns, nt = P.shape[:2]
        for i in range(0, ns - 1, r_stride):
            i2 = min(i + r_stride, ns - 1)
            for j in range(0, nt - 1, t_stride):
                j2 = min(j + t_stride, nt - 1)
                quad = [P[i, j], P[i, j2], P[i2, j2], P[i2, j]]
                verts.append(quad)
                # two-sided lambert shade from the quad normal
                n = np.cross(quad[1] - quad[0], quad[3] - quad[0])
                nn = np.linalg.norm(n)
                shade = 0.5 if nn == 0 else 0.4 + 0.6 * abs(np.dot(n / nn, light))
                facecolors.append(np.clip(base_colors[b][:3] * shade, 0, 1))

    surf = Poly3DCollection(verts, facecolors=facecolors,
                            edgecolors=(0, 0, 0, 0.12), linewidths=0.15)
    surf.set_zsort('average')               # back-to-front face ordering
    ax.add_collection3d(surf)

    # draw the rotor disk edge for reference
    circ = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.zeros_like(circ), bem.radius * np.cos(circ), bem.radius * np.sin(circ),
            'k-', lw=1.5, label='rotor disk')

    # add_collection3d does not autoscale -> set limits from the node cloud
    pts = nodes.reshape(-1, 3)
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())

    ax.set_title(f'Wake vortex surface ({bem.n_blades} blades)')
    ax.set_xlabel('x (axial, m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend(loc='upper right')
    ax.view_init(elev=18, azim=-125)     # look slightly down the axis
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    try:
        ax.set_box_aspect((3, 1, 1))
    except Exception:
        pass
    plt.tight_layout()
    plt.show()
