import numpy as np
from scipy.interpolate import interp1d
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – needed for 3-D projection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── airfoil polar ─────────────────────────────────────────────────────────────

polar_alpha = [-180, -16.062, -15.506, -15.064, -14.589, -14.109, -13.698,
               -13.237, -12.745, -12.268, -11.748, -11.183, -10.768, -10.231,
               -9.743, -9.223, -8.209, -7.187, -6.162, -5.143, -4.127, -3.106,
               -2.073, -1.04, .017, 1.025, 2.042, 3.096, 4.114, 5.126, 6.163,
               7.189, 7.713, 8.216, 8.734, 9.251, 9.558, 9.771, 10.269, 10.757,
               11.257, 11.761, 12.239, 13.224, 14.234, 15.227, 16.208, 17.201,
               18.2, 19.201, 20.189, 21.179, 22.162, 23.144, 23.547, 24.062,
               25.056, 26.06, 27.059, 28.062, 29.062, 30.056, 180]

polar_cl = [-0, -.425, -.43, -.461, -.496, -.567, -.682, -.719, -.755, -.77,
            -.774, -.77, -.756, -.736, -.711, -.68, -.612, -.529, -.434, -.337,
            -.237, -.127, -.014, .102, .217, .33, .445, .571, .683, .792, .902,
            1.014, 1.068, 1.12, 1.168, 1.207, 1.227, 1.229, 1.215, 1.192,
            1.173, 1.148, 1.126, 1.103, 1.093, 1.077, 1.06, 1.038, 1.047,
            1.043, 1.04, 1.029, 1.023, 1.007, .829, .854, .877, .89, .953,
            .976, 1.016, 1.041, 0]

polar_cd = [0.1, .225, .217, .211, .208, .201, .077, .05, .039, .033, .029,
            .025, .023, .02, .018, .017, .014, .013, .011, .01, .008, .008,
            .008, .007, .007, .008, .008, .008, .008, .009, .009, .009, .009,
            .01, .01, .01, .012, .013, .019, .025, .036, .045, .053, .066,
            .076, .089, .104, .117, .132, .152, .172, .198, .228, .261, .416,
            .436, .466, .491, .541, .574, .616, .652, 0.1]

_cl_interp = interp1d(polar_alpha, polar_cl, kind='linear', fill_value='extrapolate')
_cd_interp = interp1d(polar_alpha, polar_cd, kind='linear', fill_value='extrapolate')


def polar_airfoil(alpha):
    return float(_cl_interp(alpha)), float(_cd_interp(alpha))


def geo_blade(r_R):
    pitch = 2.0
    chord = 3.0 * (1 - r_R) + 1
    twist = -14.0 * (1 - r_R)
    return chord, twist + pitch


def velocity_3d_from_vortex_filament(gamma, xv1, xv2, xvp, core=1e-5):
    x1, y1, z1 = xv1;  x2, y2, z2 = xv2;  xp, yp, zp = xvp
    r1 = np.sqrt((xp-x1)**2 + (yp-y1)**2 + (zp-z1)**2)
    r2 = np.sqrt((xp-x2)**2 + (yp-y2)**2 + (zp-z2)**2)
    r1xr2_x = (yp-y1)*(zp-z2) - (zp-z1)*(yp-y2)
    r1xr2_y = -(xp-x1)*(zp-z2) + (zp-z1)*(xp-x2)
    r1xr2_z = (xp-x1)*(yp-y2) - (yp-y1)*(xp-x2)
    r1xr_sqr = r1xr2_x**2 + r1xr2_y**2 + r1xr2_z**2
    r0r1 = (x2-x1)*(xp-x1) + (y2-y1)*(yp-y1) + (z2-z1)*(zp-z1)
    r0r2 = (x2-x1)*(xp-x2) + (y2-y1)*(yp-y2) + (z2-z1)*(zp-z2)
    if r1xr_sqr < core**2: r1xr_sqr = core**2
    if r1 < core:          r1 = core
    if r2 < core:          r2 = core
    k = gamma / (4 * np.pi * r1xr_sqr) * (r0r1/r1 - r0r2/r2)
    return np.array([k*r1xr2_x, k*r1xr2_y, k*r1xr2_z])


def velocity_induced_single_ring(ring, controlpoint, core=1e-5):
    vel = np.zeros(3)
    for fil in ring['filaments']:
        xv1 = np.array([fil['x1'], fil['y1'], fil['z1']])
        xv2 = np.array([fil['x2'], fil['y2'], fil['z2']])
        vel += velocity_3d_from_vortex_filament(fil['Gamma'], xv1, xv2, controlpoint, core)
    return vel


def update_gamma_single_ring(ring, gamma_new, weight_new):
    for fil in ring['filaments']:
        fil['Gamma'] = fil['Gamma'] * (1 - weight_new) + weight_new * gamma_new
    return ring


def load_blade_element(v_norm, v_tan, r_R):
    vmag2        = v_norm**2 + v_tan**2
    inflow_angle = np.arctan2(v_norm, v_tan)
    chord, twist = geo_blade(r_R)
    alpha        = twist + np.degrees(inflow_angle)
    cl, cd       = polar_airfoil(alpha)
    cd           = 0.0
    lift  = 0.5 * vmag2 * cl * chord
    drag  = 0.5 * vmag2 * cd * chord
    f_norm = lift * np.cos(inflow_angle) + drag * np.sin(inflow_angle)
    f_tan  = lift * np.sin(inflow_angle) - drag * np.cos(inflow_angle)
    gamma  = 0.5 * np.sqrt(vmag2) * cl * chord
    return f_norm, f_tan, gamma


def create_rotor_geometry(span_array, radius, tip_speed_ratio, u_inf, theta_array, n_blades):
    controlpoints = []
    rings         = []

    for k_rot in range(n_blades):
        angle_rotation = 2 * np.pi / n_blades * k_rot
        cos_rot = np.cos(angle_rotation)
        sin_rot = np.sin(angle_rotation)

        for i in range(len(span_array) - 1):
            r        = (span_array[i] + span_array[i+1]) / 2
            chord, twist = geo_blade(r / radius)
            angle    = np.radians(twist)

            cp_coords = np.array([0.0, r, 0.0])
            cp_normal = np.array([np.cos(angle), 0.0, -np.sin(angle)])
            cp_tang   = np.array([-np.sin(angle), 0.0, -np.cos(angle)])

            def rot_yz(v, c=cos_rot, s=sin_rot):
                return np.array([v[0], v[1]*c - v[2]*s, v[1]*s + v[2]*c])

            controlpoints.append({
                'coordinates': rot_yz(cp_coords),
                'chord':       chord,
                'normal':      rot_yz(cp_normal),
                'tangential':  rot_yz(cp_tang),
            })

            filaments = []
            filaments.append({'x1': 0, 'y1': span_array[i],   'z1': 0,
                              'x2': 0, 'y2': span_array[i+1], 'z2': 0, 'Gamma': 0})

            chord_i, twist_i = geo_blade(span_array[i] / radius)
            angle_i = np.radians(twist_i)
            x_te = chord_i * np.sin(-angle_i)
            z_te = -chord_i * np.cos(angle_i)
            filaments.append({'x1': x_te, 'y1': span_array[i], 'z1': z_te,
                              'x2': 0,    'y2': span_array[i], 'z2': 0,   'Gamma': 0})

            for j in range(len(theta_array) - 1):
                xt = filaments[-1]['x1']; yt = filaments[-1]['y1']; zt = filaments[-1]['z1']
                dy = (np.cos(-theta_array[j+1]) - np.cos(-theta_array[j])) * span_array[i]
                dz = (np.sin(-theta_array[j+1]) - np.sin(-theta_array[j])) * span_array[i]
                dx = (theta_array[j+1] - theta_array[j]) / tip_speed_ratio * radius
                filaments.append({'x1': xt+dx, 'y1': yt+dy, 'z1': zt+dz,
                                  'x2': xt,    'y2': yt,    'z2': zt,    'Gamma': 0})

            chord_o, twist_o = geo_blade(span_array[i+1] / radius)
            angle_o = np.radians(twist_o)
            x_te2 = chord_o * np.sin(-angle_o)
            z_te2 = -chord_o * np.cos(angle_o)
            filaments.append({'x1': 0,    'y1': span_array[i+1], 'z1': 0,
                              'x2': x_te2,'y2': span_array[i+1], 'z2': z_te2, 'Gamma': 0})

            for j in range(len(theta_array) - 1):
                xt = filaments[-1]['x2']; yt = filaments[-1]['y2']; zt = filaments[-1]['z2']
                dy = (np.cos(-theta_array[j+1]) - np.cos(-theta_array[j])) * span_array[i+1]
                dz = (np.sin(-theta_array[j+1]) - np.sin(-theta_array[j])) * span_array[i+1]
                dx = (theta_array[j+1] - theta_array[j]) / tip_speed_ratio * radius
                filaments.append({'x1': xt,    'y1': yt,    'z1': zt,
                                  'x2': xt+dx, 'y2': yt+dy, 'z2': zt+dz, 'Gamma': 0})

            for fil in filaments:
                y1r = fil['y1']*cos_rot - fil['z1']*sin_rot
                z1r = fil['y1']*sin_rot + fil['z1']*cos_rot
                y2r = fil['y2']*cos_rot - fil['z2']*sin_rot
                z2r = fil['y2']*sin_rot + fil['z2']*cos_rot
                fil['y1'], fil['z1'] = y1r, z1r
                fil['y2'], fil['z2'] = y2r, z2r

            rings.append({'filaments': filaments})

    return {'controlpoints': controlpoints, 'rings': rings}


def solve_lifting_line_system_matrix_approach(rotor_wake_system, wind, omega, rotor_radius,
                                               n_iterations=1200, error_limit=0.01,
                                               conv_weight=0.3):
    controlpoints = rotor_wake_system['controlpoints']
    rings         = rotor_wake_system['rings']
    wind          = np.array(wind)
    n_cp    = len(controlpoints)
    n_rings = len(rings)

    MatrixU = np.zeros((n_cp, n_rings))
    MatrixV = np.zeros((n_cp, n_rings))
    MatrixW = np.zeros((n_cp, n_rings))

    for icp in range(n_cp):
        for jring in range(n_rings):
            rings[jring] = update_gamma_single_ring(rings[jring], 1, 1)
            vel = velocity_induced_single_ring(rings[jring], controlpoints[icp]['coordinates'])
            MatrixU[icp, jring] = vel[0]
            MatrixV[icp, jring] = vel[1]
            MatrixW[icp, jring] = vel[2]

    Gamma    = np.zeros(n_rings)
    GammaNew = np.zeros(n_rings)
    a_out = np.zeros(n_cp); aline_out = np.zeros(n_cp); r_R_out = np.zeros(n_cp)
    Fnorm_out = np.zeros(n_cp); Ftan_out = np.zeros(n_cp); Gamma_out = np.zeros(n_cp)

    for kiter in range(n_iterations):
        Gamma[:] = GammaNew
        for icp in range(n_cp):
            coords = np.array(controlpoints[icp]['coordinates'])
            r_abs  = np.linalg.norm(coords)
            u = MatrixU[icp] @ Gamma
            v = MatrixV[icp] @ Gamma
            w = MatrixW[icp] @ Gamma
            v_rot  = np.cross(np.array([-omega, 0, 0]), coords)
            vel1   = wind + np.array([u, v, w]) + v_rot
            azim_dir = np.cross(np.array([-1/r_abs, 0, 0]), coords)
            v_azim   = np.dot(azim_dir, vel1)
            v_axial  = np.dot(np.array([1, 0, 0]), vel1)
            f_norm, f_tan, gamma_new = load_blade_element(v_axial, v_azim, r_abs / rotor_radius)
            GammaNew[icp] = gamma_new
            a_out[icp]     = -(u + v_rot[0]) / wind[0]
            aline_out[icp] = v_azim / (r_abs * omega) - 1
            r_R_out[icp]   = r_abs / rotor_radius
            Fnorm_out[icp] = f_norm
            Ftan_out[icp]  = f_tan
            Gamma_out[icp] = gamma_new

        ref_error = max(np.max(np.abs(GammaNew)), 0.001)
        error     = np.max(np.abs(GammaNew - Gamma)) / ref_error
        if error < error_limit:
            print(f"Converged after {kiter+1} iterations, error={error:.6f}")
            break
        GammaNew = (1 - conv_weight) * Gamma + conv_weight * GammaNew

    return {'a': a_out, 'aline': aline_out, 'r_R': r_R_out,
            'Fnorm': Fnorm_out, 'Ftan': Ftan_out, 'Gamma': Gamma_out}


def solve_rotor_lifting_line(TSR, n_elements, n_rotations):
    s_raw         = np.linspace(0, np.pi, n_elements + 1)
    r_array_norm  = (-(np.cos(s_raw) - 1) / 2) * 0.8 + 0.2
    radius        = 50.0
    span_array    = r_array_norm * radius
    max_radius    = np.max(span_array)
    u_inf         = 1.0
    n_blades      = 3
    theta_array   = np.arange(0, n_rotations * 2 * np.pi, np.pi / 10)

    rotor_wake_system = create_rotor_geometry(
        span_array, max_radius,
        tip_speed_ratio = TSR / (1 - 0.2),
        u_inf           = u_inf,
        theta_array     = theta_array,
        n_blades        = n_blades,
    )
    results = solve_lifting_line_system_matrix_approach(
        rotor_wake_system, wind=[1, 0, 0],
        omega=TSR / max_radius, rotor_radius=max_radius,
    )
    return results, rotor_wake_system


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def make_plots(results, wake, TSR):
    r_R    = results['r_R']
    a      = results['a']
    aline  = results['aline']
    Gamma  = results['Gamma']
    Fnorm  = results['Fnorm']
    Ftan   = results['Ftan']

    # ── per-blade strip (first blade: first n_el control points) ──────────────
    n_el   = len(r_R) // 3          # 3 blades
    r_R1   = r_R[:n_el]
    a1     = a[:n_el]
    al1    = aline[:n_el]
    G1     = Gamma[:n_el]
    Fn1    = Fnorm[:n_el]
    Ft1    = Ftan[:n_el]

    dark   = "#0d1117"
    mid    = "#161b22"
    grid_c = "#30363d"
    c1, c2, c3 = "#58a6ff", "#3fb950", "#f78166"
    c4, c5     = "#d2a8ff", "#ffa657"

    plt.rcParams.update({
        "figure.facecolor":  dark,
        "axes.facecolor":    mid,
        "axes.edgecolor":    grid_c,
        "axes.labelcolor":   "#e6edf3",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "grid.color":        grid_c,
        "grid.linewidth":    0.6,
        "text.color":        "#e6edf3",
        "font.family":       "monospace",
        "legend.facecolor":  mid,
        "legend.edgecolor":  grid_c,
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1 – 2 × 2 blade-span results
    # ─────────────────────────────────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle(f"Lifting-Line Rotor Results  |  TSR = {TSR}", fontsize=13,
                  color="#e6edf3", y=0.97)

    kw  = dict(marker='o', markersize=4, linewidth=1.8)
    kw2 = dict(marker='s', markersize=4, linewidth=1.8)

    ax = axes[0, 0]
    ax.plot(r_R1, a1,  color=c1, label='Axial induction  a', **kw)
    ax.plot(r_R1, al1, color=c2, label="Tang. induction  a'", **kw2)
    ax.set_xlabel("r / R");  ax.set_ylabel("Induction factor")
    ax.set_title("Induction Factors"); ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(r_R1, G1, color=c3, **kw)
    ax.set_xlabel("r / R"); ax.set_ylabel("Γ  [m²/s]")
    ax.set_title("Bound Circulation Γ(r)"); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(r_R1, Fn1, color=c4, label='Normal', **kw)
    ax.plot(r_R1, Ft1, color=c5, label='Tangential', **kw2)
    ax.set_xlabel("r / R"); ax.set_ylabel("Force  [N/m]")
    ax.set_title("Blade Loads"); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    r_blade   = np.linspace(0.2, 1.0, 100)
    chord_arr = np.array([geo_blade(r)[0] for r in r_blade])
    twist_arr = np.array([geo_blade(r)[1] for r in r_blade])
    ax2b = ax.twinx()
    ax.plot(r_blade, chord_arr, color=c1, linewidth=2, label='Chord [m]')
    ax2b.plot(r_blade, twist_arr, color=c3, linewidth=2, linestyle='--', label='Twist+Pitch [°]')
    ax.set_xlabel("r / R"); ax.set_ylabel("Chord  [m]", color=c1)
    ax2b.set_ylabel("Twist + Pitch  [°]", color=c3)
    ax.set_title("Blade Geometry")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc='lower right'); ax.grid(True)
    ax2b.tick_params(axis='y', colors=c3)
    ax.tick_params(axis='y', colors=c1)

    #fig1.tight_layout(rect=[0, 0, 1, 0.96])
    #fig1.savefig("/mnt/user-data/outputs/rotor_results_2d.png", dpi=140, bbox_inches='tight')
    print("Saved rotor_results_2d.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2 – 3-D wake geometry (filaments coloured by radial position)
    # ─────────────────────────────────────────────────────────────────────────
    fig2 = plt.figure(figsize=(12, 9))
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.set_facecolor(mid)
    fig2.patch.set_facecolor(dark)
    ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor(grid_c)
    ax3d.yaxis.pane.set_edgecolor(grid_c)
    ax3d.zaxis.pane.set_edgecolor(grid_c)
    ax3d.tick_params(colors="#8b949e")

    cmap   = plt.cm.plasma
    radius = 50.0

    # collect all filament endpoints → radial position for colour
    all_fils = []
    for ring in wake['rings']:
        for fil in ring['filaments']:
            r_mid = 0.5 * np.sqrt((fil['y1']**2 + fil['z1']**2) +
                                   (fil['y2']**2 + fil['z2']**2))
            all_fils.append((fil, r_mid))

    r_vals = np.array([x[1] for x in all_fils])
    norm   = Normalize(vmin=r_vals.min(), vmax=r_vals.max())

    for fil, r_mid in all_fils:
        col = cmap(norm(r_mid))
        ax3d.plot([fil['x1'], fil['x2']],
                  [fil['y1'], fil['y2']],
                  [fil['z1'], fil['z2']],
                  color=col, linewidth=0.6, alpha=0.7)

    # draw rotor disc outline
    theta_disc = np.linspace(0, 2*np.pi, 120)
    ax3d.plot(np.zeros(120), radius*np.cos(theta_disc), radius*np.sin(theta_disc),
              color='white', linewidth=1.2, linestyle='--', alpha=0.5)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig2.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.1)
    cb.set_label("Radial position  [m]", color="#e6edf3")
    cb.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

    ax3d.set_xlabel("x (downstream) [m]", labelpad=8)
    ax3d.set_ylabel("y [m]",              labelpad=8)
    ax3d.set_zlabel("z [m]",              labelpad=8)
    ax3d.set_title(f"3-D Wake Geometry  |  TSR = {TSR}", fontsize=13, pad=14)
    ax3d.view_init(elev=25, azim=-55)

    fig2.tight_layout()
    #fig2.savefig("/mnt/user-data/outputs/rotor_wake_3d.png", dpi=140, bbox_inches='tight')
    #print("Saved rotor_wake_3d.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 3 – 3-D blade surface: chord + twist visualised as quads
    # ─────────────────────────────────────────────────────────────────────────
    fig3 = plt.figure(figsize=(11, 7))
    ax3b = fig3.add_subplot(111, projection='3d')
    ax3b.set_facecolor(mid)
    fig3.patch.set_facecolor(dark)
    for pane in [ax3b.xaxis.pane, ax3b.yaxis.pane, ax3b.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor(grid_c)
    ax3b.tick_params(colors="#8b949e")

    n_r     = 60
    n_blades = 3
    r_norm  = np.linspace(0.2, 1.0, n_r)
    chords  = np.array([geo_blade(r)[0] for r in r_norm])
    twists  = np.array([np.radians(geo_blade(r)[1]) for r in r_norm])
    r_abs   = r_norm * radius

    cmap2   = plt.cm.cool
    norm2   = Normalize(vmin=0, vmax=radius)

    for kb in range(n_blades):
        blade_angle = 2*np.pi/n_blades * kb
        # leading edge & trailing edge in local (x,y,z)
        le_x = -0.25 * chords * np.sin(-twists)
        le_z =  0.25 * chords * np.cos(twists)
        te_x =  0.75 * chords * np.sin(-twists)
        te_z = -0.75 * chords * np.cos(twists)

        # rotate into rotor plane
        y_rot  = r_abs * np.cos(blade_angle)
        z_rot  = r_abs * np.sin(blade_angle)

        for i in range(n_r - 1):
            # quad vertices: LE inner, LE outer, TE outer, TE inner
            xs = [le_x[i],   le_x[i+1], te_x[i+1], te_x[i]]
            ys = [y_rot[i] + (le_z[i]*-np.sin(blade_angle)),
                  y_rot[i+1] + (le_z[i+1]*-np.sin(blade_angle)),
                  y_rot[i+1] + (te_z[i+1]*-np.sin(blade_angle)),
                  y_rot[i]   + (te_z[i]*-np.sin(blade_angle))]
            zs = [z_rot[i] + (le_z[i]*np.cos(blade_angle)),
                  z_rot[i+1] + (le_z[i+1]*np.cos(blade_angle)),
                  z_rot[i+1] + (te_z[i+1]*np.cos(blade_angle)),
                  z_rot[i]   + (te_z[i]*np.cos(blade_angle))]
            col = cmap2(norm2((r_abs[i] + r_abs[i+1]) / 2))
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection([[list(zip(xs, ys, zs))[0],
                                      list(zip(xs, ys, zs))[1],
                                      list(zip(xs, ys, zs))[2],
                                      list(zip(xs, ys, zs))[3]]],
                                    alpha=0.85)
            poly.set_facecolor(col)
            poly.set_edgecolor("none")
            ax3b.add_collection3d(poly)

    sm2 = ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    cb2 = fig3.colorbar(sm2, ax=ax3b, shrink=0.5, pad=0.1)
    cb2.set_label("Span  [m]", color="#e6edf3")
    cb2.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="#8b949e")

    ax3b.set_xlim(-20, 20); ax3b.set_ylim(-radius*1.1, radius*1.1)
    ax3b.set_zlim(-radius*1.1, radius*1.1)
    ax3b.set_xlabel("x [m]", labelpad=8)
    ax3b.set_ylabel("y [m]", labelpad=8)
    ax3b.set_zlabel("z [m]", labelpad=8)
    ax3b.set_title("3-D Blade Geometry  (chord × twist)", fontsize=13, pad=14)
    ax3b.view_init(elev=30, azim=20)

    fig3.tight_layout()
    #fig3.savefig("/mnt/user-data/outputs/rotor_blade_3d.png", dpi=140, bbox_inches='tight')
    #print("Saved rotor_blade_3d.png")
    plt.show()
    #plt.close('all')


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TSR         = 8.0
    N_ELEMENTS  = 10
    N_ROTATIONS = 0.12

    results, wake = solve_rotor_lifting_line(TSR, N_ELEMENTS, N_ROTATIONS)
    make_plots(results, wake, TSR)
    #print("All plots saved.")