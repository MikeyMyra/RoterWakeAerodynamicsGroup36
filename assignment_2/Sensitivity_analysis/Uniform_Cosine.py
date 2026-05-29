import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lifting_line import BEM

bem = BEM(J=2)
tend = 5
dt = 0.1
bem.tlst = np.arange(0, tend, dt)
bem.rpm = 40

# Each entry: (label, kwargs passed to Lifting_line)
runs = [
    ('linear',  dict(resolution=10, spacing='linear',  track_convergence=True)),
    ('cosine',  dict(resolution=10, spacing='cosine',  track_convergence=True)),
]

a_out_lst     = []
aline_out_lst = []
Fnorm_out_lst = []
Ftan_out_lst  = []
Gamma_out_lst = []
r_control_lst = []
alpha_out_lst = []
conv_hist_lst = []
labels        = []

for label, kwargs in runs:
    output = bem.Lifting_line(**kwargs)
    a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, conv_iter, conv_hist, r_control, alpha_out = output

    a_out_lst.append(a_out)
    aline_out_lst.append(aline_out)
    Fnorm_out_lst.append(Fnorm_out)
    Ftan_out_lst.append(Ftan_out)
    Gamma_out_lst.append(Gamma_out)
    r_control_lst.append(r_control)
    alpha_out_lst.append(alpha_out)
    conv_hist_lst.append(conv_hist)
    labels.append(label)

blade_count = bem.n_blades


def extract_blade0(arr, station_count):
    """Return blade-0 slice, skipping hub point at index 0."""
    return np.asarray(arr[:station_count])[1:]


def get_r(r_control):
    return np.asarray(r_control)[1:]


# ------------------------------------------------------------------
# Common fine reference grid — bounded by the overlap of all runs
# so we never extrapolate outside any run's range.
# ------------------------------------------------------------------
N_REF = 300
r_min = max(get_r(rc)[0]  for rc in r_control_lst)
r_max = min(get_r(rc)[-1] for rc in r_control_lst)
r_ref = np.linspace(r_min, r_max, N_REF)


def interp_to_ref(arr, r_control):
    sc = len(r_control)
    r  = get_r(r_control)
    y  = extract_blade0(arr, sc)
    return np.interp(r_ref, r, y)


quantities = {
    'Gamma': Gamma_out_lst,
    'a':     a_out_lst,
    'aline': aline_out_lst,
    'Fnorm': Fnorm_out_lst,
    'Ftan':  Ftan_out_lst,
    'alpha': alpha_out_lst,
}
interp = {key: [interp_to_ref(lst[i], r_control_lst[i]) for i in range(len(runs))]
          for key, lst in quantities.items()}


def finish_axis(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel('r (m)')
    ax.set_ylabel(ylabel)
    try:
        blade_root_r = bem.blade_start_fraction * bem.radius
        ax.axvline(blade_root_r, color='k', linestyle='--', linewidth=1, label='blade root')
    except Exception:
        pass
    ax.legend()
    ax.grid(True)


# ------------------------------------------------------------------
# Show original (non-interpolated) r_control points as rug plots so
# it's clear where each spacing actually places its control points.
# ------------------------------------------------------------------
styles     = ['-o', '-s']
rug_colors = ['tab:blue', 'tab:orange']

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Lifting-line results — cosine vs linear spacing on common reference grid')

for i, label in enumerate(labels):
    style = styles[i % len(styles)]
    axs[0, 0].plot(r_ref, interp['Gamma'][i], style, label=label)
    axs[0, 1].plot(r_ref, interp['a'][i],     style, label=f'a  ({label})')
    axs[0, 1].plot(r_ref, interp['aline'][i], style, label=f"a' ({label})", alpha=0.6)
    axs[1, 0].plot(r_ref, interp['Fnorm'][i], style, label=f'Fnorm ({label})')
    axs[1, 0].plot(r_ref, interp['Ftan'][i],  style, label=f'Ftan  ({label})', alpha=0.6)
    axs[0, 2].plot(r_ref, interp['alpha'][i], style, label=label)

    # Rug: show where actual control points sit
    r_actual = get_r(r_control_lst[i])
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0], axs[0, 2]]:
        ax.plot(r_actual, np.full_like(r_actual, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0),
                '|', color=rug_colors[i], markersize=8, alpha=0.5)

finish_axis(axs[0, 0], 'Circulation vs radius',   r'$\Gamma$ (m²/s)')
finish_axis(axs[0, 1], 'Induction factors',        'Induction factor')
finish_axis(axs[1, 0], 'Section forces',           'Force per unit span (N/m)')
finish_axis(axs[0, 2], 'Angle of attack',          'AoA (deg)')

for i, (label, conv_hist) in enumerate(zip(labels, conv_hist_lst)):
    try:
        if conv_hist and len(conv_hist['error']) > 0:
            axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'],
                               styles[i % len(styles)], label=label)
    except Exception:
        pass

axs[1, 1].set_title('Convergence history')
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('Relative error')
axs[1, 1].legend()
axs[1, 1].grid(True)
axs[1, 2].axis('off')

plt.tight_layout()

# ------------------------------------------------------------------
# Difference plots: cosine vs linear, on common reference grid
# ------------------------------------------------------------------
diff_specs = [
    ('Gamma', r'$\Gamma$',  'Circulation'),
    ('a',     r'$a$',       'Axial induction factor'),
    ('aline', r"$a'$",      'Tangential induction factor'),
    ('Fnorm', r'$F_{norm}$','Normal force'),
    ('Ftan',  r'$F_{tan}$', 'Tangential force'),
]

lin_idx = labels.index('linear')
cos_idx = labels.index('cosine')

fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle('Percentage difference: cosine vs linear — evaluated on common reference grid')
axs2_flat = axs2.flatten()

for ax_idx, (key, sym, name) in enumerate(diff_specs):
    ax    = axs2_flat[ax_idx]
    y_lin = interp[key][lin_idx]
    y_cos = interp[key][cos_idx]
    denom = np.where(np.abs(y_lin) > 1e-10, np.abs(y_lin), np.nan)
    diff  = np.abs(y_lin - y_cos) / denom * 100
    ax.plot(r_ref, diff, '-o', label='cosine vs linear')

    # Shade tip and root regions where cosine adds extra density
    r_actual_lin = get_r(r_control_lst[lin_idx])
    dr_lin = np.mean(np.diff(r_actual_lin))
    ax.axvspan(r_ref[0],  r_ref[0]  + dr_lin, alpha=0.1, color='green', label='cosine-dense region')
    ax.axvspan(r_ref[-1] - dr_lin, r_ref[-1], alpha=0.1, color='green')

    finish_axis(ax, f'{name} difference', f'|Δ{sym}| (%)')

axs2_flat[-1].axis('off')
plt.tight_layout()
plt.show()