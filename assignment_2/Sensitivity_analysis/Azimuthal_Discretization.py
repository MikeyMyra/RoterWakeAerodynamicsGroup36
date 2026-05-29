import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lifting_line import BEM

res_lst = np.array([5, 10, 20])
bem = BEM(J=2, radius=0.7, n_blades=6, U_inf=60)
tend = 5
dt = 0.1
bem.tlst = np.arange(0, tend, dt)
bem.rpm = 40

a_out_lst     = []
aline_out_lst = []
Fnorm_out_lst = []
Ftan_out_lst  = []
Gamma_out_lst = []
r_control_lst = []
alpha_out_lst = []

for res in res_lst:
    output = bem.Lifting_line(resolution=res, track_convergence=True)
    a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, conv_iter, conv_hist, r_control, alpha_out = output

    a_out_lst.append(a_out)
    aline_out_lst.append(aline_out)
    Fnorm_out_lst.append(Fnorm_out)
    Ftan_out_lst.append(Ftan_out)
    Gamma_out_lst.append(Gamma_out)
    r_control_lst.append(r_control)
    alpha_out_lst.append(alpha_out)

blade_count = bem.n_blades


def extract_blade0(arr, station_count):
    """Return blade-0 slice, skipping hub point at index 0."""
    return np.asarray(arr[:station_count])[1:]


def get_r(r_control):
    """r values skipping hub point."""
    return np.asarray(r_control)[1:]


# ------------------------------------------------------------------
# Build common fine reference grid
# Use the innermost and outermost r values across all runs so we
# never extrapolate, then fill with N_REF points.
# ------------------------------------------------------------------
N_REF = 300
r_min = max(get_r(rc)[0]  for rc in r_control_lst)   # innermost safe point
r_max = min(get_r(rc)[-1] for rc in r_control_lst)   # outermost safe point
r_ref = np.linspace(r_min, r_max, N_REF)


def interp_to_ref(arr, r_control):
    sc = len(r_control)
    r  = get_r(r_control)
    y  = extract_blade0(arr, sc)
    return np.interp(r_ref, r, y)


# Interpolate every run onto the common grid
quantities = {
    'Gamma': Gamma_out_lst,
    'a':     a_out_lst,
    'aline': aline_out_lst,
    'Fnorm': Fnorm_out_lst,
    'Ftan':  Ftan_out_lst,
    'alpha': alpha_out_lst,
}
interp = {key: [interp_to_ref(lst[i], r_control_lst[i]) for i in range(len(res_lst))]
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
# Main overlay plots (on common grid)
# ------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Lifting-line results — all resolutions on common reference grid')

styles = ['-o', '-s', '-^']

for i, res in enumerate(res_lst):
    style = styles[i % len(styles)]
    axs[0, 0].plot(r_ref, interp['Gamma'][i], style, label=f'res={res}')
    axs[0, 1].plot(r_ref, interp['a'][i],     style, label=f'a  res={res}')
    axs[0, 1].plot(r_ref, interp['aline'][i], style, label=f"a' res={res}", alpha=0.6)
    axs[1, 0].plot(r_ref, interp['Fnorm'][i], style, label=f'Fnorm res={res}')
    axs[1, 0].plot(r_ref, interp['Ftan'][i],  style, label=f'Ftan  res={res}', alpha=0.6)
    axs[0, 2].plot(r_ref, interp['alpha'][i], style, label=f'res={res}')

finish_axis(axs[0, 0], 'Circulation vs radius',   r'$\Gamma$ (m²/s)')
finish_axis(axs[0, 1], 'Induction factors',        'Induction factor')
finish_axis(axs[1, 0], 'Section forces',           'Force per unit span (N/m)')
finish_axis(axs[0, 2], 'Angle of attack',          'AoA (deg)')

# Convergence history (raw iteration data, no interpolation needed)
for i, res in enumerate(res_lst):
    output = bem.Lifting_line(resolution=res, track_convergence=True)
    *_, conv_iter, conv_hist, _, _ = output
    try:
        if conv_hist and len(conv_hist['error']) > 0:
            axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'],
                               styles[i % len(styles)], label=f'res={res}')
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
# Difference plots vs finest resolution, on common grid
# ------------------------------------------------------------------
ref_idx   = -1   # finest resolution is the reference
ref_label = f'res={res_lst[ref_idx]}'

diff_specs = [
    ('Gamma', r'$\Gamma$',  r'Circulation'),
    ('a',     r'$a$',       r'Axial induction factor'),
    ('aline', r"$a'$",      r'Tangential induction factor'),
    ('Fnorm', r'$F_{norm}$',r'Normal force'),
    ('Ftan',  r'$F_{tan}$', r'Tangential force'),
]

fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle(f'Percentage difference vs {ref_label} — evaluated on common reference grid')
axs2_flat = axs2.flatten()

for ax_idx, (key, sym, name) in enumerate(diff_specs):
    ax = axs2_flat[ax_idx]
    y_ref = interp[key][ref_idx]
    for i in range(len(res_lst) - 1):
        y_i  = interp[key][i]
        # Avoid division by zero: use absolute ref value; mask near-zero
        denom = np.where(np.abs(y_ref) > 1e-10, np.abs(y_ref), np.nan)
        diff  = np.abs(y_ref - y_i) / denom * 100
        ax.plot(r_ref, diff, styles[i % len(styles)], label=f'res={res_lst[i]} vs {ref_label}')
    finish_axis(ax, f'{name} difference', f'|Δ{sym}| (%)')

axs2_flat[-1].axis('off')
plt.tight_layout()
plt.show()