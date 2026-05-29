import numpy as np
import matplotlib.pyplot as plt
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lifting_line import BEM

res_lst=np.array([5,10,20])  #np.linspace(0,1,10)
bem=BEM(2)
i=0
bem = BEM(J=2)
tend=5
dt=0.1
bem.tlst=np.arange(0,tend,dt)
bem.rpm=40

a_out_lst=[]
aline_out_lst=[]
Fnorm_out_lst=[]
Ftan_out_lst=[]
Gamma_out_lst=[]
r_control_lst=[]

for res in res_lst:

    output = bem.Lifting_line(resolution=res, track_convergence=True)

    # Unpack outputs
    a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, conv_iter, conv_hist, r_control, alpha_out = output

    a_out_lst.append(a_out)
    aline_out_lst.append(aline_out)
    Fnorm_out_lst.append(Fnorm_out)
    Ftan_out_lst.append(Ftan_out)
    Gamma_out_lst.append(Gamma_out)
    r_control_lst.append(r_control)

    blade_count = bem.n_blades
    station_count = len(r_control)

    if i==0:
        def plot_blade_overlay(ax, x_values, y_values, label_prefix='', style='-o'):
            x_values = np.asarray(x_values)
            y_values = np.asarray(y_values)
            x_masked = x_values[1:]
            if station_count > 0 and len(y_values) == blade_count * station_count:
                blade_series = [np.asarray(y_values[j * station_count:(j + 1) * station_count])[1:] for j in range(blade_count)]
                if len(blade_series) > 0 and all(np.allclose(blade_series[0], series) for series in blade_series[1:]):
                    ax.plot(x_masked, blade_series[0], style, label=f'{label_prefix} all blades (identical)')
                    ax.text(0.02, 0.95, f'{blade_count} blades overlap', transform=ax.transAxes,
                            va='top', ha='left', fontsize=9)
                else:
                    for blade_idx, series in enumerate(blade_series, start=1):
                        ax.plot(x_masked, series, style, label=f'{label_prefix} blade {blade_idx}')
            else:
                ax.plot(x_masked, y_values[1:], style, label=label_prefix)

        def finish_axis(ax, title, ylabel):
            ax.set_title(title)
            ax.set_xlabel('r (m)')
            ax.set_ylabel(ylabel)
            try:
                blade_root_r = bem.blade_start_fraction * bem.radius
                ax.axvline(blade_root_r, color='k', linestyle='--', linewidth=1, label='blade start')
            except Exception:
                pass
            ax.legend()
            ax.grid(True)

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        try:
            plot_blade_overlay(axs[0, 0], r_control, Gamma_out, f'Gamma (resolution={res})')
        except Exception:
            pass
        try:
            plot_blade_overlay(axs[0, 1], r_control, a_out, r'$a\,$' f'(resolution={res})')
            plot_blade_overlay(axs[0, 1], r_control, aline_out, r'$a^\prime\,$' f'(resolution={res})', style='-s')
        except Exception:
            pass
        try:
            plot_blade_overlay(axs[1, 0], r_control, Fnorm_out, r'$ F_{norm}\,$' f'(resolution={res})')
            plot_blade_overlay(axs[1, 0], r_control, Ftan_out, r'$ F_{tan}\,$' f'(resolution={res})', style='-s')
        except Exception:
            pass
        try:
            plot_blade_overlay(axs[0, 2], r_control, alpha_out, r'$AoA\,$' f'(resolution={res})', style='-^')
        except Exception:
            pass
        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error']) > 0:
                axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'], label=r'$\epsilon\,$' f'(resolution={res})')
                axs[1, 1].grid(True)
            else:
                axs[1, 1].axis('off')
        except Exception:
            axs[1, 1].axis('off')

    else:
        plot_blade_overlay(axs[0, 0], r_control, Gamma_out, f'Gamma (resolution={res})')
        plot_blade_overlay(axs[0, 1], r_control, a_out, r'$a\,$' f'(resolution={res})')
        plot_blade_overlay(axs[0, 1], r_control, aline_out, r'$a^\prime\,$' f'(resolution={res})', style='-s')
        plot_blade_overlay(axs[1, 0], r_control, Fnorm_out, r'$ F_{norm}\,$' f'(resolution={res})')
        plot_blade_overlay(axs[1, 0], r_control, Ftan_out, r'$ F_{tan}\,$' f'(resolution={res})', style='-s')
        plot_blade_overlay(axs[0, 2], r_control, alpha_out, r'$AoA\,$' f'(resolution={res})', style='-^')
        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error']) > 0:
                axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'], label=r'$\epsilon\,$' f'(resolution={res})')
                axs[1, 1].grid(True)
            else:
                axs[1, 1].axis('off')
        except Exception:
            axs[1, 1].axis('off')

    i += 1

finish_axis(axs[0, 0], 'Circulation vs radius', 'Gamma (m^2/s)')
finish_axis(axs[0, 1], 'Induction factors', 'Induction factor')
finish_axis(axs[1, 0], 'Section forces', 'Force per unit span')
finish_axis(axs[0, 2], 'Angle of attack', 'AoA (degrees)')
axs[1, 1].set_title('Convergence history')
axs[1, 1].set_ylabel('Relative error')
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].legend()

# --- Difference plots: all resolutions vs finest (res_lst[-1]), percentage ---
# r_control has station_count entries; output arrays have blade_count * station_count.
# Extract blade-0 slice from outputs: first station_count values.
# Skip index 0 (hub point) to match the main plots.
def extract(arr, sc):
    """Blade-0 slice, skip hub point."""
    return np.asarray(arr[:sc])[1:]

ref_idx   = -1
sc_ref    = len(r_control_lst[ref_idx])          # station_count for reference
r_ref     = np.asarray(r_control_lst[ref_idx])[1:]

y_ref_arrays = {
    'Gamma': extract(Gamma_out_lst[ref_idx], sc_ref),
    'a':     extract(a_out_lst[ref_idx],     sc_ref),
    'aline': extract(aline_out_lst[ref_idx], sc_ref),
    'Fnorm': extract(Fnorm_out_lst[ref_idx], sc_ref),
    'Ftan':  extract(Ftan_out_lst[ref_idx],  sc_ref),
}

diff_pairs = [
    ('Gamma', Gamma_out_lst,  r'$\Gamma$',   r'Circulation difference vs radius compared to resolution=' f'{res_lst[-1]}',                 r'$|\Delta\Gamma|$ (%)'),
    ('a',     a_out_lst,      r'$a$',         r'Axial induction factor difference vs radius compared to resolution=' f'{res_lst[-1]}',       r'$|\Delta a|$ (%)'),
    ('aline', aline_out_lst,  r"$a^\prime$",  r'Tangential induction factor difference vs radius compared to resolution=' f'{res_lst[-1]}',  r"$|\Delta a^\prime|$ (%)"),
    ('Fnorm', Fnorm_out_lst,  r'$F_{norm}$',  r'Normal force difference vs radius compared to resolution=' f'{res_lst[-1]}',                r'$|\Delta F_{norm}|$ (%)'),
    ('Ftan',  Ftan_out_lst,   r'$F_{tan}$',   r'Tangential force difference vs radius compared to resolution=' f'{res_lst[-1]}',            r'$|\Delta F_{tan}|$ (%)'),
]

for key, data_lst, label, title, ylabel in diff_pairs:
    y_ref = y_ref_arrays[key]
    fig = plt.figure()
    ax  = fig.subplots(1, 1)
    for j in range(len(res_lst) - 1):
        sc_j  = len(r_control_lst[j])
        r_j   = np.asarray(r_control_lst[j])[1:]
        y_j   = extract(data_lst[j], sc_j)
        y_j_interp = np.interp(r_ref, r_j, y_j)
        diff = abs((y_ref - y_j_interp) / y_ref) * 100
        ax.plot(r_ref, diff, '-o', label=f'{label} (res={res_lst[j]})')
    finish_axis(ax, title, ylabel)

plt.tight_layout()
plt.show()