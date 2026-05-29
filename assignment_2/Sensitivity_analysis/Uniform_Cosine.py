import numpy as np
import matplotlib.pyplot as plt
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lifting_line import BEM

a_ind_wake_lst=np.array([0,0])  #np.linspace(0,1,10)
res=20
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

for a_ind_wake in a_ind_wake_lst:

    if i==0:
        output = bem.Lifting_line(resolution=10, a_ind_wake=a_ind_wake, track_convergence=True)
    else:
        output = bem.Lifting_line(resolution=10, a_ind_wake=a_ind_wake, track_convergence=True, spacing='cosine')

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
            plot_blade_overlay(axs[0, 0], r_control, Gamma_out, 'Gamma (linear)')
        except Exception:
            pass
        try:
            plot_blade_overlay(axs[0, 1], r_control, a_out, r'$a\,$' f'(linear)')
            plot_blade_overlay(axs[0, 1], r_control, aline_out, r'$a^\prime$' f'(linear)', style='-s')
        except Exception:
            pass
        try:
            plot_blade_overlay(axs[1, 0], r_control, Fnorm_out, r'$ F_{norm}$' f'(linear)')
            plot_blade_overlay(axs[1, 0], r_control, Ftan_out, r'$ F_{tan}$' f'(linear)', style='-s')
        except Exception:
            pass
        try:
            plot_blade_overlay(axs[0, 2], r_control, alpha_out, r'$AoA\,$' f'(linear)', style='-^')
        except Exception:
            pass
        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error']) > 0:
                axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'], label=r'$\epsilon\,$' f'(linear)')
                axs[1, 1].grid(True)
            else:
                axs[1, 1].axis('off')
        except Exception:
            axs[1, 1].axis('off')

    else:
        plot_blade_overlay(axs[0, 0], r_control, Gamma_out, 'Gamma (cosine)')
        plot_blade_overlay(axs[0, 1], r_control, a_out, r'$a\,$' f'(cosine)')
        plot_blade_overlay(axs[0, 1], r_control, aline_out, r'$a^\prime\,$' f'(cosine)', style='-s')
        plot_blade_overlay(axs[1, 0], r_control, Fnorm_out, r'$ F_{norm}\,$' f'(cosine)')
        plot_blade_overlay(axs[1, 0], r_control, Ftan_out, r'$ F_{tan}\,$' f'(cosine)', style='-s')
        plot_blade_overlay(axs[0, 2], r_control, alpha_out, r'$AoA\,$' f'(cosine)', style='-^')
        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error']) > 0:
                axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'], label=r'$\epsilon\,$' f'(cosine)')
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

# --- Difference plots: cosine vs linear (percentage) ---
# r_control has station_count entries; output arrays have blade_count * station_count.
# Extract blade-0 slice from outputs: first station_count values.
# Interpolate cosine onto linear r grid since locations differ.
# Skip index 0 (hub point) to match the main plots.
def extract(arr, sc):
    """Blade-0 slice, skip hub point."""
    return np.asarray(arr[:sc])[1:]

sc_lin = len(r_control_lst[0])
sc_cos = len(r_control_lst[1])
r_lin  = np.asarray(r_control_lst[0])[1:]
r_cos  = np.asarray(r_control_lst[1])[1:]

diff_pairs = [
    (Gamma_out_lst,  r'$\Gamma\,$(cosine vs linear)',    r'Circulation difference vs radius: cosine vs linear',                  r'$|\Delta\Gamma|$ (%)'),
    (a_out_lst,      r'$a\,$(cosine vs linear)',          r'Axial induction factor difference vs radius: cosine vs linear',       r'$|\Delta a|$ (%)'),
    (aline_out_lst,  r"$a^\prime\,$(cosine vs linear)",   r'Tangential induction factor difference vs radius: cosine vs linear',  r"$|\Delta a^\prime|$ (%)"),
    (Fnorm_out_lst,  r'$F_{norm}\,$(cosine vs linear)',   r'Normal force difference vs radius: cosine vs linear',                 r'$|\Delta F_{norm}|$ (%)'),
    (Ftan_out_lst,   r'$F_{tan}\,$(cosine vs linear)',    r'Tangential force difference vs radius: cosine vs linear',             r'$|\Delta F_{tan}|$ (%)'),
]

for data_lst, label, title, ylabel in diff_pairs:
    y_lin = extract(data_lst[0], sc_lin)
    y_cos = extract(data_lst[1], sc_cos)
    y_cos_interp = np.interp(r_lin, r_cos, y_cos)
    diff = abs((y_lin - y_cos_interp) / y_lin) * 100
    fig = plt.figure()
    ax  = fig.subplots(1, 1)
    ax.plot(r_lin, diff, '-o', label=label)
    finish_axis(ax, title, ylabel)

plt.tight_layout()
plt.show()