import numpy as np
import matplotlib.pyplot as plt
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lifting_line import BEM

a_ind_wake_lst=np.array([0,0.5,1])  #np.linspace(0,1,10)
res=20
bem=BEM(2)
i=0
bem = BEM(J=2)
tend=5
dt=0.1
bem.tlst=np.arange(0,tend,dt)
# Uwake=10
bem.rpm=40

for a_ind_wake in a_ind_wake_lst:
    # bem.Lifting_line(20,a_ind_wake)
    
    # print(bem.calc_ind_filiment([0,0,0.8],0.4))
    output = bem.Lifting_line(resolution=10,a_ind_wake=a_ind_wake, track_convergence=True)

    # Unpack outputs
    a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, conv_iter, conv_hist, r_control, alpha_out = output

    blade_count = bem.n_blades
    station_count = len(r_control)


    if i==0:
        def plot_blade_overlay(ax, x_values, y_values, label_prefix='', style='-o'):
            x_values = np.asarray(x_values)
            y_values = np.asarray(y_values)

            # remove r_index == 0 control points
            x_masked = x_values[1:]

            if station_count > 0 and len(y_values) == blade_count * station_count:
                blade_series = [np.asarray(y_values[i * station_count:(i + 1) * station_count])[1:] for i in range(blade_count)]
                # if all blades identical, plot a single line
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
            # add vertical line showing blade start location
            try:
                blade_root_r = bem.blade_start_fraction * bem.radius
                ax.axvline(blade_root_r, color='k', linestyle='--', linewidth=1, label='blade start')
            except Exception:
                pass
            ax.legend()
            ax.grid(True)


        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        # Circulation
        try:
            plot_blade_overlay(axs[0, 0], r_control, Gamma_out, 'Gamma' r'$\,(a_w=$'f'{a_ind_wake})')
        except Exception:
            pass

        # Axial and azimuthal induction
        try:
            plot_blade_overlay(axs[0, 1], r_control, a_out, r'$a\,(a_w$' f'={a_ind_wake})')
            plot_blade_overlay(axs[0, 1], r_control, aline_out, r'$a^\prime\,(a_w$' f'={a_ind_wake})', style='-s')
        except Exception:
            pass

        # Forces
        try:
            plot_blade_overlay(axs[1, 0], r_control, Fnorm_out, r'$ F_{norm}\,(a_w=$' f'{a_ind_wake})')
            plot_blade_overlay(axs[1, 0], r_control, Ftan_out, r'$ F_{tan}\,(a_w=$' f'{a_ind_wake})', style='-s')
        except Exception:
            pass

        # Angle of attack
        try:
            plot_blade_overlay(axs[0, 2], r_control, alpha_out, r'$AoA\, (a_w=$' f'{a_ind_wake})', style='-^')

            # overlay BEM AoA
        except Exception:
            pass

        # Convergence history
        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error'])>0:
                axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'],label=r'$\epsilon\, (a_w=$' f'{a_ind_wake})')
                # axs[1, 1].set_title('Convergence history')
                # axs[1, 1].set_xlabel('Iteration')
                # axs[1, 1].set_ylabel('Relative error')
                axs[1, 1].grid(True)
            else:
                axs[1, 1].axis('off')
        except Exception:
            axs[1, 1].axis('off')

    else:
        plot_blade_overlay(axs[0, 0], r_control, Gamma_out, 'Gamma' r'$\,(a_w=$'f'{a_ind_wake})')


        plot_blade_overlay(axs[0, 1], r_control, a_out, r'$a\,(a_w$' f'={a_ind_wake})')
        plot_blade_overlay(axs[0, 1], r_control, aline_out, r'$a^\prime\,(a_w$' f'={a_ind_wake})', style='-s')


        plot_blade_overlay(axs[1, 0], r_control, Fnorm_out,  r'$ F_{norm}\,(a_w=$' f'{a_ind_wake})')
        plot_blade_overlay(axs[1, 0], r_control, Ftan_out, r'$ F_{tan}\,(a_w=$' f'{a_ind_wake})', style='-s')

        plot_blade_overlay(axs[0, 2], r_control, alpha_out, r'$AoA\, (a_w=$' f'{a_ind_wake})', style='-^')

        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error'])>0:
                axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'],label=r'$\epsilon\, (a_w=$' f'{a_ind_wake})')
                # axs[1, 1].set_title('Convergence history')
                # axs[1, 1].set_xlabel('Iteration')
                # axs[1, 1].set_ylabel('Relative error')
                axs[1, 1].grid(True)
            else:
                axs[1, 1].axis('off')
        except Exception:
            axs[1, 1].axis('off')


    i=i+1
finish_axis(axs[0, 0], 'Circulation vs radius', 'Gamma (m^2/s)')
finish_axis(axs[0, 1], 'Induction factors', 'Induction factor')
finish_axis(axs[1, 0], 'Section forces', 'Force per unit span')
finish_axis(axs[0, 2], 'Angle of attack', 'AoA (degrees)')
axs[1, 1].set_title('Convergence history')
axs[1, 1].set_ylabel('Relative error')
axs[1, 1].set_xlabel('Iteration')
axs[1,1].legend()







plt.tight_layout()
plt.show()





