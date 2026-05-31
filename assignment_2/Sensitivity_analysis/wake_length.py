import numpy as np
import matplotlib.pyplot as plt
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lifting_line import BEM

# tend_lst=np.array([5,10,20])  #np.linspace(0,1,10)
xend_lst=np.array([5,50,100])  #np.linspace(0,1,10)
rev_lst=np.array([0.1,0.5, 1.0,2,5])
res=10
i=0
bem = BEM(J=2, radius=0.7, n_blades=6, U_inf=60)

a_out_lst=[]
aline_out_lst=[]
Fnorm_out_lst=[]
Ftan_out_lst=[]
Gamma_out_lst=[]
conv_iter_lst=[]
conv_hist_lst=[]
r_control_lst=[]
alpha_out_lst=[]

for rev in rev_lst:

    dt=0.1
    bem.rpm=40
    omega=bem.rpm/60*2*np.pi
    tend=2*np.pi/omega*rev

    bem.tlst=np.arange(0,tend,dt)
    bem.rpm=40

    output = bem.Lifting_line(resolution=res, track_convergence=True, spacing='cosine')

    # Unpack outputs
    a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, conv_iter, conv_hist, r_control, alpha_out = output

    a_out_lst.append(a_out)
    aline_out_lst.append(aline_out)
    Fnorm_out_lst.append(Fnorm_out)
    Ftan_out_lst.append(Ftan_out)
    Gamma_out_lst.append(Gamma_out)
    conv_iter_lst.append(conv_iter)
    conv_hist_lst.append(conv_hist)
    r_control_lst.append(r_control)
    alpha_out_lst.append(alpha_out)

    blade_count = bem.n_blades
    station_count = len(r_control)

    if i==0:
        def plot_blade_overlay(ax, x_values, y_values, label_prefix='', style='-o'):

            x_values = np.asarray(x_values)
            y_values = np.asarray(y_values)

            x_masked = x_values[1:]

            if station_count > 0 and len(y_values) == blade_count * station_count:

                blade0 = np.asarray(y_values[:station_count])[1:]

                ax.plot(x_masked, blade0, style, label=label_prefix)

            else:

                ax.plot(x_masked, y_values[1:], style, label=label_prefix)
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

        # Circulation
        try:
            plot_blade_overlay(axs[0, 0], r_control, Gamma_out, 'Gamma'r'$\,(n_{rev}$'f'={rev})')
        except Exception:
            pass

        # Axial and azimuthal induction
        try:
            plot_blade_overlay(axs[0, 1], r_control, a_out, r'$a\,(n_{rev}$'f'={rev})')
            plot_blade_overlay(axs[0, 1], r_control, aline_out, r'$a^\prime\,(n_{rev}$'f'={rev})', style='-s')
        except Exception:
            pass

        # Forces
        try:
            plot_blade_overlay(axs[1, 0], r_control, Fnorm_out, r'$ F_{norm}\,(n_{rev}$'f'={rev})')
            plot_blade_overlay(axs[1, 0], r_control, Ftan_out, r'$ F_{tan}\,(n_{rev}$'f'={rev})', style='-s')
        except Exception:
            pass

        # Angle of attack
        try:
            plot_blade_overlay(axs[0, 2], r_control, alpha_out, r'$AoA\,(n_{rev}$'f'={rev})', style='-^')
        except Exception:
            pass

        # Convergence history
        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error'])>0:
                axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'], label=r'$\epsilon\,(n_{rev}$'f'={rev})')
                axs[1, 1].grid(True)
            else:
                axs[1, 1].axis('off')
        except Exception:
            axs[1, 1].axis('off')

    else:
        plot_blade_overlay(axs[0, 0], r_control, Gamma_out, 'Gamma' r'$\,(n_{rev}$'f'={rev})')

        plot_blade_overlay(axs[0, 1], r_control, a_out, r'$a\,(n_{rev}$'f'={rev})')
        plot_blade_overlay(axs[0, 1], r_control, aline_out, r'$a^\prime\,(n_{rev}$'f'={rev})', style='-s')

        plot_blade_overlay(axs[1, 0], r_control, Fnorm_out, r'$ F_{norm}\,(n_{rev}$'f'={rev})')
        plot_blade_overlay(axs[1, 0], r_control, Ftan_out, r'$ F_{tan}\,(n_{rev}$'f'={rev})', style='-s')

        plot_blade_overlay(axs[0, 2], r_control, alpha_out, r'$AoA\,(n_{rev}$'f'={rev})', style='-^')

        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error'])>0:
                axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'], label=r'$\epsilon\,(n_{rev}$'f'={rev})')
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

# --- Difference plots (all in percentage, consistent format) ---

fig = plt.figure()
ax = fig.subplots(1, 1)
for i in range(len(rev_lst) - 1):
    plot_blade_overlay(ax, r_control,
                       abs((Gamma_out_lst[-1] - Gamma_out_lst[i]) / Gamma_out_lst[-1]) * 100,
                       r'$\Gamma\,(n_{rev}=$' f'{rev_lst[i]})')
finish_axis(ax,
            r'Circulation difference vs radius compared to $n_{rev}=$' f'{rev_lst[-1]}',
            r'$|\Delta\Gamma|$ (%)')

fig = plt.figure()
ax = fig.subplots(1, 1)
for i in range(len(rev_lst) - 1):
    plot_blade_overlay(ax, r_control,
                       abs((a_out_lst[-1] - a_out_lst[i]) / a_out_lst[-1]) * 100,
                       r'$a\,(n_{rev}=$' f'{rev_lst[i]})')
finish_axis(ax,
            r'Axial induction factor difference vs radius compared to $n_{rev}=$' f'{rev_lst[-1]}',
            r'$|\Delta a|$ (%)')

fig = plt.figure()
ax = fig.subplots(1, 1)
for i in range(len(rev_lst) - 1):
    plot_blade_overlay(ax, r_control,
                       abs((aline_out_lst[-1] - aline_out_lst[i]) / aline_out_lst[-1]) * 100,
                       r'$a^\prime\,(n_{rev}=$' f'{rev_lst[i]})')
finish_axis(ax,
            r'Tangential induction factor difference vs radius compared to $n_{rev}=$' f'{rev_lst[-1]}',
            r'$|\Delta a^\prime|$ (%)')

fig = plt.figure()
ax = fig.subplots(1, 1)
for i in range(len(rev_lst) - 1):
    plot_blade_overlay(ax, r_control,
                       abs((Fnorm_out_lst[-1] - Fnorm_out_lst[i]) / Fnorm_out_lst[-1]) * 100,
                       r'$F_{norm}\,(n_{rev}=$' f'{rev_lst[i]})')
finish_axis(ax,
            r'Normal force difference vs radius compared to $n_{rev}=$' f'{rev_lst[-1]}',
            r'$|\Delta F_{norm}|$ (%)')

fig = plt.figure()
ax = fig.subplots(1, 1)
for i in range(len(rev_lst) - 1):
    plot_blade_overlay(ax, r_control,
                       abs((Ftan_out_lst[-1] - Ftan_out_lst[i]) / Ftan_out_lst[-1]) * 100,
                       r'$F_{tan}\,(n_{rev}=$' f'{rev_lst[i]})')
finish_axis(ax,
            r'Tangential force difference vs radius compared to $n_{rev}=$' f'{rev_lst[-1]}',
            r'$|\Delta F_{tan}|$ (%)')


plt.tight_layout()
plt.show()