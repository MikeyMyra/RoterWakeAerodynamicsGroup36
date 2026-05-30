import numpy as np
import matplotlib.pyplot as plt
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lifting_line import BEM

a_ind_wake_lst=np.array([0,0])  #np.linspace(0,1,10)
res=200
# bem=BEM(2)
i=0
bem = BEM(J=2, radius=0.7, n_blades=6, U_inf=60)

tend=5/60
dt=0.1
bem.tlst=np.arange(0,tend,dt)
# Uwake=10
bem.rpm=40
results={}
for a_ind_wake in a_ind_wake_lst:
    if i == 0:
        output = bem.Lifting_line(resolution=res, a_ind_wake=a_ind_wake, track_convergence=True)
        key = 'linear'
    else:
        output = bem.Lifting_line(resolution=res, a_ind_wake=a_ind_wake, track_convergence=True, spacing='cosine')
        key = 'cosine'

    a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, conv_iter, conv_hist, r_control, alpha_out = output

    # store immediately after unpacking
    results[key] = {
        'r_control': r_control,
        'Gamma': Gamma_out,
        'a': a_out,
        'aline': aline_out,
        'Fnorm': Fnorm_out,
        'Ftan': Ftan_out,
        'alpha': alpha_out,
    }

    blade_count = bem.n_blades
    station_count = len(r_control)


    if i==0:
        def plot_blade_overlay(ax, x_values, y_values, label_prefix='', style='-o'):
            x_values = np.asarray(x_values)
            y_values = np.asarray(y_values)
            sc = len(x_values)  # local, not closed-over station_count
            x_masked = x_values[1:]
            if sc > 0 and len(y_values) == blade_count * sc:
                blade_series = [np.asarray(y_values[b * sc:(b + 1) * sc])[1:] for b in range(blade_count)]
                if all(np.allclose(blade_series[0], s) for s in blade_series[1:]):
                    ax.plot(x_masked, blade_series[0], style, label=label_prefix)
                else:
                    ax.plot(x_masked, np.mean(blade_series, axis=0), style, label=label_prefix)
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


        fig_gamma, ax_gamma   = plt.subplots(figsize=(7, 5))
        fig_a,      ax_a      = plt.subplots(figsize=(7, 5))
        fig_aprime, ax_aprime = plt.subplots(figsize=(7, 5))
        fig_fnorm,  ax_fnorm  = plt.subplots(figsize=(7, 5))
        fig_ftan,   ax_ftan   = plt.subplots(figsize=(7, 5))
        fig_aoa,   ax_aoa     = plt.subplots(figsize=(7, 5))
        fig_conv,  ax_conv    = plt.subplots(figsize=(7, 5))

        # Circulation
        try:
            plot_blade_overlay(ax_gamma, r_control, Gamma_out, 'Gamma (linear)')
        except Exception:
            pass

        # Axial and azimuthal induction
        try:
            plot_blade_overlay(ax_a, r_control, a_out, r'$a\,$' f'(linear)')
            plot_blade_overlay(ax_aprime, r_control, aline_out, r'$a^\prime$' f'(linear)', style='-s')
        except Exception:
            pass

        # Forces
        try:
            plot_blade_overlay(ax_fnorm, r_control, Fnorm_out, r'$ F_{norm}$' f'(linear)')
            plot_blade_overlay(ax_ftan, r_control, Ftan_out, r'$ F_{tan}$' f'(linear)', style='-s')
        except Exception:
            pass

        # Angle of attack
        try:
            plot_blade_overlay(ax_aoa, r_control, alpha_out, r'$AoA\,$' f'(linear)', style='-^')

            # overlay BEM AoA
        except Exception:
            pass

        # Convergence history
        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error'])>0:
                ax_conv.semilogy(conv_hist['iteration'], conv_hist['error'],label=r'$\epsilon\,$' f'(linear)')
                # ax_conv.set_title('Convergence history')
                # ax_conv.set_xlabel('Iteration')
                # ax_conv.set_ylabel('Relative error')
                ax_conv.grid(True)
            else:
                ax_conv.axis('off')
        except Exception:
            ax_conv.axis('off')

    else:
        plot_blade_overlay(ax_gamma, r_control, Gamma_out, 'Gamma (cosine)' )


        plot_blade_overlay(ax_a, r_control, a_out, r'$a\,$' f'(cosine)')
        plot_blade_overlay(ax_aprime, r_control, aline_out, r'$a^\prime\,$' f'(cosine)', style='-s')


        plot_blade_overlay(ax_fnorm, r_control, Fnorm_out,  r'$ F_{norm}\,$' f'(cosine)')
        plot_blade_overlay(ax_ftan, r_control, Ftan_out, r'$ F_{tan}\,$' f'(cosine)', style='-s')

        plot_blade_overlay(ax_aoa, r_control, alpha_out, r'$AoA\,$' f'(cosine)', style='-^')

        try:
            if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error'])>0:
                ax_conv.semilogy(conv_hist['iteration'], conv_hist['error'],label=r'$\epsilon\,$' f'(cosine)')
                # ax_conv.set_title('Convergence history')
                # ax_conv.set_xlabel('Iteration')
                # ax_conv.set_ylabel('Relative error')
                ax_conv.grid(True)
            else:
                ax_conv.axis('off')
        except Exception:
            ax_conv.axis('off')


    i=i+1
data=np.loadtxt('assignment_2\\Sensitivity_analysis\\Finest_grid_cosine.txt')
# print(data[0,:])
if len(data[0,:])<res:
    np.savetxt('assignment_2\\Sensitivity_analysis\\Finest_grid_cosine.txt',np.array([r_control[1:],Gamma_out[1:res+1],Fnorm_out[1:res+1],Ftan_out[1:res+1],aline_out[1:res+1],a_out[1:res+1],alpha_out[1:res+1]]), header='rcontrol gamma ftan fnorm aprime a alpha')
# data=np.loadtxt('assignment_2\\Sensitivity_analysis\\Finest_grid_cosine.txt')
# print(data[0,:])
finish_axis(ax_gamma,  'Circulation vs radius',       r'$\Gamma$ (m²/s)')
finish_axis(ax_a,      'Axial induction factor',      'a (-)')
finish_axis(ax_aprime, 'Tangential induction factor', "a' (-)")
finish_axis(ax_fnorm,  'Normal force',                r'$F_{norm}$ (N/m)')
finish_axis(ax_ftan,   'Tangential force',            r'$F_{tan}$ (N/m)')
finish_axis(ax_aoa,    'Angle of attack',             'AoA (degrees)')
ax_conv.set_title('Convergence history')
ax_conv.set_ylabel('Relative error')
ax_conv.set_xlabel('Iteration')
ax_conv.legend()
ax_conv.grid(True)

fig_gamma.tight_layout()
fig_a.tight_layout()
fig_aprime.tight_layout()
fig_fnorm.tight_layout()
fig_ftan.tight_layout()
fig_aoa.tight_layout()
fig_conv.tight_layout()
plt.show()
# # --- Difference plots ---
# from scipy.interpolate import interp1d

# ref = results['linear']
# comp = results['cosine']

# sc_ref = len(ref['r_control'])
# sc_comp = len(comp['r_control'])

# def get_mean(data, r_control, bc):
#     sc = len(r_control)
#     blade_series = [data[b * sc:(b + 1) * sc] for b in range(bc)]
#     return np.mean(blade_series, axis=0)

# # def interp_to_ref(ref_r, comp_r, comp_mean):
# #     f = interp1d(comp_r, comp_mean, bounds_error=False, fill_value='extrapolate')
# #     return f(ref_r)

# # ref_r = ref['r_control']

# # quantities = [
# #     ('Gamma', r'$|\Delta\Gamma|$ (%)',        'Circulation difference vs linear'),
# #     ('a',     r'$|\Delta a|$ (%)',             'Axial induction difference vs linear'),
# #     ('aline', r"$|\Delta a'|$ (%)",            'Tangential induction difference vs linear'),
# #     ('Fnorm', r'$|\Delta F_{norm}|$ (%)',      'Normal force difference vs linear'),
# #     ('Ftan',  r'$|\Delta F_{tan}|$ (%)',       'Tangential force difference vs linear'),
# #     ('alpha', r'$|\Delta AoA|$ (%)',           'AoA difference vs linear'),
# # ]

# # fig, axs = plt.subplots(2, 3, figsize=(15, 8))
# # for ax, (key, ylabel, title) in zip(axs.flat, quantities):
# #     ref_mean  = get_mean(ref[key],  ref['r_control'],  blade_count)
# #     comp_mean = get_mean(comp[key], comp['r_control'], blade_count)
# #     comp_interp = interp_to_ref(ref_r, comp['r_control'], comp_mean)

# #     diff = np.abs((ref_mean - comp_interp) / (np.abs(ref_mean) + 1e-10)) * 100

# #     ax.plot(ref_r[2:], diff[2:], '-o', label='cosine vs linear')  # skip index 0 and 1
# #     ax.axvline(bem.blade_start_fraction * bem.radius, color='k', linestyle='--', linewidth=1, label='blade start')
# #     ax.set_title(title)
# #     ax.set_xlabel('r (m)')
# #     ax.set_ylabel(ylabel)
# #     ax.legend()
# #     ax.grid(True)

# # plt.tight_layout()
# # plt.show()

