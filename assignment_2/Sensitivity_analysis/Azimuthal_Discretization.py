import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lifting_line import BEM

res_lst = np.array([10,40,80,200])
bem = BEM(J=2, radius=0.7, n_blades=6, U_inf=60)
# omega=bem.rpm/60*2*np.pi
# tend=2*np.pi/omega*rev
tend = 10/60
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
conv_hist_lst = []

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
    conv_hist_lst.append(conv_hist)

blade_count = bem.n_blades


def extract_blade0(arr, r_control):
    sc = len(r_control)
    return np.asarray(arr[:sc])[1:]


def get_r(r_control):
    return np.asarray(r_control)[1:]


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
# Separate plots — one figure per quantity
# ------------------------------------------------------------------

styles = ['-o', '-s', '-^']

plot_specs = [
    ('Circulation vs radius', Gamma_out_lst, r'$\Gamma$ (m²/s)', 'Gamma'),
    ('Axial induction factor', a_out_lst, 'a', 'a'),
    ('Tangential induction factor', aline_out_lst, "a'", 'aline'),
    ('Normal force', Fnorm_out_lst, 'Force per unit span (N/m)', 'Fnorm'),
    ('Tangential force', Ftan_out_lst, 'Force per unit span (N/m)', 'Ftan'),
    ('Angle of attack', alpha_out_lst, 'AoA (deg)', 'alpha'),
]

for title, data_lst, ylabel, key in plot_specs:

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, res in enumerate(res_lst):
        style = styles[i % len(styles)]

        r = get_r(r_control_lst[i])
        rc = r_control_lst[i]

        y = extract_blade0(data_lst[i], rc)

        ax.plot(r, y, style, label=f'res={res}')

    finish_axis(ax, title, ylabel)

    plt.tight_layout()
data=np.loadtxt('assignment_2\\Sensitivity_analysis\\Finest_grid.txt')

if len(data[0,:])<res_lst[-1]:
    np.savetxt('assignment_2\\Sensitivity_analysis\\Finest_grid.txt',np.array([r_control_lst[-1][1:],Gamma_out_lst[-1][1:res_lst[-1]+1],Fnorm_out_lst[-1][1:res_lst[-1]+1],Ftan_out_lst[-1][1:res_lst[-1]+1],aline_out_lst[-1][1:res_lst[-1]+1],a_out_lst[-1][1:res_lst[-1]+1],alpha_out_lst[-1][1:res_lst[-1]+1]]), header='rcontrol gamma ftan fnorm aprime a alpha')
# data=np.loadtxt('assignment_2\\Sensitivity_analysis\\Finest_grid.txt')
# print(data[0,:])
# def interp_to_ref(arr, r_control):
#     r = get_r(r_control)
#     y = extract_blade0(arr, r_control)
#     return np.interp(r_ref, r, y)


# diff_specs = [
#     ('Gamma',     Gamma_out_lst, r'$\Gamma$',   'Circulation'),
#     ('a',         a_out_lst,     r'$a$',        'Axial induction factor'),
#     ('aline',     aline_out_lst, r"$a'$",       'Tangential induction factor'),
#     ('Fnorm',     Fnorm_out_lst, r'$F_{norm}$', 'Normal force'),
#     ('Ftan',      Ftan_out_lst,  r'$F_{tan}$',  'Tangential force'),
#     ('alpha',     alpha_out_lst, r'$\alpha$',   'Angle of attack'),
# ]
# # ------------------------------------------------------------------
# # Separate convergence plot
# # ------------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(7, 5))

# for i, res in enumerate(res_lst):

#     try:
#         ch = conv_hist_lst[i]

#         if ch and len(ch['error']) > 0:
#             ax.semilogy(
#                 ch['iteration'],
#                 ch['error'],
#                 styles[i % len(styles)],
#                 label=f'res={res}'
#             )

#     except Exception:
#         pass

# ax.set_title('Convergence history')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Relative error')
# ax.legend()
# ax.grid(True)

# plt.tight_layout()


# # ------------------------------------------------------------------
# # Difference plots — one figure per quantity
# # ------------------------------------------------------------------

# ref_idx   = -1
# ref_label = f'res={res_lst[ref_idx]}'
# r_ref     = get_r(r_control_lst[ref_idx])

# for key, lst, sym, name in diff_specs:

#     fig, ax = plt.subplots(figsize=(7, 5))

#     y_ref = interp_to_ref(lst[ref_idx], r_control_lst[ref_idx])

#     for i in range(len(res_lst) - 1):

#         y_i = interp_to_ref(lst[i], r_control_lst[i])

#         denom = np.where(
#             np.abs(y_ref) > 1e-6 * np.mean(np.abs(y_ref) + 1e-10),
#             np.abs(y_ref),
#             np.nan
#         )

#         diff = np.abs(y_ref - y_i) / denom * 100

#         ax.plot(
#             r_ref,
#             diff,
#             styles[i % len(styles)],
#             label=f'res={res_lst[i]} vs {ref_label}'
#         )

#     finish_axis(ax, f'{name} difference', f'|Δ{sym}| (%)')

#     plt.tight_layout()

plt.show()