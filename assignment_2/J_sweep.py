"""Advance-ratio sweep for the lifting-line rotor model.

Runs the lifting line over a range of advance ratios J and plots the
propeller performance coefficients:
    C_T = T / (rho n^2 D^4)
    C_Q = Q / (rho n^2 D^5)
    C_P = P / (rho n^3 D^5)
    eta = C_T * J / C_P   (propulsive efficiency)

Run from the repository root so the relative airfoil path resolves:
    python assignment_2\\J_sweep.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from Lifting_line import BEM


def run_lifting_line(J, radius=0.7, n_blades=6, U_inf=60,
                     resolution=20, a_ind_wake=-0.2, spacing='cosine',
                     tend=0.2, dt=0.005):
    """Solve the lifting line at one J and return (C_T, C_Q, C_P)."""
    bem = BEM(J=J, radius=radius, n_blades=n_blades, U_inf=U_inf)
    bem.tlst = np.arange(0, tend, dt)

    output = bem.Lifting_line(resolution=resolution, a_ind_wake=a_ind_wake,
                              track_convergence=False, spacing=spacing,
                              plot_geometry=False)
    a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, conv_iter, conv_hist, \
        r_control, alpha_out, phi_out = output

    # Integrate section forces to rotor thrust and torque (propeller convention).
    r_l = r_control[1:]
    T_LL = 0.0
    Q_LL = 0.0
    for i in range(len(r_l)):
        T_LL += Fnorm_out[i + 1] * n_blades * bem.dr[i]
        Q_LL += Ftan_out[i + 1] * n_blades * r_l[i] * bem.dr[i]

    C_T = T_LL / (bem.rho * bem.n_rps**2 * bem.D**4)
    C_Q = Q_LL / (bem.rho * bem.n_rps**2 * bem.D**5)
    C_P = (Q_LL * bem.omega) / (bem.rho * bem.n_rps**3 * bem.D**5)
    return C_T, C_Q, C_P


CACHE_FILE = 'assignment_2/J_sweep_results.npz'


def main():
    J_values = np.linspace(0.4, 2.4, 11)

    # Reuse cached solver output when available (the lifting line is slow);
    # delete J_sweep_results.npz to force a full recompute.
    if os.path.exists(CACHE_FILE):
        data = np.load(CACHE_FILE)
        J_values, CT, CQ, CP = data['J'], data['CT'], data['CQ'], data['CP']
        print(f"Loaded cached results from {CACHE_FILE}")
    else:
        CT_list, CQ_list, CP_list = [], [], []
        for J in J_values:
            C_T, C_Q, C_P = run_lifting_line(J)
            CT_list.append(C_T)
            CQ_list.append(C_Q)
            CP_list.append(C_P)
            eta_j = C_T * J / C_P if C_P != 0 else np.nan
            print(f"J = {J:5.2f} | C_T = {C_T:8.4f} | C_Q = {C_Q:8.4f} | "
                  f"C_P = {C_P:8.4f} | eta = {eta_j:7.4f}")
        CT = np.array(CT_list)
        CQ = np.array(CQ_list)
        CP = np.array(CP_list)
        np.savez(CACHE_FILE, J=J_values, CT=CT, CQ=CQ, CP=CP)
        print(f"Saved results cache to {CACHE_FILE}")

    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.where(CP != 0, CT * J_values / CP, np.nan)

    # Efficiency is only physically meaningful where the rotor acts as a
    # propeller (C_T > 0 AND C_P > 0). Outside that branch eta is negative or
    # blows up (C_P -> 0), so mask those points instead of letting a single
    # outlier wreck the axis scaling.
    eta_phys = np.where((CP > 0) & (CT > 0), eta, np.nan)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(J_values, CT, '-o', label=r'$C_T = T/(\rho n^2 D^4)$')
    ax.plot(J_values, CQ, '-s', label=r'$C_Q = Q/(\rho n^2 D^5)$')
    ax.plot(J_values, CP, '-^', label=r'$C_P = P/(\rho n^3 D^5)$')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlabel('Advance ratio $J = U_\\infty/(n D)$')
    ax.set_ylabel(r'$C_T,\; C_Q,\; C_P$')
    ax.grid(True)

    # Efficiency on a secondary axis (autoscaled to the physical branch).
    ax_eta = ax.twinx()
    ax_eta.plot(J_values, eta_phys, '-d', color='tab:red',
                label=r'$\eta = C_T J / C_P$')
    ax_eta.axhline(1.0, color='tab:red', ls=':', lw=1,
                   label=r'$\eta = 1$ (physical ceiling)')
    ax_eta.set_ylabel(r'Efficiency $\eta$', color='tab:red')
    ax_eta.tick_params(axis='y', labelcolor='tab:red')
    # Fixed, readable window. Near the C_P sign change (J ~ 0.8-1.0) eta blows
    # up because C_P -> 0; those points clip off the top rather than flattening
    # the whole curve.
    ax_eta.set_ylim(0, 3.0)

    # Combined legend.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_eta.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax.set_title('Lifting-line propeller performance vs advance ratio')
    fig.tight_layout()
    fig.savefig('assignment_2/J_sweep.png', dpi=150)
    print("Saved figure to assignment_2/J_sweep.png")
    plt.show()


if __name__ == "__main__":
    main()
