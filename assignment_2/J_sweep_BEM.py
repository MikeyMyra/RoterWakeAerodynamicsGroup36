"""Advance-ratio sweep using the BEM blade-element solver.

Same performance coefficients as J_sweep.py, but computed with the momentum /
blade-element method instead of the lifting line:
    C_T = T / (rho n^2 D^4)
    C_Q = Q / (rho n^2 D^5)
    C_P = P / (rho n^3 D^5)
    eta = C_T * J / C_P   (propulsive efficiency)

Run from the repository root so the relative airfoil path resolves:
    python assignment_2\\J_sweep_BEM.py
"""

import numpy as np
import matplotlib.pyplot as plt

from Lifting_line import BEM


def run_bem(J, radius=0.7, n_blades=6, U_inf=60,
            resolution=100, spacing='cosine', use_prandtl=True):
    """Solve the blade-element method at one J and return (C_T, C_Q, C_P)."""
    bem = BEM(J=J, radius=radius, n_blades=n_blades, U_inf=U_inf)
    bem.blade_element(resolution=resolution, spacing=spacing,
                      use_prandtl=use_prandtl)
    return bem.CT, bem.CQ, bem.CP


def main():
    J_values = np.linspace(0.4, 2.4, 21)

    CT_list, CQ_list, CP_list = [], [], []
    for J in J_values:
        C_T, C_Q, C_P = run_bem(J)
        CT_list.append(C_T)
        CQ_list.append(C_Q)
        CP_list.append(C_P)
        eta_j = C_T * J / C_P if C_P != 0 else np.nan
        print(f"J = {J:5.2f} | C_T = {C_T:8.4f} | C_Q = {C_Q:8.4f} | "
              f"C_P = {C_P:8.4f} | eta = {eta_j:7.4f}")

    CT = np.array(CT_list)
    CQ = np.array(CQ_list)
    CP = np.array(CP_list)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.where(CP != 0, CT * J_values / CP, np.nan)

    # Efficiency is only physical where the rotor acts as a propeller
    # (C_T > 0 AND C_P > 0); mask the rest so a single blow-up near C_P -> 0
    # does not wreck the axis scaling.
    eta_phys = np.where((CP > 0) & (CT > 0), eta, np.nan)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(J_values, CT, '-o', ms=4, label=r'$C_T = T/(\rho n^2 D^4)$')
    ax.plot(J_values, CQ, '-s', ms=4, label=r'$C_Q = Q/(\rho n^2 D^5)$')
    ax.plot(J_values, CP, '-^', ms=4, label=r'$C_P = P/(\rho n^3 D^5)$')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlabel('Advance ratio $J = U_\\infty/(n D)$')
    ax.set_ylabel(r'$C_T,\; C_Q,\; C_P$')
    ax.grid(True)

    ax_eta = ax.twinx()
    ax_eta.plot(J_values, eta_phys, '-d', ms=4, color='tab:red',
                label=r'$\eta = C_T J / C_P$')
    ax_eta.axhline(1.0, color='tab:red', ls=':', lw=1,
                   label=r'$\eta = 1$ (physical ceiling)')
    ax_eta.set_ylabel(r'Efficiency $\eta$', color='tab:red')
    ax_eta.tick_params(axis='y', labelcolor='tab:red')
    # eta > 1 everywhere and blows up near the C_P sign change (J ~ 0.9);
    # cap the axis so the trend stays readable and the outliers clip.
    ax_eta.set_ylim(0, 3.0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_eta.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax.set_title('BEM (blade-element) propeller performance vs advance ratio')
    fig.tight_layout()
    fig.savefig('assignment_2/J_sweep_BEM.png', dpi=150)
    print("Saved figure to assignment_2/J_sweep_BEM.png")
    plt.show()


if __name__ == "__main__":
    main()
