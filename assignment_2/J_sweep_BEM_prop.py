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

from Lifting_line_prop import BEM


def run_bem(J, radius=0.7, n_blades=6, U_inf=60,
            resolution=100, spacing='cosine', use_prandtl=True):
    """Solve the blade-element method at one J and return (C_T, C_Q, C_P)."""
    bem = BEM(J=J, radius=radius, n_blades=n_blades, U_inf=U_inf)
    bem.blade_element(resolution=resolution, spacing=spacing,
                      use_prandtl=use_prandtl)
    return bem.CT, bem.CQ, bem.CP


def zero_thrust_J(J_values, CT):
    """Advance ratio where C_T crosses zero (linear interp), or None."""
    for k in range(len(CT) - 1):
        if CT[k] > 0 >= CT[k + 1]:
            f = CT[k] / (CT[k] - CT[k + 1])
            return J_values[k] + f * (J_values[k + 1] - J_values[k])
    return None


def main():
    J_values = np.linspace(0.4, 3.2, 29)

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

    # Efficiency is only meaningful in the propeller regime (C_T > 0, C_P > 0).
    # Right at the zero-thrust J both C_T and C_P collapse to 0, so eta = J C_T/C_P
    # is a degenerate 0/0 that spikes; exclude that thin band (C_T < 0.03) too.
    eta_phys = np.where((CP > 0) & (CT > 0.03), eta, np.nan)
    J0 = zero_thrust_J(J_values, CT)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(J_values, CT, '-o', ms=4, label=r'$C_T = T/(\rho n^2 D^4)$')
    ax.plot(J_values, CQ, '-s', ms=4, label=r'$C_Q = Q/(\rho n^2 D^5)$')
    ax.plot(J_values, CP, '-^', ms=4, label=r'$C_P = P/(\rho n^3 D^5)$')
    ax.axhline(0, color='0.6', lw=0.8)
    if J0 is not None:
        ax.axvline(J0, color='0.4', ls='--', lw=1)
        ax.axvspan(J0, J_values[-1], color='0.85', alpha=0.4, zorder=0)
        ax.annotate(f'zero thrust\n$J_0\\approx{J0:.2f}$', xy=(J0, 0),
                    xytext=(J0 - 0.05, ax.get_ylim()[1] * 0.55), ha='right',
                    fontsize=9, color='0.3')
        ax.text(0.5 * (J0 + J_values[-1]), ax.get_ylim()[0] * 0.9,
                'windmill\n(C_T<0)', ha='center', va='bottom',
                fontsize=9, color='0.4')
    ax.set_xlabel('Advance ratio $J = U_\\infty/(n D)$')
    ax.set_ylabel(r'$C_T,\; C_Q,\; C_P$')
    ax.grid(True)

    ax_eta = ax.twinx()
    ax_eta.plot(J_values, eta_phys, '-d', ms=4, color='tab:red',
                label=r'$\eta = C_T J / C_P$')
    ax_eta.axhline(1.0, color='tab:red', ls=':', lw=1,
                   label=r'$\eta = 1$ (ideal ceiling)')
    ax_eta.set_ylabel(r'Efficiency $\eta$', color='tab:red')
    ax_eta.tick_params(axis='y', labelcolor='tab:red')
    ax_eta.set_ylim(0, 1.15)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_eta.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    ax.set_title('BEM (FIXED, propeller convention) performance vs J')
    fig.tight_layout()
    fig.savefig('assignment_2/J_sweep_BEM_prop.png', dpi=150)
    print("Saved figure to assignment_2/J_sweep_BEM_prop.png")
    plt.show()


if __name__ == "__main__":
    main()
