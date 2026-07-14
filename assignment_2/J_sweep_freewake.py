"""Advance-ratio sweep for the FREE-WAKE lifting-line rotor model.

Same idea as J_sweep.py, but every point is solved with the free (deforming)
wake solver Lifting_line_freewake(): the wake is convected by the freestream
plus the velocity induced by the bound circulation AND by the wake itself, so
the wake geometry relaxes to a self-consistent shape at each J.

Plots the propeller performance coefficients:
    C_T = T / (rho n^2 D^4)
    C_Q = Q / (rho n^2 D^5)
    C_P = P / (rho n^3 D^5)
    eta = C_T * J / C_P   (propulsive efficiency)

NOTE: the free wake is expensive (~2-3 min per J). Results are cached to
J_sweep_freewake_results.npz; delete that file to force a full recompute.

Run from the repository root so the relative airfoil path resolves:
    python assignment_2\\J_sweep_freewake.py
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
import matplotlib.pyplot as plt

from Lifting_line_freewake import BEM


# ---- solver settings shared by every point in the sweep ----
RESOLUTION = 20
A_IND_WAKE = 0.2          # initial helix pitch guess (the wake deforms from here)
SPACING = 'linear'
WAKE_ITERATIONS = 100
WAKE_RELAX = 0.3
WAKE_CORE = 0.10          # wake-on-wake desingularisation (stability)
TEND = 0.2
DT = 0.005


def run_lifting_line_freewake(J, radius=0.7, n_blades=6, U_inf=60,
                              resolution=RESOLUTION, a_ind_wake=A_IND_WAKE,
                              spacing=SPACING, wake_iterations=WAKE_ITERATIONS,
                              wake_relax=WAKE_RELAX, wake_core=WAKE_CORE,
                              tend=TEND, dt=DT, verbose=False):
    """Solve the free-wake lifting line at one J and return (C_T, C_Q, C_P)."""
    bem = BEM(J=J, radius=radius, n_blades=n_blades, U_inf=U_inf)
    bem.tlst = np.arange(0, tend, dt)

    output = bem.Lifting_line_freewake(resolution=resolution, a_ind_wake=a_ind_wake,
                                       spacing=spacing, wake_iterations=wake_iterations,
                                       wake_relax=wake_relax, wake_core=wake_core,
                                       verbose=verbose)
    (a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, used_iter, conv_hist,
     r_control, alpha_out, phi_out, W0, W) = output

    # Integrate section forces to rotor thrust and torque (propeller convention).
    # Fnorm_out / Ftan_out are per control point (panel midspan); skip the root
    # panel (index 0), which carries no blade.
    n_panels = len(r_control)
    T_LL = 0.0
    Q_LL = 0.0
    for p in range(1, n_panels):
        T_LL += Fnorm_out[p] * n_blades * bem.dr[p]
        Q_LL += Ftan_out[p] * n_blades * r_control[p] * bem.dr[p]

    C_T = T_LL / (bem.rho * bem.n_rps**2 * bem.D**4)
    C_Q = Q_LL / (bem.rho * bem.n_rps**2 * bem.D**5)
    C_P = (Q_LL * bem.omega) / (bem.rho * bem.n_rps**3 * bem.D**5)
    return C_T, C_Q, C_P



def _run_freewake_case(index, J, n_cases):
    """Worker function: run one J case and package timing/output for the parent process."""
    t0 = time.time()
    error = None
    try:
        C_T, C_Q, C_P = run_lifting_line_freewake(J)
    except Exception as exc:
        C_T = C_Q = C_P = np.nan
        error = str(exc)

    if np.isfinite(C_P) and C_P != 0:
        eta_j = C_T * J / C_P
    else:
        eta_j = np.nan

    return {
        "index": index,
        "n_cases": n_cases,
        "J": J,
        "CT": C_T,
        "CQ": C_Q,
        "CP": C_P,
        "eta": eta_j,
        "elapsed": time.time() - t0,
        "error": error,
    }



CACHE_FILE = 'assignment_2/J_sweep_freewake_results.npz'
FROZEN_CACHE = 'assignment_2/J_sweep_results.npz'   # optional overlay (frozen wake)


def main():
    J_values = np.linspace(0.4, 2.4, 11)

    # Reuse cached solver output when available (the free wake is very slow);
    # delete J_sweep_freewake_results.npz to force a full recompute.
    if os.path.exists(CACHE_FILE):
        data = np.load(CACHE_FILE)
        J_values, CT, CQ, CP = data['J'], data['CT'], data['CQ'], data['CP']
        print(f"Loaded cached results from {CACHE_FILE}")
    else:
        n_cases = len(J_values)
        n_workers = n_cases
        available_cores = os.cpu_count() or 1
        if available_cores < n_workers:
            print(f"Warning: {n_cases} cases requested but only {available_cores} CPU cores detected; "
                  f"the OS will time-slice workers.")
        print(f"Running {n_cases} free-wake cases in parallel ({n_workers} worker processes).")

        CT = np.full(n_cases, np.nan)
        CQ = np.full(n_cases, np.nan)
        CP = np.full(n_cases, np.nan)
        t_start = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_run_freewake_case, k, float(J), n_cases)
                for k, J in enumerate(J_values)
            ]

            for future in as_completed(futures):
                result = future.result()
                k = result["index"]
                CT[k] = result["CT"]
                CQ[k] = result["CQ"]
                CP[k] = result["CP"]

                if result["error"] is not None:
                    print(f"J = {result['J']:5.2f} | FAILED ({result['error']}); storing NaN")

                print(f"[{k+1:2d}/{result['n_cases']}] J = {result['J']:5.2f} | "
                      f"C_T = {result['CT']:8.4f} | C_Q = {result['CQ']:8.4f} | "
                      f"C_P = {result['CP']:8.4f} | eta = {result['eta']:7.4f} | "
                      f"{result['elapsed']:5.1f}s")

        np.savez(CACHE_FILE, J=J_values, CT=CT, CQ=CQ, CP=CP)
        print(f"Saved results cache to {CACHE_FILE}  (total {time.time()-t_start:.1f}s)")

    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.where(CP != 0, CT * J_values / CP, np.nan)

    # Efficiency is only physically meaningful where the rotor acts as a
    # propeller (C_T > 0 AND C_P > 0). Outside that branch eta is negative or
    # blows up (C_P -> 0), so mask those points.
    eta_phys = np.where((CP > 0) & (CT > 0), eta, np.nan)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(J_values, CT, '-o', label=r'$C_T = T/(\rho n^2 D^4)$')
    ax.plot(J_values, CQ, '-s', label=r'$C_Q = Q/(\rho n^2 D^5)$')
    ax.plot(J_values, CP, '-^', label=r'$C_P = P/(\rho n^3 D^5)$')
    ax.axhline(0, color='0.6', lw=0.8)

    # Optional: overlay the frozen-wake sweep (if its cache exists) to show the
    # effect of letting the wake deform. Drawn as faint dashed lines.
    if os.path.exists(FROZEN_CACHE):
        try:
            fz = np.load(FROZEN_CACHE)
            ax.plot(fz['J'], fz['CT'], '--o', color='C0', alpha=0.35, lw=1, ms=3, label='C_T (frozen wake)')
            ax.plot(fz['J'], fz['CQ'], '--s', color='C1', alpha=0.35, lw=1, ms=3, label='C_Q (frozen wake)')
            ax.plot(fz['J'], fz['CP'], '--^', color='C2', alpha=0.35, lw=1, ms=3, label='C_P (frozen wake)')
        except Exception:
            pass

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
    ax_eta.set_ylim(0, 3.0)

    # Combined legend.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_eta.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    ax.set_title('Free-wake lifting-line propeller performance vs advance ratio')
    fig.tight_layout()
    fig.savefig('assignment_2/J_sweep_freewake.png', dpi=150)
    print("Saved figure to assignment_2/J_sweep_freewake.png")
    plt.show()


if __name__ == "__main__":
    main()
