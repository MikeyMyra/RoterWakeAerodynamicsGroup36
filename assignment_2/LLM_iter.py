import numpy as np
from scipy.interpolate import interp1d


def lifting_line(
    bem,  # the BEM/LLM class instance — provides biot_savart, calc_ind_filiment, Make_ind_matrix
    # Rotor geometry
    radius,
    n_blades,
    blade_start_fraction,
    collective_blade_pitch,
    collective_blade_pitch_location,
    # Operating conditions
    U_inf,
    rpm,
    rho,
    # Airfoil data
    AoA,
    cl_data,
    cd_data,
    # Solver settings
    resolution       = 40,
    tolerance        = 1e-6,
    max_iterations   = 1000,
    spacing          = 'cosine',
    use_prandtl      = True,
    track_convergence = False,
    relax            = 0.25,
):
    cl_interp = interp1d(AoA, cl_data, kind='linear', fill_value='extrapolate')
    cd_interp = interp1d(AoA, cd_data, kind='linear', fill_value='extrapolate')

    omega  = (2 * np.pi * rpm) / 60
    A_disk = np.pi * radius ** 2

    # ── radial stations ───────────────────────────────────────────────────────
    if spacing == 'cosine':
        theta             = np.linspace(0, np.pi, resolution + 1)
        r_normalized_temp = 0.5 * (1 - np.cos(theta))
        r_stations_norm   = blade_start_fraction + (1 - blade_start_fraction) * r_normalized_temp
    else:
        r_stations_norm = np.linspace(blade_start_fraction, 1, resolution + 1)

    r_stations_norm = np.insert(r_stations_norm, 0, 2*r_stations_norm[0] - r_stations_norm[1])
    r_stations_norm = np.insert(r_stations_norm, 0, 0.0)

    # ── blade geometry ────────────────────────────────────────────────────────
    twist_stations      = []
    chord_norm_stations = []
    for r_norm in r_stations_norm:
        if r_norm > blade_start_fraction:
            twist_stations.append(
                -50 * r_norm + 35
                + collective_blade_pitch
                + collective_blade_pitch_location * 50 - 35
            )
            chord_norm_stations.append(0.18 - 0.06 * r_norm)
        else:
            twist_stations.append(0.0)
            chord_norm_stations.append(0.0)

    bem.r_stations_abs = r_stations_norm * radius
    bem.dr             = np.diff(bem.r_stations_abs)
    bem.resolution     = resolution

    # ── induction matrix via bem.Make_ind_matrix ──────────────────────────────
    # Make_ind_matrix needs bem.V_axial set before calling
    # TODO: confirm whether V_axial should be U_inf*(1-a) with a iterated,
    #       or just U_inf as a fixed wake convection speed.
    a_init       = 0.1
    bem.V_axial  = U_inf * (1.0 - a_init)

    # Make_ind_matrix uses bem.r_stations_abs, bem.dr, bem.resolution,
    # bem.V_axial, and bem.omega internally, and returns nothing —
    # we capture A by monkey-patching it to return both matrices.
    # Since Make_ind_matrix only stores A locally and shows a plot,
    # we need to call it and capture the result.
    # TODO: ideally modify Make_ind_matrix to return A_axial, A_tangential
    #       instead of just plotting. For now we replicate its loop here
    #       using bem.calc_ind_filiment directly, which is the actual
    #       computation kernel.

    n_mat = resolution + 1
    A_axial      = np.zeros((n_mat, n_mat))
    A_tangential = np.zeros((n_mat, n_mat))

    tend = 0.1
    dt   = 0.005
    bem.tlst  = np.arange(0, tend, dt)
    bem.omega = 1  # NOTE: hardcoded to 1 in Make_ind_matrix — TODO: should this be (2pi*rpm)/60?
    bem.xarr  = bem.tlst * bem.V_axial

    r_cp_abs = bem.r_stations_abs[:-1] + bem.dr / 2  # panel midpoints, mirrors r_stations_abs_circ

    for i in range(n_mat):
        r_vortex     = r_cp_abs[i]
        bem.dr_used  = bem.dr[i]

        bem.yarr  = (r_vortex + 0.5 * bem.dr_used) * np.sin(bem.omega * bem.tlst)
        bem.zarr  = (r_vortex + 0.5 * bem.dr_used) * np.cos(bem.omega * bem.tlst)
        bem.yarr2 = (r_vortex - 0.5 * bem.dr_used) * np.sin(bem.omega * bem.tlst)
        bem.zarr2 = (r_vortex - 0.5 * bem.dr_used) * np.cos(bem.omega * bem.tlst)

        for j in range(n_mat):
            r_p = r_cp_abs[j]
            C_ind, _ = bem.calc_ind_filiment([0, 0, r_p], r_vortex, plot=False)
            A_axial[j, i]      = C_ind[0]
            A_tangential[j, i] = C_ind[1]

    # ── active panels (blade exists beyond hub) ───────────────────────────────
    active_mask = (r_cp_abs / radius) > blade_start_fraction
    active_idx  = np.where(active_mask)[0]

    # ── initialise Γ from flat-plate estimate ─────────────────────────────────
    Gamma = np.zeros(n_mat)
    for idx in active_idx:
        r_abs     = r_cp_abs[idx]
        r_norm    = r_abs / radius
        chord_abs = chord_norm_stations[idx] * radius
        twist_deg = twist_stations[idx]
        V_ax      = U_inf * (1.0 - a_init)
        V_tan     = omega * r_abs
        V_eff     = np.sqrt(V_ax**2 + V_tan**2)
        phi       = np.arctan2(V_ax, V_tan)
        alpha     = np.rad2deg(np.deg2rad(twist_deg) - phi)
        cl        = float(cl_interp(alpha))
        Gamma[idx] = 0.5 * V_eff * chord_abs * cl

    # ── outer iteration on Γ ──────────────────────────────────────────────────
    converged_iter = max_iterations
    for outer_iter in range(max_iterations):

        u_ind = A_axial      @ Gamma
        w_ind = A_tangential @ Gamma

        Gamma_new = np.zeros(n_mat)
        for idx in active_idx:
            r_abs     = r_cp_abs[idx]
            r_norm    = r_abs / radius
            chord_abs = chord_norm_stations[idx] * radius
            twist_deg = twist_stations[idx]

            a       = u_ind[idx] / U_inf
            a_prime = w_ind[idx] / (omega * r_abs)

            V_ax  = U_inf * (1.0 - a)
            V_tan = omega * r_abs * (1.0 + a_prime)
            V_eff = np.sqrt(V_ax**2 + V_tan**2)

            phi   = np.arctan2(V_ax, V_tan)
            alpha = np.rad2deg(np.deg2rad(twist_deg) - phi)
            cl    = float(cl_interp(alpha))

            F = (bem._calculate_prandtl_factor(r_norm, a)
                 if use_prandtl else 1.0)

            Gamma_new[idx] = 0.5 * V_eff * chord_abs * cl * F

        delta = np.max(np.abs(Gamma_new - Gamma))
        Gamma = (1.0 - relax) * Gamma + relax * Gamma_new

        if delta < tolerance:
            converged_iter = outer_iter + 1
            print(f"LLM converged after {converged_iter} iterations.")
            break
    else:
        print("Warning: LLM did not converge within max_iterations.")

    # ── post-process converged solution ───────────────────────────────────────
    u_ind = A_axial      @ Gamma
    w_ind = A_tangential @ Gamma

    dT_total = 0.0
    dQ_total = 0.0

    results = {k: [] for k in (
        'r_R', 'a', 'a_prime', 'alpha', 'phi', 'cl', 'cd',
        'V_axial', 'V_tangential', 'V_effective',
        'lift', 'drag', 'F_axial', 'F_azimuth',
        'circulation', 'F_prandtl', 'dCT_dr', 'dCQ_dr', 'dCP_dr'
    )}

    convergence_history = {'CT': [], 'CQ': [], 'station': []} if track_convergence else None

    for idx in active_idx:
        r_abs     = r_cp_abs[idx]
        r_norm    = r_abs / radius
        chord_abs = chord_norm_stations[idx] * radius
        twist_deg = twist_stations[idx]
        dr_k      = bem.dr[idx]

        a       = u_ind[idx] / U_inf
        a_prime = w_ind[idx] / (omega * r_abs)

        V_ax  = U_inf * (1.0 - a)
        V_tan = omega * r_abs * (1.0 + a_prime)
        V_eff = np.sqrt(V_ax**2 + V_tan**2)

        phi   = np.arctan2(V_ax, V_tan)
        alpha = np.rad2deg(np.deg2rad(twist_deg) - phi)

        cl = float(cl_interp(alpha))
        cd = float(cd_interp(alpha))

        F = (bem._calculate_prandtl_factor(r_norm, a)
             if use_prandtl else 1.0)

        lift      = 0.5 * rho * V_eff**2 * chord_abs * cl * F
        drag      = 0.5 * rho * V_eff**2 * chord_abs * cd * F
        F_axial   = lift * np.cos(phi) + drag * np.sin(phi)
        F_azimuth = lift * np.sin(phi) - drag * np.cos(phi)

        A_a  = 2 * np.pi * r_abs * dr_k
        C_T  = (F_axial   * n_blades * dr_k) / (0.5 * rho * U_inf**2 * A_a)
        dCQ  = (F_azimuth * n_blades * r_abs * dr_k) / (0.5 * rho * U_inf**2 * A_a * radius)
        dCP  = dCQ * omega * radius / U_inf

        dT_total += F_axial   * dr_k * n_blades
        dQ_total += F_azimuth * r_abs * dr_k * n_blades

        results['r_R'].append(r_norm)
        results['a'].append(a)
        results['a_prime'].append(a_prime)
        results['alpha'].append(alpha)
        results['phi'].append(np.rad2deg(phi))
        results['cl'].append(cl)
        results['cd'].append(cd)
        results['V_axial'].append(V_ax)
        results['V_tangential'].append(V_tan)
        results['V_effective'].append(V_eff)
        results['lift'].append(lift)
        results['drag'].append(drag)
        results['F_axial'].append(F_axial)
        results['F_azimuth'].append(F_azimuth)
        results['circulation'].append(Gamma[idx])
        results['F_prandtl'].append(F)
        results['dCT_dr'].append(C_T)
        results['dCQ_dr'].append(dCQ)
        results['dCP_dr'].append(dCP)

        if track_convergence:
            convergence_history['CT'].append(dT_total / (0.5 * rho * U_inf**2 * A_disk))
            convergence_history['CQ'].append(dQ_total / (0.5 * rho * U_inf**2 * A_disk * radius))
            convergence_history['station'].append(int(idx))

    results['CT'] = dT_total / (0.5 * rho * U_inf**2 * A_disk)
    results['CQ'] = dQ_total / (0.5 * rho * U_inf**2 * A_disk * radius)
    results['CP'] = results['CQ'] * omega * radius / U_inf
    results['converged_iter']      = converged_iter
    results['convergence_history'] = convergence_history
    results['Gamma']               = Gamma
    results['r_cp_abs']            = r_cp_abs
    results['A_axial']             = A_axial
    results['A_tangential']        = A_tangential

    return results


lifting_line()