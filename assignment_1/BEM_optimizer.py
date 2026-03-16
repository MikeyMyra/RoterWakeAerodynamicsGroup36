from math import comb

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from plotter import plot


class BEM:

    def __init__(self, J):

        # Rotor specs (absolute values)
        self.radius = 0.7  # meters
        self.n_blades = 6
        self.blade_start_fraction = 0.25  # Fraction of radius (0-1)
        self.collective_blade_pitch = 46  # degrees
        self.collective_blade_pitch_location = 0.7  # Fraction of radius (0-1)

        # Operational specs
        self.J = J
        self.U_inf = 60  # m/s
        self.rpm = 60 * (self.U_inf / (J * 2 * self.radius))
        self.altitude = 2000  # meters
        self.incidence = 0
        self.rotor_yaw = 0

        # Airfoil data
        self.AoA, self.cl, self.cd, self.cm = self._get_airfoil()
        self.rho = self._get_isa_density(self.altitude)

    @staticmethod
    def _get_airfoil():

        data = []
        with open("assignment_1/ARAD8pct_polar.txt", "r") as file:
            for line in file:
                row = line.strip().split()
                data.append(row)
        data = data[2:]

        AoA = [float(row[0]) for row in data]
        cl = [float(row[1]) for row in data]
        cd = [float(row[2]) for row in data]
        cm = [float(row[3]) for row in data]

        return AoA, cl, cd, cm

    @staticmethod
    def _get_isa_density(h):

        T0, p0, L, R, g = 288.15, 101325, 0.0065, 287.05, 9.80665

        T = T0 - L * h
        p = p0 * (T / T0) ** (g / (R * L))
        rho = p / (R * T)

        return rho

    @staticmethod
    def _apply_glauert_correction(CT):

        CT1 = 1.816
        CT2 = (2 * np.sqrt(CT1)) - CT1

        if CT < 0:
            CT = 0

        if CT < CT2:
            a = 0.5 * (1 - np.sqrt(max(0, 1 - CT)))
        else:
            a = 1 + ((CT - CT1) / (4 * (np.sqrt(CT1) - 1)))

        return a

    @staticmethod
    def bezier_curve(control_points, t):
        """Evaluate a scalar Bezier curve at parameter t in [0, 1]."""
        order = len(control_points) - 1
        value = 0.0
        for i, point in enumerate(control_points):
            value += comb(order, i) * ((1 - t) ** (order - i)) * (t ** i) * point
        return value

    def _calculate_prandtl_factor(self, r_norm, a):

        mu = r_norm
        mu_root = self.blade_start_fraction

        if mu < mu_root:
            return 1.0

        if mu >= 1.00:
            return 0.0

        if a >= 1.00 or a <= 0.0:
            a = 0.5

        omega = (2 * np.pi * self.rpm) / 60
        lambda_local = (omega * r_norm * self.radius) / self.U_inf

        try:
            exponent_tip = -(self.n_blades / 2) * ((1 - mu) / mu) * np.sqrt(
                1 + (mu ** 2 * lambda_local ** 2) / ((1 - a) ** 2)
            )
            f_tip = (2 / np.pi) * np.arccos(np.exp(exponent_tip))
        except Exception:
            f_tip = 1.0

        try:
            exponent_root = -(self.n_blades / 2) * ((mu - mu_root) / mu) * np.sqrt(
                1 + (mu ** 2 * lambda_local ** 2) / ((1 - a) ** 2)
            )
            f_root = (2 / np.pi) * np.arccos(np.exp(exponent_root))
        except Exception:
            f_root = 1.0

        F_total = f_tip * f_root
        return F_total

    def _generate_radial_stations(self, resolution, spacing):

        if spacing == "cosine":
            theta = np.linspace(0, np.pi, resolution + 1)
            r_normalized_temp = 0.5 * (1 - np.cos(theta))
            r_stations_norm = self.blade_start_fraction + (1 - self.blade_start_fraction) * r_normalized_temp
        else:
            r_stations_norm = np.linspace(self.blade_start_fraction, 1, resolution + 1)

        # Keep the existing integration setup (hub point + ghost point)
        r_stations_norm = np.insert(r_stations_norm, 0, 2 * r_stations_norm[0] - r_stations_norm[1])
        r_stations_norm = np.insert(r_stations_norm, 0, 0)
        return r_stations_norm

    def _baseline_twist(self, r_norm):
        return -50 * r_norm + self.collective_blade_pitch + self.collective_blade_pitch_location * 50

    @staticmethod
    def _baseline_chord_norm(r_norm):
        return 0.18 - 0.06 * r_norm #0.07/r_norm

    def _build_blade_geometry(self, r_stations_norm, chord_control_points=None, twist_control_points=None):

        use_bezier = chord_control_points is not None and twist_control_points is not None
        twist_stations = []
        chord_norm_stations = []

        for r_norm in r_stations_norm:
            if r_norm <= self.blade_start_fraction:
                twist_stations.append(0.0)
                chord_norm_stations.append(0.0)
                continue

            if use_bezier:
                t = (r_norm - self.blade_start_fraction) / (1 - self.blade_start_fraction)
                twist_value = self.bezier_curve(twist_control_points, t)
                chord_value = self.bezier_curve(chord_control_points, t)
            else:
                twist_value = self._baseline_twist(r_norm)
                chord_value = self._baseline_chord_norm(r_norm)

            # Keep optimizer-driven geometry in physically meaningful ranges.
            twist_stations.append(float(np.clip(twist_value, -5.0, 85.0)))
            chord_norm_stations.append(float(np.clip(chord_value, 0.001, 0.5)))

        return twist_stations, chord_norm_stations

    def blade_element(
        self,
        resolution=100,
        tolerance=1e-6,
        max_iterations=1000,
        spacing="linear",
        use_prandtl=True,
        track_convergence=False,
        chord_control_points=None,
        twist_control_points=None,
        suppress_warnings=False,
    ):

        cl_interp = interp1d(self.AoA, self.cl, kind="linear", fill_value="extrapolate")
        cd_interp = interp1d(self.AoA, self.cd, kind="linear", fill_value="extrapolate")

        omega = (2 * np.pi * self.rpm) / 60

        r_stations_norm = self._generate_radial_stations(resolution=resolution, spacing=spacing)
        twist_stations, chord_norm_stations = self._build_blade_geometry(
            r_stations_norm,
            chord_control_points=chord_control_points,
            twist_control_points=twist_control_points,
        )

        r_stations_abs = r_stations_norm * self.radius
        dr = np.diff(r_stations_abs)

        A_disk = np.pi * self.radius ** 2

        self.r_R_list = []
        self.a_list = []
        self.a_prime_list = []
        self.alpha_list = []
        self.phi_list = []
        self.cl_list = []
        self.cd_list = []
        self.V_axial_list = []
        self.V_tangential_list = []
        self.V_effective_list = []
        self.lift_list = []
        self.drag_list = []
        self.F_axial_list = []
        self.F_azimuth_list = []
        self.circulation_list = []
        self.F_prandtl_list = []
        self.dCT_dr_list = []
        self.dCQ_dr_list = []
        self.dCP_dr_list = []
        self.iterations_list = []

        if track_convergence:
            self.convergence_history = {"CT": [], "CQ": [], "iteration": []}

        dT_total = 0.0
        dQ_total = 0.0

        for i, r_norm in enumerate(r_stations_norm):

            a = 0.3
            a_prime = 0.0
            r_abs = r_norm * self.radius

            if r_norm <= self.blade_start_fraction:
                self.r_R_list.append(r_norm)
                self.a_list.append(0.0)
                self.a_prime_list.append(0.0)
                self.alpha_list.append(0.0)
                self.phi_list.append(0.0)
                self.cl_list.append(0.0)
                self.cd_list.append(0.0)
                self.V_axial_list.append(0.0)
                self.V_tangential_list.append(0.0)
                self.V_effective_list.append(0.0)
                self.lift_list.append(0.0)
                self.drag_list.append(0.0)
                self.F_axial_list.append(0.0)
                self.F_azimuth_list.append(0.0)
                self.circulation_list.append(0.0)
                self.F_prandtl_list.append(0.0)
                self.dCT_dr_list.append(0.0)
                self.dCQ_dr_list.append(0.0)
                self.dCP_dr_list.append(0.0)
                self.iterations_list.append(0)
                continue

            iter_count = 0
            converged = False
            for _ in range(max_iterations):
                iter_count += 1

                V_axial = self.U_inf * (1 - a)
                V_tangential = omega * r_abs * (1 + a_prime)
                V_effective = np.sqrt(V_axial ** 2 + V_tangential ** 2)

                chord_abs = chord_norm_stations[i] * self.radius
                theta_deg = twist_stations[i]
                theta_rad = np.deg2rad(theta_deg)

                phi = np.arctan2(V_axial, V_tangential)
                alpha = np.rad2deg(theta_rad - phi)

                cl = float(cl_interp(alpha))
                cd = float(cd_interp(alpha))

                lift = 0.5 * self.rho * V_effective ** 2 * chord_abs * cl
                drag = 0.5 * self.rho * V_effective ** 2 * chord_abs * cd

                if use_prandtl:
                    F_prandtl = self._calculate_prandtl_factor(r_norm, a)
                else:
                    F_prandtl = 1.0

                F_azimuth = (lift * np.sin(phi) - drag * np.cos(phi)) * F_prandtl
                F_axial = (lift * np.cos(phi) + drag * np.sin(phi)) * F_prandtl

                A_a = 2 * np.pi * r_abs * dr[i - 1]
                C_T = (F_axial * self.n_blades * dr[i - 1]) / (0.5 * self.rho * self.U_inf ** 2 * A_a)

                a_calc = self._apply_glauert_correction(C_T)
                a_calc = np.clip(a_calc, 0, 0.95)
                a_prime_calc = (F_azimuth * self.n_blades) / (
                    2 * self.rho * (2 * np.pi * r_abs) * self.U_inf ** 2 * (1 - a_calc) * omega * r_abs
                )

                if abs(a_calc - a) < tolerance and abs(a_prime_calc - a_prime) < tolerance:
                    converged = True
                    break

                a = 0.75 * a + 0.25 * a_calc
                a_prime = 0.75 * a_prime + 0.25 * a_prime_calc

            circulation = 0.5 * V_effective * chord_abs * cl

            dT_total += F_axial * dr[i - 1] * self.n_blades
            dQ_total += F_azimuth * r_abs * dr[i - 1] * self.n_blades

            dCT = C_T
            dCQ = (F_azimuth * self.n_blades * r_abs * dr[i - 1]) / (0.5 * self.rho * self.U_inf ** 2 * A_a * self.radius)
            dCP = dCQ * omega * self.radius / self.U_inf

            self.r_R_list.append(r_norm)
            self.a_list.append(a_calc)
            self.a_prime_list.append(a_prime_calc)
            self.alpha_list.append(alpha)
            self.phi_list.append(np.rad2deg(phi))
            self.cl_list.append(cl)
            self.cd_list.append(cd)
            self.V_axial_list.append(V_axial)
            self.V_tangential_list.append(V_tangential)
            self.V_effective_list.append(V_effective)
            self.lift_list.append(lift)
            self.drag_list.append(drag)
            self.F_axial_list.append(F_axial)
            self.F_azimuth_list.append(F_azimuth)
            self.circulation_list.append(circulation)
            self.F_prandtl_list.append(F_prandtl)
            self.dCT_dr_list.append(dCT)
            self.dCQ_dr_list.append(dCQ)
            self.dCP_dr_list.append(dCP)
            self.iterations_list.append(iter_count)

            if track_convergence:
                CT_current = dT_total / (0.5 * self.rho * self.U_inf ** 2 * A_disk)
                CQ_current = dQ_total / (0.5 * self.rho * self.U_inf ** 2 * A_disk * self.radius)
                self.convergence_history["CT"].append(CT_current)
                self.convergence_history["CQ"].append(CQ_current)
                self.convergence_history["iteration"].append(i)

            if not converged and not suppress_warnings:
                print(
                    f"Warning: Blade element method did not converge within the maximum iterations for station {r_abs} for spacing {spacing}."
                )

        self.CT = dT_total / (0.5 * self.rho * self.U_inf ** 2 * A_disk)
        self.CQ = dQ_total / (0.5 * self.rho * self.U_inf ** 2 * A_disk * self.radius)
        self.CP = self.CQ * omega * self.radius / self.U_inf

    def _build_initial_bezier_control_points(self, n_control_points):
        t_points = np.linspace(0.0, 1.0, n_control_points)
        r_points = self.blade_start_fraction + (1 - self.blade_start_fraction) * t_points

        chord_cp = [self._baseline_chord_norm(r) for r in r_points]
        twist_cp = [self._baseline_twist(r) for r in r_points]

        return np.array(r_points, dtype=float), np.array(chord_cp, dtype=float), np.array(twist_cp, dtype=float)

    def optimize_bezier_geometry(
        self,
        n_control_points=10,
        resolution=60,
        spacing="cosine",
        use_prandtl=True,
        max_iterations=120,
        optimize_twist_only=False,
    ):

        r_cp, chord_cp0, twist_cp0 = self._build_initial_bezier_control_points(n_control_points)

        if optimize_twist_only:
            x0 = np.concatenate((np.array(twist_cp0, dtype=float),np.array([1]) ))
            bounds = [(-90.0, 90)] * n_control_points + [(0.01/0.07, 0.5/0.07)]
        else:
            x0 = np.concatenate([chord_cp0, twist_cp0])
            chord_bounds = [(0.0, 1)] * n_control_points
            twist_bounds = [(-90.0, 90)] * n_control_points
            bounds = chord_bounds + twist_bounds

        def unpack(x):
            if optimize_twist_only:
                chord_cp = np.array(chord_cp0, dtype=float) * x[-1]
                twist_cp = np.array(x[:-1][:n_control_points], dtype=float)
            else:
                chord_cp = np.array(x[:n_control_points], dtype=float)
                twist_cp = np.array(x[n_control_points:], dtype=float)
            return chord_cp, twist_cp

        def objective(x):
            chord_cp, twist_cp = unpack(x)

            self.blade_element(
                resolution=resolution,
                spacing=spacing,
                use_prandtl=use_prandtl,
                max_iterations=max_iterations,
                chord_control_points=chord_cp,
                twist_control_points=twist_cp,
                suppress_warnings=True,
            )

            solidity_penalty = 0.0

            # Reconstruct active chords to check local solidity
            r_active = np.array([r for r in self.r_R_list if r > self.blade_start_fraction])
            chord_active = np.array([
                self.bezier_curve(chord_cp, (r - self.blade_start_fraction) / (1 - self.blade_start_fraction))
                for r in r_active
            ])

            for r, c in zip(r_active, chord_active):
                # Calculate local solidity
                local_sigma = (self.n_blades * c) / (2 * np.pi * r)

                # If the blade takes up more than 15% of the local circumference, heavily penalize it
                if local_sigma > 0.15:
                    # Exponential penalty for exceeding the physical limit of BEM
                    solidity_penalty += 10.0 * (local_sigma - 0.15) ** 2 * r**2


            return -self.CP + 0*solidity_penalty

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 120, "ftol": 1e-5, "disp": True, "verbose": 1},
            callback=lambda result: print(f"Current Cost: {-objective(result):.6f}")
        )

        chord_opt, twist_opt = unpack(result.x)
        self.blade_element(
            resolution=resolution,
            spacing=spacing,
            use_prandtl=use_prandtl,
            max_iterations=max_iterations,
            chord_control_points=chord_opt,
            twist_control_points=twist_opt,
            suppress_warnings=True,
        )

        return {
            "success": bool(result.success),
            "message": result.message,
            "nfev": result.nfev,
            "fun": float(result.fun),
            "chord_control_points": chord_opt,
            "twist_control_points": twist_opt,
            "CT": float(self.CT),
            "CQ": float(self.CQ),
            "CP": float(self.CP),
            "eta": float(self.J * self.CT / max(self.CP, 1e-8)),
            "r_over_R": np.array(self.r_R_list),
            "chord_distribution": np.array(
                [
                    self._baseline_chord_norm(r) if r > self.blade_start_fraction else 0.0
                    for r in self.r_R_list
                ]
            ),
            "chord_distribution_optimized": np.array(
                [
                    self.bezier_curve(
                        chord_opt,
                        (r - self.blade_start_fraction) / (1 - self.blade_start_fraction),
                    ) if r > self.blade_start_fraction else 0.0
                    for r in self.r_R_list
                ]
            ),
            "twist_distribution": np.array(
                [
                    self._baseline_twist(r) if r > self.blade_start_fraction else 0.0
                    for r in self.r_R_list
                ]
            ),
            "twist_distribution_optimized": np.array(
                [
                    self.bezier_curve(
                        twist_opt,
                        (r - self.blade_start_fraction) / (1 - self.blade_start_fraction),
                    ) if r > self.blade_start_fraction else 0.0
                    for r in self.r_R_list
                ]
            ),
            "control_points_r_over_R": r_cp,
            "chord_control_points_baseline": chord_cp0,
            "chord_control_points_optimized": chord_opt,
            "twist_control_points_baseline": twist_cp0,
            "twist_control_points_optimized": twist_opt,
            "optimize_twist_only": bool(optimize_twist_only),
        }


if __name__ == "__main__":

    bem = BEM(J=2.0)

    bem.blade_element(resolution=100, use_prandtl=False)
    print(f"Baseline: CT={bem.CT:.4f}, CP={bem.CP:.4f}, eta={bem.J * bem.CT / max(bem.CP, 1e-8):.4f}")

    optimization_result = bem.optimize_bezier_geometry(
        n_control_points=10,
        resolution=70,
        spacing="cosine",
        use_prandtl=True,
        optimize_twist_only=False,
    )
    print(
        "Optimized:",
        f"CT={optimization_result['CT']:.4f}, CP={optimization_result['CP']:.4f}, eta={optimization_result['eta']:.4f}, success={optimization_result['success']}"
    )

    plot(
        "Bezier-Optimized Twist Distribution",
        optimization_result["r_over_R"],
        [optimization_result["twist_distribution"], optimization_result["twist_distribution_optimized"]],
        ["Baseline twist", "Optimized twist"],
        "r/R",
        "Twist (deg)",
    )
    plt.scatter(
        optimization_result["control_points_r_over_R"],
        optimization_result["twist_control_points_baseline"],
        color="b",
        marker="x",
        s=90,
        label="Baseline twist CP",
        zorder=5,
    )
    plt.scatter(
        optimization_result["control_points_r_over_R"],
        optimization_result["twist_control_points_optimized"],
        color="r",
        marker="x",
        s=90,
        label="Optimized twist CP",
        zorder=5,
    )
    plt.legend(loc="best")

    plot(
        "Bezier-Optimized Chord Distribution",
        optimization_result["r_over_R"],
        [optimization_result["chord_distribution"], optimization_result["chord_distribution_optimized"]],
        ["Baseline chord/R", "Optimized chord/R"],
        "r/R",
        "Chord/R (-)",
    )
    plt.scatter(
        optimization_result["control_points_r_over_R"],
        optimization_result["chord_control_points_baseline"],
        color="b",
        marker="x",
        s=90,
        label="Baseline chord CP",
        zorder=5,
    )
    plt.scatter(
        optimization_result["control_points_r_over_R"],
        optimization_result["chord_control_points_optimized"],
        color="r",
        marker="x",
        s=90,
        label="Optimized chord CP",
        zorder=5,
    )
    plt.legend(loc="best")

    plt.show()



