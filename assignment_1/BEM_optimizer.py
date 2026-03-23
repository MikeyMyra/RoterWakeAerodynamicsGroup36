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
            CT = -CT

        if CT < CT2:
            a = 0.5 * (1 - np.sqrt(1 - CT))
        else:
            a = 1 + ((CT - CT1) / (4 * (np.sqrt(CT1) - 1)))
            print('correction!')

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
        x = r_norm
        return (43.0463*x**4
            -137.5923*x**3
            +185.8450*x**2
            -157.5377*x
            +84.8556)#7.5 #-50 * r_norm + self.collective_blade_pitch + self.collective_blade_pitch_location * 50

    @staticmethod
    def _baseline_chord_norm(r_norm):
        x=r_norm
        return 5*(-2.0151*x**4
            +4.5464*x**3
            -4.0353*x**2
            +1.5364*x
            -0.0543)# 0.18 - 0.06 * r_norm #0.07/r_norm

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

        return  twist_stations, chord_norm_stations
       #  return np.array([1.55240019, 1.54433479, 1.53628997, 1.5282667 , 1.52026593,
       # 1.51228857, 1.50433554, 1.49640772, 1.48850598, 1.48063119,
       # 1.47278417, 1.46496574, 1.4571767 , 1.44941781, 1.44168983,
       # 1.4339935 , 1.42632952, 1.4186986 , 1.41110139, 1.40353854,
       # 1.39601069, 1.38851843, 1.38106235, 1.37364302, 1.36626096,
       # 1.35891671, 1.35161075, 1.34434356, 1.3371156 , 1.32992729,
       # 1.32277905, 1.31567127, 1.30860432, 1.30157855, 1.29459428,
       # 1.28765182, 1.28075147, 1.2738935 , 1.26707814, 1.26030564,
       # 1.2535762 , 1.24689002, 1.24024728, 1.23364814, 1.22709273,
       # 1.22058118, 1.21411361, 1.20769009, 1.20131071, 1.19497552,
       # 1.18868458, 1.18243791, 1.17623553, 1.17007744, 1.16396364,
       # 1.1578941 , 1.15186877, 1.14588762, 1.13995058, 1.13405758,
       # 1.12820854, 1.12240336, 1.11664194, 1.11092416, 1.1052499 ,
       # 1.09961902, 1.09403139, 1.08848684, 1.08298523, 1.07752638,
       # 1.07211012, 1.06673627, 1.06140463, 1.05611502, 1.05086722,
       # 1.04566103, 1.04049624, 1.03537262, 1.03028996, 1.02524802,
       # 1.02024657, 1.01528537, 1.01036417, 1.00548274, 1.00064081,
       # 0.99583815, 0.99107448, 0.98634956, 0.98166311, 0.97701488,
       # 0.97240459, 0.96783198, 0.96329678, 0.95879871, 0.9543375 ,
       # 0.94991287, 0.94552456, 0.94117226, 0.93685572, 0.93257465,
       # 0.92832876, 0.92411779, 0.91994144]), np.array([0.09617323, 0.10089931, 0.10554297, 0.11010123, 0.11457123,
       # 0.11895028, 0.12323584, 0.12742551, 0.13151707, 0.13550842,
       # 0.13939765, 0.14318299, 0.14686282, 0.15043568, 0.15390026,
       # 0.15725539, 0.16050005, 0.16363337, 0.16665461, 0.16956316,
       # 0.17235856, 0.17504045, 0.17760861, 0.18006294, 0.18240344,
       # 0.18463023, 0.18674352, 0.18874362, 0.19063094, 0.19240596,
       # 0.19406927, 0.19562149, 0.19706335, 0.19839563, 0.19961915,
       # 0.20073482, 0.20174356, 0.20264635, 0.20344421, 0.20413819,
       # 0.20472935, 0.20521879, 0.20560761, 0.20589694, 0.2060879 ,
       # 0.20618162, 0.20617921, 0.20608181, 0.2058905 , 0.20560637,
       # 0.2052305 , 0.20476391, 0.20420761, 0.20356259, 0.20282976,
       # 0.20201003, 0.20110424, 0.20011317, 0.19903757, 0.1978781 ,
       # 0.19663537, 0.19530991, 0.19390218, 0.19241254, 0.19084129,
       # 0.1891886 , 0.18745456, 0.18563913, 0.18374216, 0.18176337,
       # 0.17970232, 0.17755843, 0.17533096, 0.17301895, 0.17062127,
       # 0.16813654, 0.16556314, 0.16289916, 0.16014238, 0.15729022,
       # 0.15433971, 0.1512874 , 0.14812932, 0.14486086, 0.14147673,
       # 0.13797075, 0.13433575, 0.13056334, 0.12664364, 0.12256498,
       # 0.11831337, 0.11387198, 0.10922022, 0.10433258, 0.09917691,
       # 0.09371179, 0.08788244, 0.08161394, 0.07479925, 0.0672762 ,
       # 0.05877739, 0.04880038, 0.03614725]) # twist_stations, chord_norm_stations

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


            return self.CP + 0*solidity_penalty

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 120, "ftol": 1e-5, "disp": True, "verbose": 1},
            callback=lambda result: print(f"Current Cost: {objective(result):.6f}")
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
            "eta": float(self.J * self.CT / self.CP),
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
        n_control_points=4,
        resolution=100,
        spacing="cosine",
        use_prandtl=True,
        optimize_twist_only=False,
    )
    print(
        "Optimized:",
        f"CT={optimization_result['CT']:.4f}, CP={optimization_result['CP']:.4f}, eta={optimization_result['eta']:.4f}, success={optimization_result['success'], {optimization_result['message']}}",
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



