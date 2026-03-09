from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np


def plot(name, x_data, y_data_list, labels, xlabel="x", ylabel="y", colors=None):
    """Plot multiple datasets on the same figure"""
    plt.figure(figsize=(10, 6))
    if colors is None:
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    
    for i, (y_data, label) in enumerate(zip(y_data_list, labels)):
        plt.plot(x_data, y_data, marker='o', linestyle='-', 
                color=colors[i % len(colors)], label=label, markersize=4)
    
    plt.title(name, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


class BEM:
    
    def __init__(self, J):
        
        # Rotor specs (absolute values)
        self.radius = 0.7  # meters
        self.n_blades = 6
        self.blade_start_fraction = 0.25  # Fraction of radius (0-1)
        self.collective_blade_pitch = 46  # degrees
        
        # Operational specs
        self.U_inf = 60  # m/s
        self.rpm = 60 * (self.U_inf / (J * 2 * self.radius))
        self.altitude = 2000  # meters
        self.incidence = 0
        self.rotor_yaw = 0
        
        # Blade specs (normalized)
        self.r_R_norm, self.twist, self.c_R_norm = self._get_blade_specs(self.blade_start_fraction, self.collective_blade_pitch)
        
        # Airfoil data
        self.AoA, self.cl, self.cd, self.cm = self._get_airfoil()
        self.rho = self._get_isa_density(self.altitude)
    
    @staticmethod
    def _get_blade_specs(blade_start_fraction, collective_blade_pitch, resolution=100):
        
        r_R_norm = np.linspace(0, 1, resolution + 1)
        
        twist = []
        c_R_norm = []
        
        for r_norm in r_R_norm:
            if r_norm > blade_start_fraction:
                
                # Twist distribution (degrees)
                twist.append(-50 * r_norm + 35 + collective_blade_pitch)
                
                # Normalized chord distribution (c/R)
                c_R_norm.append(0.18 - 0.06 * r_norm)
                
            else:
                twist.append(0)
                c_R_norm.append(0)
        
        return r_R_norm, twist, c_R_norm
    
    @staticmethod
    def _get_airfoil():
        
        data = []
        with open("assignment_1/ARAD8pct_polar.txt", "r") as file:
            for line in file:
                row = line.strip().split()
                data.append(row)
        data = data[2:]
        
        AoA = [float(row[0]) for row in data]
        cl  = [float(row[1]) for row in data]
        cd  = [float(row[2]) for row in data]
        cm  = [float(row[3]) for row in data]
        
        return AoA, cl, cd, cm
    
    @staticmethod
    def _get_isa_density(h): 
        
        T0, p0, L, R, g = 288.15, 101325, 0.0065, 287.05, 9.80665
        
        T = T0 - L*h
        p = p0 * (T/T0)**(g/(R*L))
        rho = p/(R*T)
        
        return rho
    
    @staticmethod
    def _apply_glauert_correction(CT):
        
        CT1 = 1.816
        CT2 = (2 * np.sqrt(CT1)) - CT1
        
        if CT < 0:
            CT = 0
        
        if CT < CT2:
            a = (1/2) * (1 - np.sqrt(max(0, 1 - CT)))
        else:
            a = 1 + ((CT - CT1) / (4 * (np.sqrt(CT1) - 1)))
        
        return a
    
    def _calculate_prandtl_factor(self, r_norm, a):
        
        mu = r_norm
        mu_root = self.blade_start_fraction
        
        if mu < mu_root:
            return 1.0
        
        if mu >= 0.99:
            return 0.0
        
        if a >= 0.99 or a <= 0.0:
            a = 0.5
        
        omega = (2 * np.pi * self.rpm) / 60
        lambda_local = (omega * r_norm * self.radius) / self.U_inf
        
        try:
            exponent_tip = -(self.n_blades / 2) * ((1 - mu) / mu) * np.sqrt(1 + (mu**2 * lambda_local**2) / ((1 - a)**2))
            exponent_tip = np.clip(exponent_tip, -10, 0)
            f_tip = (2 / np.pi) * np.arccos(np.exp(exponent_tip))
        except:
            f_tip = 0.01
        
        try:
            exponent_root = -(self.n_blades / 2) * ((mu - mu_root) / mu) * np.sqrt(1 + (mu**2 * lambda_local**2) / ((1 - a)**2))
            exponent_root = np.clip(exponent_root, -10, 0)
            f_root = (2 / np.pi) * np.arccos(np.exp(exponent_root))
        except:
            f_root = 1.0
        
        F_total = f_tip * f_root
        F_total = max(F_total, 0.0001)
        
        return F_total
    
    def blade_element(self, resolution=100, tolerance=1e-5, max_iterations=100, spacing='linear', use_prandtl=True, track_convergence=False):
        
        cl_interp = interp1d(self.AoA, self.cl, kind='linear', fill_value='extrapolate')
        cd_interp = interp1d(self.AoA, self.cd, kind='linear', fill_value='extrapolate')
        
        omega = (2 * np.pi * self.rpm) / 60
        
        # Generate normalized radial stations (0 to 1)
        if spacing == 'cosine': # cosine
            theta = np.linspace(0, np.pi, resolution + 1)
            r_normalized_temp = 0.5 * (1 - np.cos(theta))
            r_stations_norm = self.blade_start_fraction + (1 - self.blade_start_fraction) * r_normalized_temp
        else:  # linear
            r_stations_norm = np.linspace(0, 1, resolution + 1)
        
        # Regenerate normalized blade properties at new radial stations
        twist_stations = []
        chord_norm_stations = []
        
        for r_norm in r_stations_norm:
            if r_norm > self.blade_start_fraction:
                twist_stations.append(-50 * r_norm + 35 + self.collective_blade_pitch)
                chord_norm_stations.append(0.18 - 0.06 * r_norm)
            else:
                twist_stations.append(0)
                chord_norm_stations.append(0)
        
        # Calculate dr in absolute units
        r_stations_abs = r_stations_norm * self.radius
        dr = np.diff(r_stations_abs)
        dr = np.append(dr, dr[-1])
        
        A_disk = np.pi * self.radius**2
        
        # Initialize tracking lists (all use NORMALIZED radius 0-1)
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
        
        # Convergence tracking
        if track_convergence:
            self.convergence_history = {'CT': [], 'CQ': [], 'iteration': []}
        
        dT_total = 0
        dQ_total = 0
        
        for i, r_norm in enumerate(r_stations_norm):
            
            a = 0.5
            a_prime = 0.5
            
            # Get absolute radius for calculations
            r_abs = r_norm * self.radius
            
            if r_norm <= self.blade_start_fraction:
                self.r_R_list.append(r_norm)
                self.a_list.append(0)
                self.a_prime_list.append(0)
                self.alpha_list.append(0)
                self.phi_list.append(0)
                self.cl_list.append(0)
                self.cd_list.append(0)
                self.V_axial_list.append(0)
                self.V_tangential_list.append(0)
                self.V_effective_list.append(0)
                self.lift_list.append(0)
                self.drag_list.append(0)
                self.F_axial_list.append(0)
                self.F_azimuth_list.append(0)
                self.circulation_list.append(0)
                self.F_prandtl_list.append(0)
                self.dCT_dr_list.append(0)
                self.dCQ_dr_list.append(0)
                self.dCP_dr_list.append(0)
                self.iterations_list.append(0)
                continue
            
            iter_count = 0
            for iteration in range(max_iterations):
                iter_count += 1
                
                # Velocities (absolute)
                V_axial = self.U_inf * (1 - a)
                V_tangential = omega * r_abs * (1 + a_prime)
                V_effective = np.sqrt(V_axial**2 + V_tangential**2)
                
                # Get blade properties (convert normalized chord to absolute)
                chord_abs = chord_norm_stations[i] * self.radius
                theta_deg = twist_stations[i]
                theta_rad = np.deg2rad(theta_deg)
                
                # Flow angles
                phi = np.arctan2(V_axial, V_tangential)
                alpha = np.rad2deg(theta_rad - phi)
                
                # Airfoil coefficients
                cl = float(cl_interp(alpha))
                cd = float(cd_interp(alpha))
                
                # Forces per unit length (absolute)
                lift = 0.5 * self.rho * V_effective**2 * chord_abs * cl
                drag = 0.5 * self.rho * V_effective**2 * chord_abs * cd
                
                # Apply Prandtl correction if enabled
                if use_prandtl:
                    F_prandtl = self._calculate_prandtl_factor(r_norm, a)
                else:
                    F_prandtl = 1.0
                
                # Corrected forces
                F_azimuth = (lift * np.sin(phi) - drag * np.cos(phi)) * F_prandtl
                F_axial = (lift * np.cos(phi) + drag * np.sin(phi)) * F_prandtl
                
                # Thrust coefficient
                A_a = 2 * np.pi * r_abs * dr[i]
                C_T = (F_axial * self.n_blades * dr[i]) / (0.5 * self.rho * self.U_inf**2 * A_a)
                
                # Apply Glauert correction for axial induction
                a_calc = self._apply_glauert_correction(C_T)
                
                # Azimuthal induction
                a_prime_calc = (F_azimuth * self.n_blades) / (2 * self.rho * (2 * np.pi * r_abs) * self.U_inf**2 * (1 - a_calc) * omega * r_abs)
                
                # Check convergence
                if abs(a_calc - a) < tolerance and abs(a_prime_calc - a_prime) < tolerance:
                    break
                
                # Relaxation of iterative variables
                a = 0.75 * a + 0.25 * a_calc
                a_prime = 0.75 * a_prime + 0.25 * a_prime_calc
            
            # Calculate circulation
            circulation = 0.5 * V_effective * chord_abs * cl
            
            # Accumulate totals
            dT_total += F_axial * dr[i] * self.n_blades
            dQ_total += F_azimuth * r_abs * dr[i] * self.n_blades
            
            # Differential coefficients
            dCT = C_T
            dCQ = (F_azimuth * self.n_blades * r_abs * dr[i]) / (0.5 * self.rho * self.U_inf**2 * A_a * self.radius)
            dCP = dCQ * omega * self.radius / self.U_inf
            
            # Store values (using NORMALIZED radius)
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
            
            # Track convergence if requested
            if track_convergence:
                CT_current = dT_total / (0.5 * self.rho * self.U_inf**2 * A_disk)
                CQ_current = dQ_total / (0.5 * self.rho * self.U_inf**2 * A_disk * self.radius)
                self.convergence_history['CT'].append(CT_current)
                self.convergence_history['CQ'].append(CQ_current)
                self.convergence_history['iteration'].append(i)
        
        # Total coefficients
        self.CT = dT_total / (0.5 * self.rho * self.U_inf**2 * A_disk)
        self.CQ = dQ_total / (0.5 * self.rho * self.U_inf**2 * A_disk * self.radius)
        self.CP = self.CQ * omega * self.radius / self.U_inf



def case_1(J_baseline):
    
    # ========== PART 1: BASELINE ANALYSIS ==========
    print("\n" + "="*60)
    print("PART 1: BASELINE BEM ANALYSIS")
    print("="*60)
    
    bem = BEM(J=J_baseline)
    bem.blade_element(resolution=100, use_prandtl=True)
    
    # 1.1 Spanwise distribution of angle of attack
    plot(
        "Spanwise Distribution: Angle of Attack",
        bem.r_R_list, [bem.alpha_list],
        ["Angle of Attack (α)"],
        "r/R", "α (deg)"
    )
    
    # 1.2 Spanwise distribution of inflow angle
    plot(
        "Spanwise Distribution: Inflow Angle",
        bem.r_R_list, [bem.phi_list],
        ["Inflow Angle (φ)"],
        "r/R", "φ (deg)"
    )
    
    # 1.3 Spanwise distribution of axial and azimuthal inductions
    plot(
        "Spanwise Distribution: Induction Factors",
        bem.r_R_list, [bem.a_list, bem.a_prime_list],
        ["Axial Induction (a)", "Azimuthal Induction (a')"],
        "r/R", "Induction Factor"
    )
    
    # 1.4 Spanwise distribution of thrust and azimuthal loading
    plot(
        "Spanwise Distribution: Thrust and Torque Loading",
        bem.r_R_list, [bem.F_axial_list, bem.F_azimuth_list],
        ["Thrust Loading (F_axial)", "Torque Loading (F_azimuth)"],
        "r/R", "Force per unit length (N/m)"
    )
    
    # 1.5 Total thrust and torque versus advance ratio
    print("\nCalculating performance vs advance ratio...")
    J_values = np.array([1.6, 2.0, 2.4])
    CT_values = []
    CQ_values = []
    CP_values = []
    
    for J in J_values:
        bem_temp = BEM(J=J)
        bem_temp.blade_element(resolution=100, use_prandtl=True)
        CT_values.append(bem_temp.CT)
        CQ_values.append(bem_temp.CQ)
        CP_values.append(bem_temp.CP)
    
    plot(
        "Total Performance vs Advance Ratio",
        J_values, [CT_values, CQ_values, CP_values],
        ["Thrust Coefficient (CT)", "Torque Coefficient (CQ)", "Power Coefficient (CP)"],
        "Advance Ratio (J)", "Coefficient"
    )
    
    print(f"\nBaseline Performance (J={J_baseline:.4f}, Radius={bem.radius}m):")
    print(f"  CT = {bem.CT:.6f}")
    print(f"  CQ = {bem.CQ:.6f}")
    print(f"  CP = {bem.CP:.6f}")

def case_2(J_baseline):
    
    # ========== PART 2: INFLUENCE OF TIP CORRECTION ==========
    print("\n" + "="*60)
    print("PART 2: INFLUENCE OF TIP CORRECTION")
    print("="*60)
    
    bem_no_prandtl = BEM(J=J_baseline)
    bem_no_prandtl.blade_element(resolution=100, use_prandtl=False)
    
    bem_with_prandtl = BEM(J=J_baseline)
    bem_with_prandtl.blade_element(resolution=100, use_prandtl=True)
    
    # 2.1 Compare thrust loading
    plot(
        "Influence of Prandtl Tip Correction on Thrust Loading",
        bem_with_prandtl.r_R_list,
        [bem_no_prandtl.F_axial_list, bem_with_prandtl.F_axial_list],
        ["Without Prandtl", "With Prandtl"],
        "r/R", "Thrust Force per unit length (N/m)"
    )
    
    # 2.2 Compare induction factors
    plot(
        "Influence of Prandtl Tip Correction on Axial Induction",
        bem_with_prandtl.r_R_list,
        [bem_no_prandtl.a_list, bem_with_prandtl.a_list],
        ["Without Prandtl", "With Prandtl"],
        "r/R", "Axial Induction Factor (a)"
    )
    
    # 2.3 Show Prandtl factor distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bem_with_prandtl.r_R_list, bem_with_prandtl.F_prandtl_list, marker='o', linestyle='-', color='green', markersize=4)
    ax.set_xlabel("r/R", fontsize=12)
    ax.set_ylabel("Prandtl Loss Factor (F)", fontsize=12)
    ax.set_title("Prandtl Tip and Root Loss Factor Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='F = 1 (no loss)')
    ax.legend()
    plt.tight_layout()
    
    print(f"\nPerformance Comparison (Radius={bem_with_prandtl.radius}m):")
    print(f"  Without Prandtl: CT = {bem_no_prandtl.CT:.6f}, CQ = {bem_no_prandtl.CQ:.6f}")
    print(f"  With Prandtl:    CT = {bem_with_prandtl.CT:.6f}, CQ = {bem_with_prandtl.CQ:.6f}")
    print(f"  CT Reduction:    {(1 - bem_with_prandtl.CT/bem_no_prandtl.CT)*100:.2f}%")

def case_3(J_baseline):
    
    # ========== PART 3: INFLUENCE OF GRID RESOLUTION AND SPACING ==========
    print("\n" + "="*60)
    print("PART 3: GRID RESOLUTION AND SPACING STUDY")
    print("="*60)
    
    # Test different resolutions
    resolutions = [10, 20, 30, 50, 75, 100, 150, 200]
    CT_linear = []
    CT_cosine = []
    
    for res in resolutions:
        bem_lin = BEM(J=J_baseline)
        bem_lin.blade_element(resolution=res, spacing='linear', use_prandtl=True)
        CT_linear.append(bem_lin.CT)
        
        bem_cos = BEM(J=J_baseline)
        bem_cos.blade_element(resolution=res, spacing='cosine', use_prandtl=True)
        CT_cosine.append(bem_cos.CT)
    
    plot(
        "Convergence of Thrust Coefficient with Grid Resolution",
        resolutions, [CT_linear, CT_cosine],
        ["Linear Spacing", "Cosine Spacing"],
        "Number of Annuli", "CT"
    )
    
    # Compare spacing methods at same resolution
    bem_linear = BEM(J=np.pi/6)
    bem_linear.blade_element(resolution=50, spacing='linear', use_prandtl=True)
    
    bem_cosine = BEM(J=np.pi/6)
    bem_cosine.blade_element(resolution=50, spacing='cosine', use_prandtl=True)
    
    plot(
        "Thrust Loading: Linear vs Cosine Spacing (50 elements)",
        bem_linear.r_R_list, [bem_linear.F_axial_list, bem_cosine.F_axial_list],
        ["Linear Spacing", "Cosine Spacing"],
        "r/R", "Thrust Force per unit length (N/m)"
    )
    
    # Show radial station distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range(len(bem_linear.r_R_list)), bem_linear.r_R_list, 'o-', label='Linear')
    ax1.set_xlabel("Station Index", fontsize=12)
    ax1.set_ylabel("r/R", fontsize=12)
    ax1.set_title("Linear Spacing Distribution", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(len(bem_cosine.r_R_list)), bem_cosine.r_R_list, 'o-', color='orange', label='Cosine')
    ax2.set_xlabel("Station Index", fontsize=12)
    ax2.set_ylabel("r/R", fontsize=12)
    ax2.set_title("Cosine Spacing Distribution", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print(f"\nGrid Resolution Study:")
    print(f"  Linear spacing (100 elements):  CT = {CT_linear[-3]:.6f}")
    print(f"  Cosine spacing (100 elements):  CT = {CT_cosine[-3]:.6f}")
    print(f"  Difference: {abs(CT_linear[-3] - CT_cosine[-3]):.6f}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    
    J_baseline = 2.1428570754 # rpm = 1200
    
    case_1(J_baseline)
    case_2(J_baseline)
    case_3(J_baseline)
    
    plt.show()