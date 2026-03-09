import matplotlib.pyplot as plt
import numpy as np

from plotter import plot
from BEM import BEM


def case_d(J_baseline):
    
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


if __name__ == "__main__":
    
    J_baseline = 2.1428570754 # rpm = 1200
    
    case_d(J_baseline=J_baseline)
    
    plt.show()