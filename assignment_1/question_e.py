import matplotlib.pyplot as plt

from plotter import plot
from BEM import BEM


def case_e(J_baseline):
    
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
        [bem_no_prandtl.r_R_list , bem_with_prandtl.r_R_list],
        [bem_no_prandtl.F_axial_list, bem_with_prandtl.F_axial_list],
        ["Without Prandtl", "With Prandtl"],
        "r/R", "Thrust Force per unit length (N/m)"
    )
    
    # 2.2 Compare induction factors
    plot(
        "Influence of Prandtl Tip Correction on Axial Induction",
        [bem_no_prandtl.r_R_list , bem_with_prandtl.r_R_list],
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


if __name__ == "__main__":
    
    J_baseline = 2.1428570754 # rpm = 1200
    
    case_e(J_baseline=J_baseline)
    
    plt.show()