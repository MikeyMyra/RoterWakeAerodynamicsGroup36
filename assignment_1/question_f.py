import matplotlib.pyplot as plt
import numpy as np

from plotter import plot
from BEM import BEM


def case_f(J_baseline):
    
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
        [bem_linear.r_R_list, bem_cosine.r_R_list], [bem_linear.F_axial_list, bem_cosine.F_axial_list],
        ["Linear Spacing", "Cosine Spacing"],
        "r/R", "Thrust Force per unit length (N/m)"
    )
    # print(res)
    # print(0/0)
    bem_lin_conv = BEM(J=J_baseline)
    bem_lin_conv.blade_element(resolution=200, spacing='linear', use_prandtl=True)
    # N_conv=len(bem_lin_conv.CT_conv_list)
    # print(np.arange(0,N_conv,1).shape)
    # print(np.array(bem_lin_conv.CT_conv_list).shape)
    # print(bem_lin_conv.CT_conv_ind)
    # print(bem_lin_conv.CT_conv_ind)
    plot('Convergence History',[bem_lin_conv.CT_conv_ind],[bem_lin_conv.CT_conv_list],['Convergence Histotry'],'Iteration point','CT')
    
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
    
    case_f(J_baseline=J_baseline)
    
    plt.show()