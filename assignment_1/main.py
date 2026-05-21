import matplotlib.pyplot as plt

from question_d import case_d
from question_e import case_e
from question_f import case_f
from plotter import plot
from BEM import BEM


if __name__ == "__main__":
    
    J_baseline = 2.1428570754 # rpm = 1200
    
    case_d(J_baseline=J_baseline)
    case_e(J_baseline=J_baseline)
    case_f(J_baseline=J_baseline)

    bem = BEM(J=2.1428570754)
    bem.blade_element(resolution=100, use_prandtl=False)

    plot(
        "Spanwise Distribution: Angle of Attack",
        bem.r_R_list, [bem.alpha_list],
        ["Angle of Attack (α)"],
        "r/R", "α (deg)"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(bem.r_R_list, bem.p_stag_up_inf_list, 'b-', marker='+', label='p_stag_upstream_inf', markersize=12)
    plt.plot(bem.r_R_list, bem.p_stag_up_list, 'r-', marker='x', label='p_stag_upstream_rotor')
    plt.plot(bem.r_R_list, bem.p_stag_down_list, 'g-', marker='o', label='p_stag_downstream_rotor', markersize=8)
    plt.plot(bem.r_R_list, bem.p_stag_down_inf_list, 'p-', marker='v', label='p_stag_downstream_inf', markersize=8)

    plt.title("Bladewise Distribution: stagnation pressure upstream and downstream", fontsize=14, fontweight='bold')
    plt.xlabel("r/R", fontsize=12)
    plt.ylabel("Pressure (Pa)", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()