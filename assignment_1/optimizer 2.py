import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ==========================================
# 1. Operational Specs & Environmental Data
# ==========================================
V_0 = 60.0  # m/s (Freestream velocity)
RPM = 1200.0  # Revolutions per minute
n_rev_sec = RPM / 60.0  # Revolutions per second (n in classical prop theory)
omega = n_rev_sec * 2 * np.pi  # Angular velocity (rad/s)
D = 1.4  # m (Propeller diameter)
R = D / 2.0  # m (Propeller radius)
n_blades = 6  # Number of blades
target_thrust = 1200.0  # N (Required thrust for optimization)

# ISA Altitude = 2000 m
altitude = 2000.0  # m
T_isa = 288.15 - 0.0065 * altitude  # Temperature (K)
p_isa = 101325 * (1 - 2.25577e-5 * altitude) ** 5.25588  # Pressure (Pa)
R_air = 287.05
rho = p_isa / (R_air * T_isa)  # Air density (kg/m^3)
c_sound = np.sqrt(1.4 * R_air * T_isa)  # Speed of sound (m/s)


# ==========================================
# 2. Aerodynamic Database (Airfoil Polar)
# ==========================================
def get_optimum_airfoil_params(Re, Ma):
    """
    Returns the characteristics of the airfoil at its maximum efficiency.
    Replace with your actual ARA-D8% interpolated data.
    """
    alpha_Emax = np.radians(7.5)  # Angle of attack for max efficiency (rad)
    cl_Emax = 1.3596  # Lift coefficient at alpha_Emax
    cd_Emax = 0.01485  # Drag coefficient at alpha_Emax
    return alpha_Emax, cl_Emax, cd_Emax


# ==========================================
# 3. Minimum Induced Loss Optimization
# ==========================================
n_sections = 103
r_array = np.linspace(0.15 * R, R * 0.99, n_sections)
dr = r_array[1] - r_array[0]

x = omega * r_array / V_0
x_R = omega * R / V_0


def evaluate_propeller_for_zeta(zeta):
    """Calculates total thrust for a given Lagrange multiplier (zeta)."""
    thrust_calc = 0.0
    for i, r in enumerate(r_array):
        phi_t = np.arctan(1.0 / x_R * (1 + zeta / 2))
        f = (n_blades / 2.0) * (1.0 - r / R) / np.sin(phi_t)
        F = (2.0 / np.pi) * np.arccos(np.exp(-f)) if f < 10 else 1.0

        phi = np.arctan(1.0 / x[i] * (1 + zeta / 2))
        V_app = np.sqrt(V_0 ** 2 + (omega * r) ** 2)

        alpha_opt, cl_opt, cd_opt = get_optimum_airfoil_params(1e6, V_app / c_sound)

        Gamma = (2 * np.pi * V_0 ** 2 * zeta * F * np.sin(phi) * np.cos(phi)) / (omega * n_blades)
        W = V_0 * (1 + zeta / 2 * (np.cos(phi) ** 2)) / np.sin(phi)
        chord = (2 * Gamma) / (W * cl_opt)

        dL = 0.5 * rho * W ** 2 * chord * cl_opt
        dD = 0.5 * rho * W ** 2 * chord * cd_opt
        dT = n_blades * (dL * np.cos(phi) - dD * np.sin(phi)) * dr

        thrust_calc += dT
    return thrust_calc


# ==========================================
# 4. Solvers & Global Coefficients
# ==========================================
print(f"Designing Optimum Propeller for Target Thrust = {target_thrust} N...")

# Find zeta that satisfies the required thrust
zeta_opt = brentq(lambda z: evaluate_propeller_for_zeta(z) - target_thrust, 0.001, 3.0)
print(f"Converged Lagrange Multiplier (Zeta): {zeta_opt:.4f}")

# Arrays to store results for plotting
optimum_chord = np.zeros(n_sections)
optimum_twist = np.zeros(n_sections)
total_thrust = 0.0
total_torque = 0.0

# Final pass to calculate geometry and integrating torque/power
for i, r in enumerate(r_array):
    phi_t = np.arctan(1.0 / x_R * (1 + zeta_opt / 2))
    f = (n_blades / 2.0) * (1.0 - r / R) / np.sin(phi_t)
    F = (2.0 / np.pi) * np.arccos(np.exp(-f)) if f < 10 else 1.0

    phi = np.arctan(1.0 / x[i] * (1 + zeta_opt / 2))
    V_app = np.sqrt(V_0 ** 2 + (omega * r) ** 2)
    alpha_opt, cl_opt, cd_opt = get_optimum_airfoil_params(1e6, V_app / c_sound)

    Gamma = (2 * np.pi * V_0 ** 2 * zeta_opt * F * np.sin(phi) * np.cos(phi)) / (omega * n_blades)
    W = V_0 * (1 + zeta_opt / 2 * (np.cos(phi) ** 2)) / np.sin(phi)

    chord = (2 * Gamma) / (W * cl_opt)
    twist = phi + alpha_opt

    optimum_chord[i] = chord
    optimum_twist[i] = twist

    # Elementary Forces [cite: 40]
    dL = 0.5 * rho * W ** 2 * chord * cl_opt
    dD = 0.5 * rho * W ** 2 * chord * cd_opt

    dT = n_blades * (dL * np.cos(phi) - dD * np.sin(phi)) * dr
    dM = n_blades * (dL * np.sin(phi) + dD * np.cos(phi)) * r * dr

    total_thrust += dT
    total_torque += dM

# Calculate Shaft Power
total_power = total_torque * omega

# Calculate Non-Dimensional Coefficients
# Advance ratio (J), Thrust coefficient (C_T), Torque coefficient (C_Q), Power coefficient (C_P)
J = V_0 / (n_rev_sec * D)
C_T = total_thrust / (rho * n_rev_sec ** 2 * D ** 4)
C_Q = total_torque / (rho * n_rev_sec ** 2 * D ** 5)
C_P = total_power / (rho * n_rev_sec ** 3 * D ** 5)  # Equivalent to C_P = 2 * pi * C_Q
efficiency = (C_T * J) / C_P

print("-" * 30)
print(f"Total Thrust: {total_thrust:.2f} N")
print(f"Total Power: {total_power / 1000:.2f} kW")
print(f"Advance Ratio (J): {J:.4f}")
print(f"Thrust Coefficient (C_T): {C_T:.4f}")
print(f"Power Coefficient (C_P): {C_P:.4f}")
print(f"Propeller Efficiency: {efficiency * 100:.2f}%")

# ==========================================
# 5. Plotting Geometry Distributions
# ==========================================
r_R_ratio = r_array / R
twist_deg = np.degrees(optimum_twist)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Chord Distribution
ax1.plot(r_R_ratio, optimum_chord, 'b-', linewidth=2)
ax1.set_title("Optimum Chord Distribution")
ax1.set_xlabel("Non-dimensional radius ($r/R$)")
ax1.set_ylabel("Chord Length (m)")
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Twist Distribution
ax2.plot(r_R_ratio, twist_deg, 'r-', linewidth=2)
ax2.set_title("Optimum Twist (Pitch) Distribution")
ax2.set_xlabel("Non-dimensional radius ($r/R$)")
ax2.set_ylabel("Twist Angle (degrees)")
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Structural check [cite: 123]
max_chord_ratio = np.max(optimum_chord) / R
if max_chord_ratio < 0.15:
    print(
        f"\nNote: bmax/R ({max_chord_ratio:.3f}) < 0.15. Consider reducing blade number 'n' for structural reasons. [cite: 123]")
elif max_chord_ratio > 0.24:
    print(
        f"\nNote: bmax/R ({max_chord_ratio:.3f}) > 0.24. Increasing blade number 'n' can produce significant improvement in aerodynamic efficiency. [cite: 123]")