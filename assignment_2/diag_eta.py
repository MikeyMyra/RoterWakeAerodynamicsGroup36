"""Diagnostic: trace where eta > 1 comes from in the BEM solver.

Runs the blade-element method at the design J and prints the dimensional
thrust-power vs shaft-power balance, checks the coefficient normalisations
against standard propeller definitions, and inspects the induction/velocity
sign conventions.
"""

import numpy as np
from Lifting_line_prop import BEM

J = 1.6
bem = BEM(J=J, radius=0.7, n_blades=6, U_inf=60)
bem.blade_element(resolution=200, spacing='cosine', use_prandtl=True)

rho, U, D, n, R = bem.rho, bem.U_inf, bem.D, bem.n_rps, bem.radius
omega = 2 * np.pi * n
B = bem.n_blades

# --- Re-integrate dimensional thrust and torque from the stored section data ---
r = np.array(bem.r_R_list) * R
Fax = np.array(bem.F_axial_list)
Faz = np.array(bem.F_azimuth_list)
a = np.array(bem.a_list)
ap = np.array(bem.a_prime_list)
phi = np.deg2rad(np.array(bem.phi_list))
alpha = np.array(bem.alpha_list)
dr = bem.dr

T = np.sum(Fax[1:] * B * dr)
Q = np.sum(Faz[1:] * r[1:] * B * dr)
P_shaft = Q * omega
P_thrust = T * U

print("=" * 70)
print(f"J = {J},  U_inf = {U} m/s,  n = {n:.3f} rev/s,  omega = {omega:.2f} rad/s")
print(f"rho = {rho:.4f},  D = {D} m,  rpm = {bem.rpm:.1f}")
print("-" * 70)
print("DIMENSIONAL BALANCE")
print(f"  T (thrust)            = {T:12.3f} N")
print(f"  Q (torque)            = {Q:12.3f} N·m")
print(f"  P_shaft = Q·omega     = {P_shaft:12.3f} W")
print(f"  P_thrust = T·U_inf    = {P_thrust:12.3f} W")
print(f"  eta = T·U / (Q·omega) = {P_thrust / P_shaft:12.4f}   <-- must be <1 (and < ideal ceiling)")
print("-" * 70)

# --- Coefficient definitions vs standard propeller convention ---
CT = T / (rho * n**2 * D**4)
CQ = Q / (rho * n**2 * D**5)
CP_from_P = P_shaft / (rho * n**3 * D**5)
print("COEFFICIENT DEFINITIONS (standard propeller convention)")
print(f"  C_T = T/(rho n^2 D^4)      = {CT:.4f}")
print(f"  C_Q = Q/(rho n^2 D^5)      = {CQ:.4f}")
print(f"  C_P = P/(rho n^3 D^5)      = {CP_from_P:.4f}")
print(f"  identity check C_P = 2*pi*C_Q : {CP_from_P:.4f} vs {2*np.pi*CQ:.4f}  "
      f"(match={np.isclose(CP_from_P, 2*np.pi*CQ)})")
print(f"  eta = J*C_T/C_P            = {J*CT/CP_from_P:.4f}")
print(f"  ideal momentum eta_i = 2/(1+sqrt(1+C_T)) = "
      f"{2/(1+np.sqrt(1+CT)):.4f}   <-- frictionless propeller ceiling")
print("-" * 70)

# --- Velocity / induction convention check at a mid station (propeller) ---
i = np.argmin(np.abs(r - 0.7 * R))
Vax = U * (1 + a[i])
Vtan = omega * r[i] * (1 - ap[i])
print(f"CONVENTION CHECK at r/R = {r[i]/R:.2f}  (propeller convention)")
print(f"  a  = {a[i]:+.4f}   a' = {ap[i]:+.4f}")
print(f"  V_axial      = U*(1 + a)      = {Vax:.3f} m/s   "
      f"(> U: slipstream accelerated, as a propeller should)")
print(f"  V_tangential = omega r (1-a') = {Vtan:.3f} m/s")
print(f"  phi = {np.rad2deg(phi[i]):.2f} deg,  alpha = {alpha[i]:.2f} deg")
print()
eta_local_prop = (1 - ap[i]) / (1 + a[i])
print(f"  frictionless local eta (1-a')/(1+a) = {eta_local_prop:.4f}  (<1, physical)")
print("=" * 70)
