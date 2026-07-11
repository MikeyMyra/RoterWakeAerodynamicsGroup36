"""Diagnostic: inspect lifting-line induction signs and eta at J=1.6."""
import numpy as np
from Lifting_line_prop import BEM

bem = BEM(J=1.6, radius=0.7, n_blades=6, U_inf=60)
bem.tlst = np.arange(0, 0.2, 0.005)
out = bem.Lifting_line(resolution=20, a_ind_wake=-0.2, spacing='cosine',
                       plot_geometry=False)
a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, _, _, r_control, alpha_out, phi_out = out

# One blade's worth of stations (drop root control point at index 0).
ncp_blade = len(r_control)
a1 = np.array(a_out[:ncp_blade])
ap1 = np.array(aline_out[:ncp_blade])
r1 = np.array(r_control)

mask = np.abs(Gamma_out[:ncp_blade]) > 1e-9
print("r/R      a (u_ind/U)   a'(swirl)   phi(deg)   alpha(deg)")
for i in range(ncp_blade):
    if mask[i]:
        print(f"{r1[i]/bem.radius:5.3f}  {a1[i]:+11.4f}  {ap1[i]:+10.4f}  "
              f"{phi_out[i]:8.2f}  {alpha_out[i]:8.2f}")

# Dimensional balance
r_l = r_control[1:]
T = sum(Fnorm_out[i+1]*bem.n_blades*bem.dr[i] for i in range(len(r_l)))
Q = sum(Ftan_out[i+1]*bem.n_blades*r_l[i]*bem.dr[i] for i in range(len(r_l)))
print(f"\nT={T:.2f} N  Q={Q:.2f} N·m  P_shaft={Q*bem.omega:.1f} W  "
      f"T·U={T*bem.U_inf:.1f} W  eta={T*bem.U_inf/(Q*bem.omega):.4f}")
print(f"mean a = {a1[mask].mean():+.4f}   mean a' = {ap1[mask].mean():+.4f}")
