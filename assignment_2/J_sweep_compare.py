"""Overlay the FIXED BEM and lifting-line performance sweeps to check that the
two methods stay aligned and both give physical (eta < 1) propeller behaviour.

LL results are read from the cache written by J_sweep_prop.py; BEM is fast so it
is recomputed inline.
"""
import numpy as np
import matplotlib.pyplot as plt

from Lifting_line_prop import BEM
from J_sweep_prop import CACHE_FILE as LL_CACHE
from J_sweep_BEM_prop import run_bem

def zero_thrust_J(J_values, CT):
    for k in range(len(CT) - 1):
        if CT[k] > 0 >= CT[k + 1]:
            f = CT[k] / (CT[k] - CT[k + 1])
            return J_values[k] + f * (J_values[k + 1] - J_values[k])
    return None

# --- Lifting line (from cache) ---
ll = np.load(LL_CACHE)
Jll, CTll, CPll = ll['J'], ll['CT'], ll['CP']
etall = np.where((CPll > 0) & (CTll > 0.03), Jll * CTll / CPll, np.nan)

# --- BEM (recompute on the same J grid as the LL sweep) ---
CTb, CPb = [], []
for J in Jll:
    C_T, _, C_P = run_bem(J)
    CTb.append(C_T)
    CPb.append(C_P)
CTb, CPb = np.array(CTb), np.array(CPb)
etab = np.where((CPb > 0) & (CTb > 0.03), Jll * CTb / CPb, np.nan)

J0b, J0l = zero_thrust_J(Jll, CTb), zero_thrust_J(Jll, CTll)

fig, (ax, axe) = plt.subplots(1, 2, figsize=(13, 5))

# Coefficients
ax.plot(Jll, CTb, '-o', color='tab:blue', label='C_T  BEM')
ax.plot(Jll, CTll, '--o', color='tab:blue', mfc='none', label='C_T  LL')
ax.plot(Jll, CPb, '-^', color='tab:green', label='C_P  BEM')
ax.plot(Jll, CPll, '--^', color='tab:green', mfc='none', label='C_P  LL')
ax.set_xlabel('J');  ax.set_ylabel('C_T, C_P')
ax.set_title('Coefficients: BEM vs lifting line (fixed)')
ax.grid(True);  ax.legend()

for J0, c in [(J0b, 'tab:red'), (J0l, 'tab:orange')]:
    if J0 is not None:
        ax.axvline(J0, color=c, ls=':', lw=1)

# Efficiency
axe.plot(Jll, etab, '-d', color='tab:red', label=r'$\eta$ BEM')
axe.plot(Jll, etall, '--d', color='tab:orange', label=r'$\eta$ LL')
axe.axhline(1.0, color='k', ls=':', lw=1, label=r'$\eta=1$ ceiling')
for J0, c, lab in [(J0b, 'tab:red', 'BEM'), (J0l, 'tab:orange', 'LL')]:
    if J0 is not None:
        axe.axvline(J0, color=c, ls=':', lw=1)
        axe.annotate(f'$J_0^{{{lab}}}\\approx{J0:.2f}$', xy=(J0, 0.05),
                     rotation=90, va='bottom', ha='right', fontsize=8, color=c)
axe.set_xlabel('J');  axe.set_ylabel(r'Efficiency $\eta$')
axe.set_ylim(0, 1.25)
axe.set_title('Efficiency: BEM vs lifting line (fixed)')
axe.grid(True);  axe.legend(loc='lower right')

fig.tight_layout()
fig.savefig('assignment_2/J_sweep_compare.png', dpi=150)
print('Saved assignment_2/J_sweep_compare.png')
plt.show()
