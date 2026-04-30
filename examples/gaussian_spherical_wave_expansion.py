import numpy as np
from matplotlib import pyplot as plt

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
)
from grid_lib.spherical_coordinates.angular_momentum import LM_to_I
from grid_lib.spherical_coordinates.potentials import (
    gaussian_spherical_wave_expansion,
)

def g00(r, r0, A, alpha):
    R0 = np.linalg.norm(r0)
    prefactor = np.sqrt(np.pi) * A / (2 * alpha * r * R0)
    return prefactor * (
        np.exp(-alpha * (r - R0) ** 2) - np.exp(-alpha * (r + R0) ** 2)
    )


# Gaussian parameters
A = 1.0
alpha = 1.0
z0 = 1.5  # centre on the z-axis

r0 = np.array([0.0, 0.0, z0])

L_max = 4
M_max = 0  # z-axis centre: only M = 0 terms are non-zero

# Radial grid
N = 150
r_max = 10.0
gll = GaussLegendreLobatto(N, Rational_map(r_max=r_max, alpha=0.4))
r = gll.r[1:-1]

# Compute radial components
g_LM = gaussian_spherical_wave_expansion(r, r0, A, alpha, L_max, M_max)

# --- Superposition of two Gaussians placed symmetrically on the z-axis ---
r0_plus = np.array([0.0, 0.0, z0])
r0_minus = np.array([0.0, 0.0, -z0])

g_LM_plus = gaussian_spherical_wave_expansion(r, r0_plus, A, alpha, L_max, M_max)
g_LM_minus = gaussian_spherical_wave_expansion(r, r0_minus, A, alpha, L_max, M_max)
g_LM_super = g_LM_plus + g_LM_minus

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

# Top panel: single Gaussian
for L in range(L_max + 1):
    I = LM_to_I(L, 0, L_max, M_max)
    ax.plot(r, g_LM[I], label=rf"$g_{{L={L},M=0}}(r)$")

ax.plot(r, g00(r, r0, A, alpha), "k--", label=r"$g_{00}(r)$ (exact)")
ax.set_xlabel(r"$r$ (a.u.)")
ax.set_ylabel(r"$g_{LM}(r)$")
ax.set_title(
    rf"Single Gaussian: $e^{{-\alpha|\mathbf{{r}}-\mathbf{{r}}_0|^2}}$, "
    rf"$\mathbf{{r}}_0 = (0,0,{z0})$, $\alpha={alpha}$"
)
ax.legend()
ax.set_xlim(0, r_max)

# Right panel: symmetric superposition
for L in range(L_max + 1):
    I = LM_to_I(L, 0, L_max, M_max)
    ax2.plot(r, g_LM_super[I], label=rf"$g_{{L={L},M=0}}(r)$")

ax2.set_xlabel(r"$r$ (a.u.)")
ax2.set_ylabel(r"$g_{LM}(r)$")
ax2.set_xlim(0, r_max)
ax2.set_title(
    rf"Superposition: $e^{{-\alpha|\mathbf{{r}}-(0,0,z_0)|^2}} + "
    rf"e^{{-\alpha|\mathbf{{r}}+(0,0,z_0)|^2}}$, "
    rf"$z_0={z0}$, $\alpha={alpha}$"
)
ax2.legend()

plt.tight_layout()
plt.show()
