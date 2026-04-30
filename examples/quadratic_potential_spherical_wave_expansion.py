import numpy as np
from matplotlib import pyplot as plt

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_momentum import LM_to_I
from grid_lib.spherical_coordinates.potentials import (
    quadratic_potential_spherical_wave_expansion,
)

# Potential parameters
omega = 1.0
z0 = 1.5  # centre on the z-axis

r0 = np.array([0.0, 0.0, z0])

L_max = 1  # exact: only L = 0 and L = 1 are non-zero
M_max = 0  # z-axis centre: only M = 0 terms contribute

# Radial grid
N = 150
r_max = 10.0
gll = GaussLegendreLobatto(N, Linear_map(r_min=0.0, r_max=r_max))
r = gll.r[1:-1]

# Compute radial components for single potential
V_LM = quadratic_potential_spherical_wave_expansion(r, r0, omega, L_max, M_max)

# Superposition: V(r; r0_+) + V(r; r0_-)  -- only even-L terms survive
r0_plus = np.array([0.0, 0.0, z0])
r0_minus = np.array([0.0, 0.0, -z0])

V_LM_plus = quadratic_potential_spherical_wave_expansion(
    r, r0_plus, omega, L_max, M_max
)
V_LM_minus = quadratic_potential_spherical_wave_expansion(
    r, r0_minus, omega, L_max, M_max
)
V_LM_super = V_LM_plus + V_LM_minus

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# Left panel: single quadratic potential
for L in range(L_max + 1):
    I = LM_to_I(L, 0, L_max, M_max)
    component = V_LM[I]
    if np.allclose(component, 0.0):
        continue
    ax1.plot(r, component, label=rf"$V_{{L={L},M=0}}(r)$")

ax1.set_xlabel(r"$r$ (a.u.)")
ax1.set_ylabel(r"$V_{LM}(r)$")
ax1.set_title(
    rf"Single quadratic: $\frac{{1}}{{2}}\omega^2|\mathbf{{r}}-\mathbf{{r}}_0|^2$, "
    rf"$\mathbf{{r}}_0=(0,0,{z0})$, $\omega={omega}$"
)
ax1.legend()
ax1.set_xlim(0, r_max)

# Right panel: symmetric superposition
for L in range(L_max + 1):
    I = LM_to_I(L, 0, L_max, M_max)
    component = V_LM_super[I]
    if np.allclose(component, 0.0):
        continue
    ax2.plot(r, component, label=rf"$V_{{L={L},M=0}}(r)$")

ax2.set_xlabel(r"$r$ (a.u.)")
ax2.set_ylabel(r"$V_{LM}(r)$")
ax2.set_title(
    rf"Superposition: $\frac{{1}}{{2}}\omega^2(|\mathbf{{r}}-\mathbf{{r}}_+|^2"
    rf"+|\mathbf{{r}}-\mathbf{{r}}_-|^2)$, "
    rf"$z_0=\pm{z0}$, $\omega={omega}$"
)
ax2.legend()
ax2.set_xlim(0, r_max)

plt.tight_layout()
plt.show()
