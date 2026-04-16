import numpy as np
from matplotlib import pyplot as plt

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_momentum import LM_to_I
from grid_lib.spherical_coordinates.potentials import (
    clamped_molecular_potential_Poisson,
    clamped_molecular_potential_quadrature,
)
from grid_lib.spherical_coordinates.radial_Coulomb import radial_Coulomb


l_max = 1
m_max = 0
L_max = 2 * l_max
M_max = 2 * m_max

Z1 = 3
Z2 = 1
a1 = np.array([0.0, 0.0, -1.525])
a2 = np.array([0.0, 0.0, 1.525])

# Make r = 1.525 an explicit FEMDVR node.
nodes = np.array([0.0, 1.525, 3.05, 6.10, 9.15, 12.20])
n_points = np.array([25, 25, 25, 25, 25])

femdvr = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)
r = femdvr.r[1:-1]
weights = femdvr.weights[1:-1]
W = radial_Coulomb(femdvr, L_max + 1)

r_idx = np.argmin(np.abs(r - 1.525))
assert np.isclose(r[r_idx], 1.525)
a = r[r_idx]

positions = np.array([a1, a2])
charges = np.array([Z1, Z2])

V_quad = clamped_molecular_potential_quadrature(
    r,
    positions=positions,
    charges=charges,
    L_max=L_max,
    M_max=M_max,
)
V_poisson = clamped_molecular_potential_Poisson(
    r,
    W,
    weights,
    positions=positions,
    charges=charges,
    L_max=L_max,
    M_max=M_max,
)

print(f"Grid point used for |a1| and |a2|: r[{r_idx}] = {r[r_idx]:.6f}")

fig, axes = plt.subplots(L_max + 1, 1, sharex=True, figsize=(8, 8))

for L in range(L_max + 1):
    ax = axes[L]
    I_L0 = LM_to_I(L, 0, L_max, M_max)

    ax.plot(r, V_quad[I_L0].real, label='quadrature')
    ax.plot(r, V_poisson[I_L0].real, '--', label='Poisson')
    ax.axvline(a, color='k', linestyle=':', linewidth=1.0)
    ax.set_ylabel(rf'$V_{{{L}0}}(r)$')
    ax.grid(True)

axes[0].legend()
axes[-1].set_xlabel(r'$r$')
fig.suptitle(
    'Clamped diatomic potential on the z-axis\n'
    r'$Z_1 = 3$, $Z_2 = 1$, $a_1 = (0,0,-1.525)$, '
    r'$a_2 = (0,0,1.525)$'
)
fig.tight_layout()
plt.show()
