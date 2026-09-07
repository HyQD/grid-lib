import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import (
    number_of_lm_states,
    LM_to_I,
    setup_y_and_ybar_sympy,
)

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from opt_einsum import contract
from grid_lib.spherical_coordinates.radial_Coulomb import radial_Coulomb

from matplotlib import pyplot as plt

l_max = 2
m_max = 0
n_l = l_max + 1
n_lm = number_of_lm_states(l_max, 0)

L_max = 2 * l_max
M_max = 0

a = 1.0
R_nuc = 2 * a
origin = "ncm"

r_start = 0.0
r_max = 10.0
element_size = 1.0

nodes = np.arange(r_start, r_max + element_size, element_size)
print(f"Nodes: {nodes}")

n_points_pr_element = 6
n_points = (
    np.ones((len(nodes) - 1,), dtype=int) * n_points_pr_element
)  # Example number of points per element

GLL_r = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

D1 = GLL_r.D1[1:-1, 1:-1]  # Exclude boundary points
D2 = GLL_r.D2[1:-1, 1:-1]  # Exclude boundary points

r = GLL_r.r[1:-1]
n_r = len(r)

nabla_sq = np.zeros((n_l, n_r, n_r))
He = np.zeros((n_l, n_r, n_r))

for l in range(l_max + 1):
    nabla_sq[l] = D2 - np.diag(l * (l + 1) / r**2)
    He[l] = -0.5 * nabla_sq[l]  # - np.diag(1 / r)


eps0, psi0 = np.linalg.eig(He[0])
idx = np.argsort(eps0)
eps0 = eps0[idx]
psi0 = psi0[:, idx]
print(f"eps0: {eps0[0]:.6f}")


weights = GLL_r.weights[1:-1]
weights_inv = 1 / weights
W = radial_Coulomb(GLL_r, L_max + 1)

y, y_bar = setup_y_and_ybar_sympy(l_max, 0, L_max, M_max)


r_idx = np.argmin(np.abs(r - a))
print(f"Index where r = a: {r_idx}, r[idx] = {r[r_idx]:.8f}, a={a:.8f}")

Wa = W[:, :, r_idx].copy()
Wa_m = W[:, :, r_idx].copy()
for L in range(L_max + 1):
    Wa[L] *= np.sqrt((2 * L + 1) / (4 * np.pi))
    Wa_m[L] *= np.sqrt((2 * L + 1) / (4 * np.pi)) * (-1) ** L

Wa /= weights[r_idx]
Wa_m /= weights[r_idx]


def compute_r_inv_l(r, a, l):
    r_min = np.minimum(r, abs(a))
    r_max = np.maximum(r, abs(a))
    if a > 0:
        return np.sqrt(4 * np.pi / (2 * l + 1)) * r_min**l / r_max ** (l + 1)
    else:
        return (
            np.sqrt(4 * np.pi / (2 * l + 1))
            * r_min**l
            / r_max ** (l + 1)
            * (-1) ** l
        )


Wa_quad = np.zeros((L_max + 1, n_r))
Wa_m_quad = np.zeros((L_max + 1, n_r))
for L in range(L_max + 1):
    Wa_quad[L] = compute_r_inv_l(r, a, L)
    Wa_m_quad[L] = compute_r_inv_l(r, a, L) * (-1) ** L

plt.figure()
for L in range(l_max+1):
    plt.subplot(211)
    plt.plot(r, -Wa[L], label=r"$W_{%d}(r;a=%.2f)$" % (L, a))
    plt.plot(r, -Wa_quad[L], '--', label=r"$W_{%d}^{\mathrm{quad}}(r;a=%.2f)$" % (L, a))
    plt.subplot(212)
    plt.plot(r, -Wa_m[L], label=r"$W_{%d}^{\mathrm{m}}(r;a=%.2f)$" % (L, a))
    plt.plot(r, -Wa_m_quad[L], '--', label=r"$W_{%d}^{\mathrm{m,quad}}(r;a=%.2f)$" % (L, a))
plt.subplot(211)
plt.grid()
plt.legend()
plt.subplot(212)
plt.grid()
plt.legend()
plt.show()