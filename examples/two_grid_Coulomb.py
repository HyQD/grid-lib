import numpy as np
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)


def W_multipole(r1, r2, alpha, L_max, q1=-1.0, q2=1.0):
    r"""
    Compute the multipole expansion of the Coulomb interaction between two particles with charge q1 and q2.

        q1*q2/|r1 - alpha * r2| = sum_{L=0}^{L_max} sum_{M=-L}^{L}
                                  4 * pi / (2 * L + 1) * sgn(alpha)^L * min(r1, |alpha| * r2)^L / (max(r1, |alpha| * r2)^(L+1))
                                    * Y^*_{LM}(\Omega_1) * Y_{LM}(\Omega_2)

    Parameters
    ----------
    r1 : ndarray
        Radial grid for the first particle.
    r2 : ndarray
        Radial grid for the second particle.
    alpha : float
        Coefficient in the Coulomb interaction 1/|r - alpha * R|.
    L_max : int
        Maximum angular momentum in the multipole expansion.

    Returns
    -------
    W1_rR : ndarray
        Multipole expansion of the Coulomb interaction.
    """
    L_vec = np.arange(L_max + 1)[:, None, None]
    r1_grid = r1[:, None]
    r2_grid = r2[None, :]

    r_min = np.minimum(r1_grid, np.abs(alpha) * r2_grid)[None, :, :]
    r_max = np.maximum(r1_grid, np.abs(alpha) * r2_grid)[None, :, :]

    W1_rR = (
        q1
        * q2
        * (4 * np.pi / (2 * L_vec + 1))
        * (r_min**L_vec)
        / (r_max ** (L_vec + 1))
    )
    return np.sign(alpha) ** L_vec * W1_rR


r_max = 12.0
n_elements_r = 4
element_size_r = r_max / n_elements_r

nodes_r = np.arange(0.0, r_max + element_size_r, element_size_r)

n_points_pr_element_r = 15
n_points_r = np.ones((n_elements_r,), dtype=int) * n_points_pr_element_r

rDVR = FEMDVR(nodes_r, n_points_r, Linear_map, GaussLegendreLobatto)

r = rDVR.r[1:-1]
D1_r = rDVR.D1[1:-1, 1:-1]
D2_r = rDVR.D2[1:-1, 1:-1]
w_r = rDVR.weights[1:-1]

R_max = 5.0
n_elements_R = 5
element_size_R = R_max / n_elements_R

nodes_R = np.arange(0.0, R_max + element_size_R, element_size_R)

n_points_pr_element_R = 15
n_points_R = np.ones((n_elements_R,), dtype=int) * n_points_pr_element_R

RDVR = FEMDVR(nodes_R, n_points_R, Linear_map, GaussLegendreLobatto)

R = RDVR.r[1:-1]
D1_R = RDVR.D1[1:-1, 1:-1]
D2_R = RDVR.D2[1:-1, 1:-1]

w_R = RDVR.weights[1:-1]

n_rpoints = len(r)
n_Rpoints = len(R)

from matplotlib import pyplot as plt
from grid_lib.spherical_coordinates.radial_Coulomb import (
    radial_Coulomb_femdvr_two_grid,
)

"""
Set up the radial functions for each L of the Coulomb interaction 
    W(bf(r), bf(R)) -> sum_{(a,q1,q2)} q1q2 / |bf(r) - a * bf(R)|
"""

gamma_q1q2 = [(2 / 3, 1.0, -1.0), (-1 / 3, 1.0, -1.0)]
L_max = 3

W_poisson = radial_Coulomb_femdvr_two_grid(rDVR, RDVR, L_max + 1, gamma_q1q2)

W_direct = sum(W_multipole(r, R, a, L_max, q1, q2) for a, q1, q2 in gamma_q1q2)
W_direct *= w_R[None, None, :]

plt.figure()
for L in range(L_max + 1):
    plt.plot(
        r,
        W_direct[L, :, n_Rpoints // 2],
        label=rf"$W_{{L={L}}}(r, R={R[n_Rpoints // 2]:.2f})$",
    )
    plt.plot(
        r,
        W_poisson[L, :, n_Rpoints // 2],
        label=rf"$W^{{\mathrm{{poisson}}}}_{{L={L}}}(r, R={R[n_Rpoints // 2]:.2f})$",
    )
plt.xlabel("r")
plt.ylabel("W_L(r, R=const)")
plt.legend()
plt.show()
