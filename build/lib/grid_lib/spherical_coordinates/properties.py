import numpy as np
from opt_einsum import contract

from grid_lib.spherical_coordinates.utils import quadrature


def expec_x_i(psi, weights, r, xi_Omega):
    """
    Compute
        \braket{\Psi | x_i | \Psi} = \braket{\Psi | r * x_i(\Omega) | \Psi}
    """
    n_lm = psi.shape[0]

    expec_xi = 0 + 0j
    tmp_xi = contract("IJ, Jk->Ik", xi_Omega, psi)
    for I in range(n_lm):
        expec_xi += quadrature(weights, r * psi[I].conj() * tmp_xi[I])
    return expec_xi


def expec_p_i(psi, dpsi_dr, weights, r, xi_Omega, H_xi_beta):
    n_lm = psi.shape[0]
    tmp_alpha = contract("IJ, Jk->Ik", xi_Omega, dpsi_dr)
    tmp_beta = contract("IJ, Jk->Ik", H_xi_beta, psi)

    expec_pi = 0 + 0j

    for I in range(n_lm):
        expec_pi -= 1j * quadrature(weights, psi[I].conj() * tmp_alpha[I])
        expec_pi -= 1j * quadrature(weights, (psi[I].conj() / r) * tmp_beta[I])

    return expec_pi


def expec_kinetic_energy(psi, T_D2psi, weights, r, lm_I):
    n_lm = psi.shape[0]

    expec_k = 0 + 0j

    for I in range(n_lm):
        l = lm_I[I][0]
        expec_k += quadrature(weights, psi[I].conj() * T_D2psi[I])
        expec_k += (l * (l + 1) / 2) * quadrature(
            weights, (psi[I].conj() / r**2) * psi[I]
        )

    return expec_k


def expec_potential_energy_spherical(psi, weights, potential):
    n_lm = psi.shape[0]

    expec_v = 0 + 0j

    for I in range(n_lm):
        expec_v += quadrature(weights, (psi[I].conj() * potential) * psi[I])

    return expec_v


def expec_potential_energy_rinv(psi, weights, r, Z=1):
    n_lm = psi.shape[0]

    expec_v = 0 + 0j

    for I in range(n_lm):
        expec_v -= Z * quadrature(weights, (psi[I].conj() / r) * psi[I])

    return expec_v


def expec_potential_energy_expansion(psi, weights, V, Z=1):
    n_lm = psi.shape[0]
    Vpsi = -Z * contract("IJk, Jk->Ik", V, psi)

    expec_v = 0 + 0j

    for I in range(n_lm):
        expec_v += quadrature(weights, psi[I].conj() * Vpsi[I])

    return expec_v


def compute_norm(psi, weights):
    n_lm = psi.shape[0]
    norm = 0 + 0j
    for I in range(n_lm):
        norm += quadrature(weights, np.abs(psi[I]) ** 2)
    return norm


def compute_overlap(psi1, psi2, weights):
    n_lm = psi1.shape[0]
    norm = 0 + 0j
    for I in range(n_lm):
        norm += quadrature(weights, psi1[I].conj() * psi2[I])
    return norm


def compute_ionization_probability(psi, weights, r, radius=20):
    """
    Here the (single) ionization probaility is taken as the probability of finding an electron outside a sphere
    of a given radius.

    P_{ionization}(t) = 1-\int_0^{radius} |\Psi(\mathfbf{r},t)|^2 r^2 dr d\Omega
    """
    n_lm = psi.shape[0]
    r_rad = r[r <= radius]
    ionization_prob = 0
    for I in range(n_lm):
        psi_rad = psi[I][r <= radius]
        ionization_prob += quadrature(weights, np.abs(psi_rad) ** 2)

    return 1 - ionization_prob
