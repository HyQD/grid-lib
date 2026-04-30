import numpy as np


def get_hydrogenic_orbitals(GLL, l_max, n_s, Z):

    """
    Compute the n_s lowest lying normalized hydrogenic orbitals for l=0,...,l_max.

    The orbitals are normalized such that
    .. math::
        \int_0^{r_{max}} |u_{n,l}(r)|^2 dr = 1 (1)
    where :math:`u_{n,l}(r)` is the radial part of the n-th eigenfunction
    .. math::
        \psi_{n,l,m}(\mathbf{r}) = r^{-1}u_{n,l}(r) Y_{lm}(\theta, \phi),
    of the hydrogenic Hamiltonian
    .. math::
        H = -\frac{1}{2} \nabla^2 - \frac{Z}{r}.

    The integral given by Eq.(1) is computed using the Gauss-Legendre-Lobatto quadrature rule.

    Parameters
    ----------
    GLL : GaussLegendreLobatto
        Gauss-Legendre-Lobatto grid object.
    l_max : int
        Maximum angular momentum quantum number.
    n_s : list
        List of number of eigenstates for each l.
    Z : float
        Nuclear charge.
    Returns
    -------
    orbitals : dict
        Dictionary of orbitals for each l.
    orbital_energy : dict
        Dictionary of orbital energies for each l.
    """

    r = GLL.r[1:-1]
    weights = GLL.weights[1:-1]
    D1 = GLL.D1
    D2 = np.dot(D1, D1)[1:-1, 1:-1]

    orbitals = dict()
    orbital_energy = dict()

    for l in range(0, l_max + 1):

        Tl = -0.5 * D2 + np.diag(l * (l + 1) / (2 * r**2))
        V = np.diag(-Z / r)
        Hl = Tl + V
        eps, U = np.linalg.eig(Hl)

        idx = np.argsort(eps)
        eps = eps[idx]
        U = U[:, idx]

        n_states = n_s[l]
        B = U[:, 0:n_states]
        for i in range(n_states):
            norm_psi = np.dot(weights, B[:, i] * B[:, i])
            B[:, i] /= np.sqrt(norm_psi)
        orbitals[f"{l}"] = B.copy()
        orbital_energy[f"{l}"] = eps[0:n_states].copy()

    return orbitals, orbital_energy


def compute_ground_state(
    angular_matrix_elements,
    radial_matrix_elements,
    potential,
    l=0,
    hermitian=True,
):
    nr = radial_matrix_elements.nr
    r = radial_matrix_elements.r
    T_D2 = -(1 / 2) * radial_matrix_elements.D2

    H0 = np.zeros((nr, nr))
    T = T_D2 + np.diag(l * (l + 1) / (2 * r**2))
    V = np.diag(potential)

    H0 = T + V

    if hermitian:
        assert np.allclose(H0, H0.T)
        eps, phi_n = np.linalg.eigh(H0)
    else:
        eps, phi_n = np.linalg.eig(H0)

    return eps, phi_n


def compute_ground_state_diatomic(
    angular_matrix_elements,
    radial_matrix_elements,
    potential,
    l_max,
    hermitian=True,
):
    nl = l_max + 1
    nr = radial_matrix_elements.nr
    r = radial_matrix_elements.r
    T_D2 = -(1 / 2) * radial_matrix_elements.D2

    clmb_ = angular_matrix_elements("1/(r-a)")
    clmb = np.zeros((nl, nl, nr, nr), dtype=np.complex128)
    clmb[:, :, np.arange(nr), np.arange(nr)] = clmb_
    clmb = clmb.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3).T
    clmb = clmb.reshape(nr * nl, nr * nl)

    H0 = np.zeros((nr * nl, nr * nl))
    for l in range(nl):
        T = T_D2 + np.diag(l * (l + 1) / (2 * r**2))
        V = np.diag(potential)

        H0[l * nr : ((l + 1) * nr), l * nr : ((l + 1) * nr)] = T + V

    H0 = H0 - clmb

    if hermitian:
        assert np.allclose(H0, H0.T)
        eps, phi_n = np.linalg.eigh(H0)
    else:
        eps, phi_n = np.linalg.eig(H0)

    return eps, phi_n
