import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    number_of_lm_states,
)
from numba import njit


def compute_mean_field_einsum(A, W, y, L_max, M_max):

    """
    Compute the radial (L,M) components of the mean-field potential W^q_s(\mathbf{r})
    given by

    W^{(q,s)}_{L,M}(r_\alpha) = \sum_{l2, m2} \sum_{l4, m4} \sum_\beta A^*_{q,\beta,l2,m2} A_{s,\beta,l4,m4} W_{L,\alpha,\beta} y(L,M,l2,m2,l4,m4)

    where W_{L,\alpha,\beta} is the radial components of the Coulomb interaction in the/a DVR basis and are determined by solving
    the radial Poisson equation.

    Parameters
    ----------
    A : np.ndarray
        Array of shape (n_active, n_r, n_lm) containing the active orbitals in the radial-DVR/spherical-harmonics basis.
    W : np.ndarray
        Array of shape (n_L, n_r, n_r) containing the radial components of the Coulomb interaction in the/a DVR basis.
    y : np.ndarray
        Array of shape (n_LM, n_lm, n_lm)
    L_max : int
        The maximum value of the L quantum number
    M_max : int
        The maximum value of the M quantum number

    Returns
    -------
    Wqs_LM : np.ndarray
        Array of shape (n_LM, n_active, n_active, n_r) containing the radial (L,M) components of the mean-field potential W^q_s(\mathbf{r}).

    """

    n_active = A.shape[0]
    n_L = L_max + 1
    n_LM = number_of_lm_states(L_max, M_max)
    n_r = W.shape[1]

    Wqs_LM = np.zeros((n_LM, n_active, n_active, n_r), dtype=A.dtype)

    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), n_L):
            I_LM = LM_to_I(L, M, L_max, M_max)
            Wqs_LM[I_LM] = np.einsum(
                "qbI, sbJ, ab, IJ -> qsa",
                A.conj(),
                A,
                W[L],
                y[I_LM],
                optimize=True,
            )

    return Wqs_LM


@njit
def XY_diag(X, Y):
    """
    Compute the diagonal of the matrix Z = X@Y^T given by
        Z_b = \sum_{J=0}^{n_{LM}-1} X_{b,J} * Y_{b,J} for b = 0, ..., n_r-1

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_r, n_lm) containing the first set of vectors.
    Y : np.ndarray
        Array of shape (n_r, n_lm) containing the second set of vectors.
    """
    n_r = X.shape[0]
    Z = np.zeros(n_r, dtype=np.complex128)
    for b in range(n_r):
        Z[b] = np.dot(X[b], Y[b])
    return Z


@njit
def compute_mean_field_numba(A, W, y, L_max, M_max):

    """
    Compute the radial (L,M) components of the mean-field potential W^q_s(\mathbf{r})
    given by

    W^{(q,s)}_{L,M}(r_\alpha) = \sum_{l2, m2} \sum_{l4, m4} \sum_\beta A^*_{q,\beta,l2,m2} A_{s,\beta,l4,m4} W_{L,\alpha,\beta} y(L,M,l2,m2,l4,m4)

    where W_{L,\alpha,\beta} is the radial components of the Coulomb interaction in the/a DVR basis and are determined by solving
    the radial Poisson equation.

    The number of independent pairs are given by
    .. math::
        n_{qs} = \frac{n_{active} (n_{active} + 1)}{2}.

    In order to loop over all independent pairs (q,s) of the mean-field potential W^q_s(\mathbf{r}) semi-efficiently we use numba.

    Parameters
    ----------
    A : np.ndarray
        Array of shape (n_active, n_r, n_lm) containing the active orbitals in the radial-DVR/spherical-harmonics basis.
    W : np.ndarray
        Array of shape (n_L, n_r, n_r) containing the radial components of the Coulomb interaction in the/a DVR basis.
    y : np.ndarray
        Array of shape (n_LM, n_lm, n_lm)
    L_max : int
        The maximum value of the L quantum number
    M_max : int
        The maximum value of the M quantum number

    Returns
    -------
    Wqs_LM : np.ndarray
        Array of shape (n_qs, n_LM, n_r) containing the radial (L,M) components for all independent pairs (q,s)
        of the mean-field potential W^q_s(\mathbf{r}).

    """

    n_active = A.shape[0]
    n_L = L_max + 1
    n_LM = y.shape[0]
    n_r = W.shape[1]

    n_qs = n_active * (n_active + 1) // 2
    Wqs_LM = np.zeros((n_qs, n_LM, n_r), dtype=A.dtype)

    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), n_L):
            I_LM = LM_to_I(L, M, L_max, M_max)
            for q in range(0, n_active):
                Aq = A[q].conj()
                tmp_bJ = np.dot(Aq, y[I_LM])
                for s in range(q, n_active):
                    As = A[s]
                    I_qs = q * (2 * n_active - q + 1) // 2 + (s - q)
                    tmp2_b = XY_diag(tmp_bJ, As)
                    res = np.dot(W[L], tmp2_b)
                    Wqs_LM[I_qs, I_LM] = res
    return Wqs_LM
