import numpy as np

from scipy.special import sph_harm, lpmv as Pml, factorial

"""
In Scipy 1.10.1 
sph_harm: sph_harm(m, n, theta, phi, out=None)
    - m: azimuthal quantum number
    - n: Degree of the harmonic, commonly denoted as l in quantum mechanics
    - theta in [0, 2pi): azimuthal angle 
    - phi in [0,pi]: polar angle

In SciPy 1.15.2 sph_harm is deprecated (and will be removed in SciPy 1.17.0) called sph_harm_y where 
sph_harm_y: sph_harm_y(n, m, theta, phi, *, diff_n=0)
    - m: azimuthal quantum number
    - n: Degree of the harmonic, commonly denoted as l in quantum mechanics
    - theta in [0,pi]: polar angle
    - phi in [0, 2pi): azimuthal angle
"""
from sympy.physics.wigner import gaunt
from numba import njit


def number_of_lm_states(l_max, m_max):
    """
    Number of lm states for a given l_max and m_max.

    Parameters
    ----------
    l_max : int
        Maximum value of the l quantum number
    m_max : int
        Maximum value of the m quantum number
    Returns
    -------
    n_lm : int
        The number of lm states
    """

    n_l = l_max + 1
    n_m = 2 * m_max + 1
    n_lm = n_l * n_m - m_max * (m_max + 1)
    return n_lm


@njit
def LM_to_I(L, M, L_max, M_max):
    """
    Map the quantum numbers (L,M) to an index I=0,...,n_lm-1.

    Parameters
    ----------
    L : int
        The L quantum number
    M : int
        The M quantum number
    L_max : int
        The maximum value of the L quantum number
    M_max : int
        The maximum value of the M quantum number

    Returns
    -------
    I_LM : int
        The index corresponding to the quantum numbers (L,M)
    """

    M_tilde = M + M_max
    n_L = L_max + 1
    if M <= 0:
        I_LM = M_tilde * (M_tilde - 1) // 2 + M_tilde * (n_L - M_max) + L + M
    else:
        I_LM = (
            (n_L + M_max * L_max - M_max * (M_max - 1) // 2)
            + (M - 1) * (n_L - M + 1)
            + (M - 1) * (M - 2) // 2
            + L
            - M
        )
    return I_LM


def setup_y_and_ybar_sympy(l_max, m_max, L_max, M_max):
    """
    Compute

             y (L,M,l1,m1,l2,m2) = \int Y^*_{l1,m1}(\Omega) Y_{L,M}(\Omega) Y_{l2,m2}(\Omega) d\Omega
                                 = (-1)^(m1) * gaunt(l1, L, l2, -m1, M, m2)
        \bar{y}(L,M,l1,m1,l2,m2) = \int Y^*_{l1,m1}(\Omega) Y^*_{L,M}(\Omega) Y_{l2,m2}(\Omega) d\Omega
                                 = (-1)^(m1+M) * gaunt(l1, L, l2, -m1, -M, m2)
    where Y_{l,m}(\Omega) are the spherical harmonics.

    Parameters
    ----------
    l_max : int
        Maximum value of the l quantum number in the expansion of the orbitals
    m_max : int
        Maximum value of the m quantum number in the expansion of the orbitals
    L_max : int
        Maximum value of the L quantum number in the multipole expansion of the Coulomb interaction
    M_max : int
        Maximum value of the M quantum number in the multipole expansion of the Coulomb interaction
    Returns
    -------
    y : np.ndarray
        The y tensor of shape (n_LM, n_lm, n_lm)
    y_bar : np.ndarray
        The y_bar tensor of shape (n_LM, n_lm, n_lm)
    """

    n_L = L_max + 1
    n_l = l_max + 1

    n_LM = number_of_lm_states(L_max, M_max)
    n_lm = number_of_lm_states(l_max, m_max)
    y = np.zeros((n_LM, n_lm, n_lm))
    y_bar = np.zeros((n_LM, n_lm, n_lm))

    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), n_L):
            I_LM = LM_to_I(L, M, L_max, M_max)
            for m1 in range(-m_max, m_max + 1):
                for l1 in range(abs(m1), n_l):
                    I_l1m1 = LM_to_I(l1, m1, l_max, m_max)
                    for m2 in range(-m_max, m_max + 1):
                        for l2 in range(abs(m2), n_l):
                            I_l2m2 = LM_to_I(l2, m2, l_max, m_max)
                            y[I_LM, I_l1m1, I_l2m2] = float(
                                gaunt(l1, L, l2, -m1, M, m2).n(64)
                            ) * (-1) ** (m1)
                            y_bar[I_LM, I_l1m1, I_l2m2] = float(
                                gaunt(l1, L, l2, -m1, -M, m2).n(64)
                            ) * (-1) ** (m1 + M)

    return y, y_bar


def setup_y_and_ybar_compact(l_max, m_max, L_max, M_max):

    n_L = L_max + 1
    n_l = l_max + 1

    n_lm = number_of_lm_states(l_max, m_max)

    y = np.zeros((n_L, n_lm, n_lm))
    y_bar = np.zeros((n_L, n_lm, n_lm))

    for L in range(n_L):
        for m1 in range(-m_max, m_max + 1):
            for l1 in range(abs(m1), n_l):
                I_l1m1 = LM_to_I(l1, m1, l_max, m_max)
                for m2 in range(-m_max, m_max + 1):
                    for l2 in range(abs(m2), n_l):
                        I_l2m2 = LM_to_I(l2, m2, l_max, m_max)
                        M12 = m1 - m2
                        M21 = m2 - m1

                        if abs(M12) <= L_max:
                            y[L, I_l1m1, I_l2m2] = float(
                                gaunt(l1, L, l2, -m1, M12, m2).n(64)
                            ) * (-1) ** (m1)
                        if abs(M21) <= L_max:
                            y_bar[L, I_l1m1, I_l2m2] = float(
                                gaunt(l1, L, l2, -m1, -M21, m2).n(64)
                            ) * (-1) ** (m1 + M21)
    return y, y_bar
