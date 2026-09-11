import numpy as np
from sympy.physics.wigner import gaunt
from numba import njit
from functools import lru_cache

try:
    import py3nj as _py3nj
except Exception:
    _py3nj = None

_ACTIVE_GAUNT_BACKEND = "auto"


def set_gaunt_backend(backend):
    """
    Select backend for Gaunt evaluations used by get_y and get_ybar.

    Parameters
    ----------
    backend : str
        One of "auto", "sympy", or "py3nj".

    Returns
    -------
    str
        The selected backend mode.
    """

    global _ACTIVE_GAUNT_BACKEND
    mode = str(backend).strip().lower()
    if mode not in ("auto", "sympy", "py3nj"):
        raise ValueError("backend must be one of: auto, sympy, py3nj")
    if mode == "py3nj" and _py3nj is None:
        raise RuntimeError("Requested backend 'py3nj' is not available")

    _ACTIVE_GAUNT_BACKEND = mode
    _gaunt_fast_cached.cache_clear()
    return _ACTIVE_GAUNT_BACKEND


def get_gaunt_backend(resolve=False):
    """
    Get current backend mode used for Gaunt evaluations.

    Parameters
    ----------
    resolve : bool, optional
        If True and mode is "auto", report the effective backend
        ("py3nj" when available, otherwise "sympy").

    Returns
    -------
    str
        Backend mode or effective backend.
    """

    if resolve and _ACTIVE_GAUNT_BACKEND == "auto":
        return "py3nj" if _py3nj is not None else "sympy"
    return _ACTIVE_GAUNT_BACKEND


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


def _scalarize(value):
    return float(np.asarray(value).reshape(-1)[0])


@lru_cache(maxsize=None)
def _gaunt_sympy_cached(l1, l2, l3, m1, m2, m3):
    return float(gaunt(l1, l2, l3, m1, m2, m3).n(64))


def _gaunt_py3nj(l1, l2, l3, m1, m2, m3):
    if _py3nj is None:
        raise RuntimeError("py3nj is not available")

    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    if l3 < abs(l1 - l2) or l3 > l1 + l2:
        return 0.0

    wigner3j = getattr(_py3nj, "wigner3j", None)
    if wigner3j is None:
        wigner3j = getattr(_py3nj, "wigner_3j", None)
    if wigner3j is None:
        raise AttributeError("py3nj does not expose wigner3j or wigner_3j")

    try:
        w000 = _scalarize(wigner3j(2 * l1, 2 * l2, 2 * l3, 0, 0, 0))
        wmmm = _scalarize(wigner3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, 2 * m3))
    except Exception:
        w000 = _scalarize(wigner3j(l1, l2, l3, 0, 0, 0))
        wmmm = _scalarize(wigner3j(l1, l2, l3, m1, m2, m3))

    prefactor = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * np.pi))
    return prefactor * w000 * wmmm


@lru_cache(maxsize=None)
def _gaunt_fast_cached(l1, l2, l3, m1, m2, m3):
    backend = get_gaunt_backend(resolve=True)

    if backend == "py3nj":
        return _gaunt_py3nj(l1, l2, l3, m1, m2, m3)

    if backend == "sympy":
        return _gaunt_sympy_cached(l1, l2, l3, m1, m2, m3)

    if _py3nj is not None:
        try:
            return _gaunt_py3nj(l1, l2, l3, m1, m2, m3)
        except Exception:
            pass
    return _gaunt_sympy_cached(l1, l2, l3, m1, m2, m3)


def _build_lm_state_tables(l_max, m_max):
    state_table = {}
    for m in range(-m_max, m_max + 1):
        l_values = np.arange(abs(m), l_max + 1, dtype=int)
        indices = np.array([LM_to_I(l, m, l_max, m_max) for l in l_values], dtype=int)
        state_table[m] = (l_values, indices)
    return state_table


def get_y(l_max, m_max, L_max, M_max):
    """
    Compute y tensor with the same signature and output shape as setup_y_and_ybar_sympy.

    Uses exact magnetic-quantum-number selection rules to skip zero Gaunt evaluations:
    m2 = m1 - M.
    """

    n_L = L_max + 1

    n_LM = number_of_lm_states(L_max, M_max)
    n_lm = number_of_lm_states(l_max, m_max)
    y = np.zeros((n_LM, n_lm, n_lm))

    state_table = _build_lm_state_tables(l_max, m_max)

    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), n_L):
            I_LM = LM_to_I(L, M, L_max, M_max)
            for m1 in range(-m_max, m_max + 1):
                m2 = m1 - M
                if m2 < -m_max or m2 > m_max:
                    continue

                l1_values, i1_values = state_table[m1]
                _, i2_values = state_table[m2]

                for l1, I_l1m1 in zip(l1_values, i1_values):
                    l2_min = max(abs(m2), abs(l1 - L))
                    l2_max = min(l_max, l1 + L)
                    if l2_min > l2_max:
                        continue

                    sign = -1.0 if (m1 % 2) else 1.0
                    start = l2_min - abs(m2)
                    stop = l2_max - abs(m2) + 1
                    for l2, I_l2m2 in zip(range(l2_min, l2_max + 1), i2_values[start:stop]):
                        y[I_LM, I_l1m1, I_l2m2] = sign * _gaunt_fast_cached(
                            int(l1), L, int(l2), -m1, M, m2
                        )

    return y


def get_ybar(l_max, m_max, L_max, M_max):
    """
    Compute y_bar tensor with the same signature and output shape as setup_y_and_ybar_sympy.

    Uses exact magnetic-quantum-number selection rules to skip zero Gaunt evaluations:
    m2 = m1 + M.
    """

    n_L = L_max + 1

    n_LM = number_of_lm_states(L_max, M_max)
    n_lm = number_of_lm_states(l_max, m_max)
    y_bar = np.zeros((n_LM, n_lm, n_lm))

    state_table = _build_lm_state_tables(l_max, m_max)

    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), n_L):
            I_LM = LM_to_I(L, M, L_max, M_max)
            for m1 in range(-m_max, m_max + 1):
                m2 = m1 + M
                if m2 < -m_max or m2 > m_max:
                    continue

                l1_values, i1_values = state_table[m1]
                _, i2_values = state_table[m2]

                for l1, I_l1m1 in zip(l1_values, i1_values):
                    l2_min = max(abs(m2), abs(l1 - L))
                    l2_max = min(l_max, l1 + L)
                    if l2_min > l2_max:
                        continue

                    sign = -1.0 if ((m1 + M) % 2) else 1.0
                    start = l2_min - abs(m2)
                    stop = l2_max - abs(m2) + 1
                    for l2, I_l2m2 in zip(range(l2_min, l2_max + 1), i2_values[start:stop]):
                        y_bar[I_LM, I_l1m1, I_l2m2] = sign * _gaunt_fast_cached(
                            int(l1), L, int(l2), -m1, -M, m2
                        )

    return y_bar


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
