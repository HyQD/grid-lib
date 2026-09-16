import numpy as np
from sympy.physics.wigner import gaunt
from numba import njit
from functools import lru_cache
from scipy.integrate import lebedev_rule
from packaging import version
import scipy


def sph_harm_y(l, m, theta, phi):
    """
    Compute Y_{l,m}(theta, phi) with a stable angle convention across SciPy versions.

    Args:
        m: magnetic quantum number
        l: orbital angular momentum quantum number
        phi: azimuthal angle in [0, 2*pi)
        theta: polar angle in [0, pi]
    """
    scipy_version = version.parse(scipy.__version__)

    if scipy_version >= version.parse("1.15.0"):
        return scipy.special.sph_harm_y(l, m, theta, phi)

    return scipy.special.sph_harm(m, l, phi, theta)


try:
    import py3nj as _py3nj
except Exception:
    _py3nj = None

_ACTIVE_GAUNT_BACKEND = "auto"


def angular_matrix_element(l1, m1, l2, m2, f, order=101):
    """
    Compute a spherical-harmonic matrix element using Lebedev quadrature.

    Evaluates the angular integral

        ∫ Y*_{l1,m1}(Ω) f(θ, φ) Y_{l2,m2}(Ω) dΩ,

    over the unit sphere, where Y_{l,m} are complex spherical harmonics
    using the convention of ``scipy.special.sph_harm_y``.

    The integral is approximated using a Lebedev quadrature rule obtained
    from ``scipy.integrate.lebedev_rule``. The Lebedev Cartesian nodes are
    converted to spherical coordinates according to

        x = sin(θ) cos(φ)
        y = sin(θ) sin(φ)
        z = cos(θ),

    where ``theta`` is the polar (colatitude) angle in [0, π] and ``phi``
    is the azimuthal angle in [0, 2π).

    Parameters
    ----------
    l1 : int
        Orbital angular-momentum quantum number of the bra spherical
        harmonic. Must satisfy ``l1 >= 0``.

    m1 : int
        Magnetic quantum number of the bra spherical harmonic. Must satisfy
        ``-l1 <= m1 <= l1``.

    l2 : int
        Orbital angular-momentum quantum number of the ket spherical
        harmonic. Must satisfy ``l2 >= 0``.

    m2 : int
        Magnetic quantum number of the ket spherical harmonic. Must satisfy
        ``-l2 <= m2 <= l2``.

    f : callable
        Function of the form ``f(theta, phi)`` representing the angular
        function or operator inserted between the two spherical harmonics.
        The function should accept NumPy arrays for ``theta`` and ``phi``
        and return either an array broadcastable to the same shape or a
        scalar. The return value may be real or complex.

    order : int, optional
        Lebedev quadrature order passed to
        ``scipy.integrate.lebedev_rule``. Higher orders use more quadrature
        points and integrate spherical polynomials of higher degree exactly.
        The default is 29.

        If ``f`` has a finite spherical-harmonic expansion containing only
        angular momenta up to ``L_f``, then a sufficient condition for exact
        integration, up to floating-point error, is approximately

            order >= l1 + l2 + L_f.

        The requested order must be one of the orders supported by SciPy.

    Returns
    -------
    value : complex
        Numerical approximation to

            ∫ Y*_{l1,m1}(Ω) f(Ω) Y_{l2,m2}(Ω) dΩ.

        The result may have a negligible imaginary component due to
        floating-point roundoff even when the exact integral is real.

    Notes
    -----
    ``scipy.special.sph_harm_y`` uses ``theta`` for the polar angle and
    ``phi`` for the azimuthal angle. This differs from the convention used
    by the older, deprecated ``scipy.special.sph_harm`` interface.

    The weights returned by ``scipy.integrate.lebedev_rule`` are normalized
    for direct integration over the unit sphere, so no additional factor of
    ``4*pi`` is required.

    Examples
    --------
    For ``f(θ, φ) = 1``, the matrix element reduces to the orthonormality
    relation for spherical harmonics:

    >>> f_unity = lambda theta, phi: np.ones_like(theta)
    >>> angular_matrix_element(2, 1, 2, 1, f_unity)
    (1+0j)

    whereas different angular-momentum states give zero, up to numerical
    roundoff:

    >>> angular_matrix_element(2, 1, 3, 1, f_unity)
    0j

    >>> fx = lambda theta, phi: np.sin(theta) * np.cos(phi)
    >>> fy = lambda theta, phi: np.sin(theta) * np.sin(phi)
    >>> fz = lambda theta, phi: np.cos(theta)
    >>> angular_matrix_element(2, 1, 2, 1, fx)
    >>> angular_matrix_element(2, 1, 2, 1, fy)
    >>> angular_matrix_element(2, 1, 2, 1, fz)

    See Also
    --------
    scipy.integrate.lebedev_rule
        Generate Lebedev quadrature nodes and weights.
    scipy.special.sph_harm_y
        Complex spherical harmonics.
    """

    # Lebedev points and weights
    xyz, w = lebedev_rule(order)
    x, y, z = xyz

    # Cartesian -> spherical
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    # Spherical harmonics
    Y1 = sph_harm_y(l1, m1, theta, phi)
    Y2 = sph_harm_y(l2, m2, theta, phi)

    # f should be vectorized: f(theta, phi)
    fv = f(theta, phi)

    return np.sum(w * np.conj(Y1) * fv * Y2)


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
        wmmm = _scalarize(
            wigner3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, 2 * m3)
        )
    except Exception:
        w000 = _scalarize(wigner3j(l1, l2, l3, 0, 0, 0))
        wmmm = _scalarize(wigner3j(l1, l2, l3, m1, m2, m3))

    prefactor = np.sqrt(
        (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * np.pi)
    )
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
        indices = np.array(
            [LM_to_I(l, m, l_max, m_max) for l in l_values], dtype=int
        )
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
                    for l2, I_l2m2 in zip(
                        range(l2_min, l2_max + 1), i2_values[start:stop]
                    ):
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
                    for l2, I_l2m2 in zip(
                        range(l2_min, l2_max + 1), i2_values[start:stop]
                    ):
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
