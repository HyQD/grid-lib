import warnings

import numpy as np
from scipy.special import erf, spherical_in

from .angular_momentum import LM_to_I, number_of_lm_states
from .utils import Ylm, cartesian_to_spherical


class Coulomb:
    def __init__(self, Z):
        self.Z = Z

    def __call__(self, r):
        return -self.Z / r


class Gaussian:
    def __init__(self, V0, alpha):
        self.V0 = V0
        self.alpha = alpha

    def __call__(self, r):
        return -self.V0 * np.exp(-self.alpha * r**2)


class Gaussian_charge_distribution:
    def __init__(self, mu):
        self.mu = mu

    def __call__(self, r):
        return -erf(self.mu * r) / r


class SAE:
    def __init__(self, Z, A, B):
        """
        Ref: 10.1103/PhysRevA.74.053412

        ------------------
        Atom Z  A    B
        ------------------
        He   2  0.00 2.083
        Ar   18 5.40 3.682
        Xe   54 44.0 3.852
        ------------------
        ------------------
        """

        self.Z = Z
        self.A = A
        self.B = B

    def __call__(self, r):
        return (
            -1
            / r
            * (
                1
                + self.A * np.exp(-r)
                + (self.Z - 1 - self.A) * np.exp(-self.B * r)
            )
        )


class SAE2:
    """
    Ref: 10.1088/2399-6528/ab9a68

    Params: See Table 1 in Ref.
    """

    def __init__(self, C0, Zc, c, a, b):
        self.C0 = C0
        self.Zc = Zc
        self.c = c
        self.a = a
        self.b = b
        self.n_a = len(a)

    def __call__(self, r):
        T1 = -self.C0 / r
        T2 = -self.Zc * np.exp(-self.c * r) / r
        T3 = np.zeros_like(r)

        for i in range(self.n_a):
            T3 -= self.a[i] * np.exp(-self.b[i] * r)

        return T1 + T2 + T3


class Erfgau:
    def __init__(self, Z, mu):
        self.Z = Z
        self.mu = mu

    def __call__(self, r):
        c = 0.923 + 1.568 * self.mu
        alpha = 0.2411 + 1.405 * self.mu
        long_range = erf(self.mu * self.Z * r) / (self.Z * r)
        return -self.Z**2 * (
            c * np.exp(-(alpha**2) * self.Z**2 * r**2) + long_range
        )


def clamped_molecular_potential_quadrature(
    r, positions, charges, L_max, M_max=None
):
    r"""
    Compute the spherical-harmonic radial coefficients of a clamped
    molecular point-charge potential from the analytic multipole kernel.

    The potential is expanded as

    .. math::
        V(\mathbf{r}) = \sum_{L,M} V_{LM}(r) Y_{LM}(\Omega),

    where for point charges located at ``R_A``,

    .. math::
        V_{LM}(r) = -\sum_A Z_A \frac{4\pi}{2L+1}
        \frac{r_<^L}{r_>^{L+1}} Y_{LM}^*(\Omega_A).

    Parameters
    ----------
    r : array_like
        Radial grid values where the coefficients are evaluated.
    positions : array_like
        Cartesian nuclear positions with shape ``(n_nuclei, 3)``.
    charges : array_like
        Nuclear charges with shape ``(n_nuclei,)``.
    L_max : int
        Maximum multipole rank in the expansion.
    M_max : int, optional
        Maximum magnetic quantum number stored. Defaults to ``L_max``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_LM, len(r))`` storing ``V_{LM}(r)`` in the
        repo's ``LM_to_I`` ordering.
    """
    r = np.asarray(r, dtype=float)
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)

    if L_max < 0:
        raise ValueError("L_max must be >= 0")

    if M_max is None:
        M_max = L_max
    if M_max < 0 or M_max > L_max:
        raise ValueError("Require 0 <= M_max <= L_max")

    if r.ndim != 1:
        raise ValueError("r must be a one-dimensional array")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_nuclei, 3)")
    if charges.ndim != 1:
        raise ValueError("charges must be a one-dimensional array")
    if len(positions) != len(charges):
        raise ValueError("positions and charges must have the same length")

    n_LM = number_of_lm_states(L_max, M_max)
    V_LM = np.zeros((n_LM, len(r)), dtype=complex)

    for position, charge in zip(positions, charges):
        R = np.linalg.norm(position)

        if R == 0:
            I_00 = LM_to_I(0, 0, L_max, M_max)
            V_LM[I_00] += -charge * np.sqrt(4 * np.pi) / r
            continue

        _, theta_R, phi_R = cartesian_to_spherical(position)
        r_min = np.minimum(r, R)
        r_max = np.maximum(r, R)

        for M in range(-M_max, M_max + 1):
            for L in range(abs(M), L_max + 1):
                I_LM = LM_to_I(L, M, L_max, M_max)
                angular_factor = Ylm(L, M, theta_R, phi_R)
                radial_factor = (r_min**L) / (r_max ** (L + 1))
                V_LM[I_LM] += (
                    -charge
                    * (4 * np.pi / (2 * L + 1))
                    * angular_factor
                    * radial_factor
                )

    if np.allclose(V_LM.imag, 0.0):
        return V_LM.real

    return V_LM


def clamped_molecular_potential_Poisson(
    r, W, weights, positions, charges, L_max, M_max=None, force=False
):
    r"""
    Compute the spherical-harmonic radial coefficients of a clamped
    molecular point-charge potential from a Poisson-solved radial kernel.

    The input ``W`` is assumed to contain the radial Coulomb components
    already including the factor ``4\pi / (2L + 1)``. For each nucleus,
    the nearest radial grid point is used in the source coordinate and the
    corresponding column is divided by the quadrature weight at that point.

    Parameters
    ----------
    r : array_like
        Radial grid values corresponding to the source coordinate of ``W``.
    W : array_like
        Array of shape ``(n_L, len(r), len(r))`` containing the radial
        Coulomb kernel.
    weights : array_like
        Quadrature weights corresponding to ``r``.
    positions : array_like
        Cartesian nuclear positions with shape ``(n_nuclei, 3)``.
    charges : array_like
        Nuclear charges with shape ``(n_nuclei,)``.
    L_max : int
        Maximum multipole rank in the expansion.
    M_max : int, optional
        Maximum magnetic quantum number stored. Defaults to ``L_max``.
    force : bool, optional
        If ``True``, issue a warning and use the nearest grid point when a
        nuclear radius does not coincide with a grid point. If ``False``
        (default), raise a ``ValueError`` instead.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_LM, len(r))`` storing ``V_{LM}(r)`` in the
        repo's ``LM_to_I`` ordering.
    """
    r = np.asarray(r, dtype=float)
    W = np.asarray(W)
    weights = np.asarray(weights, dtype=float)
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)

    if L_max < 0:
        raise ValueError("L_max must be >= 0")

    if M_max is None:
        M_max = L_max
    if M_max < 0 or M_max > L_max:
        raise ValueError("Require 0 <= M_max <= L_max")

    if r.ndim != 1:
        raise ValueError("r must be a one-dimensional array")
    if W.ndim != 3:
        raise ValueError("W must have shape (n_L, len(r), len(r))")
    if W.shape[1:] != (len(r), len(r)):
        raise ValueError("W must have shape (n_L, len(r), len(r))")
    if weights.ndim != 1 or len(weights) != len(r):
        raise ValueError("weights must be a one-dimensional array of len(r)")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_nuclei, 3)")
    if charges.ndim != 1:
        raise ValueError("charges must be a one-dimensional array")
    if len(positions) != len(charges):
        raise ValueError("positions and charges must have the same length")
    if W.shape[0] < L_max + 1:
        raise ValueError("W must contain at least L_max + 1 angular ranks")

    n_LM = number_of_lm_states(L_max, M_max)
    V_LM = np.zeros((n_LM, len(r)), dtype=complex)

    for position, charge in zip(positions, charges):
        R = np.linalg.norm(position)

        if R == 0:
            I_00 = LM_to_I(0, 0, L_max, M_max)
            V_LM[I_00] += -charge * np.sqrt(4 * np.pi) / r
            continue

        _, theta_R, phi_R = cartesian_to_spherical(position)
        r_idx = np.argmin(np.abs(r - R))

        if not np.isclose(r[r_idx], R):
            msg = (
                "Nuclear radius %.16g is not a grid point; "
                "nearest grid point is %.16g."
            ) % (R, r[r_idx])
            if not force:
                raise ValueError(
                    msg + " Pass force=True to use the nearest grid point."
                )
            warnings.warn(msg + " Using nearest grid point.", stacklevel=2)

        for M in range(-M_max, M_max + 1):
            for L in range(abs(M), L_max + 1):
                I_LM = LM_to_I(L, M, L_max, M_max)
                angular_factor = Ylm(L, M, theta_R, phi_R)
                radial_factor = W[L, :, r_idx] / weights[r_idx]
                V_LM[I_LM] += -charge * angular_factor * radial_factor

    if np.allclose(V_LM.imag, 0.0):
        return V_LM.real

    return V_LM


def gaussian_spherical_wave_expansion(r, r0, A, alpha, L_max, M_max=None):
    r"""
    Compute the spherical-harmonic radial coefficients of a Gaussian
    centered at :math:`\mathbf{r}_0`.

    The Gaussian is defined as

    .. math::
        g(\mathbf{r}) = A \, e^{-\alpha |\mathbf{r} - \mathbf{r}_0|^2}

    and is expanded as

    .. math::
        g(\mathbf{r}) = \sum_{L,M} g_{LM}(r) \, Y_{LM}(\Omega),

    where the radial components are given analytically by

    .. math::
        g_{LM}(r) = 4\pi A \, e^{-\alpha(r^2 + R_0^2)} \,
                    i_L(2\alpha r R_0) \, Y_{LM}^*(\Omega_0),

    with :math:`R_0 = |\mathbf{r}_0|`, :math:`\Omega_0` the solid angle of
    :math:`\mathbf{r}_0`, and :math:`i_L` the modified spherical Bessel
    function of the first kind. For :math:`R_0 = 0` only the
    :math:`(L, M) = (0, 0)` term survives:
    :math:`g_{00}(r) = A \sqrt{4\pi} \, e^{-\alpha r^2}`.

    Parameters
    ----------
    r : array_like
        Radial grid values where the coefficients are evaluated.
    r0 : array_like
        Cartesian centre of the Gaussian with shape ``(3,)``.
    A : float
        Amplitude of the Gaussian.
    alpha : float
        Exponent of the Gaussian (must be positive).
    L_max : int
        Maximum angular-momentum rank in the expansion.
    M_max : int, optional
        Maximum magnetic quantum number stored. Defaults to ``L_max``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_LM, len(r))`` storing :math:`g_{LM}(r)` in the
        repo's ``LM_to_I`` ordering. Returns a real array if the imaginary
        part is negligible, otherwise complex.
    """
    r = np.asarray(r, dtype=float)
    r0 = np.asarray(r0, dtype=float)

    if L_max < 0:
        raise ValueError("L_max must be >= 0")

    if M_max is None:
        M_max = L_max
    if M_max < 0 or M_max > L_max:
        raise ValueError("Require 0 <= M_max <= L_max")

    if r.ndim != 1:
        raise ValueError("r must be a one-dimensional array")
    if r0.shape != (3,):
        raise ValueError("r0 must be a Cartesian vector of shape (3,)")

    n_LM = number_of_lm_states(L_max, M_max)
    g_LM = np.zeros((n_LM, len(r)), dtype=complex)

    R0 = np.linalg.norm(r0)

    if R0 == 0:
        I_00 = LM_to_I(0, 0, L_max, M_max)
        g_LM[I_00] = A * np.sqrt(4 * np.pi) * np.exp(-alpha * r**2)
    else:
        _, theta_0, phi_0 = cartesian_to_spherical(r0)
        z = 2 * alpha * r * R0
        envelope = A * 4 * np.pi * np.exp(-alpha * (r**2 + R0**2))

        for M in range(-M_max, M_max + 1):
            for L in range(abs(M), L_max + 1):
                I_LM = LM_to_I(L, M, L_max, M_max)
                angular = np.conj(Ylm(L, M, theta_0, phi_0))
                radial = spherical_in(L, z)
                g_LM[I_LM] = envelope * radial * angular

    if np.allclose(g_LM.imag, 0.0):
        return g_LM.real

    return g_LM
