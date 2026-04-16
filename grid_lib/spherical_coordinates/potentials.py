import numpy as np
from scipy.special import erf

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
