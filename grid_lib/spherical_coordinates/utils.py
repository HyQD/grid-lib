import numpy as np
from opt_einsum import contract
import scipy.special
from packaging import version


def lowdin_orthogonalize(A, weights):

    n_active = A.shape[0]
    n_r = A.shape[1]
    n_lm = A.shape[2]

    A_ = contract("a, paJ->paJ", np.sqrt(weights), A)
    A_ = A_.reshape((n_active, n_r * n_lm)).T
    S = A_.T.conj() @ A_

    Sigma, X = np.linalg.eigh(S)
    Sm12 = X @ np.diag(Sigma ** (-0.5)) @ X.conj().T
    A_new = (A_ @ Sm12).T

    A_new = A_new.reshape((n_active, n_r, n_lm))
    A_new = contract("paJ, a->paJ", A_new, 1 / np.sqrt(weights))
    return A_new


class Counter:
    # Used to count iterations until convergence in bicgstab
    def __init__(self):
        self.counter = 0

    def __call__(self, x):
        self.counter += 1


def quadrature(weights, f):
    return np.sum(weights * f)


def mask_function(r, r_max, r0, n=4):
    mask_r = np.zeros(len(r))

    ind1 = r < r0
    ind2 = r == r_max
    ind3 = np.invert(ind1 + ind2)

    mask_r[ind1] = 1
    mask_r[ind2] = 0
    mask_r[ind3] = np.cos(np.pi * (r[ind3] - r0) / (2 * (r_max - r0))) ** (
        1 / n
    )

    return mask_r


def sph_harm_y(m, l, phi, theta):
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


def Ylm(l, m, theta, phi):
    """
    Compute the complex spherical harmonic Y_{l,m}(theta, phi).

    Args:
        l: orbital angular momentum quantum number
        m: magnetic quantum number
        theta: polar angle in [0, pi]
        phi: azimuthal angle in [0, 2*pi)
    """
    if l < 0:
        raise ValueError("l must be >= 0")
    if abs(m) > l:
        raise ValueError("Require |m| <= l")

    theta = np.asarray(theta)
    phi = np.asarray(phi)
    return sph_harm_y(m, l, phi, theta)
