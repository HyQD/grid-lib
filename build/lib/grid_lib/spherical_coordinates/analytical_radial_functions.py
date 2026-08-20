import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma, factorial, factorial2, assoc_laguerre


def Rnl_Hydrogenic(r, n, l, Z=1.0):
    """
    Eq. [6.5.13] Helgaker, Jørgensen, Olsen - Molecular Electronic-Structure Theory
    """
    norm_constant = (2 * Z / n) ** (1.5)
    norm_constant *= np.sqrt(factorial(n - l - 1) / (2 * n * factorial(n + l)))
    radial_part = (2 * Z * r / n) ** l
    radial_part *= assoc_laguerre(2 * Z * r / n, n - l - 1, 2 * l + 1)
    radial_part *= np.exp(-Z * r / n)
    return norm_constant * radial_part


def Rnl_LF(r, n, l, zeta=1.0):
    """
    Eq. [6.5.17] Helgaker, Jørgensen, Olsen - Molecular Electronic-Structure Theory
    """
    norm_constant = (2 * zeta) ** (1.5)
    norm_constant *= np.sqrt(factorial(n - l - 1) / factorial(n + l + 1))
    radial_part = (2 * zeta * r) ** l
    radial_part *= assoc_laguerre(2 * zeta * r, n - l - 1, 2 * l + 2)
    radial_part *= np.exp(-zeta * r)
    return norm_constant * radial_part


def Rn_STO(r, n, zeta=1.0):
    """
    Eq. [6.5.26] Helgaker, Jørgensen, Olsen - Molecular Electronic-Structure Theory
    """
    norm_constant = (2 * zeta) ** (1.5) / np.sqrt(gamma(2 * n + 1))
    radial_part = (2 * zeta * r) ** (n - 1) * np.exp(-zeta * r)

    return norm_constant * radial_part


def Rnl_HO(r, n, l, alpha=0.5):
    norm_constant = (2 * alpha) ** (3 / 4) / np.pi ** (0.25)
    norm_constant *= np.sqrt(
        2 ** (n + 1) * factorial(n - l - 1) / factorial2(2 * n - 1)
    )
    radial_part = (np.sqrt(2 * alpha) * r) ** l
    radial_part *= assoc_laguerre(2 * alpha * r**2, n - l - 1, l + 0.5)
    radial_part *= np.exp(-alpha * r**2)
    return norm_constant * radial_part


def Rnl_GTO(r, n, l, alpha=1.0):
    norm_constant = 2 * (2 * alpha) ** (0.75) / np.pi ** (0.25)
    norm_constant *= np.sqrt(
        2 ** (2 * n - l - 2) / factorial2(4 * n - 2 * l - 3)
    )
    radial_part = (np.sqrt(2 * alpha) * r) ** (2 * n - l - 2)
    radial_part *= np.exp(-alpha * r**2)
    return norm_constant * radial_part
