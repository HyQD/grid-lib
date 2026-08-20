import numpy as np
from grid_lib.cartesian_coordinates.potentials import Coulomb


class SincDvr:
    def __init__(self, n_dvr, dx):
        self.n_dvr = n_dvr
        self.dx = dx
        self.T = T_sinc_dvr(n_dvr, dx)


def compute_w12(x):
    """
    Compute the soft coulomb electron-electron interaction
        w(x,x') = 1/sqrt((x-x')^2 + a^2)
    """
    n_dvr = len(x)
    w = np.zeros((n_dvr, n_dvr))
    for i in range(n_dvr):
        w[i, :] = Coulomb(x[i] - x, Z=-1, alpha=1)
    return w


def compute_mean_field(w12, psi_j, psi_i):
    """
    Mean-field potential: V(x) = int psi_j^*(x') w(x,x') psi_i(x') dx'
    """
    return np.einsum("ij,j->i", w12, psi_j.conj() * psi_i)


def T_sinc_dvr(n_dvr, dx):
    """
    Sinc-DVR kinetic energy integrals: T_{ij} = <\chi_i|-0.5*d^2/dx^2|\chi_j>

    Reference:
    1. https://doi.org/10.1063/1.462100
    2. https://doi.org/10.1080/00268976.2016.1176262
    """
    T = np.zeros((n_dvr, n_dvr))
    for i in range(n_dvr):
        for j in range(n_dvr):
            if i == j:
                T[i, j] = np.pi**2 / (6 * dx**2)
            else:
                T[i, j] = (-1) ** (i - j) / (dx**2 * (i - j) ** 2)

    return T
