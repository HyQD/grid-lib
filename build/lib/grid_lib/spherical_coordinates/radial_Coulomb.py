import numpy as np
from grid_lib.spherical_coordinates.radial_poisson import (
    solve_radial_Poisson_dvr,
)


def radial_Coulomb(GLL, n_L):

    tilde_V = solve_radial_Poisson_dvr(GLL, n_L)
    r = GLL.r[1:-1]
    n_r = len(r)

    W = np.zeros((n_L, n_r, n_r))
    for L in range(0, n_L):
        W[L] = (4 * np.pi / (2 * L + 1)) * tilde_V[L] / r[:, np.newaxis]

    return W
