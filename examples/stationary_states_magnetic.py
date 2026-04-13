import numpy as np
import time
from matplotlib import pyplot as plt
import scipy as sp

print(f"Scipy version: {sp.__version__}")

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
    Linear_map,
)

from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_l,
)
from grid_lib.spherical_coordinates.radial_matrix_elements import (
    RadialMatrixElements,
)

import grid_lib as gb
print(f"Grid-lib version: {gb.__version__}")

def kron_delta(x1, x2):
    if x1 == x2:
        return 1
    else:
        return 0


N = 36
r_max = 12
gll = GaussLegendreLobatto(N, Linear_map(r_max=r_max), symmetrize=False)
weights = gll.weights

# setup radial matrix elements
radial_matrix_elements = RadialMatrixElements(gll)
potential = -radial_matrix_elements.r_inv
r = radial_matrix_elements.r
n_r = len(r)

T_D2 = -(1 / 2) * radial_matrix_elements.D2

Z = 1
B = 1
l_max = 6
m_list = [0, -1, -2]

print()
for m in m_list:

    dim = n_r * (l_max + 1 - abs(m))
    angular_matrix_elements = AngularMatrixElements_l(
        arr_to_calc=["H_Bz_Omega"], l_max=l_max, m=m
    )

    H_Bz = angular_matrix_elements("H_Bz_Omega")

    H = np.zeros((dim, dim))
    row = 0
    for l1 in range(abs(m), l_max + 1):
        for k1 in range(n_r):
            col = 0
            for l2 in range(abs(m), l_max + 1):
                for k2 in range(n_r):
                    H[row, col] = T_D2[k1, k2] * kron_delta(l1, l2)
                    H[row, col] += (
                        l2
                        * (l2 + 1)
                        / (2 * r[k2] ** 2)
                        * kron_delta(k1, k2)
                        * kron_delta(l1, l2)
                    )
                    H[row, col] += (
                        -1 / r[k2] * kron_delta(k1, k2) * kron_delta(l1, l2)
                    )
                    H[row, col] += (
                        B * m / 2 * kron_delta(k1, k2) * kron_delta(l1, l2)
                    )
                    H[row, col] += (
                        B**2
                        / 8
                        * r[k2] ** 2
                        * H_Bz[l1, l2].real
                        * kron_delta(k1, k2)
                    )
                    col += 1
            row += 1

    eps, C = np.linalg.eig(H)
    idx = np.argsort(eps)
    eps = eps[idx]
    C = C[:, idx]

    # Print binding energy as defined in Ref.[1] and groundstate energy.
    # The resulting binding energies for m=0,-1,-2 and B=1.0 are in reasonable agreement with
    # the values reported in Ref.[1] (Table 1,2, and 3).
    print(f"Lowest eigenvalue m={m}: {eps[0]:.3f}")
    print(f"Binding energy    m={m}: {0.5 * B * (abs(m) + m + 1) - eps[0]:.3f}")
    print()
