import numpy as np
from numpy.polynomial import legendre
from matplotlib import pyplot as plt
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
    Linear_map,
)

N = 5
x_min = -1
x_max = 1
GLL_x = GaussLegendreLobatto(N, Linear_map(x_min, x_max), symmetrize=False)

x_leg = GLL_x.x.copy()
x_uniform = np.linspace(x_min, x_max, 400)

x_combined = np.sort(np.concatenate([x_leg, x_uniform]))

plt.figure()
plt.title(
    "Cardinal functions for Gauss-Legendre-Lobatto grid of order N=%d" % N
)
for j in range(0, N + 1):
    g_j = GLL_x.cardinal_function(x_combined, j)
    plt.plot(x_combined, g_j, label=r"$g_{%d}(x)$" % j)
plt.plot(
    x_leg,
    np.zeros_like(x_leg),
    "x",
    label="Legendre-Lobatto points",
    color="black",
    markersize=8,
)
for x_point in x_leg:
    plt.axvline(x_point, color="gray", linestyle="--", alpha=0.5)
plt.plot(
    x_leg,
    np.ones_like(x_leg),
    "o",
    label="Cardinal function values at grid points",
    color="red",
    markersize=5,
)
plt.grid()
plt.legend()

plt.show()
