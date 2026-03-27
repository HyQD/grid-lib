import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_legendre as Legendre


class Rational_map:
    def __init__(self, r_max=30, alpha=0.4):
        self.r_max = r_max
        self.alpha = alpha
        self.L_scale = self.r_max * self.alpha / 2

    def r_x(self, x):
        return self.L_scale * (1 + x) / (1 - x + self.alpha)

    def dr_dx(self, x):
        return (2 + self.alpha) * self.L_scale / (1 - x + self.alpha) ** 2


class Linear_map:
    def __init__(self, r_min=0, r_max=30):
        self.r_min = r_min
        self.r_max = r_max

    def r_x(self, x):
        return (self.r_max - self.r_min) * (x + 1) / 2 + self.r_min

    def dr_dx(self, x):
        return (self.r_max - self.r_min) / 2 * np.ones(x.shape)


class GaussLegendreLobatto:
    def __init__(self, N, Mapping, symmetrize=True):
        self.N = N
        self.c = np.zeros((N + 1,))
        self.c[-1] = 1
        self.dc = legendre.legder(self.c)

        """
        Find the Legendre-Lobatto grid points, defined as the roots of the derivative N-th order Legendre polynomial,
            dP_N(x)/dx|_{x=x_i} = 0, i=1,...,N, 
        and x_0 = -1, x_{N+1} = 1.
        """
        self.x = np.zeros(N + 1)
        self.x[0] = -1
        self.x[-1] = 1
        self.x[1:-1] = legendre.legroots(self.dc)

        self.D1 = np.zeros((N + 1, N + 1))
        self.D2 = np.zeros((N + 1, N + 1))
        self.PN_x = legendre.legval(self.x[1:-1], self.c)
        self.PN_x2 = legendre.legval(self.x, self.c)

        self.weights = 2 / (N * (N + 1)) * np.ones(N + 1)
        if not symmetrize:
            self.weights /= self.PN_x2**2

        self.r = Mapping.r_x(self.x)
        self.r_dot = Mapping.dr_dx(self.x)
        self.weights *= self.r_dot

        for i in range(N + 1):
            for j in range(N + 1):
                if i == 0 and j == 0:
                    self.D1[i, j] = -(
                        0.25
                        * N
                        * (N + 1)
                        / np.sqrt(self.r_dot[i] * self.r_dot[j])
                    )
                elif i == N and j == N:
                    self.D1[i, j] = (
                        0.25
                        * N
                        * (N + 1)
                        / np.sqrt(self.r_dot[i] * self.r_dot[j])
                    )
                elif i == j and 1 <= j <= N - 1:
                    self.D1[i, j] = 0
                else:
                    self.D1[i, j] = 1 / (
                        (self.x[i] - self.x[j])
                        * np.sqrt(self.r_dot[i] * self.r_dot[j])
                    )
                    if not symmetrize:
                        self.D1[i, j] *= self.PN_x2[i] / self.PN_x2[j]

        self.D2 = np.dot(self.D1, self.D1)

    def cardinal_function(self, y, j):

        if np.isscalar(y) and y == self.x[j]:
            return self.weights[j]
        elif isinstance(y, np.ndarray):
            with np.errstate(divide="ignore", invalid="ignore"):
                result = (
                    -1
                    / (self.N * (self.N + 1) * self.PN_x2[j])
                    * (1 - y**2)
                    * legendre.legval(y, self.dc)
                    / (y - self.x[j])
                )
            result[y == self.x[j]] = 1.0
            return result
        else:
            return (
                -1
                / (self.N * (self.N + 1) * self.PN_x2[j])
                * (1 - y**2)
                * legendre.legval(y, self.dc)
                / (y - self.x[j])
            )
