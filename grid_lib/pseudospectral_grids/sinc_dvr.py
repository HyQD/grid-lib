import numpy as np
from matplotlib import pyplot as plt
from .pseudospectral_grid import PseudospectralGrid


class SincDVR:
    def __init__(self, x0, xN, N):
        self.x0 = x0
        self.xN = xN
        self.N = N
        self.dx = (xN - x0) / (N - 1)
        self.r = np.linspace(x0, xN, N)
        self.weights = self.dx * np.ones(N)

        self.D1 = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.D1[i, j] = 0
                else:
                    self.D1[i, j] = (-1) ** (i - j) / (self.dx * (i - j))

        self.D2 = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.D2[i, j] = -np.pi**2 / (3 * self.dx**2)
                else:
                    self.D2[i, j] = -2 * (-1) ** (i - j) / (self.dx**2 * (i - j) ** 2)


class RadialSincDVR(PseudospectralGrid):
    """Parity-adapted radial sinc DVR on ``0 < r <= r_max``."""

    def __repr__(self):
        return f"RadialSincDVR"

    def __init__(self, r_max, n_r=None, l_max=0, N=None):
        """
        Sinc DVR on r > 0 with the partial-wave boundary condition
        imposed through parity adaptation across the origin.

        Grid:
            r_j = j*dr,  j = 1,...,n_r

        The point r=0 is a boundary, not a DVR point. For angular
        momentum l, the image parity is

            s_l = (-1)**(l + 1),

        so even l uses an odd extension and odd l uses an even
        extension.
        """

        if n_r is None:
            if N is None:
                raise TypeError("n_r is required")
            n_r = N
        elif N is not None and N != n_r:
            raise ValueError("N and n_r must agree when both are provided")

        self.r_max = float(r_max)
        self.n_r = int(n_r)
        self.N = self.n_r
        self.l_max = int(l_max)

        if self.r_max <= 0.0:
            raise ValueError("r_max must be positive")
        if self.n_r < 1:
            raise ValueError("n_r must be >= 1")
        if self.l_max < 0:
            raise ValueError("l_max must be >= 0")

        self.dr = self.r_max / self.n_r

        # Integer sinc indices 1,...,n_r. The origin is a boundary, not
        # a DVR point.
        j = np.arange(1, self.n_r + 1)
        self.j = j
        self.l = np.arange(self.l_max + 1)

        # Physical grid
        self.r = self.dr * j
        self.weights = self.dr * np.ones(self.n_r)

        self.D1_l, self.D2_l = self._build_l_adapted_derivative_matrices()

        # Backward-compatible l=0 matrices for existing radial code.
        self.D1 = self.D1_l[0]
        self.D2 = self.D2_l[0]

    def _build_l_adapted_derivative_matrices(self):
        j = self.j
        I, J = np.meshgrid(j, j, indexing="ij")
        mask = I != J

        D1_direct = np.zeros((self.n_r, self.n_r))
        D1_direct[mask] = (-1.0) ** (I[mask] - J[mask]) / (
            self.dr * (I[mask] - J[mask])
        )
        D1_image = (-1.0) ** (I + J) / (self.dr * (I + J))

        D2_direct = np.zeros((self.n_r, self.n_r))
        D2_direct[mask] = (
            -2.0
            * (-1.0) ** (I[mask] - J[mask])
            / (self.dr**2 * (I[mask] - J[mask]) ** 2)
        )
        np.fill_diagonal(D2_direct, -np.pi**2 / (3.0 * self.dr**2))
        D2_image = -2.0 * (-1.0) ** (I + J) / (self.dr**2 * (I + J) ** 2)

        D1_l = np.empty((self.l_max + 1, self.n_r, self.n_r))
        D2_l = np.empty((self.l_max + 1, self.n_r, self.n_r))

        for l in range(self.l_max + 1):
            s_l = (-1.0) ** (l + 1)
            D1_l[l] = D1_direct + s_l * D1_image
            D2_l[l] = D2_direct + s_l * D2_image

        return D1_l, D2_l

    def derivative_matrices(self, l):
        """Return the first- and second-derivative matrices for one l."""
        if l < 0 or l > self.l_max:
            raise ValueError("l must satisfy 0 <= l <= l_max")
        return self.D1_l[l], self.D2_l[l]
