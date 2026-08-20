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
                    self.D2[i, j] = (
                        -2 * (-1) ** (i - j) / (self.dx**2 * (i - j) ** 2)
                    )

class RadialSincDVR(PseudospectralGrid):

    def __repr__(self):
        return f"RadialSincDVR"

    def __init__(self, r_max, N):
        """
        Sinc DVR on r > 0 with u(0) = 0 imposed through
        an odd extension across the origin.

        Grid:
            r_j = j*dr,  j = 1,...,N

        The point r=0 is a boundary, not a DVR point.
        """

        self.N = N
        self.r_max = r_max
        self.dr = r_max / N

        # Integer sinc indices 1,...,N
        j = np.arange(1, N + 1)
        self.j = j

        # Physical grid
        self.r = self.dr * j
        self.weights = self.dr * np.ones(N)

        I, J = np.meshgrid(j, j, indexing="ij")

        # ----------------------------------------------------------
        # First derivative
        #
        # D1_rad(i,j) = D1(i,j) - D1(i,-j)
        # ----------------------------------------------------------

        # image contribution: -D1(i,-j)
        self.D1 = (
            -(-1.0)**(I + J)
            / (self.dr * (I + J))
        )

        # ordinary D1(i,j), i != j
        mask = I != J

        self.D1[mask] += (
            (-1.0)**(I[mask] - J[mask])
            / (self.dr * (I[mask] - J[mask]))
        )

        # Diagonal is automatically
        #
        # D1_rad(i,i) = -1/(2*i*dr)


        # ----------------------------------------------------------
        # Second derivative
        #
        # D2_rad(i,j) = D2(i,j) - D2(i,-j)
        # ----------------------------------------------------------

        # image contribution: -D2(i,-j)
        self.D2 = (
            2.0 * (-1.0)**(I + J)
            / (self.dr**2 * (I + J)**2)
        )

        # ordinary off-diagonal D2(i,j)
        self.D2[mask] += (
            -2.0 * (-1.0)**(I[mask] - J[mask])
            / (
                self.dr**2
                * (I[mask] - J[mask])**2
            )
        )

        # exact diagonal
        np.fill_diagonal(
            self.D2,
            -np.pi**2 / (3.0 * self.dr**2)
            + 1.0 / (
                2.0 * self.dr**2 * j**2
            )
        )
