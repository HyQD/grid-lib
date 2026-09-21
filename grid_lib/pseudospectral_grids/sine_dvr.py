import numpy as np
from .pseudospectral_grid import PseudospectralGrid


import numpy as np


import numpy as np


class SineDVR:
    """
    Sine DVR on a finite interval with homogeneous Dirichlet boundaries.

    Uses N interior DVR points with uniform spacing

        dx = (x_max - x0) / N,

    so that x_j = x0 + j*dx, j = 1,...,N.  The sine basis vanishes at
    x0 and at the right boundary x_max + dx.  Thus x_max denotes the
    last DVR point, not the right boundary.

    D1 and D2 are spectral first- and second-derivative matrices.
    D1 is antisymmetric and D2 is symmetric.

    Parameters
    ----------
    x0 : float
        Left Dirichlet boundary.
    x_max : float
        Position of the last DVR grid point.
    N : int
        Number of DVR grid points.
    """

    def __init__(self, x0, x_max, N):
        self.x0 = x0
        self.x_max = x_max
        self.N = N

        # Grid spacing: N is the actual number of grid points
        self.dx = (x_max - x0) / N

        # The sine basis lives on [x0, x_right]
        self.x_right = x0 + (N + 1) * self.dx
        self.L = self.x_right - x0

        # Interior DVR grid points
        j = np.arange(1, N + 1)
        n = np.arange(1, N + 1)

        self.x = x0 + j * self.dx

        # Alias useful for radial calculations
        self.r = self.x

        # Uniform DVR quadrature weights
        self.weights = self.dx * np.ones(N)

        # ------------------------------------------------------
        # FBR -> DVR transformation
        # ------------------------------------------------------
        J, NN = np.meshgrid(j, n, indexing="ij")

        U = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * J * NN / (N + 1))

        self.U = U

        # ------------------------------------------------------
        # Second derivative
        #
        # <n|d²/dx²|m> = -(n*pi/L)² delta_nm
        # ------------------------------------------------------
        k = n * np.pi / self.L

        D2_fbr = np.diag(-(k**2))

        self.D2 = U @ D2_fbr @ U.T

        # ------------------------------------------------------
        # First derivative
        #
        # <n|d/dx|m> =
        #
        #   4*n*m / [L*(n²-m²)],  n+m odd
        #   0,                     n+m even
        #
        # ------------------------------------------------------
        n1, n2 = np.meshgrid(n, n, indexing="ij")

        D1_fbr = np.zeros((N, N))

        mask = (n1 + n2) % 2 == 1

        D1_fbr[mask] = (
            4.0
            * n1[mask]
            * n2[mask]
            / (self.L * (n1[mask] ** 2 - n2[mask] ** 2))
        )

        self.D1 = U @ D1_fbr @ U.T

        # ------------------------------------------------------
        # Consistency checks
        # ------------------------------------------------------
        assert np.allclose(
            self.D1, -self.D1.T, atol=1e-12
        ), "D1 is not antisymmetric"

        assert np.allclose(
            self.D2, self.D2.T, atol=1e-12
        ), "D2 is not symmetric"
