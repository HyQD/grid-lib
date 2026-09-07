import numpy as np
from .pseudospectral_grid import PseudospectralGrid

class RadialSineDVR(PseudospectralGrid):
    """
    Sine DVR for the reduced radial wavefunction u_l(r) = r R_l(r).

    Domain:
        0 < r < rmax

    Boundary conditions:
        u_l(0)    = 0
        u_l(rmax) = 0

    N is the number of interior DVR points.
    """

    def __repr__(self):
        return f"RadialSineDVR"

    def __init__(self, rmax, N):
        self.rmax = rmax
        self.N = N

        # Grid spacing
        self.dr = rmax / (N + 1)

        # Interior DVR points; r=0 and r=rmax are excluded
        self.r = self.dr * np.arange(1, N + 1)

        # Quadrature weights
        self.weights = self.dr * np.ones(N)

        # Basis/grid indices
        j = np.arange(1, N + 1)
        n = np.arange(1, N + 1)

        theta = np.pi * j / (N + 1)

        S = np.sin(np.outer(theta, n))
        C = np.cos(np.outer(theta, n))

        # First derivative, if needed for other purposes
        self.D1 = (
            2 * np.pi / ((N + 1) * rmax)
            * (C * n) @ S.T
        )

        # Second derivative
        self.D2 = (
            -2 * np.pi**2 / ((N + 1) * rmax**2)
            * (S * n**2) @ S.T
        )