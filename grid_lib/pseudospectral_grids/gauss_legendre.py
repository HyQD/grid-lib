import numpy as np
from numpy.polynomial import legendre

from .gauss_legendre_lobatto import Linear_map
from .pseudospectral_grid import PseudospectralGrid


class GaussLegendre(PseudospectralGrid):
    """Gauss-Legendre quadrature and nodal differentiation on an interval.

    Parameters
    ----------
    n_points : int
        Number of nodes, equal to the degree of the Legendre polynomial
        whose roots define the grid. Must be at least one.
    Mapping : Linear_map, optional
        Linear mapping from [-1, 1] to the physical interval. Defaults to
        the identity mapping. The interval must be finite and increasing.

    Attributes
    ----------
    x : ndarray, shape (n_points,)
        Reference nodes strictly inside (-1, 1).
    r : ndarray, shape (n_points,)
        Mapped nodes in the physical interval, excluding its endpoints.
    r_dot : ndarray, shape (n_points,)
        Mapping Jacobian dr/dx at each node.
    weights : ndarray, shape (n_points,)
        Quadrature weights for integration over the physical interval.
    D1, D2 : ndarray, shape (n_points, n_points)
        First and second derivatives with respect to r, acting on ordinary
        nodal values: ``D1 @ f(r)`` and ``D2 @ f(r)``.

    Notes
    -----
    Quadrature is exact for polynomials of degree up to 2*n_points - 1.
    Differentiation is exact for polynomials of degree up to n_points - 1,
    up to floating-point error. D2 is constructed as ``D1 @ D1``. A
    one-point grid has zero first and second derivative matrices.

    Examples
    --------
    >>> grid = GaussLegendre(8, Linear_map(0.0, 2.0))
    >>> np.allclose(grid.D1 @ grid.r**2, 2.0 * grid.r)
    True
    """

    def __repr__(self):
        return "GL"

    def __init__(self, n_points: int, Mapping: Linear_map | None = None):
        if isinstance(n_points, (bool, np.bool_)) or not isinstance(
            n_points, (int, np.integer)
        ):
            raise TypeError("n_points must be an integer.")
        if n_points < 1:
            raise ValueError("n_points must be at least 1.")

        if Mapping is None:
            Mapping = Linear_map(-1.0, 1.0)
        if not isinstance(Mapping, Linear_map):
            raise TypeError("Mapping must be a Linear_map instance or None.")
        if not np.all(np.isfinite([Mapping.r_min, Mapping.r_max])):
            raise ValueError("Mapping endpoints must be finite.")
        if Mapping.r_max <= Mapping.r_min:
            raise ValueError("Mapping must define a strictly increasing interval.")

        self.n_points = int(n_points)
        self.x, reference_weights = legendre.leggauss(self.n_points)
        self.r = Mapping.r_x(self.x)
        self.r_dot = Mapping.dr_dx(self.x)
        self.weights = reference_weights * self.r_dot

        # At Gauss nodes, 1/P_n'(x_i) is proportional to alternating
        # sqrt((1 - x_i**2) * w_i). These barycentric weights avoid products
        # of node differences, which can underflow for large grids.
        barycentric_weights = np.sqrt((1.0 - self.x**2) * reference_weights)
        barycentric_weights[1::2] *= -1.0
        differences = self.x[:, np.newaxis] - self.x[np.newaxis, :]
        self.D1 = np.zeros((self.n_points, self.n_points))
        np.divide(
            barycentric_weights[np.newaxis, :] / barycentric_weights[:, np.newaxis],
            differences,
            out=self.D1,
            where=~np.eye(self.n_points, dtype=bool),
        )
        np.fill_diagonal(self.D1, -self.D1.sum(axis=1))
        self.D1 /= self.r_dot[:, np.newaxis]
        self.D2 = self.D1 @ self.D1
