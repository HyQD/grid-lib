import numpy as np

from grid_lib.pseudospectral_grids.gauss_legendre import GaussLegendre


def _validate_count(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


class GaussLegendreFourierGrid:
    """Tensor-product quadrature on the unit sphere.

    Gauss-Legendre quadrature is used in ``cos(theta)`` and an equally
    spaced periodic grid in ``phi``. Here theta is the polar angle and
    phi is the azimuthal angle. Neither pole nor the duplicate azimuthal
    endpoint at 2*pi is included.

    Parameters
    ----------
    n_theta : int
        Number of Gauss-Legendre nodes. Must be at least one.
    n_phi : int
        Number of equally spaced azimuthal nodes. Must be at least one.

    Attributes
    ----------
    polar_grid : GaussLegendre
        Underlying one-dimensional grid on [-1, 1]. Its differentiation
        matrices act in cos(theta), not theta.
    cos_theta, theta : ndarray, shape (n_theta,)
        Ascending cosine nodes and corresponding descending polar angles.
    phi : ndarray, shape (n_phi,)
        Ascending azimuthal nodes in [0, 2*pi), starting at zero.
    theta_weights : ndarray, shape (n_theta,)
        Ordinary Gauss-Legendre weights on [-1, 1], summing to two.
    phi_weights : ndarray, shape (n_phi,)
        Uniform weights 2*pi/n_phi.
    weights : ndarray, shape (n_theta, n_phi)
        Physical solid-angle weights, summing to 4*pi. No additional
        sin(theta) or 4*pi factor should be applied.

    Notes
    -----
    This class supplies quadrature, not spherical-harmonic transforms or
    radial discretization. Values are ordinary nodal values, without
    square-root quadrature weights absorbed into them.

    Examples
    --------
    >>> grid = GaussLegendreFourierGrid(8, 16)
    >>> theta, phi = grid.mesh()
    >>> np.allclose(grid.integrate(np.cos(theta)**2), 4 * np.pi / 3)
    True
    """

    def __init__(self, n_theta: int, n_phi: int):
        self.n_theta = _validate_count(n_theta, "n_theta", 1)
        self.n_phi = _validate_count(n_phi, "n_phi", 1)
        self.polar_grid = GaussLegendre(self.n_theta)
        self.cos_theta = self.polar_grid.x
        self.theta = np.arccos(self.cos_theta)
        self.theta_weights = self.polar_grid.weights
        self.phi = (2 * np.pi / self.n_phi) * np.arange(self.n_phi)
        self.phi_weights = np.full(self.n_phi, 2 * np.pi / self.n_phi)
        self.weights = self.theta_weights[:, None] * self.phi_weights[None, :]

    def __repr__(self):
        return f"GaussLegendreFourierGrid(n_theta={self.n_theta}, n_phi={self.n_phi})"

    @property
    def shape(self) -> tuple[int, int]:
        """Angular array shape, with the azimuthal axis last."""
        return self.n_theta, self.n_phi

    @property
    def size(self) -> int:
        """Total number of angular nodes."""
        return self.n_theta * self.n_phi

    @classmethod
    def from_bandlimit(cls, l_max: int):
        """Choose a grid for spherical-harmonic analysis through l_max.

        Uses n_theta = l_max + 1 and n_phi = 2*l_max + 1. Products
        Y_lm.conj() * Y_l'm' are integrated exactly up to roundoff when
        both degrees are at most l_max. The cutoff must be nonnegative.

        Products of two expansions with degree at most l_orb can require
        ``from_bandlimit(2 * l_orb)``. Arbitrary non-bandlimited functions
        require convergence checks.
        """
        l_max = _validate_count(l_max, "l_max", 0)
        return cls(l_max + 1, 2 * l_max + 1)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return theta and phi arrays of shape ``shape`` (ij indexing).

        For flat quadrature consumers, flatten both returned arrays and
        ``weights`` in the same order.
        """
        return np.meshgrid(self.theta, self.phi, indexing="ij")

    def integrate(self, values):
        """Integrate nodal values over solid angle.

        Parameters
        ----------
        values : array_like
            Real or complex values with angular axes last, of shape
            (..., n_theta, n_phi). Scalars and singleton angular axes
            are broadcast. For polar-only values use (..., n_theta, 1);
            for azimuthal-only values use (..., 1, n_phi).
            Evaluate callables on ``mesh()`` before passing their values.

        Returns
        -------
        integral : scalar or ndarray
            Weighted sum over the last two axes, preserving any leading
            batch axes and complex values. No conjugation is applied.
        """
        values = np.asarray(values)
        try:
            values = np.broadcast_to(values, values.shape[:-2] + self.shape)
        except ValueError as error:
            raise ValueError(
                f"values must be broadcastable to angular shape {self.shape} "
                "on the last two axes."
            ) from error
        return np.sum(values * self.weights, axis=(-2, -1))
