r"""B-spline multipole solution of multicentre free-space Poisson problems."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.integrate import lebedev_rule
from scipy.linalg import cho_factor, cho_solve
from scipy.special import sph_harm_y

from grid_lib.basis import BSplineRadialBasis

from .becke import BeckePartition


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_integer(value, name):
    value = _nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validated_rotation(value):
    if value is None:
        return np.eye(3)
    value = np.asarray(value, dtype=float)
    if value.shape != (3, 3) or np.any(~np.isfinite(value)):
        raise ValueError("angular_rotation must be a finite 3 by 3 matrix")
    if not np.allclose(value.T @ value, np.eye(3), atol=2e-14, rtol=0.0):
        raise ValueError("angular_rotation must be orthogonal")
    if not np.isclose(np.linalg.det(value), 1.0, atol=2e-14, rtol=0.0):
        raise ValueError("angular_rotation must be proper")
    return value


def _radial_quadrature(nodes, order):
    reference_points, reference_weights = np.polynomial.legendre.leggauss(order)
    radii = []
    weights = []
    for left, right in zip(nodes[:-1], nodes[1:]):
        scale = 0.5 * (right - left)
        radii.append(0.5 * (right + left) + scale * reference_points)
        weights.append(scale * reference_weights)
    return np.concatenate(radii), np.concatenate(weights)


def _harmonics(unit_vectors, multipoles):
    x, y, z = unit_vectors.T
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    return np.column_stack(
        [sph_harm_y(L, M, theta, phi) for L, M in multipoles]
    )


def _radial_bases(value, n_centers):
    if isinstance(value, BSplineRadialBasis):
        bases = (value,) * n_centers
    else:
        bases = tuple(value)
    if len(bases) != n_centers or any(
        not isinstance(basis, BSplineRadialBasis) for basis in bases
    ):
        raise TypeError(
            "radial_bases must supply one BSplineRadialBasis per center"
        )
    for basis in bases:
        if basis.vanish_at_boundaries != (True, False):
            raise ValueError(
                "Poisson radial bases must vanish at the inner boundary "
                "and retain the outer boundary function"
            )
        if basis.r_min != 0.0:
            raise ValueError("Poisson radial bases must start at zero")
    return bases


@dataclass(frozen=True)
class _AtomicPoissonData:
    basis: BSplineRadialBasis
    radii: np.ndarray
    radial_weights: np.ndarray
    basis_values: np.ndarray
    basis_derivatives: np.ndarray
    points: np.ndarray
    partition_weights: np.ndarray
    integration_weights: np.ndarray
    factorizations: tuple
    outer_values: np.ndarray
    origin_derivatives: np.ndarray


class BeckeMulticenterPoisson:
    r"""Free-space Coulomb potential from Becke-partitioned densities.

    The input density is sampled on atom-centred radial--Lebedev grids.  Its
    Becke component around each centre is analyzed in spherical harmonics.
    Every radial multipole is then obtained from a B-spline Galerkin solution
    for ``u_LM(r) = r * V_LM(r)``:

    .. math::

        \left[-\frac{d^2}{dr^2}+\frac{L(L+1)}{r^2}\right]u_{LM}
        = 4\pi r\rho_{LM}.

    The inner condition is ``u(0)=0``.  At the finite outer radius, the
    free-space Robin condition ``u'(R)+L*u(R)/R=0`` supplies the exact
    multipole tail of the represented finite source.

    Parameters
    ----------
    centers : array_like, shape (n_centers, 3)
        Cartesian source centres.
    radial_bases : BSplineRadialBasis or sequence
        One potential basis per centre.  Each basis must use
        ``vanish_at_boundaries=(True, False)``.
    L_max : int
        Largest retained source and potential multipole degree.
    radial_order : int, optional
        Gauss--Legendre order in every radial knot interval.
    lebedev_order : int, optional
        SciPy Lebedev rule order for every atom-centred shell.
    switch_iterations : int, optional
        Becke smoothing iterations.
    angular_rotation : array_like, optional
        Proper rotation applied to every Lebedev grid.
    """

    def __init__(
        self,
        centers,
        radial_bases,
        L_max,
        *,
        radial_order=12,
        lebedev_order=29,
        switch_iterations=3,
        angular_rotation=None,
    ):
        self.partition = BeckePartition(centers, switch_iterations)
        self.centers = self.partition.centers
        self.n_centers = self.partition.n_centers
        self.radial_bases = _radial_bases(radial_bases, self.n_centers)
        self.L_max = _nonnegative_integer(L_max, "L_max")
        self.radial_order = _positive_integer(radial_order, "radial_order")
        minimum_radial_order = max(basis.degree for basis in self.radial_bases)
        if self.radial_order < minimum_radial_order:
            raise ValueError(
                "radial_order must be at least the largest B-spline degree"
            )
        self.lebedev_order = _positive_integer(
            lebedev_order, "lebedev_order"
        )
        self.angular_rotation = _validated_rotation(angular_rotation)
        self.multipoles = tuple(
            (L, M)
            for L in range(self.L_max + 1)
            for M in range(-L, L + 1)
        )
        self._indices_by_L = tuple(
            np.array(
                [k for k, (degree, _) in enumerate(self.multipoles) if degree == L]
            )
            for L in range(self.L_max + 1)
        )

        angular_points, angular_weights = lebedev_rule(self.lebedev_order)
        self.angular_unit_vectors = (
            angular_points.T @ self.angular_rotation.T
        )
        self.angular_weights = np.asarray(angular_weights)
        self._angular_harmonics = _harmonics(
            self.angular_unit_vectors, self.multipoles
        )
        self.angular_unit_vectors.setflags(write=False)
        self.angular_weights.setflags(write=False)
        self._angular_harmonics.setflags(write=False)

        atomic_data = []
        points = []
        integration_weights = []
        center_slices = []
        start = 0
        for center_index, (center, basis) in enumerate(
            zip(self.centers, self.radial_bases)
        ):
            radii, radial_weights = _radial_quadrature(
                basis.nodes, self.radial_order
            )
            basis_values, basis_derivatives = basis.evaluate(radii)
            shell_points = (
                center[None, None, :]
                + radii[:, None, None] * self.angular_unit_vectors[None, :, :]
            )
            flat_points = shell_points.reshape(-1, 3)
            partition_weights = self.partition.weights(flat_points)[
                :, center_index
            ].reshape(len(radii), -1)
            volume_weights = (
                (radial_weights * radii**2)[:, None]
                * self.angular_weights[None, :]
                * partition_weights
            )

            outer_values, _ = basis.evaluate([basis.r_max])
            _, origin_derivatives = basis.evaluate([0.0])
            factorizations = []
            for L in range(self.L_max + 1):
                stiffness = basis_derivatives.T @ (
                    radial_weights[:, None] * basis_derivatives
                )
                if L:
                    stiffness += L * (L + 1) * basis_values.T @ (
                        (radial_weights / radii**2)[:, None] * basis_values
                    )
                    stiffness += (L / basis.r_max) * np.outer(
                        outer_values[0], outer_values[0]
                    )
                factorizations.append(cho_factor(stiffness, check_finite=False))

            count = flat_points.shape[0]
            center_slices.append(slice(start, start + count))
            start += count
            points.append(flat_points)
            integration_weights.append(volume_weights.ravel())
            atomic_data.append(
                _AtomicPoissonData(
                    basis=basis,
                    radii=radii,
                    radial_weights=radial_weights,
                    basis_values=basis_values,
                    basis_derivatives=basis_derivatives,
                    points=flat_points,
                    partition_weights=partition_weights,
                    integration_weights=volume_weights,
                    factorizations=tuple(factorizations),
                    outer_values=outer_values[0],
                    origin_derivatives=origin_derivatives[0],
                )
            )

        self._atomic_data = tuple(atomic_data)
        self.center_slices = tuple(center_slices)
        self.points = np.concatenate(points)
        self.integration_weights = np.concatenate(integration_weights)
        for array in (self.points, self.integration_weights):
            array.setflags(write=False)

    @property
    def point_count(self):
        """Total number of atom-centred quadrature points."""

        return self.points.shape[0]

    def integrate(self, values):
        """Integrate scalar values supplied in the builder point ordering."""

        values = np.asarray(values)
        if values.shape != (self.point_count,):
            raise ValueError("values must contain one scalar per quadrature point")
        if np.any(~np.isfinite(values)):
            raise ValueError("values must be finite")
        return np.dot(self.integration_weights, values)

    def solve(self, density):
        """Sample a callable density and construct its Coulomb potential."""

        if not callable(density):
            raise TypeError("density must be callable")
        return self.solve_values(density(self.points))

    def solve_values(self, density_values):
        """Construct the Coulomb potential from values on ``points``."""

        density_values = np.asarray(density_values)
        if density_values.shape != (self.point_count,):
            raise ValueError(
                "density_values must contain one scalar per quadrature point"
            )
        if np.any(~np.isfinite(density_values)):
            raise ValueError("density_values must be finite")

        coefficient_sets = []
        for data, center_slice in zip(self._atomic_data, self.center_slices):
            density = density_values[center_slice].reshape(
                len(data.radii), len(self.angular_weights)
            )
            partitioned_density = density * data.partition_weights
            density_multipoles = partitioned_density @ (
                self.angular_weights[:, None]
                * self._angular_harmonics.conj()
            )
            right_hand_sides = 4.0 * np.pi * data.basis_values.T @ (
                (data.radial_weights * data.radii)[:, None]
                * density_multipoles
            )
            coefficients = np.empty(
                (len(self.multipoles), data.basis.size),
                dtype=np.result_type(density_values.dtype, complex),
            )
            for L, indices in enumerate(self._indices_by_L):
                coefficients[indices] = cho_solve(
                    data.factorizations[L],
                    right_hand_sides[:, indices],
                    check_finite=False,
                ).T
            coefficients.setflags(write=False)
            coefficient_sets.append(coefficients)

        return MulticenterPoissonPotential(
            self,
            tuple(coefficient_sets),
            self.integrate(density_values),
        )


class MulticenterPoissonPotential:
    """Coulomb potential returned by :class:`BeckeMulticenterPoisson`."""

    def __init__(self, builder, coefficients, total_charge):
        self.builder = builder
        self.coefficients = coefficients
        self.total_charge = total_charge

    def __call__(self, points, *, chunk_size=32768):
        """Evaluate the represented free-space potential at Cartesian points.

        ``chunk_size`` bounds the temporary point-by-multipole arrays used
        during evaluation.  It does not change the represented potential.
        """

        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3)")
        if np.any(~np.isfinite(points)):
            raise ValueError("points must be finite")
        point_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        chunk_size = _positive_integer(chunk_size, "chunk_size")
        result = np.zeros(
            len(flat_points), dtype=np.result_type(*self.coefficients)
        )
        for start in range(0, len(flat_points), chunk_size):
            stop = min(start + chunk_size, len(flat_points))
            result[start:stop] = self._evaluate_flat(flat_points[start:stop])
        return result.reshape(point_shape)

    def _evaluate_flat(self, flat_points):
        result = np.zeros(
            len(flat_points), dtype=np.result_type(*self.coefficients)
        )

        for center, data, coefficients in zip(
            self.builder.centers,
            self.builder._atomic_data,
            self.coefficients,
        ):
            displacements = flat_points - center
            radii = np.linalg.norm(displacements, axis=1)
            positive = radii > 0.0
            unit_vectors = np.zeros_like(displacements)
            unit_vectors[positive] = (
                displacements[positive] / radii[positive, None]
            )
            unit_vectors[~positive, 2] = 1.0
            harmonics = _harmonics(unit_vectors, self.builder.multipoles)
            radial_values = np.zeros(
                (len(flat_points), len(self.builder.multipoles)),
                dtype=coefficients.dtype,
            )

            inside = positive & (radii <= data.basis.r_max)
            if np.any(inside):
                basis_values, _ = data.basis.evaluate(radii[inside])
                radial_values[inside] = (
                    basis_values @ coefficients.T
                ) / radii[inside, None]

            outside = radii > data.basis.r_max
            if np.any(outside):
                boundary_u = data.outer_values @ coefficients.T
                for L, indices in enumerate(self.builder._indices_by_L):
                    radial_values[np.ix_(outside, indices)] = (
                        boundary_u[indices][None, :]
                        * (
                            data.basis.r_max / radii[outside, None]
                        )
                        ** L
                        / radii[outside, None]
                    )

            at_center = ~positive
            if np.any(at_center):
                monopole = self.builder._indices_by_L[0]
                radial_values[np.ix_(at_center, monopole)] = (
                    data.origin_derivatives @ coefficients[monopole].T
                )

            result += np.sum(radial_values * harmonics, axis=1)

        return result
