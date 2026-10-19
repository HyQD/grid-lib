"""Nodal differential operators in spherical coordinates.

The operators in this module act on physical scalar and vector fields.  They
do not assume an orbital representation such as ``u(r) / r`` and do not
contain density-functional-specific conventions.
"""

from dataclasses import dataclass

import numpy as np

from .angular_grid import GaussLegendreFourierGrid


@dataclass(frozen=True)
class SphericalVectorField:
    r"""Components of a vector field in the orthonormal spherical basis.

    ``radial``, ``polar``, and ``azimuthal`` are the components along
    :math:`\boldsymbol e_r`, :math:`\boldsymbol e_\theta`, and
    :math:`\boldsymbol e_\phi`, respectively.  Their last three axes must be
    ``(radial, polar, azimuthal)``; leading batch axes are allowed.
    """

    radial: np.ndarray
    polar: np.ndarray
    azimuthal: np.ndarray


class SphericalNodalDifferentialOperator:
    """Gradient and divergence on a radial--Gauss-Legendre--Fourier grid.

    Parameters
    ----------
    radial_grid
        One-dimensional grid exposing ``r``, ``weights``, and ``D1``.  ``D1``
        must differentiate ordinary nodal values with respect to ``r``.
    angular_grid : GaussLegendreFourierGrid
        Gauss-Legendre grid in ``cos(theta)`` and periodic Fourier grid in
        ``phi``.
    radial_indices : slice or one-dimensional array_like of int, optional
        Radial nodes on which the operators act.  All selected radii must be
        strictly positive.

    Notes
    -----
    Scalar fields contain ordinary nodal values.  Differential methods accept
    either active arrays of shape ``(..., n_r, n_theta, n_phi)`` or arrays on
    the complete radial grid.  Results are always returned on the active
    nodes.  Full-grid input includes boundary values in radial derivatives;
    active input restricts ``D1`` to its active rows and columns and therefore
    treats omitted values as zero.

    The poles are absent from the angular grid, while the strictly
    positive-radius requirement avoids the coordinate singularity at
    ``r = 0``.  For an FEM-DVR grid containing both endpoints,
    ``radial_indices=slice(1, -1)`` selects the usual interior while still
    allowing full-grid field values as input.

    The strong-form :meth:`divergence` uses the coordinate formula directly.
    :meth:`adjoint_divergence` is instead the negative discrete adjoint of the
    gradient under the tensor-product volume quadrature.  The latter is useful
    when a discretely variational derivative is required.
    """

    def __init__(
        self,
        radial_grid,
        angular_grid: GaussLegendreFourierGrid,
        *,
        radial_indices=None,
    ):
        if not isinstance(angular_grid, GaussLegendreFourierGrid):
            raise TypeError("angular_grid must be a GaussLegendreFourierGrid.")

        missing = [
            name for name in ("r", "weights", "D1") if not hasattr(radial_grid, name)
        ]
        if missing:
            names = ", ".join(missing)
            raise TypeError(f"radial_grid is missing required attributes: {names}.")

        full_r = np.asarray(radial_grid.r)
        full_weights = np.asarray(radial_grid.weights)
        full_d1 = np.asarray(radial_grid.D1)
        if full_r.ndim != 1:
            raise ValueError("radial_grid.r must be one-dimensional.")
        if full_weights.shape != full_r.shape:
            raise ValueError("radial_grid.weights must have the same shape as r.")
        if full_d1.shape != (full_r.size, full_r.size):
            raise ValueError("radial_grid.D1 must have shape (r.size, r.size).")
        if not np.all(np.isfinite(full_r)):
            raise ValueError("radial nodes must be finite.")
        if not np.isrealobj(full_r) or np.any(np.diff(full_r) <= 0):
            raise ValueError("radial nodes must be real and strictly increasing.")
        if full_r[0] < 0:
            raise ValueError("radial nodes must be nonnegative.")
        if not np.isrealobj(full_weights):
            raise ValueError("radial quadrature weights must be real.")
        if not np.all(np.isfinite(full_weights)) or np.any(full_weights <= 0):
            raise ValueError("radial quadrature weights must be finite and positive.")
        if not np.all(np.isfinite(full_d1)):
            raise ValueError("radial_grid.D1 must contain finite values.")

        indices = self._normalize_radial_indices(radial_indices, full_r.size)
        radii = full_r[indices]
        if np.any(radii <= 0):
            raise ValueError(
                "selected radial nodes must be strictly positive; exclude r = 0 "
                "with radial_indices."
            )

        self.angular_grid = angular_grid
        self.radial_indices = indices
        self.full_r = full_r.copy()
        self.full_radial_derivative_matrix = full_d1.copy()
        self.radial_derivative_rows = full_d1[indices, :].copy()
        self.r = radii.copy()
        self.radial_weights = full_weights[indices].copy()
        self.radial_derivative_matrix = full_d1[np.ix_(indices, indices)].copy()
        self.cos_theta = angular_grid.cos_theta.copy()
        self.sin_theta = np.sqrt(1.0 - self.cos_theta**2)
        self.polar_derivative_matrix = (
            -self.sin_theta[:, None] * angular_grid.polar_grid.D1
        )

        modes = np.fft.fftfreq(angular_grid.n_phi, d=1.0 / angular_grid.n_phi)
        if angular_grid.n_phi % 2 == 0:
            # A sampled Nyquist wave has no unique derivative on the nodes.
            # The real-valued spectral differentiation convention sets it to 0.
            modes[angular_grid.n_phi // 2] = 0.0
        self.azimuthal_modes = modes

        self.shape = (self.r.size,) + angular_grid.shape
        self.full_shape = (self.full_r.size,) + angular_grid.shape
        self.volume_weights = (
            (self.radial_weights * self.r**2)[:, None, None]
            * angular_grid.weights[None, :, :]
        )

    @staticmethod
    def _normalize_radial_indices(radial_indices, size):
        if radial_indices is None:
            indices = np.arange(size)
        elif isinstance(radial_indices, slice):
            indices = np.arange(size)[radial_indices]
        else:
            indices = np.asarray(radial_indices)
            if indices.ndim != 1:
                raise ValueError("radial_indices must be a slice or one-dimensional.")
            if np.issubdtype(indices.dtype, np.bool_):
                if indices.size != size:
                    raise ValueError(
                        "a boolean radial_indices mask must have length r.size."
                    )
                indices = np.flatnonzero(indices)
            elif not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("radial_indices must contain integers.")
            indices = indices.astype(int, copy=False)
        if indices.size == 0:
            raise ValueError("radial_indices must select at least one node.")
        if np.any(indices < 0) or np.any(indices >= size):
            raise IndexError("radial_indices contains an out-of-range index.")
        if np.unique(indices).size != indices.size:
            raise ValueError("radial_indices must not contain duplicate indices.")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("radial_indices must be in strictly increasing order.")
        return indices

    def __repr__(self):
        n_r, n_theta, n_phi = self.shape
        return (
            "SphericalNodalDifferentialOperator("
            f"n_r={n_r}, n_theta={n_theta}, n_phi={n_phi})"
        )

    def _scalar(self, values, name="values"):
        values = np.asarray(values)
        if values.ndim < 3 or values.shape[-3:] != self.shape:
            raise ValueError(
                f"{name} must have spherical grid shape {self.shape} on its "
                "last three axes."
            )
        return values

    def _differentiable_scalar(self, values, name="values"):
        values = np.asarray(values)
        if values.ndim >= 3 and values.shape[-3:] == self.shape:
            return values, False
        if values.ndim >= 3 and values.shape[-3:] == self.full_shape:
            return values, True
        allowed = str(self.shape)
        if self.full_shape != self.shape:
            allowed += f" or full radial shape {self.full_shape}"
        raise ValueError(
            f"{name} must have spherical grid shape {allowed} on its last "
            "three axes."
        )

    def _active_values(self, values, name="values"):
        values, is_full = self._differentiable_scalar(values, name)
        if is_full:
            values = values[..., self.radial_indices, :, :]
        return values

    def _vector(self, vector, *, allow_full=False):
        if not isinstance(vector, SphericalVectorField):
            raise TypeError("vector must be a SphericalVectorField.")
        raw_components = [vector.radial, vector.polar, vector.azimuthal]
        names = ["vector.radial", "vector.polar", "vector.azimuthal"]
        if allow_full:
            parsed = [
                self._differentiable_scalar(component, name)
                for component, name in zip(raw_components, names, strict=True)
            ]
            radial_extents = [is_full for _, is_full in parsed]
            if len(set(radial_extents)) != 1:
                raise ValueError(
                    "vector components must use the same active or full radial grid."
                )
            components = [component for component, _ in parsed]
            is_full = radial_extents[0]
        else:
            components = [
                self._scalar(component, name)
                for component, name in zip(raw_components, names, strict=True)
            ]
            is_full = False
        try:
            components = np.broadcast_arrays(*components)
        except ValueError as error:
            raise ValueError(
                "vector components must have broadcastable batch axes."
            ) from error
        return components, is_full

    def radial_derivative(self, values):
        """Return ``partial values / partial r`` on the active nodes."""
        values, is_full = self._differentiable_scalar(values)
        matrix = (
            self.radial_derivative_rows
            if is_full
            else self.radial_derivative_matrix
        )
        return np.einsum("ij,...jtp->...itp", matrix, values)

    def polar_derivative(self, values):
        """Return ``partial values / partial theta`` on the active nodes."""
        values = self._active_values(values)
        return np.einsum(
            "ab,...rbp->...rap", self.polar_derivative_matrix, values
        )

    def azimuthal_derivative(self, values):
        """Return the periodic ``phi`` derivative on the active nodes."""
        values = self._active_values(values)
        transformed = np.fft.fft(values, axis=-1)
        return np.fft.ifft(1j * self.azimuthal_modes * transformed, axis=-1)

    def gradient(self, values):
        """Return the physical gradient in the orthonormal spherical basis."""
        self._differentiable_scalar(values)
        inverse_r = 1.0 / self.r[:, None, None]
        inverse_r_sin = inverse_r / self.sin_theta[None, :, None]
        return SphericalVectorField(
            radial=self.radial_derivative(values),
            polar=inverse_r * self.polar_derivative(values),
            azimuthal=inverse_r_sin * self.azimuthal_derivative(values),
        )

    def gradient_squared(self, values):
        """Return ``grad(values) dot grad(values)`` without conjugation."""
        gradient = self.gradient(values)
        return gradient.radial**2 + gradient.polar**2 + gradient.azimuthal**2

    def gradient_norm_squared(self, values):
        """Return the Hermitian squared norm of ``grad(values)``."""
        gradient = self.gradient(values)
        return (
            np.abs(gradient.radial) ** 2
            + np.abs(gradient.polar) ** 2
            + np.abs(gradient.azimuthal) ** 2
        )

    def divergence(self, vector):
        """Return the strong-form spherical divergence of ``vector``."""
        (radial, polar, azimuthal), is_full = self._vector(vector, allow_full=True)
        r = self.r[:, None, None]
        input_r = self.full_r[:, None, None] if is_full else r
        sin_theta = self.sin_theta[None, :, None]
        radial_term = self.radial_derivative(input_r**2 * radial) / r**2
        polar_term = self.polar_derivative(sin_theta * polar) / (r * sin_theta)
        azimuthal_term = self.azimuthal_derivative(azimuthal) / (r * sin_theta)
        return radial_term + polar_term + azimuthal_term

    def laplacian(self, values):
        """Return the strong-form scalar Laplacian on the active nodes."""
        values, is_full = self._differentiable_scalar(values)
        active_values = (
            values[..., self.radial_indices, :, :] if is_full else values
        )
        input_r = self.full_r[:, None, None] if is_full else self.r[:, None, None]
        r = self.r[:, None, None]
        sin_theta = self.sin_theta[None, :, None]

        if is_full:
            first_radial = np.einsum(
                "ij,...jtp->...itp", self.full_radial_derivative_matrix, values
            )
        else:
            first_radial = self.radial_derivative(values)
        radial_term = self.radial_derivative(input_r**2 * first_radial) / r**2

        first_polar = self.polar_derivative(active_values)
        polar_term = self.polar_derivative(sin_theta * first_polar) / (
            r**2 * sin_theta
        )
        first_azimuthal = self.azimuthal_derivative(active_values)
        azimuthal_term = self.azimuthal_derivative(first_azimuthal) / (
            r**2 * sin_theta**2
        )
        return radial_term + polar_term + azimuthal_term

    def gradient_adjoint(self, vector):
        """Return the quadrature-weighted discrete adjoint of the gradient.

        This computes ``W**-1 G.conj().T W vector``, where ``G`` is the
        discrete physical gradient and ``W`` contains the spherical volume
        weights.  With vanishing boundary terms it approximates minus the
        continuum divergence, which is the sign needed by a variational GGA
        potential.
        """
        (radial, polar, azimuthal), _ = self._vector(vector)
        weights = self.volume_weights
        r = self.r[:, None, None]
        sin_theta = self.sin_theta[None, :, None]

        radial_part = np.einsum(
            "ij,...itp->...jtp",
            self.radial_derivative_matrix.conj(),
            weights * radial,
        )
        polar_part = np.einsum(
            "ab,...rap->...rbp",
            self.polar_derivative_matrix.conj(),
            weights * polar / r,
        )
        # The periodic Fourier differentiation matrix is skew-Hermitian.
        azimuthal_part = -self.azimuthal_derivative(
            weights * azimuthal / (r * sin_theta)
        )
        return (radial_part + polar_part + azimuthal_part) / weights

    def adjoint_divergence(self, vector):
        """Return the negative weighted adjoint of the discrete gradient."""
        return -self.gradient_adjoint(vector)

    def integrate(self, values):
        """Integrate scalar nodal values over the spherical volume."""
        values = self._scalar(values)
        return np.sum(values * self.volume_weights, axis=(-3, -2, -1))

    def inner_product(self, left, right):
        """Return the volume-quadrature inner product ``<left|right>``."""
        left = self._scalar(left, "left")
        right = self._scalar(right, "right")
        try:
            left, right = np.broadcast_arrays(left, right)
        except ValueError as error:
            raise ValueError("left and right must have broadcastable batch axes.") from error
        return self.integrate(left.conj() * right)
