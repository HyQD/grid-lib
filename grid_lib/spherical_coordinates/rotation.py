"""Rotations of complete complex spherical-harmonic expansions."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.linalg import expm
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.transform import Rotation


def _angular_cutoff(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("l_max must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("l_max must be non-negative")
    return result


def _rotation_matrix(value) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise TypeError("rotation must contain real numbers")
    matrix = np.array(raw, dtype=float, copy=True)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must contain only finite values")
    tolerance = 256.0 * np.finfo(float).eps
    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("rotation must be orthogonal")
    if not np.isclose(
        np.linalg.det(matrix),
        1.0,
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("rotation must be proper with determinant +1")
    matrix.setflags(write=False)
    return matrix


def _lm_states(l_max):
    return tuple(
        (l_value, m_value)
        for m_value in range(-l_max, l_max + 1)
        for l_value in range(abs(m_value), l_max + 1)
    )


def _angular_momentum_matrices(l_value):
    m_values = np.arange(-l_value, l_value + 1, dtype=float)
    dimension = m_values.size
    raising = np.zeros((dimension, dimension), dtype=complex)
    for column, m_value in enumerate(m_values[:-1]):
        raising[column + 1, column] = np.sqrt(
            l_value * (l_value + 1.0) - m_value * (m_value + 1.0)
        )
    lowering = raising.T.conj()
    x_component = 0.5 * (raising + lowering)
    y_component = (raising - lowering) / (2.0j)
    z_component = np.diag(m_values)
    return x_component, y_component, z_component


class SphericalHarmonicRotation:
    r"""Represent an active spatial rotation in a complete angular space.

    The state convention is

    .. math::

        (U(\mathcal R)\Psi)(\mathbf r)
        = \Psi(\mathcal R^{-1}\mathbf r).

    The channel matrix is built from
    ``exp(-1j * rotation_vector.dot(L))`` in every complete ``l`` block and
    stored in the standard grid-lib ordering: increasing signed ``m`` first,
    followed by increasing ``l``.  A complete ``m=-l,...,l`` space is
    required because a general rotation does not preserve a truncated
    magnetic subspace.

    Parameters
    ----------
    rotation : array_like
        Proper orthogonal active Cartesian rotation with shape ``(3, 3)``.
    l_max : int
        Largest spherical-harmonic angular momentum.
    """

    def __init__(self, rotation, *, l_max):
        self.l_max = _angular_cutoff(l_max)
        self.rotation = _rotation_matrix(rotation)
        self.orbital_lm = _lm_states(self.l_max)
        self.orbital_size = len(self.orbital_lm)
        self._channel_index = {
            channel: index for index, channel in enumerate(self.orbital_lm)
        }
        self._matrix = self._build_matrix()

    @classmethod
    def from_axis_angle(cls, axis, angle, *, l_max):
        """Construct a rotation from a Cartesian axis and angle in radians."""

        raw_axis = np.asarray(axis)
        if not np.issubdtype(raw_axis.dtype, np.number) or np.iscomplexobj(
            raw_axis
        ):
            raise TypeError("axis must contain real numbers")
        direction = np.array(raw_axis, dtype=float, copy=True)
        if direction.shape != (3,):
            raise ValueError("axis must have shape (3,)")
        if not np.all(np.isfinite(direction)):
            raise ValueError("axis must contain only finite values")
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("axis must be nonzero")
        if isinstance(angle, (bool, np.bool_)) or not np.isscalar(angle):
            raise TypeError("angle must be a real scalar")
        angle = float(angle)
        if not np.isfinite(angle):
            raise ValueError("angle must be finite")
        matrix = Rotation.from_rotvec(angle * direction / norm).as_matrix()
        return cls(matrix, l_max=l_max)

    def _build_matrix(self) -> csr_matrix:
        rotation_vector = Rotation.from_matrix(
            np.asarray(self.rotation)
        ).as_rotvec()
        rows = []
        columns = []
        data = []
        for l_value in range(self.l_max + 1):
            angular_momentum = _angular_momentum_matrices(l_value)
            generator = sum(
                rotation_vector[axis] * angular_momentum[axis]
                for axis in range(3)
            )
            block = expm(-1.0j * generator)
            indices = [
                self._channel_index[(l_value, m_value)]
                for m_value in range(-l_value, l_value + 1)
            ]
            for local_row, row in enumerate(indices):
                for local_column, column in enumerate(indices):
                    value = block[local_row, local_column]
                    if value != 0.0:
                        rows.append(row)
                        columns.append(column)
                        data.append(value)
        return coo_matrix(
            (data, (rows, columns)),
            shape=(self.orbital_size, self.orbital_size),
        ).tocsr()

    def channel_matrix(self) -> csr_matrix:
        """Return a defensive sparse copy of the angular rotation matrix."""

        return self._matrix.copy()

    def apply(self, states) -> np.ndarray:
        """Act on arrays whose final axis contains all angular channels."""

        values = np.asarray(states)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("states must be numeric")
        if values.ndim == 0 or values.shape[-1] != self.orbital_size:
            raise ValueError(
                "states must have final dimension " f"{self.orbital_size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("states must contain only finite values")
        flat = values.reshape(-1, self.orbital_size)
        return self._matrix.dot(flat.T).T.reshape(values.shape)

    __call__ = apply

    def inverse(self):
        """Return the inverse active rotation."""

        return type(self)(np.asarray(self.rotation).T, l_max=self.l_max)

    @property
    def metadata(self) -> dict:
        """Return a JSON-compatible rotation description."""

        return {
            "convention": "active: (U(R) psi)(r) = psi(R^-1 r)",
            "rotation": np.asarray(self.rotation).tolist(),
            "l_max": self.l_max,
            "channel_order": "signed_m_then_l",
            "complete_m_space": True,
        }


__all__ = ["SphericalHarmonicRotation"]
