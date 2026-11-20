"""Coordinate operators for reduced-radial spherical-harmonic states."""

from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix


def _angular_cutoffs(l_max, m_max):
    if isinstance(l_max, (bool, np.bool_)) or not isinstance(l_max, Integral):
        raise TypeError("l_max must be an integer")
    l_max = int(l_max)
    if l_max < 0:
        raise ValueError("l_max must be non-negative")
    if m_max is None:
        m_max = l_max
    if isinstance(m_max, (bool, np.bool_)) or not isinstance(m_max, Integral):
        raise TypeError("m_max must be an integer")
    m_max = int(m_max)
    if not 0 <= m_max <= l_max:
        raise ValueError("m_max must satisfy 0 <= m_max <= l_max")
    return l_max, m_max


def _lm_states(l_max, m_max):
    return tuple(
        (l_value, m_value)
        for m_value in range(-m_max, m_max + 1)
        for l_value in range(abs(m_value), l_max + 1)
    )


class SphericalReducedRadialOperators:
    r"""Apply axial coordinate operators to reduced-radial expansions.

    States use the convention

    .. math::

        \Psi(\mathbf r)
        = \frac{1}{r}\sum_{lm}u_{lm}(r)Y_{lm}(\hat{\mathbf r})

    and radial-first trailing shape ``(..., n_r, n_lm)``. Angular channels
    use the standard grid-lib ordering: increasing ``m`` first, followed by
    increasing ``l`` within each signed-``m`` block. Leading batch dimensions
    are arbitrary.

    Parameters
    ----------
    r : array_like
        Strictly increasing positive active radial nodes. Boundary nodes at
        ``r=0`` must be excluded.
    first_derivative : array_like
        Strong first-derivative matrix on the active radial nodes. Omitted
        boundary values are therefore treated as zero.
    l_max, m_max : int
        Orbital angular cutoffs. ``m_max`` defaults to ``l_max``.

    Notes
    -----
    The implemented actions are ``z = r cos(theta)`` and
    ``p_z = -i partial_z``. For the reduced radial functions,

    .. math::

        \partial_z
        = C\left(\partial_r-\frac{1}{r}\right)
          -\frac{B}{r},

    where ``C`` represents ``cos(theta)`` and ``B`` represents
    ``sin(theta) partial_theta`` in the spherical-harmonic basis.
    """

    def __init__(self, r, first_derivative, *, l_max, m_max=None):
        self.l_max, self.m_max = _angular_cutoffs(l_max, m_max)

        raw_radial = np.asarray(r)
        if not np.issubdtype(raw_radial.dtype, np.number) or np.iscomplexobj(
            raw_radial
        ):
            raise TypeError("r must contain real numbers")
        radial = np.array(raw_radial, dtype=float, copy=True)
        if radial.ndim != 1 or radial.size == 0:
            raise ValueError("r must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(radial)):
            raise ValueError("r must contain only finite values")
        if np.any(radial <= 0.0) or np.any(np.diff(radial) <= 0.0):
            raise ValueError("r must be positive and strictly increasing")

        derivative = np.array(first_derivative, copy=True)
        if not np.issubdtype(derivative.dtype, np.number):
            raise TypeError("first_derivative must be numeric")
        if derivative.shape != (radial.size, radial.size):
            raise ValueError(
                "first_derivative must have shape (r.size, r.size)"
            )
        if not np.all(np.isfinite(derivative)):
            raise ValueError("first_derivative must contain only finite values")

        radial.setflags(write=False)
        derivative.setflags(write=False)
        self.r = radial
        self.first_derivative = derivative
        self.orbital_lm = _lm_states(self.l_max, self.m_max)
        self.orbital_shape = (self.r.size, len(self.orbital_lm))
        self._channel_index = {
            channel: index for index, channel in enumerate(self.orbital_lm)
        }
        (
            self._cos_theta,
            self._sin_theta_dtheta,
        ) = self._build_angular_matrices()
        self._fixed_m_indices = {}
        self._fixed_m_angular = {}

    def __repr__(self):
        return (
            "SphericalReducedRadialOperators("
            f"n_r={self.r.size}, l_max={self.l_max}, m_max={self.m_max})"
        )

    def _build_angular_matrices(self):
        rows = []
        columns = []
        cos_coefficients = []
        derivative_coefficients = []
        for column, (l_value, m_value) in enumerate(self.orbital_lm):
            upward = (l_value + 1, m_value)
            if upward in self._channel_index:
                coefficient = np.sqrt(
                    ((l_value + 1) ** 2 - m_value**2)
                    / ((2 * l_value + 1) * (2 * l_value + 3))
                )
                rows.append(self._channel_index[upward])
                columns.append(column)
                cos_coefficients.append(coefficient)
                derivative_coefficients.append(l_value * coefficient)

            downward = (l_value - 1, m_value)
            if downward in self._channel_index:
                coefficient = np.sqrt(
                    (l_value**2 - m_value**2)
                    / ((2 * l_value - 1) * (2 * l_value + 1))
                )
                rows.append(self._channel_index[downward])
                columns.append(column)
                cos_coefficients.append(coefficient)
                derivative_coefficients.append(-(l_value + 1) * coefficient)

        shape = (len(self.orbital_lm),) * 2
        cos_theta = csr_matrix(
            (cos_coefficients, (rows, columns)), shape=shape
        )
        sin_theta_dtheta = csr_matrix(
            (derivative_coefficients, (rows, columns)), shape=shape
        )
        return cos_theta, sin_theta_dtheta

    def _validated_m(self, m):
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
            raise TypeError("m must be an integer")
        m = int(m)
        if abs(m) > self.m_max:
            raise ValueError(f"abs(m) must not exceed {self.m_max}")
        return m

    def _indices_for_m(self, m):
        m = self._validated_m(m)
        if m not in self._fixed_m_indices:
            indices = np.asarray(
                [
                    index
                    for index, (_, magnetic) in enumerate(self.orbital_lm)
                    if magnetic == m
                ],
                dtype=int,
            )
            indices.setflags(write=False)
            self._fixed_m_indices[m] = indices
        return self._fixed_m_indices[m]

    def _angular_matrix(self, name, m=None):
        if name == "cos_theta":
            matrix = self._cos_theta
        elif name == "sin_theta_dtheta":
            matrix = self._sin_theta_dtheta
        else:
            raise ValueError(f"unknown angular matrix {name!r}")
        if m is None:
            return matrix
        m = self._validated_m(m)
        key = (name, m)
        if key not in self._fixed_m_angular:
            indices = self._indices_for_m(m)
            self._fixed_m_angular[key] = matrix[indices][:, indices].tocsr()
        return self._fixed_m_angular[key]

    def cos_theta_matrix(self, m=None):
        r"""Return a sparse copy of ``cos(theta)`` in channel space."""

        return self._angular_matrix("cos_theta", m).copy()

    def sin_theta_dtheta_matrix(self, m=None):
        r"""Return a sparse copy of ``sin(theta) partial_theta``."""

        return self._angular_matrix("sin_theta_dtheta", m).copy()

    def fixed_m_shape(self, m):
        """Return the radial-first state shape for signed magnetic number ``m``."""

        m = self._validated_m(m)
        return (self.r.size, self.l_max - abs(m) + 1)

    @staticmethod
    def _validated_values(values, shape, name="states"):
        result = np.asarray(values)
        if not np.issubdtype(result.dtype, np.number):
            raise TypeError(f"{name} must be numeric")
        if result.ndim < 2 or result.shape[-2:] != shape:
            raise ValueError(f"{name} must have trailing shape {shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result

    def validate_state(self, states):
        """Return a validated full-angular radial-first state array."""

        return self._validated_values(states, self.orbital_shape)

    def validate_fixed_m_state(self, states, m=0):
        """Return a validated fixed-``m`` radial-first state array."""

        return self._validated_values(states, self.fixed_m_shape(m))

    @staticmethod
    def _apply_angular(values, matrix):
        flat = values.reshape(-1, values.shape[-1])
        return matrix.dot(flat.T).T.reshape(values.shape)

    def _apply_z(self, values, cos_theta):
        result = self._apply_angular(values, cos_theta)
        radial_shape = (1,) * (values.ndim - 2) + (self.r.size, 1)
        return result * self.r.reshape(radial_shape)

    def _apply_pz(self, values, cos_theta, sin_theta_dtheta):
        derivative = np.einsum(
            "ab,...bj->...aj",
            self.first_derivative,
            values,
            optimize=True,
        )
        radial_shape = (1,) * (values.ndim - 2) + (self.r.size, 1)
        divided = values / self.r.reshape(radial_shape)
        derivative_tilde = derivative - divided
        result = self._apply_angular(derivative_tilde, cos_theta)
        result -= self._apply_angular(divided, sin_theta_dtheta)
        return -1j * result

    def apply_z(self, states):
        r"""Apply ``z = r cos(theta)`` to full angular expansions."""

        values = self.validate_state(states)
        return self._apply_z(values, self._cos_theta)

    def apply_z_fixed_m(self, states, m=0):
        r"""Apply ``z = r cos(theta)`` in one signed-``m`` block."""

        values = self.validate_fixed_m_state(states, m)
        return self._apply_z(values, self._angular_matrix("cos_theta", m))

    def apply_pz(self, states):
        r"""Apply ``p_z = -i partial_z`` to full angular expansions."""

        values = self.validate_state(states)
        return self._apply_pz(
            values,
            self._cos_theta,
            self._sin_theta_dtheta,
        )

    def apply_pz_fixed_m(self, states, m=0):
        r"""Apply ``p_z = -i partial_z`` in one signed-``m`` block."""

        values = self.validate_fixed_m_state(states, m)
        return self._apply_pz(
            values,
            self._angular_matrix("cos_theta", m),
            self._angular_matrix("sin_theta_dtheta", m),
        )
