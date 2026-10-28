"""Actions of local potentials represented by spherical multipoles."""

from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix

from .angular_momentum import LM_to_I, get_y_value, number_of_lm_states
from .potentials import (
    clamped_molecular_potential_Poisson,
    clamped_molecular_potential_quadrature,
)


def _angular_cutoffs(l_max, m_max, L_max, M_max):
    for name, value in (("l_max", l_max), ("m_max", m_max)):
        if not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    l_max = int(l_max)
    m_max = int(m_max)
    if l_max < 0 or not 0 <= m_max <= l_max:
        raise ValueError("Require l_max >= 0 and 0 <= m_max <= l_max")

    if L_max is None:
        L_max = 2 * l_max
    elif not isinstance(L_max, Integral):
        raise TypeError("L_max must be an integer")
    L_max = int(L_max)
    if M_max is None:
        M_max = min(L_max, 2 * m_max)
    elif not isinstance(M_max, Integral):
        raise TypeError("M_max must be an integer")
    M_max = int(M_max)
    if L_max < 0 or not 0 <= M_max <= L_max:
        raise ValueError("Require L_max >= 0 and 0 <= M_max <= L_max")
    return l_max, m_max, L_max, M_max


def _lm_states(l_max, m_max):
    return tuple(
        (l, m)
        for m in range(-m_max, m_max + 1)
        for l in range(abs(m), l_max + 1)
    )


class SphericalMultipolePotential:
    r"""Local scalar potential acting on spherical-harmonic channels.

    The potential and a reduced radial orbital are represented as

    .. math::

        V(\mathbf r) = \sum_{LM} V_{LM}(r)Y_{LM}(\Omega),
        \qquad
        \phi(\mathbf r) = \frac{1}{r}\sum_{lm}u_{lm}(r)Y_{lm}(\Omega).

    Arrays supplied to :meth:`apply` use radial-first storage with shape
    ``(..., n_r, n_lm)``.  The leading axes are arbitrary batch axes.  No
    radial quadrature weights are included in the coefficients or action.

    Parameters
    ----------
    r : array_like
        Strictly increasing non-negative radial grid.
    coefficients : array_like
        Multipole coefficients with shape ``(n_LM, n_r)`` in ``LM_to_I``
        ordering.
    l_max, m_max : int
        Orbital angular cutoffs.
    L_max, M_max : int
        Potential multipole cutoffs.
    """

    def __init__(
        self,
        r,
        coefficients,
        *,
        l_max,
        m_max=None,
        L_max=None,
        M_max=None,
    ):
        if m_max is None:
            m_max = l_max
        self.l_max, self.m_max, self.L_max, self.M_max = _angular_cutoffs(
            l_max, m_max, L_max, M_max
        )

        radial_grid = np.array(r, dtype=float, copy=True)
        if radial_grid.ndim != 1 or radial_grid.size == 0:
            raise ValueError("r must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(radial_grid)):
            raise ValueError("r must contain only finite values")
        if np.any(radial_grid < 0.0) or np.any(np.diff(radial_grid) <= 0.0):
            raise ValueError("r must be non-negative and strictly increasing")

        values = np.array(coefficients, copy=True)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("coefficients must be numeric")
        expected = (
            number_of_lm_states(self.L_max, self.M_max),
            radial_grid.size,
        )
        if values.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError("coefficients must contain only finite values")

        radial_grid.setflags(write=False)
        values.setflags(write=False)
        self.r = radial_grid
        self.coefficients = values
        self.orbital_lm = _lm_states(self.l_max, self.m_max)
        self.multipole_lm = _lm_states(self.L_max, self.M_max)
        self.orbital_shape = (self.r.size, len(self.orbital_lm))
        self._couplings = self._build_active_couplings()

    @classmethod
    def from_point_charges(
        cls,
        r,
        positions,
        charges,
        *,
        l_max,
        m_max=None,
        L_max=None,
        M_max=None,
    ):
        """Construct a point-charge potential from analytic multipoles."""
        if m_max is None:
            m_max = l_max
        l_max, m_max, L_max, M_max = _angular_cutoffs(
            l_max, m_max, L_max, M_max
        )
        coefficients = clamped_molecular_potential_quadrature(
            r, positions, charges, L_max, M_max
        )
        return cls(
            r,
            coefficients,
            l_max=l_max,
            m_max=m_max,
            L_max=L_max,
            M_max=M_max,
        )

    from_point_charges_analytic = from_point_charges

    @classmethod
    def from_point_charges_poisson(
        cls,
        r,
        W,
        weights,
        positions,
        charges,
        *,
        l_max,
        m_max=None,
        L_max=None,
        M_max=None,
        force=False,
    ):
        """Construct a point-charge potential from Poisson-solved kernels."""
        if m_max is None:
            m_max = l_max
        l_max, m_max, L_max, M_max = _angular_cutoffs(
            l_max, m_max, L_max, M_max
        )
        coefficients = clamped_molecular_potential_Poisson(
            r,
            W,
            weights,
            positions,
            charges,
            L_max,
            M_max,
            force=force,
        )
        return cls(
            r,
            coefficients,
            l_max=l_max,
            m_max=m_max,
            L_max=L_max,
            M_max=M_max,
        )

    @property
    def is_axisymmetric(self):
        """Whether all numerically nonzero multipoles have ``M = 0``."""
        scale = max(1.0, float(np.max(np.abs(self.coefficients))))
        tolerance = 100.0 * np.finfo(float).eps * scale
        return all(
            M == 0 or np.all(np.abs(self.coefficients[index]) <= tolerance)
            for index, (_, M) in enumerate(self.multipole_lm)
        )

    def _build_active_couplings(self):
        n_lm = len(self.orbital_lm)
        couplings = []
        for component, (L, M) in enumerate(self.multipole_lm):
            if not np.any(self.coefficients[component] != 0):
                continue
            rows = []
            columns = []
            data = []
            for row, (l1, m1) in enumerate(self.orbital_lm):
                m2 = m1 - M
                if not -self.m_max <= m2 <= self.m_max:
                    continue
                for l2 in range(abs(m2), self.l_max + 1):
                    value = get_y_value(L, M, l1, m1, l2, m2)
                    if value != 0.0:
                        rows.append(row)
                        columns.append(LM_to_I(l2, m2, self.l_max, self.m_max))
                        data.append(value)
            matrix = csr_matrix((data, (rows, columns)), shape=(n_lm, n_lm))
            if matrix.nnz:
                couplings.append((component, matrix))
        return tuple(couplings)

    def apply(self, orbitals):
        """Apply the local potential to one or more orbital arrays."""
        values = np.asarray(orbitals)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("orbitals must be numeric")
        if values.ndim < 2 or values.shape[-2:] != self.orbital_shape:
            raise ValueError(
                "orbitals must have trailing shape " f"{self.orbital_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("orbitals must contain only finite values")

        dtype = np.result_type(values.dtype, self.coefficients.dtype, float)
        result = np.zeros(values.shape, dtype=dtype)
        flat = values.reshape(-1, values.shape[-1])
        radial_axis = (1,) * (values.ndim - 2) + (self.r.size, 1)
        for component, coupling in self._couplings:
            transformed = coupling.dot(flat.T).T.reshape(values.shape)
            coefficient = self.coefficients[component].reshape(radial_axis)
            result += coefficient * transformed
        return result

    __call__ = apply

    def angular_matrix(self):
        """Return the pointwise channel matrices with shape ``(n_r,n_lm,n_lm)``."""
        n_lm = len(self.orbital_lm)
        dtype = np.result_type(self.coefficients.dtype, float)
        matrix = np.zeros((self.r.size, n_lm, n_lm), dtype=dtype)
        for component, coupling in self._couplings:
            matrix += self.coefficients[component, :, None, None] * coupling.toarray()
        return matrix

    def fixed_m_matrix(self, m):
        """Return an axial pointwise block for a declared magnetic number."""
        if not self.is_axisymmetric:
            raise ValueError("fixed-m blocks require an axisymmetric potential")
        if not isinstance(m, Integral):
            raise TypeError("m must be an integer")
        m = int(m)
        if abs(m) > self.m_max:
            raise ValueError("Require abs(m) <= m_max")
        indices = [
            LM_to_I(l, m, self.l_max, self.m_max)
            for l in range(abs(m), self.l_max + 1)
        ]
        return self.angular_matrix()[:, indices][:, :, indices]
