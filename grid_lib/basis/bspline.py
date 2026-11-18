"""Locally supported B-spline bases on finite radial intervals."""

from numbers import Integral

import numpy as np
from scipy.interpolate import BSpline


def _validated_integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validated_boundary_flags(value):
    try:
        flags = tuple(value)
    except TypeError as error:
        raise TypeError(
            "vanish_at_boundaries must contain two booleans"
        ) from error
    if len(flags) != 2 or any(
        not isinstance(flag, (bool, np.bool_)) for flag in flags
    ):
        raise TypeError("vanish_at_boundaries must contain two booleans")
    return bool(flags[0]), bool(flags[1])


class BSplineRadialBasis:
    r"""Open, clamped B-spline basis on a finite radial interval.

    Interior knot multiplicity is chosen from the requested continuity.  A
    degree-``p`` basis with continuity ``c`` repeats every interior radial
    node ``p - c`` times.  The basis therefore has compact support while its
    first derivatives can be evaluated at arbitrary radii independently of
    any integration rule.

    By default, the first and last clamped B-splines are omitted.  This
    imposes homogeneous values at both finite boundaries and matches the
    reduced radial functions used by finite-box spherical calculations.  The
    boundary choice does not impose the stronger angular-momentum-dependent
    condition :math:`u_l(r) \sim r^{l+1}` at the origin.

    Parameters
    ----------
    nodes : array_like
        Strictly increasing, nonnegative radial breakpoints.  The first and
        last entries define the finite interval.
    degree : int, optional
        Polynomial degree.  It must be positive.
    continuity : int, optional
        Continuity at every interior node.  It must satisfy
        ``0 <= continuity < degree``.
    vanish_at_boundaries : pair of bool, optional
        Whether to omit the clamped basis function that is nonzero at the
        inner and outer boundary, respectively.
    """

    def __init__(
        self,
        nodes,
        *,
        degree=7,
        continuity=1,
        vanish_at_boundaries=(True, True),
    ):
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("nodes must be a one-dimensional array of length >= 2")
        if np.any(~np.isfinite(nodes)):
            raise ValueError("nodes must be finite")
        if nodes[0] < 0.0:
            raise ValueError("radial nodes must be nonnegative")
        if np.any(np.diff(nodes) <= 0.0):
            raise ValueError("nodes must be strictly increasing")

        degree = _validated_integer(degree, "degree", 1)
        continuity = _validated_integer(continuity, "continuity", 0)
        if continuity >= degree:
            raise ValueError("continuity must be less than degree")
        vanish_at_boundaries = _validated_boundary_flags(
            vanish_at_boundaries
        )

        interior_multiplicity = degree - continuity
        knots = np.concatenate(
            (
                np.repeat(nodes[0], degree + 1),
                np.repeat(nodes[1:-1], interior_multiplicity),
                np.repeat(nodes[-1], degree + 1),
            )
        )
        full_size = knots.size - degree - 1
        first = int(vanish_at_boundaries[0])
        stop = full_size - int(vanish_at_boundaries[1])
        if first >= stop:
            raise ValueError("boundary constraints remove every B-spline")
        kept_full_indices = np.arange(first, stop, dtype=int)

        support_left = knots[kept_full_indices]
        support_right = knots[kept_full_indices + degree + 1]
        interval_active_indices = []
        for left, right in zip(nodes[:-1], nodes[1:]):
            active = (support_left < right) & (support_right > left)
            indices = np.flatnonzero(active)
            indices.setflags(write=False)
            interval_active_indices.append(indices)

        self.nodes = np.array(nodes, copy=True)
        self.knots = np.array(knots, copy=True)
        self.degree = degree
        self.continuity = continuity
        self.vanish_at_boundaries = vanish_at_boundaries
        self.size = kept_full_indices.size
        self.full_size = full_size
        self.n_intervals = nodes.size - 1
        self.r_min = float(nodes[0])
        self.r_max = float(nodes[-1])
        self._kept_full_indices = kept_full_indices
        self._interval_active_indices = tuple(interval_active_indices)
        for array in (self.nodes, self.knots, self._kept_full_indices):
            array.setflags(write=False)

    def locate(self, radii):
        """Return the interval containing each radius, or ``-1`` outside."""

        radii = np.asarray(radii, dtype=float)
        if np.any(~np.isfinite(radii)):
            raise ValueError("radii must be finite")
        intervals = np.searchsorted(self.nodes, radii, side="right") - 1
        intervals = np.where(
            radii == self.nodes[-1], self.n_intervals - 1, intervals
        )
        outside = (radii < self.nodes[0]) | (radii > self.nodes[-1])
        return np.where(outside, -1, intervals).astype(int, copy=False)

    def active_indices(self, interval_index):
        """Return retained basis indices active inside one radial interval."""

        interval_index = self._validated_interval_index(interval_index)
        return self._interval_active_indices[interval_index]

    def evaluate(self, radii):
        """Evaluate all retained basis functions and their first derivatives.

        The returned arrays have shape ``radii.shape + (size,)``.
        """

        radii = self._validated_radii(radii)
        return self._evaluate_full_indices(
            radii, self._kept_full_indices
        )

    def evaluate_interval(self, interval_index, radii):
        """Evaluate only basis functions active in one radial interval.

        Returns
        -------
        values, derivatives, indices
            Values and first radial derivatives have shape
            ``radii.shape + (n_active,)``.  ``indices`` maps their final axis
            to the retained global basis ordering.
        """

        interval_index = self._validated_interval_index(interval_index)
        radii = self._validated_radii(radii)
        left = self.nodes[interval_index]
        right = self.nodes[interval_index + 1]
        scale = max(1.0, abs(left), abs(right))
        tolerance = 64.0 * np.finfo(float).eps * scale
        if np.any(radii < left - tolerance) or np.any(radii > right + tolerance):
            raise ValueError("radii lie outside the requested interval")
        radii = np.clip(radii, left, right)
        active = self._interval_active_indices[interval_index]
        full_indices = self._kept_full_indices[active]
        values, derivatives = self._evaluate_full_indices(
            radii, full_indices
        )
        return values, derivatives, active

    def _validated_interval_index(self, value):
        value = _validated_integer(value, "interval_index", 0)
        if value >= self.n_intervals:
            raise IndexError("interval_index is outside the radial basis")
        return value

    def _validated_radii(self, radii):
        radii = np.asarray(radii, dtype=float)
        if np.any(~np.isfinite(radii)):
            raise ValueError("radii must be finite")
        if np.any(radii < self.r_min) or np.any(radii > self.r_max):
            raise ValueError("radii lie outside the radial basis")
        return radii

    def _evaluate_full_indices(self, radii, full_indices):
        shape = radii.shape
        flat_radii = radii.reshape(-1)
        value_design = BSpline.design_matrix(
            flat_radii,
            self.knots,
            self.degree,
            extrapolate=False,
        )
        values = value_design[:, full_indices].toarray()

        lower_design = BSpline.design_matrix(
            flat_radii,
            self.knots,
            self.degree - 1,
            extrapolate=False,
        )
        lower_left = lower_design[:, full_indices].toarray()
        lower_right = lower_design[:, full_indices + 1].toarray()
        # With the degree-p clamped knot vector, the degree-(p-1) design
        # matrix contains one additional zero-support endpoint function.
        # SciPy assigns the exact outer endpoint to that final column.  The
        # derivative of the degree-p basis instead needs the limit from the
        # final nonempty interval, represented by the preceding column.
        at_outer_boundary = flat_radii == self.r_max
        lower_left[at_outer_boundary, :] = (
            full_indices == self.full_size - 1
        )
        lower_right[at_outer_boundary, :] = (
            full_indices + 1 == self.full_size - 1
        )
        left_denominators = (
            self.knots[full_indices + self.degree]
            - self.knots[full_indices]
        )
        right_denominators = (
            self.knots[full_indices + self.degree + 1]
            - self.knots[full_indices + 1]
        )
        left_coefficients = np.divide(
            self.degree,
            left_denominators,
            out=np.zeros_like(left_denominators),
            where=left_denominators != 0.0,
        )
        right_coefficients = np.divide(
            self.degree,
            right_denominators,
            out=np.zeros_like(right_denominators),
            where=right_denominators != 0.0,
        )
        derivatives = (
            lower_left * left_coefficients
            - lower_right * right_coefficients
        )
        return (
            values.reshape(shape + (full_indices.size,)),
            derivatives.reshape(shape + (full_indices.size,)),
        )
