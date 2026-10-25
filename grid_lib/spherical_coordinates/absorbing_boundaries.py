"""Absorbing-boundary profiles for radial spherical-coordinate grids."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


def mask_function(r, r_max, r0, n=4):
    r"""Return a radial cosine mask.

    The mask is one for ``r < r0`` and zero at ``r_max``. Between these
    radii it is

    ``cos(pi * (r - r0) / (2 * (r_max - r0)))**(1 / n)``.

    This function retains the established positional API from
    :mod:`grid_lib.spherical_coordinates.utils`.
    """

    nodes = _radial_nodes(r)
    r_max = _positive_real(r_max, "r_max")
    r0 = _finite_real(r0, "r0")
    n = _positive_real(n, "n")
    if r0 < 0.0:
        raise ValueError("r0 must be nonnegative")
    if r0 >= r_max:
        raise ValueError("r0 must be smaller than r_max")
    if np.any(nodes < 0.0) or np.any(nodes > r_max):
        raise ValueError("radial nodes must lie in the interval [0, r_max]")

    values = np.ones(nodes.shape, dtype=float)
    absorbing = nodes >= r0
    phase = np.pi * (nodes[absorbing] - r0) / (2.0 * (r_max - r0))
    values[absorbing] = np.cos(phase) ** (1.0 / n)
    values[nodes == r_max] = 0.0
    return values


@dataclass(frozen=True)
class RadialCosineMask:
    r"""Cosine mask equal to one below ``r0`` and zero at ``r_max``."""

    r0: float
    n: float = 4.0

    def __post_init__(self) -> None:
        r0 = _finite_real(self.r0, "r0")
        n = _positive_real(self.n, "n")
        if r0 < 0.0:
            raise ValueError("r0 must be nonnegative")
        object.__setattr__(self, "r0", r0)
        object.__setattr__(self, "n", n)

    def __call__(self, radial_grid, r_max):
        """Evaluate the mask on ``radial_grid``."""

        return mask_function(radial_grid, r_max, self.r0, self.n)

    values = __call__


@dataclass(frozen=True)
class CubicCAP:
    r"""Cubic complex absorbing potential near the outer radial boundary.

    The potential is zero through ``alpha * r_max`` and is
    ``-1j * eta * (r - alpha * r_max)**3`` beyond that point.
    """

    alpha: float
    eta: float = 1.0

    def __post_init__(self) -> None:
        alpha = _finite_real(self.alpha, "alpha")
        eta = _positive_real(self.eta, "eta")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "eta", eta)

    def __call__(self, radial_grid, r_max):
        """Evaluate the CAP on ``radial_grid``."""

        nodes = _radial_nodes(radial_grid)
        r_max = _positive_real(r_max, "r_max")
        if np.any(nodes < 0.0) or np.any(nodes > r_max):
            raise ValueError("radial nodes must lie in the interval [0, r_max]")
        displacement = np.maximum(nodes - self.alpha * r_max, 0.0)
        return -1j * self.eta * displacement**3

    values = __call__


def _radial_nodes(values) -> np.ndarray:
    nodes = np.asarray(values)
    if (
        nodes.ndim != 1
        or np.iscomplexobj(nodes)
        or not np.issubdtype(nodes.dtype, np.number)
        or not np.all(np.isfinite(nodes))
    ):
        raise ValueError("radial_grid must be a finite real one-dimensional array")
    return nodes.astype(float, copy=False)


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_real(value, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
