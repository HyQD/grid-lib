from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .setup_grids import setup_femdvr_uniform, setup_grid

GridBasis1D = Literal["sinc", "gll", "femdvr", "femdvr_uniform"]


class Grid1D:
    """Convenience wrapper for one-dimensional pseudospectral grids.

    Parameters
    ----------
    grid
        Tuple of (grid_basis, grid_params). Supported grid bases are
        "sinc", "gll", "femdvr", and "femdvr_uniform".

    Notes
    -----
    If "remove_boundaries" is provided in grid_params, it is applied only to
    non-sinc grids and defaults to True.
    """

    valid_grid_bases = {"sinc", "gll", "femdvr", "femdvr_uniform"}

    def __init__(
        self,
        grid: tuple[GridBasis1D, dict[str, Any]],
    ) -> None:
        grid_basis, grid_params = grid

        self.grid_basis = grid_basis
        self.grid_params = dict(grid_params)
        self.remove_boundaries = self._pop_remove_boundaries()

        self._validate_grid_basis()
        self.grid = self._setup_grid()
        self.x, self.weights, self.D1, self.D2 = self._extract_grid_arrays()
        self.n_grid = self.x.size
        self.dx = self._compute_dx()

    def _validate_grid_basis(self) -> None:
        if self.grid_basis not in self.valid_grid_bases:
            valid = ", ".join(sorted(self.valid_grid_bases))
            raise ValueError(
                f"Unknown grid_basis {self.grid_basis!r}. "
                f"Supported grid bases are: {valid}."
            )

    def _pop_remove_boundaries(self) -> bool:
        return bool(self.grid_params.pop("remove_boundaries", True))

    def _setup_grid(self):
        if self.grid_basis == "femdvr_uniform":
            params = self._normalized_femdvr_uniform_params()
            return setup_femdvr_uniform(**params)

        return setup_grid(self.grid_basis, self.grid_params)

    def _normalized_femdvr_uniform_params(self) -> dict[str, Any]:
        params = dict(self.grid_params)

        required = {"x_min", "x_max", "n_elements", "points_per_element"}
        missing = required.difference(params)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(
                "femdvr_uniform grid_params missing required keys: "
                f"{missing_keys}."
            )

        params.setdefault("symmetrize", False)
        return params

    def _extract_grid_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.grid_basis == "sinc":
            x = np.asarray(self.grid.x)
            weights = np.asarray(self.grid.weights)
            D1 = np.asarray(self.grid.D1)
            D2 = np.asarray(self.grid.D2)
        else:
            x = np.asarray(self.grid.r)
            weights = np.asarray(self.grid.weights)
            D1 = np.asarray(self.grid.D1)
            D2 = np.asarray(self.grid.D2)

            if self.remove_boundaries:
                x = x[1:-1]
                weights = weights[1:-1]
                D1 = D1[1:-1, 1:-1]
                D2 = D2[1:-1, 1:-1]

        self._validate_grid_arrays(x, weights, D1, D2)
        return x, weights, D1, D2

    @staticmethod
    def _validate_grid_arrays(
        x: np.ndarray,
        weights: np.ndarray,
        D1: np.ndarray,
        D2: np.ndarray,
    ) -> None:
        if x.ndim != 1:
            raise ValueError("Grid points must be a one-dimensional array.")
        if weights.shape != x.shape:
            raise ValueError("Grid weights must have the same shape as x.")
        if D1.shape != (x.size, x.size):
            raise ValueError("D1 must have shape (n_grid, n_grid).")
        if D2.shape != (x.size, x.size):
            raise ValueError("D2 must have shape (n_grid, n_grid).")
        if x.size < 2:
            raise ValueError("Grid must contain at least two points.")

    def _compute_dx(self) -> float:
        return float(self.x[1] - self.x[0])


__all__ = ["GridBasis1D", "Grid1D"]
