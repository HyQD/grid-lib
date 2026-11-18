"""Geometry-only Becke partitions for distinct Cartesian centres."""

from numbers import Integral

import numpy as np


class BeckePartition:
    r"""Smooth geometry-only partition of unity around atomic centres.

    For each ordered pair of centres, the elliptic coordinate and switch
    favouring centre ``A`` are

    .. math::

        \mu_{AB} = \frac{r_A-r_B}{R_{AB}}, \qquad
        s_{AB} = \frac{1-f(\mu_{AB})}{2}.

    The polynomial ``f(x) = (3*x - x**3)/2`` is applied repeatedly before
    pair switches are multiplied and normalized.  Atomic-size corrections
    are intentionally not part of this first reusable implementation.

    Parameters
    ----------
    centers : array_like, shape (n_centers, 3)
        Distinct finite Cartesian centres.
    switch_iterations : int, optional
        Number of repeated smoothing operations.  The standard Becke value
        is three.
    """

    def __init__(self, centers, switch_iterations=3):
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1:] != (3,):
            raise ValueError("centers must have shape (n_centers, 3)")
        if centers.shape[0] == 0:
            raise ValueError("at least one center is required")
        if np.any(~np.isfinite(centers)):
            raise ValueError("centers must be finite")
        if isinstance(switch_iterations, (bool, np.bool_)) or not isinstance(
            switch_iterations, Integral
        ):
            raise TypeError("switch_iterations must be an integer")
        switch_iterations = int(switch_iterations)
        if switch_iterations < 1:
            raise ValueError("switch_iterations must be positive")

        vectors = centers[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(vectors, axis=-1)
        off_diagonal = ~np.eye(len(centers), dtype=bool)
        if np.any(distances[off_diagonal] == 0.0):
            raise ValueError("centers must be distinct")

        self.centers = np.array(centers, copy=True)
        self.pair_distances = distances
        self.switch_iterations = switch_iterations
        self.centers.setflags(write=False)
        self.pair_distances.setflags(write=False)

    @property
    def n_centers(self):
        """Number of centres in the partition."""

        return self.centers.shape[0]

    def weights(self, points):
        """Evaluate normalized centre weights at Cartesian points."""

        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3)")
        if np.any(~np.isfinite(points)):
            raise ValueError("points must be finite")

        displacements = points[..., None, :] - self.centers
        center_distances = np.linalg.norm(displacements, axis=-1)
        safe_pair_distances = np.array(self.pair_distances, copy=True)
        diagonal = np.diag_indices(self.n_centers)
        safe_pair_distances[diagonal] = 1.0
        pair_variables = (
            center_distances[..., :, None] - center_distances[..., None, :]
        ) / safe_pair_distances
        pair_variables = np.clip(pair_variables, -1.0, 1.0)
        pair_variables[..., diagonal[0], diagonal[1]] = 0.0

        smoothed = pair_variables
        for _ in range(self.switch_iterations):
            smoothed = 1.5 * smoothed - 0.5 * smoothed**3
        switches = 0.5 * (1.0 - smoothed)
        switches[..., diagonal[0], diagonal[1]] = 1.0
        unnormalized = np.prod(switches, axis=-1)
        normalization = np.sum(unnormalized, axis=-1, keepdims=True)
        if np.any(normalization <= 0.0):
            raise FloatingPointError("Becke weight normalization vanished")
        return unnormalized / normalization
