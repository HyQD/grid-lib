import numpy as np
import pytest

from grid_lib.pseudospectral_grids import Grid1D


def test_system_grid_1d_sinc_shapes():
    system = Grid1D(("sinc", {"x0": -1.0, "xN": 1.0, "N": 11}))

    assert system.x.shape == (11,)
    assert system.weights.shape == (11,)
    assert system.D1.shape == (11, 11)
    assert system.D2.shape == (11, 11)


def test_system_grid_1d_gll_boundary_trimming():
    with_boundaries = Grid1D(("gll", {"N": 6, "x0": 0.0, "xN": 2.0, "remove_boundaries": False}))
    without_boundaries = Grid1D(("gll", {"N": 6, "x0": 0.0, "xN": 2.0, "remove_boundaries": True}))

    assert without_boundaries.n_grid == with_boundaries.n_grid - 2
    assert without_boundaries.D1.shape[0] == with_boundaries.D1.shape[0] - 2


def test_system_grid_1d_femdvr_uniform_matches_expected_size():
    system = Grid1D(
        (
            "femdvr_uniform",
            {
                "x_min": -2.0,
                "x_max": 2.0,
                "n_elements": 2,
                "points_per_element": 4,
            },
        )
    )

    # FEM-DVR has shared endpoints across neighboring elements.
    n_total = 2 * (4 - 1) + 1
    assert system.grid.r.size == n_total
    assert system.n_grid == n_total - 2


def test_system_grid_1d_rejects_invalid_basis():
    with pytest.raises(ValueError, match="Unknown grid_basis"):
        Grid1D(("invalid", {}))


def test_system_grid_1d_validates_required_uniform_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        Grid1D(("femdvr_uniform", {"x_min": -1.0, "x_max": 1.0}))


def test_system_grid_1d_dx_for_uniform_sinc_grid():
    system = Grid1D(("sinc", {"x0": 0.0, "xN": 2.0, "N": 5}))
    assert np.isclose(system.dx, 0.5)
