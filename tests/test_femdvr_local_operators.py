import numpy as np
import pytest

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)


@pytest.fixture
def femdvr():
    return FEMDVR(
        nodes=np.array([0.0, 0.7, 2.0, 4.5]),
        n_points=np.array([4, 5, 6]),
        Mapping=Linear_map,
        element_class=GaussLegendreLobatto,
    )


def test_element_topology_and_left_bridge_ownership(femdvr):
    expected_maps = (
        np.array([0, 1, 2, 3]),
        np.array([3, 4, 5, 6, 7]),
        np.array([7, 8, 9, 10, 11, 12]),
    )

    assert femdvr.r.size == 13
    np.testing.assert_array_equal(femdvr.bridge_indices, [3, 7])
    for dof_map, expected, element in zip(
        femdvr.element_dof_maps, expected_maps, femdvr.dvr
    ):
        np.testing.assert_array_equal(dof_map, expected)
        np.testing.assert_allclose(femdvr.r[dof_map], element.r)

    ownership_count = np.zeros(femdvr.r.size, dtype=int)
    for dof_map, owned in zip(
        femdvr.element_dof_maps, femdvr.element_owned_dof_masks
    ):
        ownership_count[dof_map[owned]] += 1
    np.testing.assert_array_equal(ownership_count, 1)
    np.testing.assert_array_equal(femdvr.element_owned_dof_maps[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(femdvr.element_owned_dof_maps[1], [4, 5, 6, 7])
    np.testing.assert_array_equal(
        femdvr.element_owned_dof_maps[2], [8, 9, 10, 11, 12]
    )
    np.testing.assert_array_equal(
        femdvr.dof_owner,
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    )


def test_gather_and_scatter_add_bridge_contributions(femdvr):
    values = np.arange(2 * femdvr.r.size * 3).reshape(2, femdvr.r.size, 3)
    local_values = femdvr.element_values(values, axis=1)

    for local, dof_map in zip(local_values, femdvr.element_dof_maps):
        np.testing.assert_array_equal(local, values[:, dof_map, :])

    assembled = femdvr.assemble_element_vectors(local_values, axis=1)
    multiplicity = np.ones(femdvr.r.size, dtype=int)
    multiplicity[femdvr.bridge_indices] = 2
    np.testing.assert_array_equal(
        assembled, values * multiplicity[np.newaxis, :, np.newaxis]
    )


def test_local_weak_matrices_assemble_to_existing_global_operators(femdvr):
    mass = femdvr.assemble_local_matrix(femdvr.local_mass_matrices).toarray()
    first_derivative = femdvr.assemble_local_matrix(
        femdvr.local_first_derivative_matrices
    ).toarray()
    stiffness = femdvr.assemble_local_matrix(
        femdvr.local_stiffness_matrices
    ).toarray()

    np.testing.assert_allclose(mass, femdvr.S, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        first_derivative, femdvr.S @ femdvr.D1, atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(
        stiffness, -femdvr.S @ femdvr.D2, atol=1e-14, rtol=1e-14
    )


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_local_actions_match_dense_global_operators(femdvr, axis):
    rng = np.random.default_rng(3417)
    shape = [2, 3, 4]
    shape[axis] = femdvr.r.size
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def dense_action(matrix):
        moved = np.moveaxis(values, axis, -1)
        result = np.einsum("ij,...j->...i", matrix, moved)
        return np.moveaxis(result, -1, axis)

    np.testing.assert_allclose(
        femdvr.apply_local_mass(values, axis=axis),
        dense_action(femdvr.S),
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        femdvr.apply_local_first_derivative(values, axis=axis),
        dense_action(femdvr.D1),
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        femdvr.apply_local_first_derivative(
            values, representation="weak", axis=axis
        ),
        dense_action(femdvr.S @ femdvr.D1),
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        femdvr.apply_local_first_derivative(
            values, representation="symmetric", axis=axis
        ),
        dense_action(femdvr.D1_symmetric),
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        femdvr.apply_local_second_derivative(values, axis=axis),
        dense_action(femdvr.D2),
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        femdvr.apply_local_second_derivative(
            values, representation="symmetric", axis=axis
        ),
        dense_action(femdvr.D2_symmetric),
        atol=2e-14,
        rtol=2e-14,
    )


def test_sparse_operators_match_existing_dense_operators(femdvr):
    np.testing.assert_allclose(femdvr.sparse_mass_matrix().toarray(), femdvr.S)
    np.testing.assert_allclose(
        femdvr.sparse_first_derivative_matrix().toarray(), femdvr.D1
    )
    np.testing.assert_allclose(
        femdvr.sparse_first_derivative_matrix(
            representation="symmetric"
        ).toarray(),
        femdvr.D1_symmetric,
    )
    np.testing.assert_allclose(
        femdvr.sparse_stiffness_matrix(representation="strong").toarray(),
        -femdvr.D2,
    )
    np.testing.assert_allclose(
        femdvr.sparse_second_derivative_matrix().toarray(), femdvr.D2
    )
    np.testing.assert_allclose(
        femdvr.sparse_second_derivative_matrix(
            representation="symmetric"
        ).toarray(),
        femdvr.D2_symmetric,
    )


def test_local_operator_validation(femdvr):
    with pytest.raises(ValueError, match="expected 13"):
        femdvr.element_values(np.ones(12))
    with pytest.raises(ValueError, match="representation"):
        femdvr.apply_local_first_derivative(
            np.ones(femdvr.r.size), representation="unknown"
        )
    with pytest.raises(ValueError, match="expected 3"):
        femdvr.assemble_local_matrix(femdvr.local_mass_matrices[:2])
