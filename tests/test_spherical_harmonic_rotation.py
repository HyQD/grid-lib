import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates import (
    SphericalHarmonicRotation,
    SphericalReducedRadialOperators,
)


def _operators(l_max=3):
    dvr = FEMDVR(
        np.array([0.0, 1.0, 3.0]),
        np.array([5, 5]),
        Linear_map,
        GaussLegendreLobatto,
    )
    return SphericalReducedRadialOperators(
        dvr.r[1:-1],
        dvr.D1[1:-1, 1:-1],
        l_max=l_max,
        m_max=l_max,
    )


def test_identity_and_axial_rotation_follow_active_convention():
    l_max = 4
    identity = SphericalHarmonicRotation(np.eye(3), l_max=l_max)
    np.testing.assert_allclose(
        identity.channel_matrix().toarray(), np.eye((l_max + 1) ** 2)
    )

    angle = 0.37
    axial = SphericalHarmonicRotation.from_axis_angle(
        [0.0, 0.0, 1.0], angle, l_max=l_max
    )
    expected = np.diag(
        [np.exp(-1.0j * m * angle) for _, m in axial.orbital_lm]
    )
    np.testing.assert_allclose(
        axial.channel_matrix().toarray(),
        expected,
        atol=2.0e-15,
        rtol=2.0e-15,
    )


def test_rotation_is_unitary_and_inverse_recovers_complex_batches():
    rotation = SphericalHarmonicRotation.from_axis_angle(
        [1.0, -2.0, 0.5], 0.83, l_max=4
    )
    matrix = rotation.channel_matrix().toarray()
    np.testing.assert_allclose(
        matrix.conj().T @ matrix,
        np.eye(rotation.orbital_size),
        atol=3.0e-15,
        rtol=3.0e-15,
    )

    generator = np.random.default_rng(25)
    states = generator.normal(size=(2, 3, rotation.orbital_size)) + (
        1.0j * generator.normal(size=(2, 3, rotation.orbital_size))
    )
    recovered = rotation.inverse()(rotation(states))
    np.testing.assert_allclose(
        recovered, states, atol=5.0e-15, rtol=5.0e-15
    )


def test_rotation_composition_matches_cartesian_composition():
    first_matrix = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    second_matrix = Rotation.from_rotvec([-0.4, 0.15, 0.05]).as_matrix()
    first = SphericalHarmonicRotation(first_matrix, l_max=3)
    second = SphericalHarmonicRotation(second_matrix, l_max=3)
    combined = SphericalHarmonicRotation(
        second_matrix @ first_matrix, l_max=3
    )

    np.testing.assert_allclose(
        combined.channel_matrix().toarray(),
        second.channel_matrix().toarray()
        @ first.channel_matrix().toarray(),
        atol=3.0e-15,
        rtol=3.0e-15,
    )


@pytest.mark.parametrize(
    "matrix_getter",
    [
        lambda operators, component: operators.coordinate_angular_matrix(
            component
        ),
        lambda operators, component: operators.momentum_angular_matrices(
            component
        )[0],
        lambda operators, component: operators.momentum_angular_matrices(
            component
        )[1],
    ],
)
def test_cartesian_angular_matrices_transform_as_vectors(matrix_getter):
    operators = _operators(l_max=3)
    cartesian = Rotation.from_rotvec([0.2, -0.3, 0.4]).as_matrix()
    rotation = SphericalHarmonicRotation(cartesian, l_max=3)
    channel_rotation = rotation.channel_matrix().toarray()
    matrices = [
        matrix_getter(operators, component).toarray()
        for component in "xyz"
    ]

    for component in range(3):
        transformed = (
            channel_rotation.conj().T
            @ matrices[component]
            @ channel_rotation
        )
        expected = sum(
            cartesian[component, other] * matrices[other]
            for other in range(3)
        )
        np.testing.assert_allclose(
            transformed,
            expected,
            atol=8.0e-15,
            rtol=8.0e-15,
        )


def test_rotation_validation_and_defensive_matrix_copy():
    rotation = SphericalHarmonicRotation(np.eye(3), l_max=2)
    matrix = rotation.channel_matrix()
    matrix.data[:] = 0.0
    assert rotation.channel_matrix().nnz > 0

    with pytest.raises(ValueError, match="orthogonal"):
        SphericalHarmonicRotation(np.diag([1.0, 1.0, 2.0]), l_max=2)
    with pytest.raises(ValueError, match="determinant"):
        SphericalHarmonicRotation(np.diag([1.0, 1.0, -1.0]), l_max=2)
    with pytest.raises(ValueError, match="shape"):
        SphericalHarmonicRotation(np.eye(2), l_max=2)
    with pytest.raises(ValueError, match="nonzero"):
        SphericalHarmonicRotation.from_axis_angle(
            [0.0, 0.0, 0.0], 0.2, l_max=2
        )
    with pytest.raises(ValueError, match="final dimension"):
        rotation(np.zeros((2, rotation.orbital_size + 1)))
