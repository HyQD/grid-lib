import numpy as np
import pytest

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates import SphericalReducedRadialOperators
from grid_lib.spherical_coordinates.momentum_operators import MomentumOperators


@pytest.fixture
def femdvr_data():
    dvr = FEMDVR(
        np.array([0.0, 1.0, 3.0, 6.0]),
        np.array([6, 6, 6]),
        Linear_map,
        GaussLegendreLobatto,
    )
    return (
        dvr.r[1:-1],
        dvr.weights[1:-1],
        dvr.D1[1:-1, 1:-1],
    )


@pytest.fixture
def operators(femdvr_data):
    r, _, first_derivative = femdvr_data
    return SphericalReducedRadialOperators(
        r,
        first_derivative,
        l_max=3,
        m_max=2,
    )


def test_channel_order_and_angular_matrices_match_legacy_operator(
    operators, femdvr_data
):
    r, weights, first_derivative = femdvr_data
    legacy = MomentumOperators(3, 2, r, weights, first_derivative)

    assert operators.orbital_lm == (
        (2, -2),
        (3, -2),
        (1, -1),
        (2, -1),
        (3, -1),
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 2),
    )
    np.testing.assert_allclose(
        operators.cos_theta_matrix().toarray(), legacy.pz_lm_1
    )
    np.testing.assert_allclose(
        operators.sin_theta_dtheta_matrix().toarray(), legacy.pz_lm_2
    )

    generator = np.random.default_rng(8)
    states = generator.normal(size=(4, *operators.orbital_shape))
    states = states + 1j * generator.normal(size=states.shape)
    np.testing.assert_allclose(
        operators.apply_pz(states),
        legacy.pz(states),
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_matrix_accessors_return_defensive_sparse_copies(operators):
    matrix = operators.cos_theta_matrix()
    matrix.data[:] = 0.0
    coordinate = operators.coordinate_angular_matrix("x")
    coordinate.data[:] = 0.0
    radial, angular = operators.momentum_angular_matrices("y")
    radial.data[:] = 0.0
    angular.data[:] = 0.0

    assert operators.cos_theta_matrix().nnz > 0
    assert operators.sin_theta_dtheta_matrix(m=1).shape == (3, 3)
    assert operators.coordinate_angular_matrix("x").nnz > 0
    assert operators.momentum_angular_matrices("y")[0].nnz > 0
    assert operators.momentum_angular_matrices("y")[1].nnz > 0


def test_transverse_actions_on_an_s_orbital(operators):
    radial_values = operators.r * (6.0 - operators.r)
    radial_derivative = 6.0 - 2.0 * operators.r
    derivative_tilde = radial_derivative - radial_values / operators.r
    state = np.zeros(operators.orbital_shape)
    channel_index = {
        channel: index for index, channel in enumerate(operators.orbital_lm)
    }
    state[:, channel_index[(0, 0)]] = radial_values

    expected_x = np.zeros_like(state, dtype=complex)
    expected_y = np.zeros_like(state, dtype=complex)
    expected_px = np.zeros_like(state, dtype=complex)
    expected_py = np.zeros_like(state, dtype=complex)
    for m_value, x_coefficient, y_coefficient in (
        (-1, 1.0 / np.sqrt(6.0), 1.0j / np.sqrt(6.0)),
        (1, -1.0 / np.sqrt(6.0), 1.0j / np.sqrt(6.0)),
    ):
        index = channel_index[(1, m_value)]
        expected_x[:, index] = (
            x_coefficient * operators.r * radial_values
        )
        expected_y[:, index] = (
            y_coefficient * operators.r * radial_values
        )
        expected_px[:, index] = -1.0j * x_coefficient * derivative_tilde
        expected_py[:, index] = -1.0j * y_coefficient * derivative_tilde

    np.testing.assert_allclose(operators.apply_x(state), expected_x)
    np.testing.assert_allclose(operators.apply_y(state), expected_y)
    np.testing.assert_allclose(operators.apply_px(state), expected_px)
    np.testing.assert_allclose(operators.apply_py(state), expected_py)


def test_cartesian_coordinate_norm_identity_away_from_cutoff(operators):
    generator = np.random.default_rng(51)
    state = np.zeros(operators.orbital_shape, dtype=complex)
    for index, (l_value, _) in enumerate(operators.orbital_lm):
        if l_value <= operators.l_max - 2:
            state[:, index] = generator.normal(size=operators.r.size) + (
                1.0j * generator.normal(size=operators.r.size)
            )

    radius_squared = sum(
        action(action(state))
        for action in (operators.apply_x, operators.apply_y, operators.apply_z)
    )
    expected = operators.r[:, None] ** 2 * state
    np.testing.assert_allclose(
        radius_squared,
        expected,
        atol=5.0e-14,
        rtol=5.0e-14,
    )


def test_actions_have_analytic_selection_rules_and_radial_terms(operators):
    radial_values = operators.r * (6.0 - operators.r)
    radial_derivative = 6.0 - 2.0 * operators.r
    state = np.zeros(operators.orbital_shape)
    channel_index = {
        channel: index for index, channel in enumerate(operators.orbital_lm)
    }
    state[:, channel_index[(2, 1)]] = radial_values

    z_action = operators.apply_z(state)
    expected_z = np.zeros_like(z_action)
    expected_z[:, channel_index[(1, 1)]] = (
        np.sqrt(3.0 / 15.0) * operators.r * radial_values
    )
    expected_z[:, channel_index[(3, 1)]] = (
        np.sqrt(8.0 / 35.0) * operators.r * radial_values
    )
    np.testing.assert_allclose(z_action, expected_z, atol=2.0e-14)

    pz_action = operators.apply_pz(state)
    expected_pz = np.zeros_like(pz_action, dtype=np.complex128)
    expected_pz[:, channel_index[(1, 1)]] = -1j * np.sqrt(3.0 / 15.0) * (
        radial_derivative + 2.0 * radial_values / operators.r
    )
    expected_pz[:, channel_index[(3, 1)]] = -1j * np.sqrt(8.0 / 35.0) * (
        radial_derivative - 3.0 * radial_values / operators.r
    )
    np.testing.assert_allclose(pz_action, expected_pz, atol=2.0e-13)


@pytest.mark.parametrize("m", [-2, -1, 0, 1, 2])
def test_full_and_fixed_m_actions_agree_for_complex_batches(operators, m):
    generator = np.random.default_rng(30 + m)
    fixed = generator.normal(
        size=(2, 3, *operators.fixed_m_shape(m))
    ) + 1j * generator.normal(size=(2, 3, *operators.fixed_m_shape(m)))
    full = np.zeros((2, 3, *operators.orbital_shape), dtype=np.complex128)
    indices = [
        index
        for index, (_, magnetic) in enumerate(operators.orbital_lm)
        if magnetic == m
    ]
    full[..., indices] = fixed

    full_z = operators.apply_z(full)
    full_pz = operators.apply_pz(full)

    np.testing.assert_allclose(
        full_z[..., indices],
        operators.apply_z_fixed_m(fixed, m),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        full_pz[..., indices],
        operators.apply_pz_fixed_m(fixed, m),
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    complement = np.ones(full.shape[-1], dtype=bool)
    complement[indices] = False
    np.testing.assert_allclose(full_z[..., complement], 0.0, atol=2.0e-15)
    np.testing.assert_allclose(full_pz[..., complement], 0.0, atol=2.0e-14)


@pytest.mark.parametrize(
    "action_name",
    ["apply_x", "apply_y", "apply_z", "apply_px", "apply_py", "apply_pz"],
)
def test_actions_are_hermitian_in_the_femdvr_metric(
    operators, femdvr_data, action_name
):
    _, weights, _ = femdvr_data
    generator = np.random.default_rng(84)
    left = generator.normal(size=operators.orbital_shape) + 1j * generator.normal(
        size=operators.orbital_shape
    )
    right = generator.normal(size=operators.orbital_shape) + 1j * generator.normal(
        size=operators.orbital_shape
    )
    action = getattr(operators, action_name)

    left_action_right = np.sum(
        weights[:, None] * left.conj() * action(right)
    )
    action_left_right = np.sum(
        weights[:, None] * action(left).conj() * right
    )

    assert left_action_right == pytest.approx(
        action_left_right, abs=3.0e-13, rel=3.0e-13
    )


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"r": [0.0, 1.0]}, ValueError, "positive"),
        ({"r": [1.0, 0.5]}, ValueError, "strictly increasing"),
        ({"r": [0.5 + 0.1j, 1.0]}, TypeError, "real numbers"),
        ({"first_derivative": np.eye(3)}, ValueError, "shape"),
        ({"l_max": -1}, ValueError, "non-negative"),
        ({"m_max": 4}, ValueError, "0 <= m_max"),
    ],
)
def test_constructor_validation(arguments, error, message):
    parameters = {
        "r": np.array([0.5, 1.0]),
        "first_derivative": np.eye(2),
        "l_max": 2,
        "m_max": 1,
    }
    parameters.update(arguments)

    with pytest.raises(error, match=message):
        SphericalReducedRadialOperators(**parameters)


def test_state_validation(operators):
    with pytest.raises(ValueError, match="trailing shape"):
        operators.apply_z(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="trailing shape"):
        operators.apply_pz_fixed_m(np.zeros(operators.orbital_shape), m=0)
    with pytest.raises(ValueError, match="finite"):
        invalid = np.zeros(operators.orbital_shape)
        invalid[0, 0] = np.nan
        operators.apply_z(invalid)

    with pytest.raises(ValueError, match="component"):
        operators.coordinate_angular_matrix("invalid")
