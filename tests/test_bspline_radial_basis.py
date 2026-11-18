import numpy as np
import pytest
from scipy.interpolate import BSpline

from grid_lib.basis import BSplineRadialBasis


def _scatter_local(basis, interval, radius):
    values, derivatives, indices = basis.evaluate_interval(interval, radius)
    global_values = np.zeros(basis.size)
    global_derivatives = np.zeros(basis.size)
    global_values[indices] = values
    global_derivatives[indices] = derivatives
    return global_values, global_derivatives


def test_open_basis_has_requested_continuity_and_boundary_values():
    basis = BSplineRadialBasis(
        [0.0, 0.5, 1.5, 3.0], degree=6, continuity=4
    )

    assert basis.degree == 6
    assert basis.continuity == 4
    assert basis.n_intervals == 3
    assert basis.r_min == 0.0
    assert basis.r_max == 3.0
    assert basis.size == basis.full_size - 2
    values, _ = basis.evaluate([basis.r_min, basis.r_max])
    np.testing.assert_array_equal(values, 0.0)


def test_unconstrained_basis_is_a_partition_of_unity():
    basis = BSplineRadialBasis(
        [0.0, 0.4, 1.0, 2.0],
        degree=5,
        continuity=3,
        vanish_at_boundaries=(False, False),
    )
    radii = np.linspace(0.0, 2.0, 101)

    values, derivatives = basis.evaluate(radii)

    np.testing.assert_allclose(np.sum(values, axis=-1), 1.0, atol=2e-15)
    np.testing.assert_allclose(np.sum(derivatives, axis=-1), 0.0, atol=2e-14)


def test_first_derivatives_match_centered_finite_differences():
    basis = BSplineRadialBasis(
        [0.0, 0.7, 1.2, 2.0], degree=7, continuity=4
    )


@pytest.mark.parametrize(
    ("degree", "continuity"), [(1, 0), (2, 0), (5, 3), (7, 4), (8, 4)]
)
def test_values_and_derivatives_match_scipy_reference_at_boundaries(
    degree, continuity
):
    basis = BSplineRadialBasis(
        [0.0, 0.7, 1.2, 2.0], degree=degree, continuity=continuity
    )
    radii = np.array([0.0, 0.7, 0.9, 1.2, 2.0])
    coefficients = np.eye(basis.full_size)
    reference = BSpline(
        basis.knots, coefficients, basis.degree, extrapolate=False
    )

    values, derivatives = basis.evaluate(radii)
    first = int(basis.vanish_at_boundaries[0])
    stop = basis.full_size - int(basis.vanish_at_boundaries[1])

    np.testing.assert_allclose(values, reference(radii)[:, first:stop])
    np.testing.assert_allclose(
        derivatives, reference.derivative()(radii)[:, first:stop]
    )
    radii = np.array([0.2, 0.9, 1.6])
    step = 1.0e-6

    _, derivatives = basis.evaluate(radii)
    values_plus, _ = basis.evaluate(radii + step)
    values_minus, _ = basis.evaluate(radii - step)

    np.testing.assert_allclose(
        derivatives,
        (values_plus - values_minus) / (2.0 * step),
        atol=2e-9,
        rtol=2e-9,
    )


def test_interval_evaluation_matches_global_evaluation():
    basis = BSplineRadialBasis(
        [0.0, 0.5, 1.0, 2.0], degree=6, continuity=4
    )
    radii = np.array([0.55, 0.73, 0.99])

    local_values, local_derivatives, indices = basis.evaluate_interval(1, radii)
    values, derivatives = basis.evaluate(radii)

    np.testing.assert_allclose(local_values, values[:, indices])
    np.testing.assert_allclose(local_derivatives, derivatives[:, indices])
    inactive = np.setdiff1d(np.arange(basis.size), indices)
    np.testing.assert_array_equal(values[:, inactive], 0.0)
    np.testing.assert_array_equal(derivatives[:, inactive], 0.0)


def test_neighboring_interval_views_agree_at_knot():
    basis = BSplineRadialBasis(
        [0.0, 1.0, 2.0], degree=6, continuity=4
    )

    left_values, left_derivatives = _scatter_local(basis, 0, 1.0)
    right_values, right_derivatives = _scatter_local(basis, 1, 1.0)

    np.testing.assert_allclose(left_values, right_values, atol=2e-15)
    np.testing.assert_allclose(left_derivatives, right_derivatives, atol=2e-14)


def test_locate_preserves_shape_and_marks_outside_radii():
    basis = BSplineRadialBasis([0.0, 1.0, 3.0], degree=4, continuity=2)

    np.testing.assert_array_equal(
        basis.locate([[-0.1, 0.0, 0.4], [1.0, 3.0, 3.1]]),
        [[-1, 0, 0], [1, 1, -1]],
    )
    assert basis.locate(0.2).shape == ()


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    [
        ({"nodes": [0.0]}, ValueError, "length"),
        ({"nodes": [0.0, 1.0, 0.5]}, ValueError, "increasing"),
        ({"nodes": [-1.0, 1.0]}, ValueError, "nonnegative"),
        ({"nodes": [0.0, np.inf]}, ValueError, "finite"),
        ({"nodes": [0.0, 1.0], "degree": True}, TypeError, "integer"),
        ({"nodes": [0.0, 1.0], "degree": 0}, ValueError, "at least"),
        (
            {"nodes": [0.0, 1.0], "degree": 4, "continuity": 4},
            ValueError,
            "less than",
        ),
        (
            {"nodes": [0.0, 1.0], "vanish_at_boundaries": (1, 0)},
            TypeError,
            "booleans",
        ),
    ],
)
def test_constructor_rejects_invalid_inputs(kwargs, exception, match):
    with pytest.raises(exception, match=match):
        BSplineRadialBasis(**kwargs)


def test_evaluation_rejects_wrong_interval_or_radii():
    basis = BSplineRadialBasis([0.0, 1.0, 2.0], degree=5, continuity=3)

    with pytest.raises(IndexError, match="outside"):
        basis.evaluate_interval(2, [1.5])
    with pytest.raises(ValueError, match="requested interval"):
        basis.evaluate_interval(0, [1.5])
    with pytest.raises(ValueError, match="outside"):
        basis.evaluate([-0.1])
    with pytest.raises(ValueError, match="finite"):
        basis.locate([np.nan])


def test_interval_evaluation_clips_roundoff_at_finite_boundaries():
    basis = BSplineRadialBasis([0.0, 1.0, 2.0], degree=5, continuity=3)
    inner_roundoff = np.nextafter(0.0, -np.inf)
    outer_roundoff = np.nextafter(2.0, np.inf)

    inner_values = basis.evaluate_interval(0, [inner_roundoff])[:2]
    outer_values = basis.evaluate_interval(1, [outer_roundoff])[:2]

    for actual, expected in zip(
        inner_values, basis.evaluate_interval(0, [0.0])[:2]
    ):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        outer_values, basis.evaluate_interval(1, [2.0])[:2]
    ):
        np.testing.assert_array_equal(actual, expected)


def test_public_topology_arrays_are_read_only():
    basis = BSplineRadialBasis([0.0, 1.0, 2.0], degree=5, continuity=3)

    assert not basis.nodes.flags.writeable
    assert not basis.knots.flags.writeable
    assert not basis.active_indices(0).flags.writeable
