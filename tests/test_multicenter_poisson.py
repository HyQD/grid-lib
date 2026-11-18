import numpy as np
import pytest
from scipy.special import erf

from grid_lib import BSplineRadialBasis
from grid_lib.multicenter import BeckeMulticenterPoisson


def _basis(nodes=None):
    if nodes is None:
        nodes = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    return BSplineRadialBasis(
        nodes,
        degree=8,
        continuity=4,
        vanish_at_boundaries=(True, False),
    )


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _gaussian(alpha, center):
    center = np.asarray(center)

    def density(points):
        displacement = points - center
        return (alpha / np.pi) ** 1.5 * np.exp(
            -alpha * np.sum(displacement**2, axis=-1)
        )

    return density


def _gaussian_potential(points, alpha, center):
    distances = np.linalg.norm(np.asarray(points) - center, axis=-1)
    values = np.empty_like(distances)
    positive = distances > 0.0
    values[positive] = (
        erf(np.sqrt(alpha) * distances[positive]) / distances[positive]
    )
    values[~positive] = 2.0 * np.sqrt(alpha / np.pi)
    return values


def test_single_center_gaussian_has_correct_charge_potential_and_tail():
    alpha = 1.0
    builder = BeckeMulticenterPoisson(
        [[0.0, 0.0, 0.0]],
        _basis(),
        L_max=0,
        radial_order=18,
        lebedev_order=17,
    )
    potential = builder.solve(_gaussian(alpha, [0.0, 0.0, 0.0]))
    radii = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0])
    points = np.column_stack((radii, np.zeros((len(radii), 2))))
    exact = _gaussian_potential(points, alpha, np.zeros(3))

    values = potential(points)
    sampled_density = _gaussian(alpha, [0.0, 0.0, 0.0])(builder.points)
    coulomb_energy = 0.5 * builder.integrate(
        sampled_density * potential(builder.points)
    )

    assert potential.total_charge == pytest.approx(1.0, abs=2e-14)
    np.testing.assert_allclose(values.imag, 0.0, atol=2e-15)
    np.testing.assert_allclose(values.real, exact, atol=5e-7, rtol=0.0)
    assert values[-1].real == pytest.approx(1.0 / 20.0, abs=3e-15)
    assert coulomb_energy.real == pytest.approx(
        np.sqrt(alpha / (2.0 * np.pi)), abs=5e-11
    )
    assert coulomb_energy.imag == pytest.approx(0.0, abs=2e-15)


def test_sampled_complex_density_is_linear_and_matches_callable_path():
    builder = BeckeMulticenterPoisson(
        [[0.0, 0.0, 0.0]],
        _basis(),
        L_max=0,
        radial_order=12,
        lebedev_order=9,
    )
    density = _gaussian(0.8, [0.0, 0.0, 0.0])
    scale = 1.0 - 0.37j
    sampled = scale * density(builder.points)

    from_values = builder.solve_values(sampled)
    from_callable = builder.solve(lambda points: scale * density(points))
    targets = np.array([[0.2, 0.1, 0.3], [2.0, -0.4, 0.5]])

    np.testing.assert_allclose(from_values(targets), from_callable(targets))
    np.testing.assert_allclose(
        from_values(targets, chunk_size=1),
        from_values(targets),
        atol=2e-16,
        rtol=2e-16,
    )
    assert from_values.total_charge == pytest.approx(scale, abs=2e-12)


def _two_center_case():
    centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source = np.array([0.2, 0.3, -0.1])
    targets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, -0.2, 0.3],
            [1.5, 0.2, -0.1],
            [-2.0, 0.4, 0.2],
            [4.0, -1.0, 0.5],
            [12.0, 3.0, -2.0],
        ]
    )
    nodes = [
        0.0,
        0.0625,
        0.125,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        3.0,
        4.0,
        6.0,
        8.0,
        10.0,
    ]
    alpha = 1.3
    density = _gaussian(alpha, source)
    exact = _gaussian_potential(targets, alpha, source)
    return centers, source, targets, _basis(nodes), density, exact


def test_displaced_density_converges_with_multipole_cutoff():
    centers, _, targets, basis, density, exact = _two_center_case()
    errors = []
    for L_max in (2, 6, 10):
        builder = BeckeMulticenterPoisson(
            centers,
            basis,
            L_max=L_max,
            radial_order=12,
            lebedev_order=29,
        )
        potential = builder.solve(density)
        errors.append(np.max(np.abs(potential(targets).real - exact)))

    assert errors[1] < 0.02 * errors[0]
    assert errors[2] < 0.15 * errors[1]
    assert errors[2] < 4e-5


def test_simultaneous_spatial_and_grid_rotation_is_covariant():
    centers, source, targets, basis, density, _ = _two_center_case()
    rotation = _rotation([1.0, 2.0, 3.0], 0.57)
    reference_builder = BeckeMulticenterPoisson(
        centers,
        basis,
        L_max=10,
        radial_order=12,
        lebedev_order=29,
    )
    reference = reference_builder.solve(density)

    rotated_centers = centers @ rotation.T
    rotated_source = source @ rotation.T
    rotated_targets = targets @ rotation.T
    rotated_builder = BeckeMulticenterPoisson(
        rotated_centers,
        basis,
        L_max=10,
        radial_order=12,
        lebedev_order=29,
        angular_rotation=rotation,
    )
    rotated = rotated_builder.solve(_gaussian(1.3, rotated_source))

    assert rotated.total_charge == pytest.approx(
        reference.total_charge, abs=2e-15
    )
    np.testing.assert_allclose(
        rotated(rotated_targets), reference(targets), atol=2e-14, rtol=0.0
    )


def test_angular_orientation_error_decreases_with_lebedev_order():
    centers, _, targets, basis, density, _ = _two_center_case()
    rotation = _rotation([1.0, -2.0, 0.5], 0.81)
    envelopes = []
    charge_errors = []
    for order in (9, 29):
        values = []
        charges = []
        for angular_rotation in (None, rotation):
            builder = BeckeMulticenterPoisson(
                centers,
                basis,
                L_max=10,
                radial_order=12,
                lebedev_order=order,
                angular_rotation=angular_rotation,
            )
            potential = builder.solve(density)
            values.append(potential(targets).real)
            charges.append(potential.total_charge.real)
        envelopes.append(np.max(np.abs(values[1] - values[0])))
        charge_errors.append(np.max(np.abs(np.asarray(charges) - 1.0)))

    assert envelopes[1] < 1e-7
    assert envelopes[1] < 1e-5 * envelopes[0]
    assert charge_errors[1] < 1e-9
    assert charge_errors[1] < 1e-5 * charge_errors[0]


def test_radial_quadrature_converges_total_charge():
    centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source = np.array([0.2, 0.3, -0.1])
    basis = BSplineRadialBasis(
        [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        degree=4,
        continuity=2,
        vanish_at_boundaries=(True, False),
    )
    density = _gaussian(10.0, source)
    errors = []
    for radial_order in (4, 6, 8, 10):
        builder = BeckeMulticenterPoisson(
            centers,
            basis,
            L_max=0,
            radial_order=radial_order,
            lebedev_order=41,
        )
        errors.append(abs(builder.integrate(density(builder.points)) - 1.0))

    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 1e-8


def test_three_centers_support_distinct_radial_bases():
    centers = np.array(
        [[0.0, 0.0, 0.0], [1.43, 1.1, 0.0], [-1.43, 1.1, 0.0]]
    )
    source = np.array([0.1, 0.15, 0.05])
    central_basis = _basis(
        [0.0, 0.1, 0.2, 0.4, 0.8, 1.5, 2.5, 4.0, 6.0, 8.0, 10.0]
    )
    outer_basis = _basis(
        [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    )
    density = _gaussian(2.0, source)
    targets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.1, 0.2],
            centers[1],
            [-1.0, 2.0, 0.5],
            [4.0, -1.0, 0.3],
            [12.0, 2.0, -1.0],
        ]
    )
    builder = BeckeMulticenterPoisson(
        centers,
        (central_basis, outer_basis, outer_basis),
        L_max=10,
        radial_order=14,
        lebedev_order=41,
    )

    potential = builder.solve(density)
    exact = _gaussian_potential(targets, 2.0, source)

    assert len(builder.center_slices) == 3
    assert potential.total_charge == pytest.approx(1.0, abs=3e-11)
    np.testing.assert_allclose(
        potential(targets).real, exact, atol=7e-5, rtol=0.0
    )


def test_coulomb_bilinear_form_is_positive_and_converges_to_reciprocity():
    centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    basis = _basis(
        [0.0, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    )
    reciprocity_errors = []
    self_energies = None
    for L_max in (2, 4, 8):
        builder = BeckeMulticenterPoisson(
            centers,
            basis,
            L_max=L_max,
            radial_order=12,
            lebedev_order=29,
        )
        density_a = _gaussian(1.1, [-0.2, 0.3, 0.1])(builder.points)
        density_b = _gaussian(1.7, [0.5, -0.1, 0.2])(builder.points)
        potential_a = builder.solve_values(density_a)(builder.points)
        potential_b = builder.solve_values(density_b)(builder.points)
        coupling_ab = builder.integrate(density_a * potential_b)
        coupling_ba = builder.integrate(density_b * potential_a)
        reciprocity_errors.append(abs(coupling_ab - coupling_ba))
        self_energies = (
            builder.integrate(density_a * potential_a).real,
            builder.integrate(density_b * potential_b).real,
        )

    assert np.all(np.diff(reciprocity_errors) < 0.0)
    assert reciprocity_errors[-1] < 1e-6
    assert np.all(np.asarray(self_energies) > 0.0)


def test_builder_validates_basis_quadrature_and_density_values():
    wrong_boundaries = BSplineRadialBasis(
        [0.0, 1.0, 2.0],
        degree=4,
        continuity=2,
        vanish_at_boundaries=(True, True),
    )
    with pytest.raises(ValueError, match="retain the outer"):
        BeckeMulticenterPoisson([[0.0, 0.0, 0.0]], wrong_boundaries, 0)

    basis = _basis()
    with pytest.raises(ValueError, match="at least the largest"):
        BeckeMulticenterPoisson(
            [[0.0, 0.0, 0.0]], basis, 0, radial_order=7
        )
    with pytest.raises(ValueError, match="nonnegative"):
        BeckeMulticenterPoisson([[0.0, 0.0, 0.0]], basis, -1)
    with pytest.raises(TypeError, match="one BSplineRadialBasis"):
        BeckeMulticenterPoisson(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [basis], 0
        )
    with pytest.raises(ValueError, match="proper"):
        BeckeMulticenterPoisson(
            [[0.0, 0.0, 0.0]],
            basis,
            0,
            angular_rotation=np.diag([-1.0, 1.0, 1.0]),
        )

    builder = BeckeMulticenterPoisson(
        [[0.0, 0.0, 0.0]], basis, 0, radial_order=8, lebedev_order=5
    )
    with pytest.raises(TypeError, match="callable"):
        builder.solve(np.ones(builder.point_count))
    with pytest.raises(ValueError, match="one scalar"):
        builder.solve_values(np.ones(builder.point_count - 1))
    with pytest.raises(ValueError, match="finite"):
        values = np.ones(builder.point_count)
        values[0] = np.nan
        builder.solve_values(values)
