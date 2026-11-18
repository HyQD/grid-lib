import numpy as np
import pytest

from grid_lib.multicenter import BeckePartition


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


def test_weights_form_partition_and_select_the_centers():
    centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    partition = BeckePartition(centers)
    points = np.array(
        [centers[0], centers[1], [0.0, 0.0, 0.0], [0.3, -0.2, 0.8]]
    )

    weights = partition.weights(points)

    np.testing.assert_allclose(np.sum(weights, axis=-1), 1.0, atol=2e-15)
    np.testing.assert_array_equal(weights[0], [1.0, 0.0])
    np.testing.assert_array_equal(weights[1], [0.0, 1.0])
    np.testing.assert_allclose(weights[2], [0.5, 0.5], atol=2e-15)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)


def test_weights_are_translation_and_rotation_covariant():
    rng = np.random.default_rng(8271)
    centers = rng.normal(size=(4, 3))
    points = rng.normal(size=(19, 3))
    rotation = _rotation([1.0, -2.0, 3.0], 0.71)
    translation = np.array([0.4, -1.2, 2.1])
    reference = BeckePartition(centers).weights(points)

    transformed = BeckePartition(
        centers @ rotation.T + translation
    ).weights(points @ rotation.T + translation)

    np.testing.assert_allclose(transformed, reference, atol=3e-15, rtol=0.0)


@pytest.mark.parametrize(
    ("centers", "exception", "match"),
    [
        ([], ValueError, "shape"),
        ([[0.0, 0.0]], ValueError, "shape"),
        ([[0.0, 0.0, np.inf]], ValueError, "finite"),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], ValueError, "distinct"),
    ],
)
def test_partition_rejects_invalid_centers(centers, exception, match):
    with pytest.raises(exception, match=match):
        BeckePartition(centers)


def test_partition_validates_switches_and_points():
    with pytest.raises(TypeError, match="integer"):
        BeckePartition([[0.0, 0.0, 0.0]], switch_iterations=2.5)
    with pytest.raises(ValueError, match="positive"):
        BeckePartition([[0.0, 0.0, 0.0]], switch_iterations=0)

    partition = BeckePartition([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        partition.weights([0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        partition.weights([[0.0, np.nan, 0.0]])
