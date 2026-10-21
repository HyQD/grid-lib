import grid_lib.spherical_coordinates as spherical_coordinates
from grid_lib.spherical_coordinates import angular_momentum


def test_angular_momentum_helpers_are_publicly_exported():
    names = (
        "LM_to_I",
        "angular_matrix_element",
        "get_y",
        "get_y_value",
        "get_ybar",
        "get_ybar_value",
        "number_of_lm_states",
    )

    for name in names:
        assert name in spherical_coordinates.__all__
        assert getattr(spherical_coordinates, name) is getattr(
            angular_momentum, name
        )
