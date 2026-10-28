import grid_lib.spherical_coordinates as spherical_coordinates
from grid_lib.spherical_coordinates import angular_momentum
from grid_lib.spherical_coordinates.multipole_potential import (
    SphericalMultipolePotential,
)


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


def test_spherical_multipole_potential_is_publicly_exported():
    assert "SphericalMultipolePotential" in spherical_coordinates.__all__
    assert (
        spherical_coordinates.SphericalMultipolePotential
        is SphericalMultipolePotential
    )
