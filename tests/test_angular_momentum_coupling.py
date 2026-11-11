import math

import numpy as np
import pytest

from grid_lib.spherical_coordinates import angular_momentum
from grid_lib.spherical_coordinates.angular_momentum import (
    clebsch_gordan,
    get_coupling_backend,
    set_coupling_backend,
    wigner_6j,
)


@pytest.fixture(autouse=True)
def restore_coupling_backend():
    previous = get_coupling_backend()
    try:
        yield
    finally:
        set_coupling_backend(previous)


@pytest.mark.parametrize(
    "quantum_numbers, expected",
    [
        ((0, 0, 0, 0, 0, 0), 1.0),
        ((1, 0, 1, 0, 0, 0), -1.0 / math.sqrt(3.0)),
        ((1, 1, 1, -1, 1, 0), 1.0 / math.sqrt(2.0)),
        ((1, 1, 1, 0, 2, 1), 1.0 / math.sqrt(2.0)),
    ],
)
def test_clebsch_gordan_known_condon_shortley_values(
    quantum_numbers,
    expected,
):
    set_coupling_backend("sympy")

    assert clebsch_gordan(*quantum_numbers) == pytest.approx(expected)


@pytest.mark.parametrize(
    "quantum_numbers",
    [
        (1, 2, 1, 0, 1, 2),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 1, 0, 3, 0),
    ],
)
def test_clebsch_gordan_forbidden_combinations_are_exactly_zero(
    quantum_numbers,
):
    assert clebsch_gordan(*quantum_numbers) == 0.0


@pytest.mark.parametrize(
    "quantum_numbers, expected",
    [
        ((1, 1, 0, 1, 1, 0), 1.0 / 3.0),
        ((1, 1, 1, 1, 1, 1), 1.0 / 6.0),
        ((1, 2, 1, 1, 1, 1), 1.0 / 6.0),
    ],
)
def test_wigner_6j_known_values(quantum_numbers, expected):
    set_coupling_backend("sympy")

    assert wigner_6j(*quantum_numbers) == pytest.approx(expected)


def test_wigner_6j_forbidden_triangle_is_exactly_zero():
    assert wigner_6j(0, 0, 1, 0, 0, 0) == 0.0


def test_sympy_and_py3nj_clebsch_gordan_backends_agree():
    try:
        set_coupling_backend("py3nj")
    except RuntimeError:
        pytest.skip("py3nj is not installed")

    fast = {}
    for j1 in range(4):
        for j2 in range(4):
            for J in range(abs(j1 - j2), j1 + j2 + 1):
                for m1 in range(-j1, j1 + 1):
                    for m2 in range(-j2, j2 + 1):
                        M = m1 + m2
                        if abs(M) <= J:
                            key = (j1, m1, j2, m2, J, M)
                            fast[key] = clebsch_gordan(*key)

    set_coupling_backend("sympy")
    reference = {key: clebsch_gordan(*key) for key in fast}

    assert fast.keys() == reference.keys()
    np.testing.assert_allclose(
        tuple(fast.values()),
        tuple(reference.values()),
        atol=2.0e-15,
        rtol=2.0e-15,
    )


@pytest.mark.parametrize(
    "quantum_numbers",
    [
        (1, 1, 0, 1, 1, 0),
        (1, 1, 1, 1, 1, 1),
        (1, 2, 1, 1, 1, 1),
        (2, 2, 2, 2, 2, 2),
        (2, 3, 2, 3, 2, 3),
    ],
)
def test_sympy_and_py3nj_wigner_6j_backends_agree(quantum_numbers):
    try:
        set_coupling_backend("py3nj")
    except RuntimeError:
        pytest.skip("py3nj is not installed")
    fast = wigner_6j(*quantum_numbers)

    set_coupling_backend("sympy")
    reference = wigner_6j(*quantum_numbers)

    assert fast == pytest.approx(reference, abs=2.0e-15, rel=2.0e-15)


def test_coupling_backend_selection_and_cache():
    assert set_coupling_backend(" SYMPY ") == "sympy"
    assert get_coupling_backend() == "sympy"
    assert get_coupling_backend(resolve=True) == "sympy"

    angular_momentum._clebsch_gordan_cached.cache_clear()
    before = angular_momentum._clebsch_gordan_cached.cache_info()
    clebsch_gordan(1, 0, 1, 0, 0, 0)
    clebsch_gordan(1, 0, 1, 0, 0, 0)
    after = angular_momentum._clebsch_gordan_cached.cache_info()

    assert after.misses == before.misses + 1
    assert after.hits == before.hits + 1

    with pytest.raises(ValueError, match="auto, sympy, py3nj"):
        set_coupling_backend("invalid")


@pytest.mark.parametrize(
    "arguments, error, message",
    [
        ((-1, 0, 1, 0, 0, 0), ValueError, "j1"),
        ((True, 0, 1, 0, 0, 0), TypeError, "j1"),
        ((1.5, 0, 1, 0, 0, 0), TypeError, "j1"),
        ((1, True, 1, 0, 0, 0), TypeError, "m1"),
        ((1, 0.5, 1, 0, 0, 0), TypeError, "m1"),
    ],
)
def test_clebsch_gordan_rejects_invalid_quantum_number_types(
    arguments,
    error,
    message,
):
    with pytest.raises(error, match=message):
        clebsch_gordan(*arguments)


@pytest.mark.parametrize(
    "arguments, error, message",
    [
        ((-1, 0, 0, 0, 0, 0), ValueError, "j1"),
        ((True, 0, 0, 0, 0, 0), TypeError, "j1"),
        ((0.5, 0, 0, 0, 0, 0), TypeError, "j1"),
    ],
)
def test_wigner_6j_rejects_invalid_quantum_numbers(arguments, error, message):
    with pytest.raises(error, match=message):
        wigner_6j(*arguments)


def test_auto_backend_falls_back_when_py3nj_evaluation_fails(monkeypatch):
    class IncompletePy3nj:
        pass

    monkeypatch.setattr(angular_momentum, "_py3nj", IncompletePy3nj())
    set_coupling_backend("auto")

    assert clebsch_gordan(1, 0, 1, 0, 0, 0) == pytest.approx(-1.0 / math.sqrt(3.0))
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 6.0)
