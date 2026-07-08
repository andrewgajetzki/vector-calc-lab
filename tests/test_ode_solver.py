import pytest

from src.ode_solver import (
    classify_roots,
    constant_forcing,
    exponential_forcing,
    sinusoidal_forcing,
    solve_homogeneous_second_order,
    solve_linear_second_order,
)


def test_two_distinct_real_roots_solution():
    solution = solve_homogeneous_second_order(1, 3, 2)

    assert classify_roots(1, 3, 2) == "two distinct real roots"
    assert solution.root_type == "two distinct real roots"
    assert solution.characteristic_equation == (
        "Characteristic equation: r^2 + 3r + 2 = 0"
    )
    assert solution.roots == "Roots: r = -1, -2"
    assert solution.general_solution == "General solution: y = C1*e^(-x) + C2*e^(-2x)"


def test_repeated_real_root_solution():
    solution = solve_homogeneous_second_order(1, -2, 1)

    assert classify_roots(1, -2, 1) == "repeated real root"
    assert solution.root_type == "repeated real root"
    assert solution.characteristic_equation == (
        "Characteristic equation: r^2 - 2r + 1 = 0"
    )
    assert solution.roots == "Roots: r = 1"
    assert solution.general_solution == "General solution: y = (C1 + C2*x)*e^(x)"


def test_complex_conjugate_roots_solution():
    solution = solve_homogeneous_second_order(1, 2, 5)

    assert classify_roots(1, 2, 5) == "complex conjugate roots"
    assert solution.root_type == "complex conjugate roots"
    assert solution.characteristic_equation == (
        "Characteristic equation: r^2 + 2r + 5 = 0"
    )
    assert solution.roots == "Roots: r = -1 + 2i, -1 - 2i"
    assert solution.general_solution == (
        "General solution: y = e^(-x)*(C1*cos(2x) + C2*sin(2x))"
    )


def test_linear_second_order_constant_forcing_solution():
    solution = solve_linear_second_order(1, 0, -1, constant_forcing(2))

    assert solution.equation == "Equation: y'' - y = 2"
    assert solution.complementary_solution == (
        "Complementary solution: y_c = C1*e^(x) + C2*e^(-x)"
    )
    assert solution.particular_solution == "Particular solution: y_p = -2"
    assert solution.general_solution == (
        "General solution: y = C1*e^(x) + C2*e^(-x) - 2"
    )


def test_linear_second_order_exponential_resonance_solution():
    solution = solve_linear_second_order(1, -2, 1, exponential_forcing(1, 1))

    assert solution.equation == "Equation: y'' - 2y' + y = e^(x)"
    assert solution.complementary_solution == (
        "Complementary solution: y_c = (C1 + C2*x)*e^(x)"
    )
    assert solution.particular_solution == (
        "Particular solution: y_p = 0.5*x^2*e^(x)"
    )
    assert solution.general_solution == (
        "General solution: y = (C1 + C2*x)*e^(x) + 0.5*x^2*e^(x)"
    )


def test_linear_second_order_sinusoidal_forcing_solution():
    solution = solve_linear_second_order(
        1,
        0,
        1,
        sinusoidal_forcing(cosine_coefficient=1, frequency=2),
    )

    assert solution.equation == "Equation: y'' + y = cos(2x)"
    assert solution.complementary_solution == (
        "Complementary solution: y_c = C1*cos(x) + C2*sin(x)"
    )
    assert solution.particular_solution == (
        "Particular solution: y_p = -0.333333*cos(2x)"
    )
    assert solution.general_solution == (
        "General solution: y = C1*cos(x) + C2*sin(x) - 0.333333*cos(2x)"
    )


def test_linear_second_order_resonant_sinusoidal_forcing_solution():
    solution = solve_linear_second_order(
        1,
        0,
        1,
        sinusoidal_forcing(cosine_coefficient=1, frequency=1),
    )

    assert solution.particular_solution == "Particular solution: y_p = x*(0.5*sin(x))"
    assert solution.general_solution == (
        "General solution: y = C1*cos(x) + C2*sin(x) + x*(0.5*sin(x))"
    )


def test_zero_a_raises_value_error():
    with pytest.raises(ValueError, match="nonzero"):
        solve_homogeneous_second_order(0, 1, 1)


def test_invalid_frequency_raises_value_error():
    with pytest.raises(ValueError, match="frequency"):
        sinusoidal_forcing(frequency=0)
