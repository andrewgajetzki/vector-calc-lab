import pytest

from src.ode_solver import classify_roots, solve_homogeneous_second_order


def test_two_distinct_real_roots_solution():
    solution = solve_homogeneous_second_order(1, 3, 2)

    assert classify_roots(1, 3, 2) == "two distinct real roots"
    assert solution.root_type == "two distinct real roots"
    assert solution.characteristic_equation == "Characteristic equation: r^2 + 3r + 2 = 0"
    assert solution.roots == "Roots: r = -1, -2"
    assert solution.general_solution == "General solution: y = C1*e^(-x) + C2*e^(-2x)"


def test_repeated_real_root_solution():
    solution = solve_homogeneous_second_order(1, -2, 1)

    assert classify_roots(1, -2, 1) == "repeated real root"
    assert solution.root_type == "repeated real root"
    assert solution.characteristic_equation == "Characteristic equation: r^2 - 2r + 1 = 0"
    assert solution.roots == "Roots: r = 1"
    assert solution.general_solution == "General solution: y = (C1 + C2*x)*e^(x)"


def test_complex_conjugate_roots_solution():
    solution = solve_homogeneous_second_order(1, 2, 5)

    assert classify_roots(1, 2, 5) == "complex conjugate roots"
    assert solution.root_type == "complex conjugate roots"
    assert solution.characteristic_equation == "Characteristic equation: r^2 + 2r + 5 = 0"
    assert solution.roots == "Roots: r = -1 + 2i, -1 - 2i"
    assert solution.general_solution == "General solution: y = e^(-x)*(C1*cos(2x) + C2*sin(2x))"


def test_zero_a_raises_value_error():
    with pytest.raises(ValueError, match="nonzero"):
        solve_homogeneous_second_order(0, 1, 1)