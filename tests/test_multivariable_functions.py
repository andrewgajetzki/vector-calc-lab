import math

import pytest

from src.multivariable_functions import make_function_2d, make_function_3d
from src.vectors import Vector2D, Vector3D


def assert_vector2d(vector: Vector2D, x: float, y: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)


def assert_vector3d(vector: Vector3D, x: float, y: float, z: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)
    assert vector.z == pytest.approx(z)


def test_function_2d_evaluates_partials_and_gradient():
    function = make_function_2d(lambda x, y: x**2 + 3 * x * y + y**2)

    assert function.value_at(1, 2) == pytest.approx(11)
    assert function.partial_x(1, 2) == pytest.approx(8)
    assert function.partial_y(1, 2) == pytest.approx(7)
    assert_vector2d(function.gradient(1, 2), 8, 7)


def test_function_2d_second_partials_hessian_and_critical_points():
    function = make_function_2d(lambda x, y: x**2 + 3 * x * y + y**2)

    assert function.second_partial_xx(1, 2) == pytest.approx(2, rel=1e-4)
    assert function.second_partial_yy(1, 2) == pytest.approx(2, rel=1e-4)
    assert function.second_partial_xy(1, 2) == pytest.approx(3, rel=1e-5)
    assert function.second_partial_yx(1, 2) == pytest.approx(3, rel=1e-5)
    hessian = function.hessian(1, 2)
    assert hessian[0][0] == pytest.approx(2, rel=1e-4)
    assert hessian[0][1] == pytest.approx(3, rel=1e-5)
    assert hessian[1][0] == pytest.approx(3, rel=1e-5)
    assert hessian[1][1] == pytest.approx(2, rel=1e-4)

    minimum = make_function_2d(lambda x, y: x**2 + y**2)
    maximum = make_function_2d(lambda x, y: -(x**2) - y**2)
    saddle = make_function_2d(lambda x, y: x**2 - y**2)
    inconclusive = make_function_2d(lambda x, y: x**4 + y**4)

    assert minimum.classify_critical_point(0, 0) == "local minimum"
    assert maximum.classify_critical_point(0, 0) == "local maximum"
    assert saddle.classify_critical_point(0, 0) == "saddle point"
    assert inconclusive.classify_critical_point(0, 0) == "inconclusive"


def test_function_2d_finds_critical_points_and_rectangle_extrema():
    function = make_function_2d(lambda x, y: (x - 1) ** 2 + (y + 2) ** 2)

    critical_points = function.find_critical_points((-3, 3), (-4, 2))
    extrema = function.absolute_extrema_on_rectangle((-1, 3), (-3, 1))

    assert len(critical_points) == 1
    critical_point = critical_points[0]
    assert critical_point.point.x == pytest.approx(1)
    assert critical_point.point.y == pytest.approx(-2)
    assert critical_point.value == pytest.approx(0)
    assert critical_point.classification == "local minimum"
    assert extrema.minimum.value == pytest.approx(0)
    assert extrema.maximum.value == pytest.approx(13)


def test_function_2d_lagrange_extrema_for_circle_constraint():
    function = make_function_2d(lambda x, y: x + y)

    extrema = function.lagrange_extrema(
        lambda x, y: x**2 + y**2 - 1,
        (-1.2, 1.2),
        (-1.2, 1.2),
    )

    coordinate = math.sqrt(0.5)
    assert len(extrema.candidates) == 2
    assert extrema.minimum.value == pytest.approx(-math.sqrt(2), rel=1e-5)
    assert extrema.minimum.point.x == pytest.approx(-coordinate, rel=1e-5)
    assert extrema.minimum.point.y == pytest.approx(-coordinate, rel=1e-5)
    assert extrema.maximum.value == pytest.approx(math.sqrt(2), rel=1e-5)
    assert extrema.maximum.point.x == pytest.approx(coordinate, rel=1e-5)
    assert extrema.maximum.point.y == pytest.approx(coordinate, rel=1e-5)


def test_function_2d_directional_derivative_tangent_plane_and_linearization():
    function = make_function_2d(lambda x, y: x**2 + 3 * x * y + y**2)

    tangent = function.tangent_plane(1, 2)

    assert function.directional_derivative(1, 2, Vector2D(3, 4)) == pytest.approx(10.4)
    assert tangent.x_slope == pytest.approx(8)
    assert tangent.y_slope == pytest.approx(7)
    assert tangent.as_text() == "z - 11 = 8(x - 1) + 7(y - 2)"
    assert tangent.evaluate(1.1, 1.9) == pytest.approx(11.1)
    assert function.linear_approximation(1, 2, 1.1, 1.9) == pytest.approx(11.1)
    assert function.differential(1, 2, 0.1, -0.1) == pytest.approx(0.1)


def test_function_2d_chain_rule_for_one_parameter_path():
    function = make_function_2d(lambda x, y: x**2 * y + y)

    derivative = function.chain_rule_derivative(
        lambda t: t**2,
        lambda t: t + 1,
        t=2,
    )

    assert derivative == pytest.approx(113)


def test_function_2d_chain_rule_for_two_parameter_substitution():
    function = make_function_2d(lambda x, y: x * y + y**2)

    partials = function.chain_rule_partials(
        lambda u, v: u + v**2,
        lambda u, v: u * v,
        u=2,
        v=3,
    )

    assert_vector2d(partials, 75, 82)


def test_function_2d_estimates_limits_and_continuity():
    continuous = make_function_2d(lambda x, y: x**2 + y**2)

    estimate = continuous.limit_at(0, 0)

    assert estimate.converged
    assert estimate.value == pytest.approx(0, abs=1e-4)
    assert continuous.is_continuous_at(0, 0)
    assert "converged" in estimate.as_text()


def test_function_2d_detects_path_dependent_limit_evidence():
    def ratio(x: float, y: float) -> float:
        if x == 0 and y == 0:
            return 0
        return x * y / (x**2 + y**2)

    function = make_function_2d(ratio)

    estimate = function.limit_at(0, 0)
    x_axis = function.limit_along_path(lambda t: t, lambda t: 0)
    diagonal = function.limit_along_path(lambda t: t, lambda t: t)

    assert not estimate.converged
    assert not function.is_continuous_at(0, 0)
    assert x_axis.value == pytest.approx(0)
    assert diagonal.value == pytest.approx(0.5)


def test_function_2d_detects_removable_discontinuity_evidence():
    function = make_function_2d(
        lambda x, y: 1 if x == 0 and y == 0 else x**2 + y**2
    )

    assert not function.is_continuous_at(0, 0)


def test_function_3d_evaluates_partials_gradient_and_hessian():
    function = make_function_3d(lambda x, y, z: x**2 + y * z + z**3)

    assert function.value_at(1, 2, 3) == pytest.approx(34)
    assert function.partial_x(1, 2, 3) == pytest.approx(2)
    assert function.partial_y(1, 2, 3) == pytest.approx(3)
    assert function.partial_z(1, 2, 3) == pytest.approx(29)
    assert_vector3d(function.gradient(1, 2, 3), 2, 3, 29)
    assert function.second_partial_xx(1, 2, 3) == pytest.approx(2, rel=1e-4)
    assert function.second_partial_yy(1, 2, 3) == pytest.approx(0, abs=1e-5)
    assert function.second_partial_zz(1, 2, 3) == pytest.approx(18, rel=1e-4)
    assert function.second_partial_xy(1, 2, 3) == pytest.approx(0, abs=1e-4)
    assert function.second_partial_xz(1, 2, 3) == pytest.approx(0, abs=1e-4)
    assert function.second_partial_yz(1, 2, 3) == pytest.approx(1, rel=1e-5)


def test_function_3d_directional_derivative_linearization_and_differential():
    function = make_function_3d(lambda x, y, z: x**2 + y * z + z**3)

    assert function.directional_derivative(1, 2, 3, Vector3D(0, 0, 2)) == pytest.approx(
        29
    )
    assert function.linear_approximation(1, 2, 3, 1.1, 1.9, 3.05) == pytest.approx(
        35.35
    )
    assert function.differential(1, 2, 3, 0.1, -0.1, 0.05) == pytest.approx(1.35)


def test_function_3d_classifies_and_finds_critical_points():
    minimum = make_function_3d(lambda x, y, z: (x - 1) ** 2 + (y + 1) ** 2 + z**2)
    maximum = make_function_3d(lambda x, y, z: -(x**2) - y**2 - z**2)
    saddle = make_function_3d(lambda x, y, z: x**2 + y**2 - z**2)

    critical_points = minimum.find_critical_points((-2, 2), (-2, 2), (-1, 1))

    assert minimum.classify_critical_point(1, -1, 0) == "local minimum"
    assert maximum.classify_critical_point(0, 0, 0) == "local maximum"
    assert saddle.classify_critical_point(0, 0, 0) == "saddle point"
    assert len(critical_points) == 1
    assert critical_points[0].point.x == pytest.approx(1)
    assert critical_points[0].point.y == pytest.approx(-1)
    assert critical_points[0].point.z == pytest.approx(0)
    assert critical_points[0].classification == "local minimum"


def test_function_3d_box_extrema_and_lagrange_extrema():
    box_function = make_function_3d(
        lambda x, y, z: (x - 1) ** 2 + (y + 1) ** 2 + (z - 0.5) ** 2
    )
    sphere_function = make_function_3d(lambda x, y, z: x + y + z)

    box_extrema = box_function.absolute_extrema_on_box((0, 2), (-2, 0), (0, 1))
    constrained_extrema = sphere_function.lagrange_extrema(
        lambda x, y, z: x**2 + y**2 + z**2 - 1,
        (-1.2, 1.2),
        (-1.2, 1.2),
        (-1.2, 1.2),
    )

    coordinate = 1 / math.sqrt(3)
    assert box_extrema.minimum.value == pytest.approx(0)
    assert box_extrema.maximum.value == pytest.approx(2.25)
    assert len(constrained_extrema.candidates) == 2
    assert constrained_extrema.minimum.value == pytest.approx(-math.sqrt(3), rel=1e-5)
    assert constrained_extrema.minimum.point.x == pytest.approx(-coordinate, rel=1e-5)
    assert constrained_extrema.maximum.value == pytest.approx(math.sqrt(3), rel=1e-5)
    assert constrained_extrema.maximum.point.x == pytest.approx(coordinate, rel=1e-5)


def test_function_3d_chain_rule_for_one_parameter_path():
    function = make_function_3d(lambda x, y, z: x * y + z**2)

    derivative = function.chain_rule_derivative(
        lambda t: t,
        lambda t: t**2,
        lambda t: t + 1,
        t=2,
    )

    assert derivative == pytest.approx(18)


def test_function_3d_chain_rule_for_two_parameter_substitution():
    function = make_function_3d(lambda x, y, z: x + y * z)

    partials = function.chain_rule_partials(
        lambda u, v: u * v,
        lambda u, v: u + v,
        lambda u, v: u - v,
        u=3,
        v=2,
    )

    assert_vector2d(partials, 8, -1)


def test_function_3d_estimates_limits_and_continuity():
    function = make_function_3d(lambda x, y, z: x + y + z)

    estimate = function.limit_at(1, 2, 3)

    assert estimate.converged
    assert estimate.value == pytest.approx(6)
    assert function.is_continuous_at(1, 2, 3)


def test_function_3d_detects_path_dependent_limit_evidence():
    def ratio(x: float, y: float, z: float) -> float:
        denominator = x**2 + y**2 + z**2
        if denominator == 0:
            return 0
        return x**2 / denominator

    function = make_function_3d(ratio)

    estimate = function.limit_at(0, 0, 0)
    x_axis = function.limit_along_path(lambda t: t, lambda t: 0, lambda t: 0)
    y_axis = function.limit_along_path(lambda t: 0, lambda t: t, lambda t: 0)

    assert not estimate.converged
    assert x_axis.value == pytest.approx(1)
    assert y_axis.value == pytest.approx(0)


def test_level_surface_tangent_plane_uses_gradient_as_normal():
    sphere_level = make_function_3d(lambda x, y, z: x**2 + y**2 + z**2)

    plane = sphere_level.level_surface_tangent_plane(1, 2, 2)

    assert_vector3d(plane.normal, 2, 4, 4)
    assert plane.scalar_equation() == "2x + 4y + 4z = 18"


def test_zero_direction_and_zero_gradient_raise_value_error():
    function_2d = make_function_2d(lambda x, y: x + y)
    function_3d = make_function_3d(lambda x, y, z: x + y + z)
    constant_3d = make_function_3d(lambda x, y, z: 5)

    with pytest.raises(ValueError, match="nonzero direction"):
        function_2d.directional_derivative(0, 0, Vector2D(0, 0))
    with pytest.raises(ValueError, match="nonzero direction"):
        function_3d.directional_derivative(0, 0, 0, Vector3D(0, 0, 0))
    with pytest.raises(ValueError, match="nonzero gradient"):
        constant_3d.level_surface_tangent_plane(0, 0, 0)


def test_limit_helpers_validate_inputs():
    function = make_function_2d(lambda x, y: x + y)

    with pytest.raises(ValueError, match="tolerance"):
        function.limit_at(0, 0, tolerance=0)
    with pytest.raises(ValueError, match="side"):
        function.limit_along_path(lambda t: t, lambda t: t, side="later")
