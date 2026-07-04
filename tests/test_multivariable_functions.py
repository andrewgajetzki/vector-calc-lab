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
