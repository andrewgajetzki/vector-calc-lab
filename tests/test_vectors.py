import math

import pytest

from src.vectors import (
    Point2D,
    Point3D,
    Vector2D,
    Vector3D,
    make_vector2d,
    make_vector3d,
)


def assert_vector2d(vector: Vector2D, x: float, y: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)


def assert_vector3d(vector: Vector3D, x: float, y: float, z: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)
    assert vector.z == pytest.approx(z)


def test_plane_vector_from_components_and_points():
    vector = make_vector2d(3, 4)
    start = Point2D(1, -2)
    end = Point2D(4, 2)

    assert vector.components() == (3, 4)
    assert vector.as_text() == "<3, 4>"
    assert_vector2d(start.vector_to(end), 3, 4)
    assert_vector2d(Vector2D.from_points(start, end), 3, 4)
    assert start.distance_to(end) == pytest.approx(5)


def test_plane_vector_arithmetic_and_scaling():
    first = make_vector2d(3, 4)
    second = make_vector2d(-1, 2)

    assert_vector2d(first + second, 2, 6)
    assert_vector2d(first - second, 4, 2)
    assert_vector2d(-first, -3, -4)
    assert_vector2d(2 * first, 6, 8)
    assert_vector2d(first / 2, 1.5, 2)


def test_plane_vector_magnitude_direction_unit_and_polar_form():
    vector = make_vector2d(3, 4)
    polar_vector = Vector2D.from_polar(2, math.pi / 2)

    assert vector.magnitude() == pytest.approx(5)
    assert vector.direction_angle() == pytest.approx(math.atan2(4, 3))
    assert vector.to_polar() == pytest.approx((5, math.atan2(4, 3)))
    assert_vector2d(vector.unit(), 0.6, 0.8)
    assert_vector2d(polar_vector, 0, 2)


def test_plane_vector_dot_cross_angle_and_orientation():
    x_axis = make_vector2d(1, 0)
    y_axis = make_vector2d(0, 1)

    assert x_axis.dot(y_axis) == pytest.approx(0)
    assert x_axis.cross_z(y_axis) == pytest.approx(1)
    assert x_axis.determinant(y_axis) == pytest.approx(1)
    assert x_axis.angle_with(y_axis) == pytest.approx(math.pi / 2)
    assert_vector2d(x_axis.perpendicular_left(), 0, 1)
    assert_vector2d(x_axis.perpendicular_right(), 0, -1)


def test_plane_vector_projection_parallel_orthogonal_and_areas():
    vector = make_vector2d(3, 4)
    x_axis_scaled = make_vector2d(4, 0)

    assert vector.scalar_projection_onto(x_axis_scaled) == pytest.approx(3)
    assert_vector2d(vector.projection_onto(x_axis_scaled), 3, 0)
    assert make_vector2d(2, 2).is_parallel_to(make_vector2d(-4, -4))
    assert make_vector2d(2, 2).is_orthogonal_to(make_vector2d(1, -1))
    assert make_vector2d(3, 0).parallelogram_area_with(make_vector2d(0, 4)) == 12
    assert make_vector2d(3, 0).triangle_area_with(make_vector2d(0, 4)) == 6


def test_zero_plane_vector_operations_raise_value_error():
    zero = make_vector2d(0, 0)

    with pytest.raises(ValueError, match="direction angle"):
        zero.direction_angle()
    with pytest.raises(ValueError, match="unit vector"):
        zero.unit()
    with pytest.raises(ValueError, match="undefined"):
        zero.angle_with(make_vector2d(1, 0))
    with pytest.raises(ValueError, match="project onto"):
        make_vector2d(1, 0).projection_onto(zero)
    with pytest.raises(ValueError, match="divide"):
        make_vector2d(1, 0) / 0


def test_space_vector_basics():
    vector = make_vector3d(1, 2, 3)
    other = make_vector3d(-2, 0, 5)
    start = Point3D(1, -1, 2)
    end = Point3D(4, 3, 14)

    assert vector.components() == (1, 2, 3)
    assert vector.as_text() == "<1, 2, 3>"
    assert vector.magnitude() == pytest.approx(math.sqrt(14))
    assert vector.dot(other) == pytest.approx(13)
    assert_vector3d(vector.cross(other), 10, -11, 4)
    assert_vector3d(Vector3D.from_points(start, end), 3, 4, 12)
    assert start.distance_to(end) == pytest.approx(13)


def test_space_vector_projection_angle_parallel_orthogonal_and_areas():
    vector = make_vector3d(1, 2, 3)
    z_axis_scaled = make_vector3d(0, 0, 2)

    assert vector.scalar_projection_onto(z_axis_scaled) == pytest.approx(3)
    assert_vector3d(vector.projection_onto(z_axis_scaled), 0, 0, 3)
    assert make_vector3d(1, 0, 0).angle_with(make_vector3d(0, 1, 0)) == pytest.approx(
        math.pi / 2
    )
    assert make_vector3d(1, 2, 3).is_parallel_to(make_vector3d(2, 4, 6))
    assert make_vector3d(1, 2, 3).is_orthogonal_to(make_vector3d(2, -1, 0))
    assert (
        make_vector3d(3, 0, 0).parallelogram_area_with(make_vector3d(0, 4, 0))
        == 12
    )
    assert make_vector3d(3, 0, 0).triangle_area_with(make_vector3d(0, 4, 0)) == 6
