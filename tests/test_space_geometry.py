import math

import pytest

from src.space_geometry import (
    Line3D,
    Plane3D,
    line_from_points,
    make_line,
    make_plane,
    plane_from_points,
    plane_from_scalar_equation,
)
from src.vectors import Point3D, make_vector3d


def assert_point3d(point: Point3D, x: float, y: float, z: float) -> None:
    assert point.x == pytest.approx(x)
    assert point.y == pytest.approx(y)
    assert point.z == pytest.approx(z)


def test_line_equations_from_point_and_direction():
    line = make_line(Point3D(1, 2, -1), make_vector3d(2, -1, 3))

    assert_point3d(line.point_at(2), 5, 0, 5)
    assert line.vector_equation() == "r(t) = (1, 2, -1) + t<2, -1, 3>"
    assert line.parametric_equations() == "x = 1 + 2t, y = 2 - t, z = -1 + 3t"
    assert line.symmetric_equations() == "(x - 1)/2 = (y - 2)/-1 = (z + 1)/3"
    assert line.contains_point(Point3D(5, 0, 5))
    assert not line.contains_point(Point3D(5, 1, 5))


def test_line_from_points_and_zero_direction_validation():
    start = Point3D(1, -2, 3)
    end = Point3D(4, 2, 15)

    line = line_from_points(start, end)

    assert line.direction.components() == (3, 4, 12)
    with pytest.raises(ValueError, match="direction vector"):
        Line3D.from_points(start, start)


def test_line_distances_angles_and_zero_component_symmetric_equations():
    x_axis = make_line(Point3D(0, 0, 0), make_vector3d(1, 0, 0))
    y_axis_offset = make_line(Point3D(0, 1, 1), make_vector3d(0, 1, 0))
    vertical_y_line = make_line(Point3D(1, 2, 3), make_vector3d(0, 4, 0))

    assert x_axis.distance_to_point(Point3D(0, 3, 4)) == pytest.approx(5)
    assert x_axis.distance_to_line(y_axis_offset) == pytest.approx(1)
    assert x_axis.angle_with_line(y_axis_offset) == pytest.approx(math.pi / 2)
    assert x_axis.is_orthogonal_to(y_axis_offset)
    assert vertical_y_line.symmetric_equations() == "(y - 2)/4; x = 1, z = 3"


def test_plane_equations_from_point_and_normal():
    plane = make_plane(Point3D(1, 2, -1), make_vector3d(2, -1, 3))

    assert plane.coefficients() == pytest.approx((2, -1, 3, -3))
    assert plane.scalar_equation() == "2x - y + 3z = -3"
    assert plane.point_normal_form() == "2(x - 1) - (y - 2) + 3(z + 1) = 0"
    assert plane.contains_point(Point3D(1, 2, -1))
    assert not plane.contains_point(Point3D(1, 2, 0))
    assert plane.distance_to_point(Point3D(1, 2, 0)) == pytest.approx(
        3 / math.sqrt(14)
    )


def test_plane_from_points_and_scalar_equation():
    xy_plane = plane_from_points(
        Point3D(0, 0, 0),
        Point3D(1, 0, 0),
        Point3D(0, 1, 0),
    )
    plane = plane_from_scalar_equation(2, -1, 3, 5)

    assert xy_plane.scalar_equation() == "z = 0"
    assert xy_plane.contains_point(Point3D(2, -3, 0))
    assert plane.coefficients() == pytest.approx((2, -1, 3, 5))
    assert plane.scalar_equation() == "2x - y + 3z = 5"
    with pytest.raises(ValueError, match="normal vector"):
        Plane3D.from_points(Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(2, 2, 2))
    with pytest.raises(ValueError, match="normal vector"):
        plane_from_scalar_equation(0, 0, 0, 1)


def test_line_plane_intersections():
    xy_plane = plane_from_scalar_equation(0, 0, 1, 0)
    crossing_line = make_line(Point3D(1, 1, 1), make_vector3d(0, 0, -1))
    parallel_line = make_line(Point3D(1, 1, 1), make_vector3d(1, 0, 0))
    contained_line = make_line(Point3D(1, 1, 0), make_vector3d(1, 0, 0))

    intersection = crossing_line.intersection_with_plane(xy_plane)

    assert_point3d(intersection, 1, 1, 0)
    assert xy_plane.intersection_with_line(parallel_line) is None
    with pytest.raises(ValueError, match="not unique"):
        contained_line.intersection_with_plane(xy_plane)


def test_plane_plane_intersections_and_angles():
    yz_plane = plane_from_scalar_equation(1, 0, 0, 0)
    xz_plane = plane_from_scalar_equation(0, 1, 0, 0)
    offset_yz_plane = plane_from_scalar_equation(1, 0, 0, 2)
    x_axis = make_line(Point3D(0, 0, 0), make_vector3d(1, 0, 0))
    z_axis = make_line(Point3D(0, 0, 0), make_vector3d(0, 0, 1))

    intersection = yz_plane.intersection_with_plane(xz_plane)

    assert intersection is not None
    assert_point3d(intersection.point, 0, 0, 0)
    assert intersection.direction.is_parallel_to(make_vector3d(0, 0, 1))
    assert yz_plane.intersection_with_plane(offset_yz_plane) is None
    assert yz_plane.angle_with_plane(xz_plane) == pytest.approx(math.pi / 2)
    assert yz_plane.is_orthogonal_to(xz_plane)
    assert yz_plane.is_parallel_to(offset_yz_plane)
    assert yz_plane.angle_with_line(x_axis) == pytest.approx(math.pi / 2)
    assert yz_plane.angle_with_line(z_axis) == pytest.approx(0)
