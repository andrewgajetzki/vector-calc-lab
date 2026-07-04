import math

import pytest

from src.space_coordinates import (
    CylindricalPoint,
    SphericalPoint,
    cylindrical_points_table,
    cylindrical_to_rectangular,
    cylindrical_to_rectangular_table,
    cylindrical_to_spherical,
    normalize_angle,
    rectangular_to_cylindrical,
    rectangular_to_spherical,
    spherical_points_table,
    spherical_to_cylindrical,
    spherical_to_rectangular,
    spherical_to_rectangular_table,
)
from src.vectors import Point3D


def assert_point3d(point: Point3D, x: float, y: float, z: float) -> None:
    assert point.x == pytest.approx(x)
    assert point.y == pytest.approx(y)
    assert point.z == pytest.approx(z)


def assert_cylindrical(
    point: CylindricalPoint,
    r: float,
    theta: float,
    z: float,
) -> None:
    assert point.r == pytest.approx(r)
    assert point.theta == pytest.approx(theta)
    assert point.z == pytest.approx(z)


def assert_spherical(
    point: SphericalPoint,
    rho: float,
    theta: float,
    phi: float,
) -> None:
    assert point.rho == pytest.approx(rho)
    assert point.theta == pytest.approx(theta)
    assert point.phi == pytest.approx(phi)


def test_cylindrical_to_rectangular_converts_coordinates():
    point = cylindrical_to_rectangular(2, math.pi / 2, 3)

    assert_point3d(point, 0, 2, 3)


def test_rectangular_to_cylindrical_converts_coordinates():
    point = rectangular_to_cylindrical(0, -3, 4)

    assert_cylindrical(point, 3, -math.pi / 2, 4)


def test_spherical_to_rectangular_converts_coordinates():
    point = spherical_to_rectangular(2, math.pi / 2, math.pi / 2)

    assert_point3d(point, 0, 2, 0)


def test_rectangular_to_spherical_converts_coordinates():
    positive_z_axis = rectangular_to_spherical(0, 0, 3)
    negative_z_axis = rectangular_to_spherical(0, 0, -3)
    origin = rectangular_to_spherical(0, 0, 0)

    assert_spherical(positive_z_axis, 3, 0, 0)
    assert_spherical(negative_z_axis, 3, 0, math.pi)
    assert_spherical(origin, 0, 0, 0)


def test_cylindrical_and_spherical_dataclass_conversion_methods():
    cylindrical = CylindricalPoint(3, math.pi / 6, 4)
    spherical = SphericalPoint(5, math.pi / 6, math.atan2(3, 4))

    assert cylindrical.as_text() == "(3, 0.523599, 4)"
    assert spherical.as_text() == "(5, 0.523599, 0.643501)"
    assert_point3d(cylindrical.to_rectangular(), 3 * math.sqrt(3) / 2, 1.5, 4)
    assert_spherical(cylindrical.to_spherical(), 5, math.pi / 6, math.atan2(3, 4))
    assert_cylindrical(spherical.to_cylindrical(), 3, math.pi / 6, 4)
    assert_point3d(spherical.to_rectangular(), 3 * math.sqrt(3) / 2, 1.5, 4)


def test_direct_cylindrical_and_spherical_conversions():
    spherical = cylindrical_to_spherical(3, math.pi / 4, 4)
    cylindrical = spherical_to_cylindrical(5, math.pi / 4, math.atan2(3, 4))

    assert_spherical(spherical, 5, math.pi / 4, math.atan2(3, 4))
    assert_cylindrical(cylindrical, 3, math.pi / 4, 4)


def test_angle_normalization_and_volume_element_factors():
    cylindrical = CylindricalPoint(3, 0, 4)
    spherical = SphericalPoint(2, 0, math.pi / 2)

    assert normalize_angle(-math.pi / 2) == pytest.approx(3 * math.pi / 2)
    assert cylindrical.volume_element_factor() == pytest.approx(3)
    assert spherical.volume_element_factor() == pytest.approx(4)


def test_coordinate_tables_format_values():
    cylindrical_points = [
        CylindricalPoint(2, 0, 3),
        CylindricalPoint(1, math.pi / 2, -1),
    ]
    spherical_points = [
        SphericalPoint(2, 0, 0),
        SphericalPoint(1, math.pi / 2, math.pi / 2),
    ]

    assert cylindrical_points_table(cylindrical_points) == (
        "r | theta | z\n"
        "--|-------|--\n"
        "2 | 0 | 3\n"
        "1 | 1.5708 | -1"
    )
    assert spherical_points_table(spherical_points) == (
        "rho | theta | phi\n"
        "----|-------|----\n"
        "2 | 0 | 0\n"
        "1 | 1.5708 | 1.5708"
    )
    assert cylindrical_to_rectangular_table(cylindrical_points) == (
        "r | theta | z_cyl | x | y | z_rect\n"
        "--|-------|-------|---|---|-------\n"
        "2 | 0 | 3 | 2 | 0 | 3\n"
        "1 | 1.5708 | -1 | 0 | 1 | -1"
    )
    assert spherical_to_rectangular_table(spherical_points) == (
        "rho | theta | phi | x | y | z\n"
        "----|-------|-----|---|---|--\n"
        "2 | 0 | 0 | 0 | 0 | 2\n"
        "1 | 1.5708 | 1.5708 | 0 | 1 | 0"
    )
