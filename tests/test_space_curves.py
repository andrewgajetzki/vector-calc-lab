import math

import pytest

from src.space_curves import (
    SpaceCurvePoint,
    make_space_curve,
    make_vector_valued_function,
    space_curve_points_table,
)
from src.vectors import Vector3D


def assert_vector3d(vector: Vector3D, x: float, y: float, z: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)
    assert vector.z == pytest.approx(z)


def test_space_curve_evaluates_and_samples_points():
    curve = make_space_curve(lambda t: t, lambda t: t**2, lambda t: t**3)

    point = curve.point_at(2)
    points = curve.sample(0, 2, steps=2)

    assert point == SpaceCurvePoint(t=2, x=2, y=4, z=8)
    assert point.as_text() == "(2, 4, 8)"
    assert point.position_vector().components() == (2, 4, 8)
    assert point.to_point().z == 8
    assert [sample.t for sample in points] == [0, 1, 2]
    assert [sample.z for sample in points] == [0, 1, 8]


def test_vector_valued_function_alias_and_position_vector():
    curve = make_vector_valued_function(math.cos, math.sin, lambda t: t)

    assert_vector3d(curve.position_vector(0), 1, 0, 0)


def test_velocity_acceleration_speed_and_tangent_line():
    curve = make_space_curve(lambda t: t**2, lambda t: t**3, lambda t: 2 * t)

    assert_vector3d(curve.velocity(2), 4, 12, 2)
    assert_vector3d(curve.acceleration(2), 2, 12, 0)
    assert curve.speed(2) == pytest.approx(math.sqrt(164))

    tangent = curve.tangent_line(2)
    assert tangent.point.x == pytest.approx(4)
    assert tangent.point.y == pytest.approx(8)
    assert tangent.point.z == pytest.approx(4)
    assert_vector3d(tangent.direction, 4, 12, 2)


def test_helix_unit_tangent_curvature_normal_binormal_and_torsion():
    helix = make_space_curve(math.cos, math.sin, lambda t: t)

    assert helix.speed(0) == pytest.approx(math.sqrt(2))
    assert_vector3d(helix.unit_tangent(0), 0, 1 / math.sqrt(2), 1 / math.sqrt(2))
    assert helix.curvature(0) == pytest.approx(0.5, rel=1e-4)
    assert_vector3d(helix.unit_normal(0), -1, 0, 0)
    assert_vector3d(helix.binormal(0), 0, -1 / math.sqrt(2), 1 / math.sqrt(2))
    assert helix.torsion(0) == pytest.approx(0.5, rel=1e-4)


def test_arc_length_and_acceleration_components():
    helix = make_space_curve(math.cos, math.sin, lambda t: t)

    assert helix.arc_length(0, math.pi) == pytest.approx(math.sqrt(2) * math.pi)
    assert helix.tangential_acceleration(0) == pytest.approx(0, abs=1e-5)
    assert helix.normal_acceleration(0) == pytest.approx(1, rel=1e-4)


def test_space_curve_points_table_formats_values():
    curve = make_space_curve(lambda t: t, lambda t: t**2, lambda t: t**3)

    assert space_curve_points_table(curve.sample(0, 1, steps=2)) == (
        "t | x(t) | y(t) | z(t)\n"
        "--|------|------|-----\n"
        "0 | 0 | 0 | 0\n"
        "0.5 | 0.5 | 0.25 | 0.125\n"
        "1 | 1 | 1 | 1"
    )


def test_zero_velocity_operations_raise_value_error():
    stationary = make_space_curve(lambda t: 1, lambda t: 2, lambda t: 3)
    line = make_space_curve(lambda t: t, lambda t: 0, lambda t: 0)

    with pytest.raises(ValueError, match="unit tangent"):
        stationary.unit_tangent(0)
    with pytest.raises(ValueError, match="tangent line"):
        stationary.tangent_line(0)
    with pytest.raises(ValueError, match="Curvature"):
        stationary.curvature(0)
    with pytest.raises(ValueError, match="unit normal"):
        line.unit_normal(0)
    with pytest.raises(ValueError, match="Torsion"):
        line.torsion(0)
