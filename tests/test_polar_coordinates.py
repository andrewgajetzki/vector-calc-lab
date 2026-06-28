import math

import pytest

from src.polar_coordinates import (
    CartesianPoint,
    PolarPoint,
    cartesian_to_polar,
    make_polar_curve,
    normalize_angle,
    polar_points_table,
    polar_to_cartesian,
    polar_to_cartesian_table,
)


def test_polar_to_cartesian_converts_coordinates():
    point = polar_to_cartesian(2, math.pi / 2)

    assert point.x == pytest.approx(0)
    assert point.y == pytest.approx(2)


def test_cartesian_to_polar_converts_coordinates():
    point = cartesian_to_polar(0, -3)

    assert point.r == pytest.approx(3)
    assert point.theta == pytest.approx(-math.pi / 2)


def test_dataclass_conversion_methods():
    cartesian = PolarPoint(5, 0).to_cartesian()
    polar = CartesianPoint(3, 4).to_polar()

    assert cartesian.x == pytest.approx(5)
    assert cartesian.y == pytest.approx(0)
    assert polar.r == pytest.approx(5)
    assert polar.theta == pytest.approx(math.atan2(4, 3))


def test_equivalent_polar_points_include_negative_radius_form():
    equivalents = PolarPoint(2, math.pi / 6).equivalent(turns=1)

    assert any(
        point.r == pytest.approx(2) and point.theta == pytest.approx(math.pi / 6)
        for point in equivalents
    )
    assert any(
        point.r == pytest.approx(-2) and point.theta == pytest.approx(7 * math.pi / 6)
        for point in equivalents
    )


def test_normalize_angle_uses_zero_to_two_pi_range():
    assert normalize_angle(-math.pi / 2) == pytest.approx(3 * math.pi / 2)
    assert normalize_angle(5 * math.pi) == pytest.approx(math.pi)


def test_polar_curve_evaluates_and_samples_points():
    curve = make_polar_curve(lambda theta: 1 + math.cos(theta))

    assert curve.point_at(0).r == pytest.approx(2)
    assert curve.cartesian_point_at(0).x == pytest.approx(2)

    points = curve.sample(0, math.pi, steps=2)
    assert [point.theta for point in points] == pytest.approx([0, math.pi / 2, math.pi])
    assert [point.r for point in points] == pytest.approx([2, 1, 0])


def test_polar_curve_approximates_radius_derivative():
    curve = make_polar_curve(lambda theta: theta**2)

    assert curve.dr_dtheta(3) == pytest.approx(6)


def test_polar_curve_area_uses_half_integral_of_radius_squared():
    circle = make_polar_curve(lambda theta: 2)
    spiral = make_polar_curve(lambda theta: theta)

    assert circle.area(0, math.pi / 2) == pytest.approx(math.pi)
    assert circle.area(math.pi / 2, 0) == pytest.approx(math.pi)
    assert spiral.area(0, 1) == pytest.approx(1 / 6, rel=1e-6)


def test_polar_curve_arc_length_uses_polar_formula():
    circle = make_polar_curve(lambda theta: 2)
    exponential_spiral = make_polar_curve(math.exp)

    assert circle.arc_length(0, math.pi / 2) == pytest.approx(math.pi)
    assert circle.arc_length(math.pi / 2, 0) == pytest.approx(math.pi)
    assert exponential_spiral.arc_length(0, 1) == pytest.approx(
        math.sqrt(2) * (math.e - 1),
        rel=1e-6,
    )


def test_polar_curve_can_be_used_as_parametric_curve():
    polar_curve = make_polar_curve(lambda theta: 2)
    parametric_curve = polar_curve.to_parametric_curve()

    assert parametric_curve.point_at(math.pi / 2).x == pytest.approx(0)
    assert parametric_curve.point_at(math.pi / 2).y == pytest.approx(2)
    assert parametric_curve.arc_length(0, math.pi / 2) == pytest.approx(math.pi, rel=1e-6)


def test_polar_tables_format_values():
    points = [PolarPoint(2, 0), PolarPoint(1, math.pi / 2)]

    assert polar_points_table(points) == "theta | r\n------|--\n0 | 2\n1.5708 | 1"
    assert polar_to_cartesian_table(points) == (
        "theta | r | x | y\n"
        "------|---|---|--\n"
        "0 | 2 | 2 | 0\n"
        "1.5708 | 1 | 0 | 1"
    )
