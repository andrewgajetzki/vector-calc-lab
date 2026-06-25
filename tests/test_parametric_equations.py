import math

import pytest

from src.parametric_equations import make_curve, points_table


def test_point_at_evaluates_parametric_curve():
    curve = make_curve(lambda t: t**2 - 1, lambda t: 2 * t + 3)

    point = curve.point_at(2)

    assert point.t == 2
    assert point.x == 3
    assert point.y == 7


def test_sample_returns_evenly_spaced_points_inclusive():
    curve = make_curve(lambda t: t, lambda t: t**2)

    points = curve.sample(0, 1, steps=4)

    assert [point.t for point in points] == [0, 0.25, 0.5, 0.75, 1]
    assert [point.x for point in points] == [0, 0.25, 0.5, 0.75, 1]
    assert [point.y for point in points] == [0, 0.0625, 0.25, 0.5625, 1]


def test_slope_uses_dy_dt_over_dx_dt():
    curve = make_curve(lambda t: t**2 - 1, lambda t: 2 * t + 3)

    assert curve.slope(2) == pytest.approx(0.5)


def test_second_derivative_uses_parametric_formula():
    curve = make_curve(lambda t: t**2, lambda t: t**3)

    assert curve.second_derivative(2) == pytest.approx(0.375, rel=1e-4)


def test_tangent_and_normal_lines_at_parameter_value():
    curve = make_curve(lambda t: t**2, lambda t: t**3)

    tangent = curve.tangent_line(2)
    normal = curve.normal_line(2)

    assert tangent.point.x == 4
    assert tangent.point.y == 8
    assert tangent.slope == pytest.approx(3)
    assert tangent.as_text() == "y - 8 = 3(x - 4)"
    assert normal.slope == pytest.approx(-1 / 3)
    assert normal.as_text() == "y - 8 = -0.333333(x - 4)"


def test_normal_line_is_vertical_when_tangent_is_horizontal():
    curve = make_curve(lambda t: t, lambda t: t**2)

    normal = curve.normal_line(0)

    assert normal.slope is None
    assert normal.as_text() == "x = 0"


def test_concavity_uses_second_derivative_sign():
    concave_up_curve = make_curve(lambda t: t, lambda t: t**2)
    concave_down_curve = make_curve(lambda t: t, lambda t: -(t**2))

    assert concave_up_curve.concavity(1) == "concave up"
    assert concave_down_curve.concavity(1) == "concave down"


def test_speed_and_arc_length_for_unit_circle_quarter_arc():
    curve = make_curve(math.cos, math.sin)

    assert curve.speed(0) == pytest.approx(1)
    assert curve.arc_length(0, math.pi / 2) == pytest.approx(math.pi / 2, rel=1e-6)


def test_signed_area_under_parametric_curve():
    curve = make_curve(lambda t: t, lambda t: t**2)

    assert curve.signed_area_under_curve(0, 1) == pytest.approx(1 / 3, rel=1e-6)


def test_surface_area_about_coordinate_axes():
    horizontal_segment = make_curve(lambda t: t, lambda t: 2)
    vertical_segment = make_curve(lambda t: 2, lambda t: t)

    assert horizontal_segment.surface_area_about_x_axis(0, 3) == pytest.approx(12 * math.pi)
    assert horizontal_segment.surface_area_about_x_axis(3, 0) == pytest.approx(12 * math.pi)
    assert vertical_segment.surface_area_about_y_axis(0, 3) == pytest.approx(12 * math.pi)


def test_undefined_slope_raises_value_error():
    curve = make_curve(lambda t: 3, lambda t: t)

    with pytest.raises(ValueError, match="dx/dt is zero"):
        curve.slope(1)


def test_points_table_formats_values():
    curve = make_curve(lambda t: t, lambda t: t**2)

    assert points_table(curve.sample(0, 1, steps=2)) == (
        "t | x(t) | y(t)\n"
        "--|------|-----\n"
        "0 | 0 | 0\n"
        "0.5 | 0.5 | 0.25\n"
        "1 | 1 | 1"
    )
