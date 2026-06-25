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

    assert curve.second_derivative(2) == pytest.approx(0.75, rel=1e-4)


def test_speed_and_arc_length_for_unit_circle_quarter_arc():
    curve = make_curve(math.cos, math.sin)

    assert curve.speed(0) == pytest.approx(1)
    assert curve.arc_length(0, math.pi / 2) == pytest.approx(math.pi / 2, rel=1e-6)


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
