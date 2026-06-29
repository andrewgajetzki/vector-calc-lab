import math

import pytest

from src.conic_sections import Point, classify_conic, make_conic


def assert_point(point: Point, x: float, y: float) -> None:
    assert point.x == pytest.approx(x)
    assert point.y == pytest.approx(y)


def test_classify_conic_from_general_coefficients():
    assert classify_conic(1, 0, 1, -4, 6, -12) == "circle"
    assert classify_conic(9, 0, 4, -36, 24, -72) == "ellipse"
    assert classify_conic(4, 0, -9, 0, 0, -36) == "hyperbola"
    assert classify_conic(1, 0, 0, 0, -4, 0) == "parabola"
    assert classify_conic(1, 0, -1) == "degenerate conic"
    assert classify_conic(0, 0, 0, 2, 3, -1) == "not a conic"


def test_conic_evaluates_implicit_equation():
    circle = make_conic(x_squared=1, y_squared=1, x=-4, y=6, constant=-12)

    assert circle.evaluate(2, 2) == pytest.approx(0)
    assert circle.evaluate(2, -3) == pytest.approx(-25)


def test_circle_standard_form_and_features():
    circle = make_conic(x_squared=1, y_squared=1, x=-4, y=6, constant=-12)

    features = circle.standard_form()

    assert features.kind == "circle"
    assert features.standard_form == "(x - 2)^2 + (y + 3)^2 = 25"
    assert_point(features.center, 2, -3)
    assert features.radius == pytest.approx(5)


def test_ellipse_standard_form_and_features():
    ellipse = make_conic(x_squared=9, y_squared=4, x=-36, y=24, constant=-72)

    features = ellipse.standard_form()

    assert features.kind == "ellipse"
    assert features.standard_form == "(x - 2)^2/16 + (y + 3)^2/36 = 1"
    assert features.orientation == "vertical major axis"
    assert_point(features.center, 2, -3)
    assert features.semi_major_axis == pytest.approx(6)
    assert features.semi_minor_axis == pytest.approx(4)
    assert features.focal_distance == pytest.approx(math.sqrt(20))
    assert features.eccentricity == pytest.approx(math.sqrt(20) / 6)
    assert_point(features.vertices[0], 2, -9)
    assert_point(features.vertices[1], 2, 3)
    assert_point(features.foci[0], 2, -3 - math.sqrt(20))
    assert_point(features.foci[1], 2, -3 + math.sqrt(20))


def test_hyperbola_standard_form_and_features():
    hyperbola = make_conic(x_squared=4, y_squared=-9, constant=-36)

    features = hyperbola.standard_form()

    assert features.kind == "hyperbola"
    assert features.standard_form == "x^2/9 - y^2/4 = 1"
    assert features.orientation == "horizontal transverse axis"
    assert_point(features.center, 0, 0)
    assert features.semi_transverse_axis == pytest.approx(3)
    assert features.semi_conjugate_axis == pytest.approx(2)
    assert features.focal_distance == pytest.approx(math.sqrt(13))
    assert features.eccentricity == pytest.approx(math.sqrt(13) / 3)
    assert_point(features.vertices[0], -3, 0)
    assert_point(features.vertices[1], 3, 0)
    assert_point(features.foci[0], -math.sqrt(13), 0)
    assert_point(features.foci[1], math.sqrt(13), 0)
    assert features.asymptotes == ("y = 0.666667x", "y = -0.666667x")


def test_parabola_standard_form_and_features():
    parabola = make_conic(x_squared=1, y=-4)

    features = parabola.standard_form()

    assert features.kind == "parabola"
    assert features.standard_form == "x^2 = 4y"
    assert features.orientation == "opens up"
    assert_point(features.vertex, 0, 0)
    assert features.focal_parameter == pytest.approx(1)
    assert_point(features.foci[0], 0, 1)
    assert features.directrix == "y = -1"


def test_horizontal_parabola_standard_form_and_features():
    parabola = make_conic(y_squared=1, x=-8, y=-6, constant=9)

    features = parabola.standard_form()

    assert features.standard_form == "(y - 3)^2 = 8x"
    assert features.orientation == "opens right"
    assert_point(features.vertex, 0, 3)
    assert features.focal_parameter == pytest.approx(2)
    assert_point(features.foci[0], 2, 3)
    assert features.directrix == "x = -2"


def test_rotated_conic_classifies_but_standard_form_requires_axis_alignment():
    conic = make_conic(x_squared=1, xy=1, y_squared=-1, constant=-1)

    assert conic.classification() == "hyperbola"
    assert not conic.is_axis_aligned()
    with pytest.raises(ValueError, match="axis-aligned"):
        conic.standard_form()
