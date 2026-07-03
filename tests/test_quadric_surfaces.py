import pytest

from src.quadric_surfaces import QuadricSurface, classify_quadric, make_quadric
from src.vectors import Point3D


def assert_point3d(point: Point3D, x: float, y: float, z: float) -> None:
    assert point.x == pytest.approx(x)
    assert point.y == pytest.approx(y)
    assert point.z == pytest.approx(z)


def test_quadric_evaluates_implicit_equation():
    sphere = make_quadric(
        x_squared=1,
        y_squared=1,
        z_squared=1,
        x=-2,
        y=4,
        z=-6,
        constant=10,
    )

    assert sphere.evaluate(1, -2, 5) == pytest.approx(0)
    assert sphere.evaluate(1, -2, 3) == pytest.approx(-4)


def test_sphere_standard_form_and_features():
    sphere = make_quadric(
        x_squared=1,
        y_squared=1,
        z_squared=1,
        x=-2,
        y=4,
        z=-6,
        constant=10,
    )

    features = sphere.standard_form()

    assert sphere.classification() == "sphere"
    assert features.kind == "sphere"
    assert features.standard_form == (
        "(x - 1)^2/4 + (y + 2)^2/4 + (z - 3)^2/4 = 1"
    )
    assert_point3d(features.center, 1, -2, 3)
    assert features.radius == pytest.approx(2)


def test_ellipsoid_standard_form_and_features():
    ellipsoid = make_quadric(x_squared=36, y_squared=16, z_squared=9, constant=-144)

    features = ellipsoid.standard_form()

    assert classify_quadric(36, 16, 9, constant=-144) == "ellipsoid"
    assert features.standard_form == "x^2/4 + y^2/9 + z^2/16 = 1"
    assert features.x_radius == pytest.approx(2)
    assert features.y_radius == pytest.approx(3)
    assert features.z_radius == pytest.approx(4)


def test_hyperboloid_standard_forms_and_axes():
    one_sheet = make_quadric(
        x_squared=1 / 9,
        y_squared=1 / 4,
        z_squared=-1 / 16,
        constant=-1,
    )
    two_sheets = make_quadric(
        x_squared=-1 / 4,
        y_squared=-1 / 16,
        z_squared=1 / 9,
        constant=-1,
    )

    one_sheet_features = one_sheet.standard_form()
    two_sheets_features = two_sheets.standard_form()

    assert one_sheet_features.kind == "hyperboloid of one sheet"
    assert one_sheet_features.standard_form == "x^2/9 + y^2/4 - z^2/16 = 1"
    assert one_sheet_features.axis == "z-axis"
    assert two_sheets_features.kind == "hyperboloid of two sheets"
    assert two_sheets_features.standard_form == "z^2/9 - x^2/4 - y^2/16 = 1"
    assert two_sheets_features.axis == "z-axis"


def test_elliptic_cone_standard_form_and_vertex():
    cone = make_quadric(
        x_squared=1 / 4,
        y_squared=1 / 9,
        z_squared=-1 / 16,
    )

    features = cone.standard_form()

    assert features.kind == "elliptic cone"
    assert features.standard_form == "x^2/4 + y^2/9 - z^2/16 = 0"
    assert_point3d(features.vertex, 0, 0, 0)


def test_paraboloid_standard_forms_and_orientation():
    elliptic = make_quadric(x_squared=1 / 4, y_squared=1 / 9, z=-1)
    hyperbolic = make_quadric(x_squared=1 / 4, y_squared=-1 / 9, z=-1)

    elliptic_features = elliptic.standard_form()
    hyperbolic_features = hyperbolic.standard_form()

    assert elliptic_features.kind == "elliptic paraboloid"
    assert elliptic_features.standard_form == "x^2/4 + y^2/9 = z"
    assert elliptic_features.axis == "z-axis"
    assert elliptic_features.orientation == "opens in the positive z-direction"
    assert_point3d(elliptic_features.vertex, 0, 0, 0)
    assert hyperbolic_features.kind == "hyperbolic paraboloid"
    assert hyperbolic_features.standard_form == "x^2/4 - y^2/9 = z"
    assert hyperbolic_features.orientation == "saddle along the z-axis"


def test_cylinder_standard_forms_and_axes():
    elliptic = make_quadric(x_squared=1 / 4, y_squared=1 / 9, constant=-1)
    circular = make_quadric(x_squared=1 / 4, y_squared=1 / 4, constant=-1)
    hyperbolic = make_quadric(x_squared=1 / 4, y_squared=-1 / 9, constant=-1)
    parabolic = make_quadric(x_squared=1, y=-4)

    elliptic_features = elliptic.standard_form()
    circular_features = circular.standard_form()
    hyperbolic_features = hyperbolic.standard_form()
    parabolic_features = parabolic.standard_form()

    assert elliptic_features.kind == "elliptic cylinder"
    assert elliptic_features.standard_form == "x^2/4 + y^2/9 = 1"
    assert elliptic_features.axis == "z-axis"
    assert circular_features.kind == "circular cylinder"
    assert circular_features.radius == pytest.approx(2)
    assert hyperbolic_features.kind == "hyperbolic cylinder"
    assert hyperbolic_features.standard_form == "x^2/4 - y^2/9 = 1"
    assert parabolic_features.kind == "parabolic cylinder"
    assert parabolic_features.standard_form == "x^2/4 = y"
    assert parabolic_features.axis == "z-axis"


def test_rotated_quadric_classifies_but_standard_form_requires_axis_alignment():
    rotated = QuadricSurface(
        x_squared=1,
        y_squared=1,
        z_squared=1,
        xy=2,
        constant=-1,
    )

    assert rotated.classification() == "rotated quadric surface"
    assert not rotated.is_axis_aligned()
    with pytest.raises(ValueError, match="axis-aligned"):
        rotated.standard_form()
