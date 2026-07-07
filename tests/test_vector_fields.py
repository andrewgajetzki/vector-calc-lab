import math

import pytest

from src.vector_fields import (
    make_vector_field_2d,
    make_vector_field_3d,
    vector_field_2d_table,
    vector_field_3d_table,
)
from src.vectors import Point2D, Point3D, Vector2D, Vector3D


def assert_vector2d(vector: Vector2D, x: float, y: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)


def assert_vector3d(vector: Vector3D, x: float, y: float, z: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)
    assert vector.z == pytest.approx(z)


def test_vector_field_2d_evaluates_and_samples():
    field = make_vector_field_2d(lambda x, y: x * y, lambda x, y: x**2 - y)

    value = field.value_at(2, 3)
    point = field.point_at(2, 3)
    samples = field.sample((0, 1), (0, 1), x_steps=1, y_steps=1)

    assert_vector2d(value, 6, 1)
    assert field.magnitude_at(2, 3) == pytest.approx(math.sqrt(37))
    assert_vector2d(
        field.unit_at(2, 3),
        6 / math.sqrt(37),
        1 / math.sqrt(37),
    )
    assert point.as_text() == "F(2, 3) = <6, 1>"
    assert len(samples) == 4
    assert samples[-1].as_text() == "F(1, 1) = <1, 0>"
    assert vector_field_2d_table(samples) == (
        "x | y | P | Q\n"
        "--|---|---|--\n"
        "0 | 0 | 0 | 0\n"
        "0 | 1 | 0 | -1\n"
        "1 | 0 | 0 | 1\n"
        "1 | 1 | 1 | 0"
    )


def test_vector_field_2d_divergence_curl_and_jacobian():
    field = make_vector_field_2d(lambda x, y: x * y, lambda x, y: x**2 - y)

    jacobian = field.jacobian_matrix(2, 3)

    assert jacobian[0][0] == pytest.approx(3)
    assert jacobian[0][1] == pytest.approx(2)
    assert jacobian[1][0] == pytest.approx(4)
    assert jacobian[1][1] == pytest.approx(-1)
    assert field.divergence(2, 3) == pytest.approx(2)
    assert field.curl_z(2, 3) == pytest.approx(2)


def test_vector_field_2d_line_integral():
    linear = make_vector_field_2d(lambda x, y: y, lambda x, y: x)
    circulation = make_vector_field_2d(lambda x, y: -y, lambda x, y: x)

    assert linear.line_integral(
        lambda t: t,
        lambda t: t,
        (0, 1),
        segments=1,
    ) == pytest.approx(1)
    assert linear.line_integral(
        lambda t: t,
        lambda t: t,
        (1, 0),
        segments=1,
    ) == pytest.approx(-1)
    assert circulation.line_integral(
        math.cos,
        math.sin,
        (0, 2 * math.pi),
        segments=400,
    ) == pytest.approx(2 * math.pi, rel=1e-5)


def test_vector_field_2d_detects_conservative_fields():
    conservative = make_vector_field_2d(
        lambda x, y: 2 * x * y,
        lambda x, y: x**2 + 3 * y**2,
    )
    circulation = make_vector_field_2d(lambda x, y: -y, lambda x, y: x)

    assert conservative.is_conservative_at(2, 1)
    assert conservative.is_conservative_on_rectangle((-1, 1), (-1, 1))
    assert not circulation.is_conservative_at(1, 1)
    assert not circulation.is_conservative_on_rectangle((-1, 1), (-1, 1))


def test_vector_field_2d_potential_difference():
    field = make_vector_field_2d(
        lambda x, y: 2 * x * y,
        lambda x, y: x**2 + 3 * y**2,
    )

    assert field.potential_difference(
        Point2D(0, 0),
        Point2D(2, 1),
        segments=400,
    ) == pytest.approx(5, rel=1e-5)
    assert field.potential_at(2, 1, segments=400) == pytest.approx(5, rel=1e-5)
    assert field.potential_difference(
        Point2D(2, 1),
        Point2D(0, 0),
        segments=400,
    ) == pytest.approx(-5, rel=1e-5)


def test_vector_field_3d_evaluates_and_samples():
    field = make_vector_field_3d(
        lambda x, y, z: x * y,
        lambda x, y, z: y * z,
        lambda x, y, z: z * x,
    )

    value = field.value_at(2, 3, 4)
    point = field.point_at(2, 3, 4)
    samples = field.sample((0, 1), (0, 0), (0, 1), 1, 1, 1)

    assert_vector3d(value, 6, 12, 8)
    assert field.magnitude_at(2, 3, 4) == pytest.approx(math.sqrt(244))
    assert_vector3d(
        field.unit_at(2, 3, 4),
        6 / math.sqrt(244),
        12 / math.sqrt(244),
        8 / math.sqrt(244),
    )
    assert point.as_text() == "F(2, 3, 4) = <6, 12, 8>"
    assert len(samples) == 8
    assert vector_field_3d_table(samples[:2]) == (
        "x | y | z | P | Q | R\n"
        "--|---|---|---|---|--\n"
        "0 | 0 | 0 | 0 | 0 | 0\n"
        "0 | 0 | 1 | 0 | 0 | 0"
    )


def test_vector_field_3d_divergence_curl_and_jacobian():
    field = make_vector_field_3d(
        lambda x, y, z: x * y,
        lambda x, y, z: y * z,
        lambda x, y, z: z * x,
    )

    jacobian = field.jacobian_matrix(2, 3, 4)

    assert jacobian[0] == pytest.approx((3, 2, 0))
    assert jacobian[1] == pytest.approx((0, 4, 3))
    assert jacobian[2] == pytest.approx((4, 0, 2))
    assert field.divergence(2, 3, 4) == pytest.approx(9)
    assert_vector3d(field.curl(2, 3, 4), -3, -4, -2)


def test_vector_field_3d_line_integral():
    field = make_vector_field_3d(
        lambda x, y, z: y,
        lambda x, y, z: z,
        lambda x, y, z: x,
    )

    assert field.line_integral(
        lambda t: t,
        lambda t: t,
        lambda t: t,
        (0, 1),
        segments=1,
    ) == pytest.approx(1.5)
    assert field.line_integral(
        lambda t: t,
        lambda t: t,
        lambda t: t,
        (1, 0),
        segments=1,
    ) == pytest.approx(-1.5)


def test_vector_field_3d_detects_conservative_fields():
    conservative = make_vector_field_3d(
        lambda x, y, z: y + z,
        lambda x, y, z: x + z,
        lambda x, y, z: x + y,
    )
    circulation = make_vector_field_3d(
        lambda x, y, z: -y,
        lambda x, y, z: x,
        lambda x, y, z: z,
    )

    assert conservative.is_conservative_at(1, 2, 3)
    assert conservative.is_conservative_on_box((-1, 1), (-1, 1), (-1, 1))
    assert not circulation.is_conservative_at(1, 2, 3)
    assert not circulation.is_conservative_on_box((-1, 1), (-1, 1), (-1, 1))


def test_vector_field_3d_potential_difference():
    field = make_vector_field_3d(
        lambda x, y, z: y + z,
        lambda x, y, z: x + z,
        lambda x, y, z: x + y,
    )

    assert field.potential_difference(
        Point3D(0, 0, 0),
        Point3D(1, 2, 3),
        segments=400,
    ) == pytest.approx(11, rel=1e-5)
    assert field.potential_at(1, 2, 3, segments=400) == pytest.approx(11, rel=1e-5)
    assert field.potential_difference(
        Point3D(1, 2, 3),
        Point3D(0, 0, 0),
        segments=400,
    ) == pytest.approx(-11, rel=1e-5)


def test_vector_field_validation():
    zero_2d = make_vector_field_2d(lambda x, y: 0, lambda x, y: 0)
    field_2d = make_vector_field_2d(lambda x, y: x, lambda x, y: y)
    zero_3d = make_vector_field_3d(
        lambda x, y, z: 0,
        lambda x, y, z: 0,
        lambda x, y, z: 0,
    )
    field_3d = make_vector_field_3d(
        lambda x, y, z: x,
        lambda x, y, z: y,
        lambda x, y, z: z,
    )

    with pytest.raises(ValueError, match="unit vector"):
        zero_2d.unit_at(0, 0)
    with pytest.raises(ValueError, match="unit vector"):
        zero_3d.unit_at(0, 0, 0)
    with pytest.raises(ValueError, match="x_steps"):
        field_2d.sample((0, 1), (0, 1), x_steps=0, y_steps=1)
    with pytest.raises(ValueError, match="x_bounds"):
        field_2d.sample((1, 0), (0, 1), x_steps=1, y_steps=1)
    with pytest.raises(ValueError, match="h"):
        field_2d.divergence(0, 0, h=0)
    with pytest.raises(ValueError, match="segments"):
        field_2d.line_integral(lambda t: t, lambda t: t, (0, 1), segments=0)
    with pytest.raises(ValueError, match="h"):
        field_2d.line_integral(lambda t: t, lambda t: t, (0, 1), h=0)
    with pytest.raises(ValueError, match="tolerance"):
        field_2d.is_conservative_at(0, 0, tolerance=0)
    with pytest.raises(ValueError, match="x_steps"):
        field_2d.is_conservative_on_rectangle((0, 1), (0, 1), x_steps=0)
    with pytest.raises(ValueError, match="segments"):
        field_2d.potential_difference(Point2D(0, 0), Point2D(1, 1), segments=0)
    with pytest.raises(ValueError, match="z_steps"):
        field_3d.sample((0, 1), (0, 1), (0, 1), 1, 1, 0)
    with pytest.raises(ValueError, match="z_bounds"):
        field_3d.sample((0, 1), (0, 1), (1, 0), 1, 1, 1)
    with pytest.raises(ValueError, match="h"):
        field_3d.curl(0, 0, 0, h=0)
    with pytest.raises(ValueError, match="segments"):
        field_3d.line_integral(
            lambda t: t,
            lambda t: t,
            lambda t: t,
            (0, 1),
            segments=0,
        )
    with pytest.raises(ValueError, match="h"):
        field_3d.line_integral(
            lambda t: t,
            lambda t: t,
            lambda t: t,
            (0, 1),
            h=0,
        )
    with pytest.raises(ValueError, match="tolerance"):
        field_3d.is_conservative_at(0, 0, 0, tolerance=0)
    with pytest.raises(ValueError, match="z_steps"):
        field_3d.is_conservative_on_box((0, 1), (0, 1), (0, 1), z_steps=0)
    with pytest.raises(ValueError, match="segments"):
        field_3d.potential_difference(
            Point3D(0, 0, 0),
            Point3D(1, 1, 1),
            segments=0,
        )
