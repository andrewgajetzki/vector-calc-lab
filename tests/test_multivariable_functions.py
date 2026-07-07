import math

import pytest

from src.multivariable_functions import make_function_2d, make_function_3d
from src.vectors import Vector2D, Vector3D


def assert_vector2d(vector: Vector2D, x: float, y: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)


def assert_vector3d(vector: Vector3D, x: float, y: float, z: float) -> None:
    assert vector.x == pytest.approx(x)
    assert vector.y == pytest.approx(y)
    assert vector.z == pytest.approx(z)


def test_function_2d_evaluates_partials_and_gradient():
    function = make_function_2d(lambda x, y: x**2 + 3 * x * y + y**2)

    assert function.value_at(1, 2) == pytest.approx(11)
    assert function.partial_x(1, 2) == pytest.approx(8)
    assert function.partial_y(1, 2) == pytest.approx(7)
    assert_vector2d(function.gradient(1, 2), 8, 7)


def test_function_2d_second_partials_hessian_and_critical_points():
    function = make_function_2d(lambda x, y: x**2 + 3 * x * y + y**2)

    assert function.second_partial_xx(1, 2) == pytest.approx(2, rel=1e-4)
    assert function.second_partial_yy(1, 2) == pytest.approx(2, rel=1e-4)
    assert function.second_partial_xy(1, 2) == pytest.approx(3, rel=1e-5)
    assert function.second_partial_yx(1, 2) == pytest.approx(3, rel=1e-5)
    hessian = function.hessian(1, 2)
    assert hessian[0][0] == pytest.approx(2, rel=1e-4)
    assert hessian[0][1] == pytest.approx(3, rel=1e-5)
    assert hessian[1][0] == pytest.approx(3, rel=1e-5)
    assert hessian[1][1] == pytest.approx(2, rel=1e-4)

    minimum = make_function_2d(lambda x, y: x**2 + y**2)
    maximum = make_function_2d(lambda x, y: -(x**2) - y**2)
    saddle = make_function_2d(lambda x, y: x**2 - y**2)
    inconclusive = make_function_2d(lambda x, y: x**4 + y**4)

    assert minimum.classify_critical_point(0, 0) == "local minimum"
    assert maximum.classify_critical_point(0, 0) == "local maximum"
    assert saddle.classify_critical_point(0, 0) == "saddle point"
    assert inconclusive.classify_critical_point(0, 0) == "inconclusive"


def test_function_2d_finds_critical_points_and_rectangle_extrema():
    function = make_function_2d(lambda x, y: (x - 1) ** 2 + (y + 2) ** 2)

    critical_points = function.find_critical_points((-3, 3), (-4, 2))
    extrema = function.absolute_extrema_on_rectangle((-1, 3), (-3, 1))

    assert len(critical_points) == 1
    critical_point = critical_points[0]
    assert critical_point.point.x == pytest.approx(1)
    assert critical_point.point.y == pytest.approx(-2)
    assert critical_point.value == pytest.approx(0)
    assert critical_point.classification == "local minimum"
    assert extrema.minimum.value == pytest.approx(0)
    assert extrema.maximum.value == pytest.approx(13)


def test_function_2d_lagrange_extrema_for_circle_constraint():
    function = make_function_2d(lambda x, y: x + y)

    extrema = function.lagrange_extrema(
        lambda x, y: x**2 + y**2 - 1,
        (-1.2, 1.2),
        (-1.2, 1.2),
    )

    coordinate = math.sqrt(0.5)
    assert len(extrema.candidates) == 2
    assert extrema.minimum.value == pytest.approx(-math.sqrt(2), rel=1e-5)
    assert extrema.minimum.point.x == pytest.approx(-coordinate, rel=1e-5)
    assert extrema.minimum.point.y == pytest.approx(-coordinate, rel=1e-5)
    assert extrema.maximum.value == pytest.approx(math.sqrt(2), rel=1e-5)
    assert extrema.maximum.point.x == pytest.approx(coordinate, rel=1e-5)
    assert extrema.maximum.point.y == pytest.approx(coordinate, rel=1e-5)


def test_function_2d_directional_derivative_tangent_plane_and_linearization():
    function = make_function_2d(lambda x, y: x**2 + 3 * x * y + y**2)

    tangent = function.tangent_plane(1, 2)

    assert function.directional_derivative(1, 2, Vector2D(3, 4)) == pytest.approx(10.4)
    assert tangent.x_slope == pytest.approx(8)
    assert tangent.y_slope == pytest.approx(7)
    assert tangent.as_text() == "z - 11 = 8(x - 1) + 7(y - 2)"
    assert tangent.evaluate(1.1, 1.9) == pytest.approx(11.1)
    assert function.linear_approximation(1, 2, 1.1, 1.9) == pytest.approx(11.1)
    assert function.differential(1, 2, 0.1, -0.1) == pytest.approx(0.1)


def test_function_2d_double_integral_over_rectangle():
    constant = make_function_2d(lambda x, y: 3)
    linear = make_function_2d(lambda x, y: x + y)
    quadratic = make_function_2d(lambda x, y: x**2 + y**2)

    assert constant.double_integral_over_rectangle(
        (0, 2),
        (1, 4),
        x_segments=1,
        y_segments=1,
    ) == pytest.approx(18)
    assert linear.double_integral_over_rectangle(
        (0, 2),
        (0, 3),
        x_segments=3,
        y_segments=4,
    ) == pytest.approx(15)
    assert quadratic.double_integral_over_rectangle(
        (0, 1),
        (0, 1),
        x_segments=100,
        y_segments=100,
    ) == pytest.approx(2 / 3, rel=1e-4)


def test_function_2d_double_integral_over_type_i_region():
    constant = make_function_2d(lambda x, y: 1)
    linear = make_function_2d(lambda x, y: x + y)

    assert constant.double_integral_type_i(
        (0, 1),
        lambda x: 0,
        lambda x: x,
        x_segments=4,
        y_segments=4,
    ) == pytest.approx(0.5)
    assert linear.double_integral_type_i(
        (0, 1),
        lambda x: 0,
        lambda x: x,
        x_segments=400,
        y_segments=20,
    ) == pytest.approx(0.5, rel=1e-5)


def test_function_2d_double_integral_over_type_ii_region():
    constant = make_function_2d(lambda x, y: 1)
    linear = make_function_2d(lambda x, y: x + y)

    assert constant.double_integral_type_ii(
        (0, 1),
        lambda y: y,
        lambda y: 1,
        y_segments=4,
        x_segments=4,
    ) == pytest.approx(0.5)
    assert linear.double_integral_type_ii(
        (0, 1),
        lambda y: y,
        lambda y: 1,
        y_segments=400,
        x_segments=20,
    ) == pytest.approx(0.5, rel=1e-5)


def test_function_2d_double_integral_over_polar_region():
    constant = make_function_2d(lambda x, y: 1)
    radial = make_function_2d(lambda x, y: x**2 + y**2)

    assert constant.double_integral_polar(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 2,
        theta_segments=16,
        r_segments=1,
    ) == pytest.approx(4 * math.pi)
    assert radial.double_integral_polar(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 1,
        theta_segments=32,
        r_segments=400,
    ) == pytest.approx(math.pi / 2, rel=1e-5)
    assert constant.double_integral_polar(
        (0, 1),
        lambda theta: 0,
        lambda theta: theta,
        theta_segments=400,
        r_segments=1,
    ) == pytest.approx(1 / 6, rel=1e-5)


def test_function_2d_double_integral_change_of_variables():
    constant = make_function_2d(lambda x, y: 1)
    radial = make_function_2d(lambda x, y: x**2 + y**2)

    assert constant.double_integral_change_of_variables(
        (0, 1),
        (0, 1),
        lambda u, v: 2 * u,
        lambda u, v: 3 * v,
        u_segments=1,
        v_segments=1,
    ) == pytest.approx(6)
    assert constant.double_integral_change_of_variables(
        (0, 1),
        (0, 1),
        lambda u, v: u,
        lambda u, v: -v,
        jacobian=lambda u, v: -1,
        u_segments=1,
        v_segments=1,
    ) == pytest.approx(1)
    assert radial.double_integral_change_of_variables(
        (0, 2 * math.pi),
        (0, 1),
        lambda theta, radius: radius * math.cos(theta),
        lambda theta, radius: radius * math.sin(theta),
        jacobian=lambda theta, radius: radius,
        u_segments=32,
        v_segments=400,
    ) == pytest.approx(math.pi / 2, rel=1e-5)


def test_function_2d_mass_properties_over_rectangle():
    density = make_function_2d(lambda x, y: 2)

    properties = density.mass_properties_over_rectangle(
        (0, 2),
        (0, 3),
        x_segments=100,
        y_segments=100,
    )

    assert properties.mass == pytest.approx(12)
    assert properties.center_of_mass.x == pytest.approx(1)
    assert properties.center_of_mass.y == pytest.approx(1.5)
    assert properties.first_moment_x == pytest.approx(18)
    assert properties.first_moment_y == pytest.approx(12)
    assert properties.moment_of_inertia_x == pytest.approx(36, rel=1e-4)
    assert properties.moment_of_inertia_y == pytest.approx(16, rel=1e-4)
    assert properties.polar_moment_of_inertia == pytest.approx(52, rel=1e-4)
    assert properties.as_text() == "mass = 12, center = (1, 1.5)"


def test_function_2d_mass_properties_over_general_regions():
    density = make_function_2d(lambda x, y: 1)

    type_i = density.mass_properties_type_i(
        (0, 1),
        lambda x: 0,
        lambda x: x,
        x_segments=400,
        y_segments=1,
    )
    type_ii = density.mass_properties_type_ii(
        (0, 1),
        lambda y: y,
        lambda y: 1,
        y_segments=400,
        x_segments=1,
    )

    assert type_i.mass == pytest.approx(0.5, rel=1e-5)
    assert type_i.center_of_mass.x == pytest.approx(2 / 3, rel=1e-5)
    assert type_i.center_of_mass.y == pytest.approx(1 / 3, rel=1e-5)
    assert type_ii.mass == pytest.approx(0.5, rel=1e-5)
    assert type_ii.center_of_mass.x == pytest.approx(2 / 3, rel=1e-5)
    assert type_ii.center_of_mass.y == pytest.approx(1 / 3, rel=1e-5)


def test_function_2d_mass_properties_over_polar_region():
    density = make_function_2d(lambda x, y: 1)

    properties = density.mass_properties_polar(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 2,
        theta_segments=32,
        r_segments=400,
    )

    assert properties.mass == pytest.approx(4 * math.pi)
    assert properties.center_of_mass.x == pytest.approx(0, abs=1e-15)
    assert properties.center_of_mass.y == pytest.approx(0, abs=1e-15)
    assert properties.moment_of_inertia_x == pytest.approx(4 * math.pi, rel=1e-5)
    assert properties.moment_of_inertia_y == pytest.approx(4 * math.pi, rel=1e-5)
    assert properties.polar_moment_of_inertia == pytest.approx(8 * math.pi, rel=1e-5)


def test_function_2d_chain_rule_for_one_parameter_path():
    function = make_function_2d(lambda x, y: x**2 * y + y)

    derivative = function.chain_rule_derivative(
        lambda t: t**2,
        lambda t: t + 1,
        t=2,
    )

    assert derivative == pytest.approx(113)


def test_function_2d_chain_rule_for_two_parameter_substitution():
    function = make_function_2d(lambda x, y: x * y + y**2)

    partials = function.chain_rule_partials(
        lambda u, v: u + v**2,
        lambda u, v: u * v,
        u=2,
        v=3,
    )

    assert_vector2d(partials, 75, 82)


def test_function_2d_estimates_limits_and_continuity():
    continuous = make_function_2d(lambda x, y: x**2 + y**2)

    estimate = continuous.limit_at(0, 0)

    assert estimate.converged
    assert estimate.value == pytest.approx(0, abs=1e-4)
    assert continuous.is_continuous_at(0, 0)
    assert "converged" in estimate.as_text()


def test_function_2d_detects_path_dependent_limit_evidence():
    def ratio(x: float, y: float) -> float:
        if x == 0 and y == 0:
            return 0
        return x * y / (x**2 + y**2)

    function = make_function_2d(ratio)

    estimate = function.limit_at(0, 0)
    x_axis = function.limit_along_path(lambda t: t, lambda t: 0)
    diagonal = function.limit_along_path(lambda t: t, lambda t: t)

    assert not estimate.converged
    assert not function.is_continuous_at(0, 0)
    assert x_axis.value == pytest.approx(0)
    assert diagonal.value == pytest.approx(0.5)


def test_function_2d_detects_removable_discontinuity_evidence():
    function = make_function_2d(
        lambda x, y: 1 if x == 0 and y == 0 else x**2 + y**2
    )

    assert not function.is_continuous_at(0, 0)


def test_function_3d_evaluates_partials_gradient_and_hessian():
    function = make_function_3d(lambda x, y, z: x**2 + y * z + z**3)

    assert function.value_at(1, 2, 3) == pytest.approx(34)
    assert function.partial_x(1, 2, 3) == pytest.approx(2)
    assert function.partial_y(1, 2, 3) == pytest.approx(3)
    assert function.partial_z(1, 2, 3) == pytest.approx(29)
    assert_vector3d(function.gradient(1, 2, 3), 2, 3, 29)
    assert function.second_partial_xx(1, 2, 3) == pytest.approx(2, rel=1e-4)
    assert function.second_partial_yy(1, 2, 3) == pytest.approx(0, abs=1e-5)
    assert function.second_partial_zz(1, 2, 3) == pytest.approx(18, rel=1e-4)
    assert function.second_partial_xy(1, 2, 3) == pytest.approx(0, abs=1e-4)
    assert function.second_partial_xz(1, 2, 3) == pytest.approx(0, abs=1e-4)
    assert function.second_partial_yz(1, 2, 3) == pytest.approx(1, rel=1e-5)


def test_function_3d_directional_derivative_linearization_and_differential():
    function = make_function_3d(lambda x, y, z: x**2 + y * z + z**3)

    assert function.directional_derivative(1, 2, 3, Vector3D(0, 0, 2)) == pytest.approx(
        29
    )
    assert function.linear_approximation(1, 2, 3, 1.1, 1.9, 3.05) == pytest.approx(
        35.35
    )
    assert function.differential(1, 2, 3, 0.1, -0.1, 0.05) == pytest.approx(1.35)


def test_function_3d_triple_integral_over_box():
    constant = make_function_3d(lambda x, y, z: 2)
    linear = make_function_3d(lambda x, y, z: x + y + z)
    quadratic = make_function_3d(lambda x, y, z: x**2 + y**2 + z**2)

    assert constant.triple_integral_over_box(
        (0, 2),
        (0, 3),
        (0, 4),
        x_segments=1,
        y_segments=1,
        z_segments=1,
    ) == pytest.approx(48)
    assert linear.triple_integral_over_box(
        (0, 1),
        (0, 1),
        (0, 1),
        x_segments=3,
        y_segments=4,
        z_segments=5,
    ) == pytest.approx(1.5)
    assert quadratic.triple_integral_over_box(
        (0, 1),
        (0, 1),
        (0, 1),
        x_segments=50,
        y_segments=50,
        z_segments=50,
    ) == pytest.approx(1, rel=3e-4)


def test_function_3d_triple_integral_over_iterated_region():
    constant = make_function_3d(lambda x, y, z: 1)
    linear = make_function_3d(lambda x, y, z: x + y + z)

    assert constant.triple_integral_iterated(
        (0, 1),
        lambda x: 0,
        lambda x: x,
        lambda x, y: 0,
        lambda x, y: y,
        x_segments=400,
        y_segments=4,
        z_segments=1,
    ) == pytest.approx(1 / 6, rel=1e-5)
    assert linear.triple_integral_iterated(
        (0, 1),
        lambda x: 0,
        lambda x: x,
        lambda x, y: 0,
        lambda x, y: 1,
        x_segments=400,
        y_segments=4,
        z_segments=1,
    ) == pytest.approx(0.75, rel=1e-5)


def test_function_3d_triple_integral_over_cylindrical_region():
    constant = make_function_3d(lambda x, y, z: 1)
    radial = make_function_3d(lambda x, y, z: x**2 + y**2)

    assert constant.triple_integral_cylindrical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 2,
        lambda theta, radius: 0,
        lambda theta, radius: 3,
        theta_segments=16,
        r_segments=1,
        z_segments=1,
    ) == pytest.approx(12 * math.pi)
    assert radial.triple_integral_cylindrical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 1,
        lambda theta, radius: 0,
        lambda theta, radius: 1,
        theta_segments=16,
        r_segments=400,
        z_segments=1,
    ) == pytest.approx(math.pi / 2, rel=1e-5)
    assert constant.triple_integral_cylindrical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 1,
        lambda theta, radius: 0,
        lambda theta, radius: radius,
        theta_segments=16,
        r_segments=400,
        z_segments=1,
    ) == pytest.approx(2 * math.pi / 3, rel=1e-5)


def test_function_3d_triple_integral_over_spherical_region():
    constant = make_function_3d(lambda x, y, z: 1)
    radial = make_function_3d(lambda x, y, z: x**2 + y**2 + z**2)

    assert constant.triple_integral_spherical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: math.pi,
        lambda theta, phi: 0,
        lambda theta, phi: 1,
        theta_segments=16,
        phi_segments=100,
        rho_segments=100,
    ) == pytest.approx(4 * math.pi / 3, rel=5e-4)
    assert constant.triple_integral_spherical(
        (0, math.pi / 2),
        lambda theta: 0,
        lambda theta: math.pi / 2,
        lambda theta, phi: 0,
        lambda theta, phi: 1,
        theta_segments=16,
        phi_segments=100,
        rho_segments=100,
    ) == pytest.approx(math.pi / 6, rel=5e-4)
    assert radial.triple_integral_spherical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: math.pi,
        lambda theta, phi: 0,
        lambda theta, phi: 1,
        theta_segments=16,
        phi_segments=100,
        rho_segments=400,
    ) == pytest.approx(4 * math.pi / 5, rel=5e-4)


def test_function_3d_triple_integral_change_of_variables():
    constant = make_function_3d(lambda x, y, z: 1)
    radial = make_function_3d(lambda x, y, z: x**2 + y**2 + z**2)

    assert constant.triple_integral_change_of_variables(
        (0, 1),
        (0, 1),
        (0, 1),
        lambda u, v, w: 2 * u,
        lambda u, v, w: 3 * v,
        lambda u, v, w: 4 * w,
        u_segments=1,
        v_segments=1,
        w_segments=1,
    ) == pytest.approx(24)
    assert constant.triple_integral_change_of_variables(
        (0, 1),
        (0, 1),
        (0, 1),
        lambda u, v, w: u,
        lambda u, v, w: v,
        lambda u, v, w: -w,
        jacobian=lambda u, v, w: -1,
        u_segments=1,
        v_segments=1,
        w_segments=1,
    ) == pytest.approx(1)
    assert radial.triple_integral_change_of_variables(
        (0, 2 * math.pi),
        (0, math.pi),
        (0, 1),
        lambda theta, phi, rho: rho * math.sin(phi) * math.cos(theta),
        lambda theta, phi, rho: rho * math.sin(phi) * math.sin(theta),
        lambda theta, phi, rho: rho * math.cos(phi),
        jacobian=lambda theta, phi, rho: rho**2 * math.sin(phi),
        u_segments=16,
        v_segments=100,
        w_segments=400,
    ) == pytest.approx(4 * math.pi / 5, rel=5e-4)


def test_function_3d_mass_properties_over_box():
    density = make_function_3d(lambda x, y, z: 1)

    properties = density.mass_properties_over_box(
        (0, 1),
        (0, 1),
        (0, 1),
        x_segments=40,
        y_segments=40,
        z_segments=40,
    )

    assert properties.mass == pytest.approx(1)
    assert properties.center_of_mass.x == pytest.approx(0.5)
    assert properties.center_of_mass.y == pytest.approx(0.5)
    assert properties.center_of_mass.z == pytest.approx(0.5)
    assert properties.first_moment_yz == pytest.approx(0.5)
    assert properties.first_moment_xz == pytest.approx(0.5)
    assert properties.first_moment_xy == pytest.approx(0.5)
    assert properties.moment_of_inertia_x == pytest.approx(2 / 3, rel=5e-4)
    assert properties.moment_of_inertia_y == pytest.approx(2 / 3, rel=5e-4)
    assert properties.moment_of_inertia_z == pytest.approx(2 / 3, rel=5e-4)
    assert properties.moment_of_inertia_origin == pytest.approx(1, rel=5e-4)
    assert properties.as_text() == "mass = 1, center = (0.5, 0.5, 0.5)"


def test_function_3d_mass_properties_over_iterated_region():
    density = make_function_3d(lambda x, y, z: 1)

    properties = density.mass_properties_iterated(
        (0, 1),
        lambda x: 0,
        lambda x: x,
        lambda x, y: 0,
        lambda x, y: y,
        x_segments=400,
        y_segments=40,
        z_segments=1,
    )

    assert properties.mass == pytest.approx(1 / 6, rel=1e-5)
    assert properties.center_of_mass.x == pytest.approx(3 / 4, rel=1e-5)
    assert properties.center_of_mass.y == pytest.approx(1 / 2, rel=5e-4)
    assert properties.center_of_mass.z == pytest.approx(1 / 4, rel=5e-4)


def test_function_3d_mass_properties_over_cylindrical_region():
    density = make_function_3d(lambda x, y, z: 1)

    properties = density.mass_properties_cylindrical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 1,
        lambda theta, radius: 0,
        lambda theta, radius: 2,
        theta_segments=16,
        r_segments=400,
        z_segments=1,
    )

    assert properties.mass == pytest.approx(2 * math.pi)
    assert properties.center_of_mass.x == pytest.approx(0, abs=1e-15)
    assert properties.center_of_mass.y == pytest.approx(0, abs=1e-15)
    assert properties.center_of_mass.z == pytest.approx(1)
    assert properties.moment_of_inertia_z == pytest.approx(math.pi, rel=1e-5)


def test_function_3d_mass_properties_over_spherical_region():
    density = make_function_3d(lambda x, y, z: 1)

    properties = density.mass_properties_spherical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: math.pi,
        lambda theta, phi: 0,
        lambda theta, phi: 1,
        theta_segments=16,
        phi_segments=80,
        rho_segments=80,
    )

    assert properties.mass == pytest.approx(4 * math.pi / 3, rel=1e-3)
    assert properties.center_of_mass.x == pytest.approx(0, abs=1e-12)
    assert properties.center_of_mass.y == pytest.approx(0, abs=1e-12)
    assert properties.center_of_mass.z == pytest.approx(0, abs=1e-12)
    assert properties.moment_of_inertia_z == pytest.approx(8 * math.pi / 15, rel=1e-3)


def test_function_3d_classifies_and_finds_critical_points():
    minimum = make_function_3d(lambda x, y, z: (x - 1) ** 2 + (y + 1) ** 2 + z**2)
    maximum = make_function_3d(lambda x, y, z: -(x**2) - y**2 - z**2)
    saddle = make_function_3d(lambda x, y, z: x**2 + y**2 - z**2)

    critical_points = minimum.find_critical_points((-2, 2), (-2, 2), (-1, 1))

    assert minimum.classify_critical_point(1, -1, 0) == "local minimum"
    assert maximum.classify_critical_point(0, 0, 0) == "local maximum"
    assert saddle.classify_critical_point(0, 0, 0) == "saddle point"
    assert len(critical_points) == 1
    assert critical_points[0].point.x == pytest.approx(1)
    assert critical_points[0].point.y == pytest.approx(-1)
    assert critical_points[0].point.z == pytest.approx(0)
    assert critical_points[0].classification == "local minimum"


def test_function_3d_box_extrema_and_lagrange_extrema():
    box_function = make_function_3d(
        lambda x, y, z: (x - 1) ** 2 + (y + 1) ** 2 + (z - 0.5) ** 2
    )
    sphere_function = make_function_3d(lambda x, y, z: x + y + z)

    box_extrema = box_function.absolute_extrema_on_box((0, 2), (-2, 0), (0, 1))
    constrained_extrema = sphere_function.lagrange_extrema(
        lambda x, y, z: x**2 + y**2 + z**2 - 1,
        (-1.2, 1.2),
        (-1.2, 1.2),
        (-1.2, 1.2),
    )

    coordinate = 1 / math.sqrt(3)
    assert box_extrema.minimum.value == pytest.approx(0)
    assert box_extrema.maximum.value == pytest.approx(2.25)
    assert len(constrained_extrema.candidates) == 2
    assert constrained_extrema.minimum.value == pytest.approx(-math.sqrt(3), rel=1e-5)
    assert constrained_extrema.minimum.point.x == pytest.approx(-coordinate, rel=1e-5)
    assert constrained_extrema.maximum.value == pytest.approx(math.sqrt(3), rel=1e-5)
    assert constrained_extrema.maximum.point.x == pytest.approx(coordinate, rel=1e-5)


def test_function_3d_chain_rule_for_one_parameter_path():
    function = make_function_3d(lambda x, y, z: x * y + z**2)

    derivative = function.chain_rule_derivative(
        lambda t: t,
        lambda t: t**2,
        lambda t: t + 1,
        t=2,
    )

    assert derivative == pytest.approx(18)


def test_function_3d_chain_rule_for_two_parameter_substitution():
    function = make_function_3d(lambda x, y, z: x + y * z)

    partials = function.chain_rule_partials(
        lambda u, v: u * v,
        lambda u, v: u + v,
        lambda u, v: u - v,
        u=3,
        v=2,
    )

    assert_vector2d(partials, 8, -1)


def test_function_3d_estimates_limits_and_continuity():
    function = make_function_3d(lambda x, y, z: x + y + z)

    estimate = function.limit_at(1, 2, 3)

    assert estimate.converged
    assert estimate.value == pytest.approx(6)
    assert function.is_continuous_at(1, 2, 3)


def test_function_3d_detects_path_dependent_limit_evidence():
    def ratio(x: float, y: float, z: float) -> float:
        denominator = x**2 + y**2 + z**2
        if denominator == 0:
            return 0
        return x**2 / denominator

    function = make_function_3d(ratio)

    estimate = function.limit_at(0, 0, 0)
    x_axis = function.limit_along_path(lambda t: t, lambda t: 0, lambda t: 0)
    y_axis = function.limit_along_path(lambda t: 0, lambda t: t, lambda t: 0)

    assert not estimate.converged
    assert x_axis.value == pytest.approx(1)
    assert y_axis.value == pytest.approx(0)


def test_level_surface_tangent_plane_uses_gradient_as_normal():
    sphere_level = make_function_3d(lambda x, y, z: x**2 + y**2 + z**2)

    plane = sphere_level.level_surface_tangent_plane(1, 2, 2)

    assert_vector3d(plane.normal, 2, 4, 4)
    assert plane.scalar_equation() == "2x + 4y + 4z = 18"


def test_zero_direction_and_zero_gradient_raise_value_error():
    function_2d = make_function_2d(lambda x, y: x + y)
    function_3d = make_function_3d(lambda x, y, z: x + y + z)
    constant_3d = make_function_3d(lambda x, y, z: 5)

    with pytest.raises(ValueError, match="nonzero direction"):
        function_2d.directional_derivative(0, 0, Vector2D(0, 0))
    with pytest.raises(ValueError, match="nonzero direction"):
        function_3d.directional_derivative(0, 0, 0, Vector3D(0, 0, 0))
    with pytest.raises(ValueError, match="nonzero gradient"):
        constant_3d.level_surface_tangent_plane(0, 0, 0)


def test_limit_helpers_validate_inputs():
    function = make_function_2d(lambda x, y: x + y)
    zero_density = make_function_2d(lambda x, y: 0)
    function_3d = make_function_3d(lambda x, y, z: x + y + z)
    zero_density_3d = make_function_3d(lambda x, y, z: 0)

    with pytest.raises(ValueError, match="tolerance"):
        function.limit_at(0, 0, tolerance=0)
    with pytest.raises(ValueError, match="side"):
        function.limit_along_path(lambda t: t, lambda t: t, side="later")
    with pytest.raises(ValueError, match="x_segments"):
        function.double_integral_over_rectangle((0, 1), (0, 1), x_segments=0)
    with pytest.raises(ValueError, match="x_bounds"):
        function.double_integral_over_rectangle((1, 0), (0, 1))
    with pytest.raises(ValueError, match="y_segments"):
        function.double_integral_type_i(
            (0, 1),
            lambda x: 0,
            lambda x: x,
            y_segments=0,
        )
    with pytest.raises(ValueError, match="y_bounds"):
        function.double_integral_type_i(
            (0, 1),
            lambda x: 1,
            lambda x: 0,
        )
    with pytest.raises(ValueError, match="theta_segments"):
        function.double_integral_polar(
            (0, 1),
            lambda theta: 0,
            lambda theta: 1,
            theta_segments=0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        function.double_integral_polar(
            (0, 1),
            lambda theta: -1,
            lambda theta: 1,
        )
    with pytest.raises(ValueError, match="u_bounds"):
        function.double_integral_change_of_variables(
            (1, 0),
            (0, 1),
            lambda u, v: u,
            lambda u, v: v,
        )
    with pytest.raises(ValueError, match="h"):
        function.double_integral_change_of_variables(
            (0, 1),
            (0, 1),
            lambda u, v: u,
            lambda u, v: v,
            h=0,
        )
    with pytest.raises(ValueError, match="Mass"):
        zero_density.mass_properties_over_rectangle((0, 1), (0, 1))
    with pytest.raises(ValueError, match="z_segments"):
        function_3d.triple_integral_over_box(
            (0, 1),
            (0, 1),
            (0, 1),
            z_segments=0,
        )
    with pytest.raises(ValueError, match="z_bounds"):
        function_3d.triple_integral_iterated(
            (0, 1),
            lambda x: 0,
            lambda x: 1,
            lambda x, y: 1,
            lambda x, y: 0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        function_3d.triple_integral_cylindrical(
            (0, 1),
            lambda theta: -1,
            lambda theta: 1,
            lambda theta, radius: 0,
            lambda theta, radius: 1,
        )
    with pytest.raises(ValueError, match="phi_segments"):
        function_3d.triple_integral_spherical(
            (0, 1),
            lambda theta: 0,
            lambda theta: 1,
            lambda theta, phi: 0,
            lambda theta, phi: 1,
            phi_segments=0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        function_3d.triple_integral_spherical(
            (0, 1),
            lambda theta: 0,
            lambda theta: 1,
            lambda theta, phi: -1,
            lambda theta, phi: 1,
        )
    with pytest.raises(ValueError, match="w_bounds"):
        function_3d.triple_integral_change_of_variables(
            (0, 1),
            (0, 1),
            (1, 0),
            lambda u, v, w: u,
            lambda u, v, w: v,
            lambda u, v, w: w,
        )
    with pytest.raises(ValueError, match="h"):
        function_3d.triple_integral_change_of_variables(
            (0, 1),
            (0, 1),
            (0, 1),
            lambda u, v, w: u,
            lambda u, v, w: v,
            lambda u, v, w: w,
            h=0,
        )
    with pytest.raises(ValueError, match="Mass"):
        zero_density_3d.mass_properties_over_box((0, 1), (0, 1), (0, 1))
