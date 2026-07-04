"""Study helpers for Calculus 3 topics."""

from __future__ import annotations

import math

from src.conic_sections import make_conic
from src.parametric_equations import make_curve, points_table
from src.polar_coordinates import make_polar_curve, polar_to_cartesian_table
from src.quadric_surfaces import make_quadric
from src.space_coordinates import (
    CylindricalPoint,
    SphericalPoint,
    cylindrical_to_rectangular_table,
    spherical_to_rectangular_table,
)
from src.space_curves import make_space_curve, space_curve_points_table
from src.space_geometry import (
    line_from_points,
    make_line,
    make_plane,
    plane_from_scalar_equation,
)
from src.vectors import Point2D, Point3D, make_vector2d, make_vector3d


def run_demo() -> None:
    """Show the available calculus helpers on sample curves."""
    run_vector_demo()
    print()
    run_space_curve_demo()
    print()
    run_space_geometry_demo()
    print()
    run_parametric_demo()
    print()
    run_polar_demo()
    print()
    run_space_coordinates_demo()
    print()
    run_conic_demo()
    print()
    run_quadric_demo()


def run_vector_demo() -> None:
    """Show the vector tools on sample plane and space vectors."""
    start = Point2D(1, -2)
    end = Point2D(4, 2)
    displacement = start.vector_to(end)
    first = make_vector2d(3, 4)
    second = make_vector2d(-2, 1)

    print("Vectors")
    print()
    print("Example plane vectors: u = <3, 4>, v = <-2, 1>")
    print()
    print(f"Vector from (1, -2) to (4, 2): {displacement.as_text()}")
    print(f"|u|: {first.magnitude():.6g}")
    print(f"Direction angle of u: {first.direction_angle():.6g} radians")
    print(f"u dot v: {first.dot(second):.6g}")
    print(f"2D cross z-component of u and v: {first.cross_z(second):.6g}")
    print(f"Angle between u and v: {first.angle_with(second):.6g} radians")
    print(f"Projection of u onto v: {first.projection_onto(second).as_text()}")
    print(
        "Parallelogram area spanned by u and v: "
        f"{first.parallelogram_area_with(second):.6g}"
    )
    print(
        "Space cross product <1, 0, 0> x <0, 1, 0>: "
        f"{make_vector3d(1, 0, 0).cross(make_vector3d(0, 1, 0)).as_text()}"
    )


def run_space_curve_demo() -> None:
    """Show vector-valued function and space-curve tools."""
    curve = make_space_curve(math.cos, math.sin, lambda t: t)

    print("Vector-Valued Functions and Space Curves")
    print()
    print("Example curve: r(t) = <cos(t), sin(t), t>")
    print()
    print(space_curve_points_table(curve.sample(start=0, stop=math.pi / 2, steps=2)))
    print()
    print(f"Velocity at t = 0: {curve.velocity(0).as_text()}")
    print(f"Acceleration at t = 0: {curve.acceleration(0).as_text()}")
    print(f"Speed at t = 0: {curve.speed(0):.6g}")
    print(f"Unit tangent at t = 0: {curve.unit_tangent(0).as_text()}")
    print(f"Unit normal at t = 0: {curve.unit_normal(0).as_text()}")
    print(f"Binormal at t = 0: {curve.binormal(0).as_text()}")
    print(f"Curvature at t = 0: {curve.curvature(0):.6g}")
    print(f"Torsion at t = 0: {curve.torsion(0):.6g}")
    print(f"Arc length from t = 0 to t = pi: {curve.arc_length(0, math.pi):.6g}")
    print(f"Tangent line at t = 0: {curve.tangent_line(0).parametric_equations()}")


def run_space_geometry_demo() -> None:
    """Show the line and plane tools in space."""
    line = line_from_points(Point3D(1, 2, -1), Point3D(3, 1, 2))
    plane = make_plane(Point3D(1, 2, -1), make_vector3d(2, -1, 3))
    xy_plane = plane_from_scalar_equation(0, 0, 1, 0)
    crossing_line = make_line(Point3D(1, 1, 1), make_vector3d(0, 0, -1))

    print("Lines and Planes in Space")
    print()
    print("Example line through (1, 2, -1) and (3, 1, 2)")
    print()
    print(line.vector_equation())
    print(line.parametric_equations())
    print(line.symmetric_equations())
    print()
    print("Example plane through (1, 2, -1) with normal <2, -1, 3>")
    print()
    print(plane.scalar_equation())
    print(plane.point_normal_form())
    print(
        "Distance from plane to (1, 2, 0): "
        f"{plane.distance_to_point(Point3D(1, 2, 0)):.6g}"
    )
    intersection = crossing_line.intersection_with_plane(xy_plane)
    print(f"Line-plane intersection with z = 0: {intersection}")


def run_parametric_demo() -> None:
    """Show the parametric-equation tools on a sample curve."""
    curve = make_curve(
        x_of_t=lambda t: t**2 - 1,
        y_of_t=lambda t: 2 * t + 3,
    )

    print("Parametric Equations")
    print()
    print("Example curve: x = t^2 - 1, y = 2t + 3")
    print()
    print(points_table(curve.sample(start=-2, stop=2, steps=4)))
    print()
    print(f"Slope dy/dx at t = 2: {curve.slope(2):.6g}")
    print(f"Second derivative d2y/dx2 at t = 2: {curve.second_derivative(2):.6g}")
    print(f"Tangent line at t = 2: {curve.tangent_line(2).as_text()}")
    print(f"Normal line at t = 2: {curve.normal_line(2).as_text()}")
    print(f"Concavity at t = 2: {curve.concavity(2)}")
    print(f"Speed at t = 2: {curve.speed(2):.6g}")
    print(
        "Signed area under the curve from t = -2 to t = 2: "
        f"{curve.signed_area_under_curve(-2, 2):.6g}"
    )
    print(f"Approximate arc length from t = -2 to t = 2: {curve.arc_length(-2, 2):.6g}")
    print(
        "Approximate surface area about the x-axis from t = 0 to t = 2: "
        f"{curve.surface_area_about_x_axis(0, 2):.6g}"
    )


def run_polar_demo() -> None:
    """Show the polar-coordinate tools on a sample curve."""
    curve = make_polar_curve(lambda theta: 1 + theta)

    print("Polar Coordinates")
    print()
    print("Example curve: r = 1 + theta")
    print()
    print(polar_to_cartesian_table(curve.sample(start=0, stop=2, steps=4)))
    print()
    parametric_curve = curve.to_parametric_curve()
    print(
        "Equivalent parametric point at theta = 1: "
        f"({parametric_curve.point_at(1).x:.6g}, "
        f"{parametric_curve.point_at(1).y:.6g})"
    )
    print(f"dr/dtheta at theta = 1: {curve.dr_dtheta(1):.6g}")
    print(
        "Approximate polar area from theta = 0 to theta = 2: "
        f"{curve.area(0, 2):.6g}"
    )
    print(
        "Approximate polar arc length from theta = 0 to theta = 2: "
        f"{curve.arc_length(0, 2):.6g}"
    )


def run_space_coordinates_demo() -> None:
    """Show cylindrical and spherical coordinate conversions."""
    cylindrical = CylindricalPoint(2, math.pi / 2, 3)
    spherical = SphericalPoint(4, math.pi / 2, math.pi / 3)

    print("Cylindrical and Spherical Coordinates")
    print()
    print("Example cylindrical point: (r, theta, z) = (2, pi/2, 3)")
    print()
    print(cylindrical_to_rectangular_table([cylindrical]))
    print(f"Cylindrical volume factor r: {cylindrical.volume_element_factor():.6g}")
    print(f"As spherical coordinates: {cylindrical.to_spherical().as_text()}")
    print()
    print("Example spherical point: (rho, theta, phi) = (4, pi/2, pi/3)")
    print()
    print(spherical_to_rectangular_table([spherical]))
    print(
        "Spherical volume factor rho^2*sin(phi): "
        f"{spherical.volume_element_factor():.6g}"
    )
    print(f"As cylindrical coordinates: {spherical.to_cylindrical().as_text()}")


def run_conic_demo() -> None:
    """Show the conic-section tools on sample equations."""
    circle = make_conic(x_squared=1, y_squared=1, x=-4, y=6, constant=-12)
    parabola = make_conic(x_squared=1, y=-4)

    print("Conic Sections")
    print()
    print("Example equation: x^2 + y^2 - 4x + 6y - 12 = 0")
    print()
    print(circle.standard_form().as_text())
    print()
    print("Example equation: x^2 - 4y = 0")
    print()
    print(parabola.standard_form().as_text())


def run_quadric_demo() -> None:
    """Show the quadric-surface tools on sample equations."""
    ellipsoid = make_quadric(x_squared=36, y_squared=16, z_squared=9, constant=-144)
    paraboloid = make_quadric(x_squared=1 / 4, y_squared=1 / 9, z=-1)

    print("Quadric Surfaces")
    print()
    print("Example equation: 36x^2 + 16y^2 + 9z^2 - 144 = 0")
    print()
    print(ellipsoid.standard_form().as_text())
    print()
    print("Example equation: x^2/4 + y^2/9 - z = 0")
    print()
    print(paraboloid.standard_form().as_text())


if __name__ == "__main__":
    run_demo()
