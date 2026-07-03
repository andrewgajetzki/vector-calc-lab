"""Study helpers for Calculus 3 topics."""

from __future__ import annotations

from src.conic_sections import make_conic
from src.parametric_equations import make_curve, points_table
from src.polar_coordinates import make_polar_curve, polar_to_cartesian_table
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
    run_space_geometry_demo()
    print()
    run_parametric_demo()
    print()
    run_polar_demo()
    print()
    run_conic_demo()


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


if __name__ == "__main__":
    run_demo()
