"""Utilities for lines and planes in three-dimensional space."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.vectors import Point3D, Vector3D


_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Line3D:
    """A line in space, described by a point and direction vector."""

    point: Point3D
    direction: Vector3D

    def __post_init__(self) -> None:
        if self.direction.is_zero():
            raise ValueError("A line direction vector must be nonzero.")

    @classmethod
    def from_points(cls, start: Point3D, end: Point3D) -> "Line3D":
        """Create a line through two distinct points."""
        return cls(point=start, direction=start.vector_to(end))

    def point_at(self, parameter: float) -> Point3D:
        """Evaluate the line at parameter value ``t``."""
        return Point3D(
            x=self.point.x + parameter * self.direction.x,
            y=self.point.y + parameter * self.direction.y,
            z=self.point.z + parameter * self.direction.z,
        )

    def vector_equation(self, parameter: str = "t") -> str:
        """Return the vector equation of the line."""
        return (
            f"r({parameter}) = {_format_point(self.point)} + "
            f"{parameter}{_format_vector(self.direction)}"
        )

    def parametric_equations(self, parameter: str = "t") -> str:
        """Return parametric equations for the line."""
        return ", ".join(
            [
                _parameter_equation("x", self.point.x, self.direction.x, parameter),
                _parameter_equation("y", self.point.y, self.direction.y, parameter),
                _parameter_equation("z", self.point.z, self.direction.z, parameter),
            ]
        )

    def symmetric_equations(self) -> str:
        """Return symmetric equations, with fixed-coordinate notes when needed."""
        moving_terms: list[str] = []
        fixed_terms: list[str] = []
        for variable, offset, coefficient in (
            ("x", self.point.x, self.direction.x),
            ("y", self.point.y, self.direction.y),
            ("z", self.point.z, self.direction.z),
        ):
            if _is_close(coefficient):
                fixed_terms.append(f"{variable} = {_format_number(offset)}")
            else:
                moving_terms.append(_symmetric_term(variable, offset, coefficient))

        equation = " = ".join(moving_terms)
        if fixed_terms:
            return f"{equation}; {', '.join(fixed_terms)}"
        return equation

    def contains_point(
        self,
        point: Point3D,
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether a point lies on this line."""
        offset = self.point.vector_to(point)
        if offset.is_zero(tolerance):
            return True
        return self.direction.cross(offset).magnitude() <= tolerance

    def distance_to_point(self, point: Point3D) -> float:
        """Return the shortest distance from this line to a point."""
        offset = self.point.vector_to(point)
        return offset.cross(self.direction).magnitude() / self.direction.magnitude()

    def is_parallel_to(
        self,
        other: "Line3D",
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether two lines have parallel direction vectors."""
        return self.direction.is_parallel_to(other.direction, tolerance)

    def is_orthogonal_to(
        self,
        other: "Line3D",
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether two lines have orthogonal direction vectors."""
        return self.direction.is_orthogonal_to(other.direction, tolerance)

    def angle_with_line(self, other: "Line3D") -> float:
        """Return the smaller angle between two lines in radians."""
        return _acute_angle_between_vectors(self.direction, other.direction)

    def distance_to_line(self, other: "Line3D") -> float:
        """Return the shortest distance between this line and another line."""
        cross = self.direction.cross(other.direction)
        point_offset = self.point.vector_to(other.point)
        if cross.is_zero():
            return self.distance_to_point(other.point)
        return abs(point_offset.dot(cross)) / cross.magnitude()

    def intersection_with_plane(self, plane: "Plane3D") -> Point3D | None:
        """Return the unique intersection point with a plane, if one exists."""
        denominator = plane.normal.dot(self.direction)
        numerator = plane.constant() - plane.normal.dot(_point_as_vector(self.point))
        if _is_close(denominator):
            if plane.contains_point(self.point):
                raise ValueError(
                    "The line lies in the plane; intersection is not unique."
                )
            return None
        return self.point_at(numerator / denominator)

    def as_text(self) -> str:
        """Return the common equation forms for this line."""
        return "\n".join(
            [
                f"Vector equation: {self.vector_equation()}",
                f"Parametric equations: {self.parametric_equations()}",
                f"Symmetric equations: {self.symmetric_equations()}",
            ]
        )


@dataclass(frozen=True)
class Plane3D:
    """A plane in space, described by a point and normal vector."""

    point: Point3D
    normal: Vector3D

    def __post_init__(self) -> None:
        if self.normal.is_zero():
            raise ValueError("A plane normal vector must be nonzero.")

    @classmethod
    def from_point_and_normal(cls, point: Point3D, normal: Vector3D) -> "Plane3D":
        """Create a plane from a point and normal vector."""
        return cls(point=point, normal=normal)

    @classmethod
    def from_points(
        cls,
        first: Point3D,
        second: Point3D,
        third: Point3D,
    ) -> "Plane3D":
        """Create a plane through three noncollinear points."""
        first_direction = first.vector_to(second)
        second_direction = first.vector_to(third)
        return cls(point=first, normal=first_direction.cross(second_direction))

    @classmethod
    def from_scalar_equation(
        cls,
        x: float,
        y: float,
        z: float,
        constant: float,
    ) -> "Plane3D":
        """Create a plane from ``Ax + By + Cz = D``."""
        normal = Vector3D(x=x, y=y, z=z)
        if normal.is_zero():
            raise ValueError("A plane normal vector must be nonzero.")
        if not _is_close(x):
            point = Point3D(constant / x, 0, 0)
        elif not _is_close(y):
            point = Point3D(0, constant / y, 0)
        else:
            point = Point3D(0, 0, constant / z)
        return cls(point=point, normal=normal)

    def constant(self) -> float:
        """Return ``D`` in the scalar equation ``Ax + By + Cz = D``."""
        return self.normal.dot(_point_as_vector(self.point))

    def coefficients(self) -> tuple[float, float, float, float]:
        """Return ``(A, B, C, D)`` for ``Ax + By + Cz = D``."""
        return (self.normal.x, self.normal.y, self.normal.z, self.constant())

    def evaluate(self, point: Point3D) -> float:
        """Evaluate ``Ax + By + Cz - D`` at a point."""
        return self.normal.dot(_point_as_vector(point)) - self.constant()

    def contains_point(
        self,
        point: Point3D,
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether a point lies on this plane."""
        return abs(self.evaluate(point)) <= tolerance

    def scalar_equation(self) -> str:
        """Return the scalar equation ``Ax + By + Cz = D``."""
        terms = _scalar_terms(
            (
                (self.normal.x, "x"),
                (self.normal.y, "y"),
                (self.normal.z, "z"),
            )
        )
        return f"{terms} = {_format_number(self.constant())}"

    def point_normal_form(self) -> str:
        """Return point-normal form for the plane."""
        terms = _scalar_terms(
            (
                (self.normal.x, _linear_variable("x", self.point.x)),
                (self.normal.y, _linear_variable("y", self.point.y)),
                (self.normal.z, _linear_variable("z", self.point.z)),
            )
        )
        return f"{terms} = 0"

    def distance_to_point(self, point: Point3D) -> float:
        """Return the shortest distance from this plane to a point."""
        return abs(self.evaluate(point)) / self.normal.magnitude()

    def is_parallel_to(
        self,
        other: "Plane3D",
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether two planes have parallel normal vectors."""
        return self.normal.is_parallel_to(other.normal, tolerance)

    def is_orthogonal_to(
        self,
        other: "Plane3D",
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether two planes have orthogonal normal vectors."""
        return self.normal.is_orthogonal_to(other.normal, tolerance)

    def angle_with_plane(self, other: "Plane3D") -> float:
        """Return the smaller angle between two planes in radians."""
        return _acute_angle_between_vectors(self.normal, other.normal)

    def angle_with_line(self, line: Line3D) -> float:
        """Return the smaller angle between this plane and a line in radians."""
        ratio = abs(self.normal.dot(line.direction)) / (
            self.normal.magnitude() * line.direction.magnitude()
        )
        return math.asin(_clamp(ratio, -1.0, 1.0))

    def intersection_with_line(self, line: Line3D) -> Point3D | None:
        """Return the unique intersection point with a line, if one exists."""
        return line.intersection_with_plane(self)

    def intersection_with_plane(self, other: "Plane3D") -> Line3D | None:
        """Return the line of intersection with another plane, if unique."""
        direction = self.normal.cross(other.normal)
        if direction.is_zero():
            return None

        first_x, first_y, first_z, first_d = self.coefficients()
        second_x, second_y, second_z, second_d = other.coefficients()

        if abs(direction.x) >= abs(direction.y) and abs(direction.x) >= abs(direction.z):
            y, z = _solve_two_by_two(
                first_y,
                first_z,
                first_d,
                second_y,
                second_z,
                second_d,
            )
            point = Point3D(0, y, z)
        elif abs(direction.y) >= abs(direction.z):
            x, z = _solve_two_by_two(
                first_x,
                first_z,
                first_d,
                second_x,
                second_z,
                second_d,
            )
            point = Point3D(x, 0, z)
        else:
            x, y = _solve_two_by_two(
                first_x,
                first_y,
                first_d,
                second_x,
                second_y,
                second_d,
            )
            point = Point3D(x, y, 0)

        return Line3D(point=point, direction=direction)

    def as_text(self) -> str:
        """Return the common equation forms for this plane."""
        return "\n".join(
            [
                f"Scalar equation: {self.scalar_equation()}",
                f"Point-normal form: {self.point_normal_form()}",
            ]
        )


def make_line(point: Point3D, direction: Vector3D) -> Line3D:
    """Create a line from a point and direction vector."""
    return Line3D(point=point, direction=direction)


def line_from_points(start: Point3D, end: Point3D) -> Line3D:
    """Create a line through two distinct points."""
    return Line3D.from_points(start, end)


def make_plane(point: Point3D, normal: Vector3D) -> Plane3D:
    """Create a plane from a point and normal vector."""
    return Plane3D(point=point, normal=normal)


def plane_from_points(first: Point3D, second: Point3D, third: Point3D) -> Plane3D:
    """Create a plane through three noncollinear points."""
    return Plane3D.from_points(first, second, third)


def plane_from_scalar_equation(
    x: float,
    y: float,
    z: float,
    constant: float,
) -> Plane3D:
    """Create a plane from ``Ax + By + Cz = D``."""
    return Plane3D.from_scalar_equation(x, y, z, constant)


def _point_as_vector(point: Point3D) -> Vector3D:
    return Vector3D(point.x, point.y, point.z)


def _acute_angle_between_vectors(first: Vector3D, second: Vector3D) -> float:
    denominator = first.magnitude() * second.magnitude()
    if _is_close(denominator):
        raise ValueError("The angle with the zero vector is undefined.")
    cosine = abs(first.dot(second)) / denominator
    return math.acos(_clamp(cosine, -1.0, 1.0))


def _solve_two_by_two(
    first_a: float,
    first_b: float,
    first_c: float,
    second_a: float,
    second_b: float,
    second_c: float,
) -> tuple[float, float]:
    determinant = first_a * second_b - second_a * first_b
    if _is_close(determinant):
        raise ValueError("The system does not have a unique two-variable solution.")
    first_value = (first_c * second_b - second_c * first_b) / determinant
    second_value = (first_a * second_c - second_a * first_c) / determinant
    return first_value, second_value


def _parameter_equation(
    variable: str,
    initial: float,
    coefficient: float,
    parameter: str,
) -> str:
    if _is_close(coefficient):
        return f"{variable} = {_format_number(initial)}"
    if _is_close(initial):
        return f"{variable} = {_parameter_term(coefficient, parameter)}"
    sign = "+" if coefficient > 0 else "-"
    return (
        f"{variable} = {_format_number(initial)} {sign} "
        f"{_parameter_term(abs(coefficient), parameter)}"
    )


def _parameter_term(coefficient: float, parameter: str) -> str:
    if _is_close(coefficient, 1.0):
        return parameter
    if _is_close(coefficient, -1.0):
        return f"-{parameter}"
    return f"{_format_number(coefficient)}{parameter}"


def _symmetric_term(variable: str, offset: float, coefficient: float) -> str:
    expression = _linear_variable(variable, offset)
    if _is_close(coefficient, 1.0):
        return expression
    return f"{expression}/{_format_number(coefficient)}"


def _linear_variable(variable: str, offset: float) -> str:
    if _is_close(offset):
        return variable
    if offset > 0:
        return f"({variable} - {_format_number(offset)})"
    return f"({variable} + {_format_number(abs(offset))})"


def _scalar_terms(terms: tuple[tuple[float, str], ...]) -> str:
    formatted_terms: list[str] = []
    for coefficient, expression in terms:
        if _is_close(coefficient):
            continue
        formatted_terms.append(_scalar_term(coefficient, expression, not formatted_terms))
    if not formatted_terms:
        return "0"
    return "".join(formatted_terms)


def _scalar_term(coefficient: float, expression: str, is_first: bool) -> str:
    sign = "-" if coefficient < 0 else "+"
    magnitude = abs(coefficient)
    if _is_close(magnitude, 1.0):
        term = expression
    else:
        term = f"{_format_number(magnitude)}{expression}"
    if is_first:
        return f"-{term}" if sign == "-" else term
    return f" {sign} {term}"


def _format_point(point: Point3D) -> str:
    return (
        f"({_format_number(point.x)}, {_format_number(point.y)}, "
        f"{_format_number(point.z)})"
    )


def _format_vector(vector: Vector3D) -> str:
    return (
        f"<{_format_number(vector.x)}, {_format_number(vector.y)}, "
        f"{_format_number(vector.z)}>"
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _is_close(
    value: float,
    target: float = 0.0,
    tolerance: float = _TOLERANCE,
) -> bool:
    return math.isclose(value, target, abs_tol=tolerance)


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"
