"""Utilities for classifying and studying conic sections."""

from __future__ import annotations

import math
from dataclasses import dataclass


_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Point:
    """A point in the rectangular plane."""

    x: float
    y: float


@dataclass(frozen=True)
class ConicFeatures:
    """Standard-form details for an axis-aligned conic section."""

    kind: str
    standard_form: str
    center: Point | None = None
    vertex: Point | None = None
    orientation: str | None = None
    radius: float | None = None
    semi_major_axis: float | None = None
    semi_minor_axis: float | None = None
    semi_transverse_axis: float | None = None
    semi_conjugate_axis: float | None = None
    focal_parameter: float | None = None
    focal_distance: float | None = None
    eccentricity: float | None = None
    vertices: tuple[Point, ...] = ()
    co_vertices: tuple[Point, ...] = ()
    foci: tuple[Point, ...] = ()
    asymptotes: tuple[str, ...] = ()
    directrix: str | None = None

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        rows = [f"Type: {self.kind}", f"Standard form: {self.standard_form}"]
        if self.center is not None:
            rows.append(f"Center: {_format_point(self.center)}")
        if self.vertex is not None:
            rows.append(f"Vertex: {_format_point(self.vertex)}")
        if self.orientation is not None:
            rows.append(f"Orientation: {self.orientation}")
        if self.radius is not None:
            rows.append(f"Radius: {_format_number(self.radius)}")
        if self.semi_major_axis is not None:
            rows.append(f"Semi-major axis: {_format_number(self.semi_major_axis)}")
        if self.semi_minor_axis is not None:
            rows.append(f"Semi-minor axis: {_format_number(self.semi_minor_axis)}")
        if self.semi_transverse_axis is not None:
            rows.append(
                f"Semi-transverse axis: {_format_number(self.semi_transverse_axis)}"
            )
        if self.semi_conjugate_axis is not None:
            rows.append(
                f"Semi-conjugate axis: {_format_number(self.semi_conjugate_axis)}"
            )
        if self.focal_parameter is not None:
            rows.append(f"Focal parameter p: {_format_number(self.focal_parameter)}")
        if self.focal_distance is not None:
            rows.append(f"Focal distance c: {_format_number(self.focal_distance)}")
        if self.eccentricity is not None:
            rows.append(f"Eccentricity: {_format_number(self.eccentricity)}")
        if self.vertices:
            rows.append(f"Vertices: {_format_points(self.vertices)}")
        if self.co_vertices:
            rows.append(f"Co-vertices: {_format_points(self.co_vertices)}")
        if self.foci:
            rows.append(f"Foci: {_format_points(self.foci)}")
        if self.asymptotes:
            rows.append(f"Asymptotes: {'; '.join(self.asymptotes)}")
        if self.directrix is not None:
            rows.append(f"Directrix: {self.directrix}")
        return "\n".join(rows)


@dataclass(frozen=True)
class ConicSection:
    """A conic from ``Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0``."""

    x_squared: float = 0.0
    xy: float = 0.0
    y_squared: float = 0.0
    x: float = 0.0
    y: float = 0.0
    constant: float = 0.0

    def evaluate(self, x: float, y: float) -> float:
        """Evaluate the implicit conic equation at ``(x, y)``."""
        return (
            self.x_squared * x**2
            + self.xy * x * y
            + self.y_squared * y**2
            + self.x * x
            + self.y * y
            + self.constant
        )

    def discriminant(self) -> float:
        """Return ``B^2 - 4AC`` for the conic."""
        return self.xy**2 - 4 * self.x_squared * self.y_squared

    def determinant(self) -> float:
        """Return the determinant of the symmetric conic matrix."""
        return _determinant(
            (
                (self.x_squared, self.xy / 2, self.x / 2),
                (self.xy / 2, self.y_squared, self.y / 2),
                (self.x / 2, self.y / 2, self.constant),
            )
        )

    def classification(self) -> str:
        """Classify the conic from its coefficients."""
        return classify_conic(
            self.x_squared,
            self.xy,
            self.y_squared,
            self.x,
            self.y,
            self.constant,
        )

    def is_axis_aligned(self) -> bool:
        """Return whether the conic has no ``xy`` rotation term."""
        return _is_close(self.xy)

    def standard_form(self) -> ConicFeatures:
        """Return standard-form features for an axis-aligned nondegenerate conic."""
        if not self.is_axis_aligned():
            raise ValueError(
                "Standard form is only supported for axis-aligned conics "
                "with xy coefficient equal to zero."
            )

        kind = self.classification()
        if kind == "circle":
            return self._circle_features()
        if kind == "ellipse":
            return self._ellipse_features()
        if kind == "hyperbola":
            return self._hyperbola_features()
        if kind == "parabola":
            return self._parabola_features()

        raise ValueError(f"Standard form is not available for {kind}.")

    def _centered_parts(self) -> tuple[Point, float, float, float]:
        if _is_close(self.x_squared) or _is_close(self.y_squared):
            raise ValueError("Both squared coefficients must be nonzero.")

        center = Point(
            x=-self.x / (2 * self.x_squared),
            y=-self.y / (2 * self.y_squared),
        )
        completed_constant = (
            self.constant
            - self.x**2 / (4 * self.x_squared)
            - self.y**2 / (4 * self.y_squared)
        )
        right_side = -completed_constant
        x_denominator = right_side / self.x_squared
        y_denominator = right_side / self.y_squared
        return center, right_side, x_denominator, y_denominator

    def _circle_features(self) -> ConicFeatures:
        center, _, x_denominator, y_denominator = self._centered_parts()
        if (
            x_denominator <= 0
            or y_denominator <= 0
            or not _is_close(x_denominator, y_denominator)
        ):
            raise ValueError("This circle does not have a positive real radius.")

        radius = math.sqrt(x_denominator)
        standard_form = (
            f"{_squared_variable('x', center.x)} + "
            f"{_squared_variable('y', center.y)} = {_format_number(radius**2)}"
        )
        return ConicFeatures(
            kind="circle",
            standard_form=standard_form,
            center=center,
            radius=radius,
        )

    def _ellipse_features(self) -> ConicFeatures:
        center, _, x_denominator, y_denominator = self._centered_parts()
        if x_denominator <= 0 or y_denominator <= 0:
            raise ValueError("This ellipse does not have positive real axes.")

        semi_axis_x = math.sqrt(x_denominator)
        semi_axis_y = math.sqrt(y_denominator)
        standard_form = (
            f"{_squared_fraction('x', center.x, x_denominator)} + "
            f"{_squared_fraction('y', center.y, y_denominator)} = 1"
        )

        if semi_axis_x >= semi_axis_y:
            orientation = "horizontal major axis"
            semi_major_axis = semi_axis_x
            semi_minor_axis = semi_axis_y
            focal_distance = math.sqrt(semi_axis_x**2 - semi_axis_y**2)
            vertices = (
                Point(center.x - semi_axis_x, center.y),
                Point(center.x + semi_axis_x, center.y),
            )
            co_vertices = (
                Point(center.x, center.y - semi_axis_y),
                Point(center.x, center.y + semi_axis_y),
            )
            foci = (
                Point(center.x - focal_distance, center.y),
                Point(center.x + focal_distance, center.y),
            )
        else:
            orientation = "vertical major axis"
            semi_major_axis = semi_axis_y
            semi_minor_axis = semi_axis_x
            focal_distance = math.sqrt(semi_axis_y**2 - semi_axis_x**2)
            vertices = (
                Point(center.x, center.y - semi_axis_y),
                Point(center.x, center.y + semi_axis_y),
            )
            co_vertices = (
                Point(center.x - semi_axis_x, center.y),
                Point(center.x + semi_axis_x, center.y),
            )
            foci = (
                Point(center.x, center.y - focal_distance),
                Point(center.x, center.y + focal_distance),
            )

        return ConicFeatures(
            kind="ellipse",
            standard_form=standard_form,
            center=center,
            orientation=orientation,
            semi_major_axis=semi_major_axis,
            semi_minor_axis=semi_minor_axis,
            focal_distance=focal_distance,
            eccentricity=focal_distance / semi_major_axis,
            vertices=vertices,
            co_vertices=co_vertices,
            foci=foci,
        )

    def _hyperbola_features(self) -> ConicFeatures:
        center, _, x_denominator, y_denominator = self._centered_parts()
        if _is_close(x_denominator) or _is_close(y_denominator):
            raise ValueError("This hyperbola has a zero denominator.")
        if x_denominator * y_denominator >= 0:
            raise ValueError("This hyperbola does not have one positive term.")

        if x_denominator > 0:
            orientation = "horizontal transverse axis"
            semi_transverse_axis = math.sqrt(x_denominator)
            semi_conjugate_axis = math.sqrt(abs(y_denominator))
            focal_distance = math.sqrt(x_denominator + abs(y_denominator))
            standard_form = (
                f"{_squared_fraction('x', center.x, x_denominator)} - "
                f"{_squared_fraction('y', center.y, abs(y_denominator))} = 1"
            )
            vertices = (
                Point(center.x - semi_transverse_axis, center.y),
                Point(center.x + semi_transverse_axis, center.y),
            )
            foci = (
                Point(center.x - focal_distance, center.y),
                Point(center.x + focal_distance, center.y),
            )
            asymptote_slope = semi_conjugate_axis / semi_transverse_axis
        else:
            orientation = "vertical transverse axis"
            semi_transverse_axis = math.sqrt(y_denominator)
            semi_conjugate_axis = math.sqrt(abs(x_denominator))
            focal_distance = math.sqrt(y_denominator + abs(x_denominator))
            standard_form = (
                f"{_squared_fraction('y', center.y, y_denominator)} - "
                f"{_squared_fraction('x', center.x, abs(x_denominator))} = 1"
            )
            vertices = (
                Point(center.x, center.y - semi_transverse_axis),
                Point(center.x, center.y + semi_transverse_axis),
            )
            foci = (
                Point(center.x, center.y - focal_distance),
                Point(center.x, center.y + focal_distance),
            )
            asymptote_slope = semi_transverse_axis / semi_conjugate_axis

        return ConicFeatures(
            kind="hyperbola",
            standard_form=standard_form,
            center=center,
            orientation=orientation,
            semi_transverse_axis=semi_transverse_axis,
            semi_conjugate_axis=semi_conjugate_axis,
            focal_distance=focal_distance,
            eccentricity=focal_distance / semi_transverse_axis,
            vertices=vertices,
            foci=foci,
            asymptotes=_asymptotes(center, asymptote_slope),
        )

    def _parabola_features(self) -> ConicFeatures:
        if not _is_close(self.x_squared) and _is_close(self.y_squared):
            return self._vertical_parabola_features()
        if _is_close(self.x_squared) and not _is_close(self.y_squared):
            return self._horizontal_parabola_features()
        raise ValueError("A parabola must have exactly one squared term.")

    def _vertical_parabola_features(self) -> ConicFeatures:
        if _is_close(self.y):
            raise ValueError("This equation does not describe a vertical parabola.")

        h = -self.x / (2 * self.x_squared)
        completed_constant = self.constant - self.x**2 / (4 * self.x_squared)
        k = -completed_constant / self.y
        p = -self.y / (4 * self.x_squared)
        vertex = Point(h, k)
        focus = Point(h, k + p)
        standard_form = (
            f"{_squared_variable('x', h)} = "
            f"{_coefficient_expression(4 * p, _linear_variable('y', k))}"
        )
        return ConicFeatures(
            kind="parabola",
            standard_form=standard_form,
            vertex=vertex,
            orientation="opens up" if p > 0 else "opens down",
            focal_parameter=p,
            foci=(focus,),
            directrix=f"y = {_format_number(k - p)}",
        )

    def _horizontal_parabola_features(self) -> ConicFeatures:
        if _is_close(self.x):
            raise ValueError("This equation does not describe a horizontal parabola.")

        k = -self.y / (2 * self.y_squared)
        completed_constant = self.constant - self.y**2 / (4 * self.y_squared)
        h = -completed_constant / self.x
        p = -self.x / (4 * self.y_squared)
        vertex = Point(h, k)
        focus = Point(h + p, k)
        standard_form = (
            f"{_squared_variable('y', k)} = "
            f"{_coefficient_expression(4 * p, _linear_variable('x', h))}"
        )
        return ConicFeatures(
            kind="parabola",
            standard_form=standard_form,
            vertex=vertex,
            orientation="opens right" if p > 0 else "opens left",
            focal_parameter=p,
            foci=(focus,),
            directrix=f"x = {_format_number(h - p)}",
        )


def make_conic(
    x_squared: float = 0.0,
    xy: float = 0.0,
    y_squared: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    constant: float = 0.0,
) -> ConicSection:
    """Create a conic from ``Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0``."""
    return ConicSection(
        x_squared=x_squared,
        xy=xy,
        y_squared=y_squared,
        x=x,
        y=y,
        constant=constant,
    )


def classify_conic(
    x_squared: float,
    xy: float,
    y_squared: float,
    x: float = 0.0,
    y: float = 0.0,
    constant: float = 0.0,
) -> str:
    """Classify a conic from ``Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0``."""
    if _is_close(x_squared) and _is_close(xy) and _is_close(y_squared):
        return "not a conic"

    conic = ConicSection(x_squared, xy, y_squared, x, y, constant)
    if _is_close(conic.determinant()):
        return "degenerate conic"

    discriminant = conic.discriminant()
    if discriminant < -_TOLERANCE:
        if _is_close(xy) and _is_close(x_squared, y_squared):
            return "circle"
        return "ellipse"
    if discriminant > _TOLERANCE:
        return "hyperbola"
    return "parabola"


def _determinant(matrix: tuple[tuple[float, float, float], ...]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _is_close(value: float, target: float = 0.0) -> bool:
    return math.isclose(value, target, abs_tol=_TOLERANCE)


def _squared_fraction(variable: str, offset: float, denominator: float) -> str:
    expression = _squared_variable(variable, offset)
    if _is_close(denominator, 1.0):
        return expression
    return f"{expression}/{_format_number(denominator)}"


def _squared_variable(variable: str, offset: float) -> str:
    return f"{_linear_variable(variable, offset)}^2"


def _linear_variable(variable: str, offset: float) -> str:
    if _is_close(offset):
        return variable
    if offset > 0:
        return f"({variable} - {_format_number(offset)})"
    return f"({variable} + {_format_number(abs(offset))})"


def _coefficient_expression(coefficient: float, expression: str) -> str:
    if _is_close(coefficient, 1.0):
        return expression
    if _is_close(coefficient, -1.0):
        return f"-{expression}"
    return f"{_format_number(coefficient)}{expression}"


def _asymptotes(center: Point, slope: float) -> tuple[str, str]:
    left = _linear_variable("y", center.y)
    right = _linear_variable("x", center.x)
    return (
        f"{left} = {_coefficient_expression(slope, right)}",
        f"{left} = {_coefficient_expression(-slope, right)}",
    )


def _format_point(point: Point) -> str:
    return f"({_format_number(point.x)}, {_format_number(point.y)})"


def _format_points(points: tuple[Point, ...]) -> str:
    return ", ".join(_format_point(point) for point in points)


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"
