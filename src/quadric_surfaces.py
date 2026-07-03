"""Utilities for classifying and studying quadric surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.vectors import Point3D


_TOLERANCE = 1e-10


@dataclass(frozen=True)
class QuadricFeatures:
    """Standard-form details for an axis-aligned quadric surface."""

    kind: str
    standard_form: str
    center: Point3D | None = None
    vertex: Point3D | None = None
    axis: str | None = None
    orientation: str | None = None
    radius: float | None = None
    x_denominator: float | None = None
    y_denominator: float | None = None
    z_denominator: float | None = None
    x_radius: float | None = None
    y_radius: float | None = None
    z_radius: float | None = None

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        rows = [f"Type: {self.kind}", f"Standard form: {self.standard_form}"]
        if self.center is not None:
            rows.append(f"Center: {_format_point(self.center)}")
        if self.vertex is not None:
            rows.append(f"Vertex: {_format_point(self.vertex)}")
        if self.axis is not None:
            rows.append(f"Axis: {self.axis}")
        if self.orientation is not None:
            rows.append(f"Orientation: {self.orientation}")
        if self.radius is not None:
            rows.append(f"Radius: {_format_number(self.radius)}")
        if self.x_radius is not None:
            rows.append(f"x-radius: {_format_number(self.x_radius)}")
        if self.y_radius is not None:
            rows.append(f"y-radius: {_format_number(self.y_radius)}")
        if self.z_radius is not None:
            rows.append(f"z-radius: {_format_number(self.z_radius)}")
        return "\n".join(rows)


@dataclass(frozen=True)
class QuadricSurface:
    """A second-degree surface in ``x``, ``y``, and ``z``."""

    x_squared: float = 0.0
    y_squared: float = 0.0
    z_squared: float = 0.0
    xy: float = 0.0
    xz: float = 0.0
    yz: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    constant: float = 0.0

    def evaluate(self, x: float, y: float, z: float) -> float:
        """Evaluate the implicit quadric equation at ``(x, y, z)``."""
        return (
            self.x_squared * x**2
            + self.y_squared * y**2
            + self.z_squared * z**2
            + self.xy * x * y
            + self.xz * x * z
            + self.yz * y * z
            + self.x * x
            + self.y * y
            + self.z * z
            + self.constant
        )

    def is_axis_aligned(self) -> bool:
        """Return whether the quadric has no mixed-product rotation terms."""
        return _is_close(self.xy) and _is_close(self.xz) and _is_close(self.yz)

    def classification(self) -> str:
        """Classify this quadric surface."""
        if not self.is_axis_aligned():
            return "rotated quadric surface"
        return self.standard_form().kind

    def standard_form(self) -> QuadricFeatures:
        """Return standard-form features for an axis-aligned quadric surface."""
        if not self.is_axis_aligned():
            raise ValueError(
                "Standard form is only supported for axis-aligned quadrics "
                "with xy, xz, and yz coefficients equal to zero."
            )

        completed = _complete_squares(self)
        squared_variables = [
            variable
            for variable in _VARIABLES
            if not _is_close(completed.squared_coefficients[variable])
        ]
        linear_variables = [
            variable
            for variable in _VARIABLES
            if not _is_close(completed.linear_coefficients[variable])
        ]

        if len(squared_variables) == 3 and not linear_variables:
            return _central_surface_features(completed)
        if len(squared_variables) == 2 and not linear_variables:
            return _cylinder_features(completed, squared_variables)
        if len(squared_variables) == 2 and len(linear_variables) == 1:
            return _paraboloid_features(completed, squared_variables, linear_variables[0])
        if len(squared_variables) == 1 and len(linear_variables) == 1:
            return _parabolic_cylinder_features(
                completed,
                squared_variables[0],
                linear_variables[0],
            )

        raise ValueError("This quadric surface is degenerate or not supported.")


@dataclass(frozen=True)
class _CompletedQuadric:
    squared_coefficients: dict[str, float]
    linear_coefficients: dict[str, float]
    offsets: dict[str, float]
    constant: float


def make_quadric(
    x_squared: float = 0.0,
    y_squared: float = 0.0,
    z_squared: float = 0.0,
    xy: float = 0.0,
    xz: float = 0.0,
    yz: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    constant: float = 0.0,
) -> QuadricSurface:
    """Create a quadric surface from second-degree coefficients."""
    return QuadricSurface(
        x_squared=x_squared,
        y_squared=y_squared,
        z_squared=z_squared,
        xy=xy,
        xz=xz,
        yz=yz,
        x=x,
        y=y,
        z=z,
        constant=constant,
    )


def classify_quadric(
    x_squared: float = 0.0,
    y_squared: float = 0.0,
    z_squared: float = 0.0,
    xy: float = 0.0,
    xz: float = 0.0,
    yz: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    constant: float = 0.0,
) -> str:
    """Classify a quadric surface from its coefficients."""
    return make_quadric(
        x_squared=x_squared,
        y_squared=y_squared,
        z_squared=z_squared,
        xy=xy,
        xz=xz,
        yz=yz,
        x=x,
        y=y,
        z=z,
        constant=constant,
    ).classification()


def _complete_squares(surface: QuadricSurface) -> _CompletedQuadric:
    squared_coefficients = {
        "x": surface.x_squared,
        "y": surface.y_squared,
        "z": surface.z_squared,
    }
    linear_coefficients = {"x": surface.x, "y": surface.y, "z": surface.z}
    offsets = {"x": 0.0, "y": 0.0, "z": 0.0}
    completed_constant = surface.constant

    for variable in _VARIABLES:
        squared = squared_coefficients[variable]
        linear = linear_coefficients[variable]
        if _is_close(squared):
            continue
        offsets[variable] = -linear / (2 * squared)
        completed_constant -= linear**2 / (4 * squared)
        linear_coefficients[variable] = 0.0

    return _CompletedQuadric(
        squared_coefficients=squared_coefficients,
        linear_coefficients=linear_coefficients,
        offsets=offsets,
        constant=completed_constant,
    )


def _central_surface_features(completed: _CompletedQuadric) -> QuadricFeatures:
    right_side = -completed.constant
    center = _point_from_offsets(completed.offsets)
    coefficients = completed.squared_coefficients

    if _is_close(right_side):
        signs = [_sign(coefficients[variable]) for variable in _VARIABLES]
        if len(set(signs)) == 1:
            return QuadricFeatures(
                kind="point",
                standard_form=_equation(_central_terms(completed, right_side=0), "0"),
                center=center,
            )
        return QuadricFeatures(
            kind="elliptic cone",
            standard_form=_equation(_cone_terms(completed), "0"),
            center=center,
            vertex=center,
        )

    normalized_terms = _normalized_central_terms(completed, right_side)
    positive_variables = [
        variable for variable, sign, _, _ in normalized_terms if sign > 0
    ]
    negative_variables = [
        variable for variable, sign, _, _ in normalized_terms if sign < 0
    ]

    if len(positive_variables) == 3:
        denominators = _denominators_from_terms(normalized_terms)
        kind = "sphere" if _same_denominators(denominators.values()) else "ellipsoid"
        radius = (
            math.sqrt(next(iter(denominators.values()))) if kind == "sphere" else None
        )
        return QuadricFeatures(
            kind=kind,
            standard_form=_equation(_terms_from_normalized(normalized_terms), "1"),
            center=center,
            radius=radius,
            x_denominator=denominators["x"],
            y_denominator=denominators["y"],
            z_denominator=denominators["z"],
            x_radius=math.sqrt(denominators["x"]),
            y_radius=math.sqrt(denominators["y"]),
            z_radius=math.sqrt(denominators["z"]),
        )

    if len(positive_variables) == 2 and len(negative_variables) == 1:
        return QuadricFeatures(
            kind="hyperboloid of one sheet",
            standard_form=_equation(_terms_from_normalized(normalized_terms), "1"),
            center=center,
            axis=f"{negative_variables[0]}-axis",
        )

    if len(positive_variables) == 1 and len(negative_variables) == 2:
        return QuadricFeatures(
            kind="hyperboloid of two sheets",
            standard_form=_equation(_terms_from_normalized(normalized_terms), "1"),
            center=center,
            axis=f"{positive_variables[0]}-axis",
        )

    raise ValueError("This central quadric has no real surface.")


def _cylinder_features(
    completed: _CompletedQuadric,
    squared_variables: list[str],
) -> QuadricFeatures:
    right_side = -completed.constant
    if _is_close(right_side):
        raise ValueError("This cylinder is degenerate or not supported.")

    normalized_terms = _normalized_terms(completed, squared_variables, right_side)
    positive_count = sum(1 for _, sign, _, _ in normalized_terms if sign > 0)
    axis = _missing_variable(squared_variables)

    if positive_count == 2:
        denominators = _denominators_from_terms(normalized_terms)
        kind = (
            "circular cylinder"
            if _same_denominators(denominators.values())
            else "elliptic cylinder"
        )
        radius = (
            math.sqrt(next(iter(denominators.values())))
            if kind == "circular cylinder"
            else None
        )
        return QuadricFeatures(
            kind=kind,
            standard_form=_equation(_terms_from_normalized(normalized_terms), "1"),
            center=_point_from_offsets(completed.offsets),
            axis=f"{axis}-axis",
            radius=radius,
        )

    if positive_count == 1:
        return QuadricFeatures(
            kind="hyperbolic cylinder",
            standard_form=_equation(_terms_from_normalized(normalized_terms), "1"),
            center=_point_from_offsets(completed.offsets),
            axis=f"{axis}-axis",
        )

    raise ValueError("This cylinder has no real surface.")


def _paraboloid_features(
    completed: _CompletedQuadric,
    squared_variables: list[str],
    linear_variable: str,
) -> QuadricFeatures:
    if linear_variable in squared_variables:
        raise ValueError("A paraboloid needs one linear variable without a square term.")

    coefficients = completed.squared_coefficients
    signs = [_sign(coefficients[variable]) for variable in squared_variables]
    if signs[0] == signs[1]:
        return _elliptic_paraboloid_features(
            completed,
            squared_variables,
            linear_variable,
        )
    return _hyperbolic_paraboloid_features(completed, squared_variables, linear_variable)


def _elliptic_paraboloid_features(
    completed: _CompletedQuadric,
    squared_variables: list[str],
    linear_variable: str,
) -> QuadricFeatures:
    coefficients = completed.squared_coefficients
    linear = completed.linear_coefficients[linear_variable]
    multiplier = _sign(coefficients[squared_variables[0]])
    rhs_coefficient = -multiplier * linear
    if _is_close(rhs_coefficient):
        raise ValueError("A paraboloid needs a nonzero linear coefficient.")

    terms = [
        (
            variable,
            1,
            abs(rhs_coefficient) / abs(coefficients[variable]),
            completed.offsets[variable],
        )
        for variable in squared_variables
    ]
    rhs = _oriented_linear_expression(
        linear_variable,
        _linear_vertex_coordinate(completed, linear_variable),
        rhs_coefficient,
    )
    vertex = _paraboloid_vertex(completed, linear_variable)
    return QuadricFeatures(
        kind="elliptic paraboloid",
        standard_form=_equation(_terms_from_normalized(terms), rhs),
        vertex=vertex,
        axis=f"{linear_variable}-axis",
        orientation=_orientation(linear_variable, rhs_coefficient),
    )


def _hyperbolic_paraboloid_features(
    completed: _CompletedQuadric,
    squared_variables: list[str],
    linear_variable: str,
) -> QuadricFeatures:
    coefficients = completed.squared_coefficients
    linear = completed.linear_coefficients[linear_variable]
    rhs_coefficient = -linear
    if _is_close(rhs_coefficient):
        raise ValueError("A paraboloid needs a nonzero linear coefficient.")

    terms = [
        (
            variable,
            _sign(coefficients[variable]),
            abs(rhs_coefficient) / abs(coefficients[variable]),
            completed.offsets[variable],
        )
        for variable in squared_variables
    ]
    rhs = _oriented_linear_expression(
        linear_variable,
        _linear_vertex_coordinate(completed, linear_variable),
        rhs_coefficient,
    )
    vertex = _paraboloid_vertex(completed, linear_variable)
    return QuadricFeatures(
        kind="hyperbolic paraboloid",
        standard_form=_equation(_terms_from_normalized(terms), rhs),
        vertex=vertex,
        axis=f"{linear_variable}-axis",
        orientation=f"saddle along the {linear_variable}-axis",
    )


def _parabolic_cylinder_features(
    completed: _CompletedQuadric,
    squared_variable: str,
    linear_variable: str,
) -> QuadricFeatures:
    if squared_variable == linear_variable:
        raise ValueError(
            "A parabolic cylinder needs one squared and one linear variable."
        )

    coefficient = completed.squared_coefficients[squared_variable]
    linear = completed.linear_coefficients[linear_variable]
    multiplier = _sign(coefficient)
    rhs_coefficient = -multiplier * linear
    if _is_close(rhs_coefficient):
        raise ValueError("A parabolic cylinder needs a nonzero linear coefficient.")

    term = (
        squared_variable,
        1,
        abs(rhs_coefficient) / abs(coefficient),
        completed.offsets[squared_variable],
    )
    rhs = _oriented_linear_expression(
        linear_variable,
        _linear_vertex_coordinate(completed, linear_variable),
        rhs_coefficient,
    )
    axis = _missing_variable([squared_variable, linear_variable])
    return QuadricFeatures(
        kind="parabolic cylinder",
        standard_form=_equation(_terms_from_normalized([term]), rhs),
        axis=f"{axis}-axis",
        orientation=_orientation(linear_variable, rhs_coefficient),
    )


def _normalized_central_terms(
    completed: _CompletedQuadric,
    right_side: float,
) -> list[tuple[str, int, float, float]]:
    return _normalized_terms(completed, list(_VARIABLES), right_side)


def _normalized_terms(
    completed: _CompletedQuadric,
    variables: list[str],
    right_side: float,
) -> list[tuple[str, int, float, float]]:
    terms = []
    for variable in variables:
        coefficient = completed.squared_coefficients[variable]
        terms.append(
            (
                variable,
                _sign(coefficient / right_side),
                abs(right_side / coefficient),
                completed.offsets[variable],
            )
        )
    return terms


def _central_terms(
    completed: _CompletedQuadric,
    right_side: float,
) -> list[tuple[str, int, float, float]]:
    if _is_close(right_side):
        return _cone_terms(completed)
    return _normalized_central_terms(completed, right_side)


def _cone_terms(completed: _CompletedQuadric) -> list[tuple[str, int, float, float]]:
    terms = []
    for variable in _VARIABLES:
        coefficient = completed.squared_coefficients[variable]
        terms.append(
            (
                variable,
                _sign(coefficient),
                1 / abs(coefficient),
                completed.offsets[variable],
            )
        )
    return terms


def _terms_from_normalized(terms: list[tuple[str, int, float, float]]) -> str:
    sorted_terms = sorted(terms, key=lambda term: (-term[1], _VARIABLES.index(term[0])))
    formatted_terms: list[str] = []
    for variable, sign, denominator, offset in sorted_terms:
        formatted_terms.append(
            _signed_term(
                sign,
                _squared_fraction(variable, offset, denominator),
                not formatted_terms,
            )
        )
    return "".join(formatted_terms)


def _equation(left: list[tuple[str, int, float, float]] | str, right: str) -> str:
    left_text = _terms_from_normalized(left) if isinstance(left, list) else left
    return f"{left_text} = {right}"


def _denominators_from_terms(
    terms: list[tuple[str, int, float, float]],
) -> dict[str, float]:
    return {variable: denominator for variable, _, denominator, _ in terms}


def _same_denominators(denominators: object) -> bool:
    values = list(denominators)
    if not values:
        return False
    return all(_is_close(value, values[0]) for value in values)


def _paraboloid_vertex(
    completed: _CompletedQuadric,
    linear_variable: str,
) -> Point3D:
    coordinates = dict(completed.offsets)
    coordinates[linear_variable] = _linear_vertex_coordinate(completed, linear_variable)
    return Point3D(coordinates["x"], coordinates["y"], coordinates["z"])


def _linear_vertex_coordinate(
    completed: _CompletedQuadric,
    variable: str,
) -> float:
    return -completed.constant / completed.linear_coefficients[variable]


def _oriented_linear_expression(
    variable: str,
    offset: float,
    coefficient: float,
) -> str:
    expression = _linear_variable(variable, offset)
    if coefficient > 0:
        return expression
    return f"-{expression}"


def _orientation(variable: str, coefficient: float) -> str:
    direction = "positive" if coefficient > 0 else "negative"
    return f"opens in the {direction} {variable}-direction"


def _missing_variable(variables: list[str]) -> str:
    missing = [variable for variable in _VARIABLES if variable not in variables]
    if len(missing) != 1:
        raise ValueError("Expected exactly one missing variable.")
    return missing[0]


def _point_from_offsets(offsets: dict[str, float]) -> Point3D:
    return Point3D(offsets["x"], offsets["y"], offsets["z"])


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


def _signed_term(sign: int, expression: str, is_first: bool) -> str:
    if is_first:
        return expression if sign > 0 else f"-{expression}"
    operator = "+" if sign > 0 else "-"
    return f" {operator} {expression}"


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _is_close(value: float, target: float = 0.0) -> bool:
    return math.isclose(value, target, abs_tol=_TOLERANCE)


def _format_point(point: Point3D) -> str:
    return (
        f"({_format_number(point.x)}, {_format_number(point.y)}, "
        f"{_format_number(point.z)})"
    )


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"


_VARIABLES = ("x", "y", "z")
