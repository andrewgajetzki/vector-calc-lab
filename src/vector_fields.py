"""Utilities for vector fields in the plane and in space."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from src.vectors import Point2D, Point3D, Vector2D, Vector3D


NumberFunction2D = Callable[[float, float], float]
NumberFunction3D = Callable[[float, float, float], float]


@dataclass(frozen=True)
class VectorFieldPoint2D:
    """A sampled vector from a plane vector field."""

    point: Point2D
    vector: Vector2D

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"F({_format_number(self.point.x)}, {_format_number(self.point.y)}) = "
            f"{self.vector.as_text()}"
        )


@dataclass(frozen=True)
class VectorFieldPoint3D:
    """A sampled vector from a space vector field."""

    point: Point3D
    vector: Vector3D

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"F({_format_number(self.point.x)}, {_format_number(self.point.y)}, "
            f"{_format_number(self.point.z)}) = {self.vector.as_text()}"
        )


@dataclass(frozen=True)
class VectorField2D:
    """A plane vector field ``F(x, y) = <P(x, y), Q(x, y)>``."""

    p_component: NumberFunction2D
    q_component: NumberFunction2D

    def value_at(self, x: float, y: float) -> Vector2D:
        """Evaluate the vector field at ``(x, y)``."""
        return Vector2D(self.p_component(x, y), self.q_component(x, y))

    def point_at(self, x: float, y: float) -> VectorFieldPoint2D:
        """Evaluate the vector field and keep the base point."""
        return VectorFieldPoint2D(point=Point2D(x, y), vector=self.value_at(x, y))

    def magnitude_at(self, x: float, y: float) -> float:
        """Return ``|F(x, y)|``."""
        return self.value_at(x, y).magnitude()

    def unit_at(self, x: float, y: float) -> Vector2D:
        """Return the unit vector in the direction of ``F(x, y)``."""
        return self.value_at(x, y).unit()

    def sample(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        x_steps: int,
        y_steps: int,
    ) -> list[VectorFieldPoint2D]:
        """Sample the field on an evenly spaced rectangular grid."""
        x_values = _linspace(
            _validate_bounds(x_bounds, "x_bounds"),
            x_steps,
            "x_steps",
        )
        y_values = _linspace(
            _validate_bounds(y_bounds, "y_bounds"),
            y_steps,
            "y_steps",
        )
        return [self.point_at(x, y) for x in x_values for y in y_values]

    def jacobian_matrix(
        self,
        x: float,
        y: float,
        h: float = 1e-5,
    ) -> tuple[tuple[float, float], ...]:
        """Approximate the matrix of first partial derivatives of ``F``."""
        return (
            (
                _partial_x_2d(self.p_component, x, y, h),
                _partial_y_2d(self.p_component, x, y, h),
            ),
            (
                _partial_x_2d(self.q_component, x, y, h),
                _partial_y_2d(self.q_component, x, y, h),
            ),
        )

    def divergence(self, x: float, y: float, h: float = 1e-5) -> float:
        """Approximate ``div F = P_x + Q_y`` at ``(x, y)``."""
        return _partial_x_2d(self.p_component, x, y, h) + _partial_y_2d(
            self.q_component,
            x,
            y,
            h,
        )

    def curl_z(self, x: float, y: float, h: float = 1e-5) -> float:
        """Approximate the scalar curl ``Q_x - P_y`` in the plane."""
        return _partial_x_2d(self.q_component, x, y, h) - _partial_y_2d(
            self.p_component,
            x,
            y,
            h,
        )

    def line_integral(
        self,
        x_of_t: Callable[[float], float],
        y_of_t: Callable[[float], float],
        t_bounds: tuple[float, float],
        segments: int = 1000,
        h: float = 1e-5,
    ) -> float:
        """Approximate the work integral ``integral_C F dot dr``."""
        start, stop = t_bounds

        def integrand(t: float) -> float:
            velocity = Vector2D(
                _central_difference(x_of_t, t, h),
                _central_difference(y_of_t, t, h),
            )
            return self.value_at(x_of_t(t), y_of_t(t)).dot(velocity)

        return _trapezoid_rule(integrand, start, stop, segments)

    def is_conservative_at(
        self,
        x: float,
        y: float,
        tolerance: float = 1e-5,
        h: float = 1e-5,
    ) -> bool:
        """Return whether numerical evidence suggests ``F`` is conservative here."""
        _validate_positive(tolerance, "tolerance")
        return abs(self.curl_z(x, y, h)) <= tolerance

    def is_conservative_on_rectangle(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        x_steps: int = 4,
        y_steps: int = 4,
        tolerance: float = 1e-5,
        h: float = 1e-5,
    ) -> bool:
        """Check sampled curl values over a rectangular region."""
        _validate_positive(tolerance, "tolerance")
        x_values = _linspace(
            _validate_bounds(x_bounds, "x_bounds"),
            x_steps,
            "x_steps",
        )
        y_values = _linspace(
            _validate_bounds(y_bounds, "y_bounds"),
            y_steps,
            "y_steps",
        )
        return all(
            self.is_conservative_at(x, y, tolerance, h)
            for x in x_values
            for y in y_values
        )

    def potential_difference(
        self,
        start: Point2D,
        end: Point2D,
        segments: int = 1000,
    ) -> float:
        """Approximate ``phi(end) - phi(start)`` along an axis-aligned path."""
        x_integral = _trapezoid_rule(
            lambda x: self.p_component(x, start.y),
            start.x,
            end.x,
            segments,
        )
        y_integral = _trapezoid_rule(
            lambda y: self.q_component(end.x, y),
            start.y,
            end.y,
            segments,
        )
        return x_integral + y_integral

    def potential_at(
        self,
        x: float,
        y: float,
        base_point: Point2D | None = None,
        segments: int = 1000,
    ) -> float:
        """Approximate a potential value with ``phi(base_point) = 0``."""
        if base_point is None:
            base_point = Point2D(0.0, 0.0)
        return self.potential_difference(base_point, Point2D(x, y), segments)


@dataclass(frozen=True)
class VectorField3D:
    """A space vector field ``F(x, y, z) = <P, Q, R>``."""

    p_component: NumberFunction3D
    q_component: NumberFunction3D
    r_component: NumberFunction3D

    def value_at(self, x: float, y: float, z: float) -> Vector3D:
        """Evaluate the vector field at ``(x, y, z)``."""
        return Vector3D(
            self.p_component(x, y, z),
            self.q_component(x, y, z),
            self.r_component(x, y, z),
        )

    def point_at(self, x: float, y: float, z: float) -> VectorFieldPoint3D:
        """Evaluate the vector field and keep the base point."""
        return VectorFieldPoint3D(
            point=Point3D(x, y, z),
            vector=self.value_at(x, y, z),
        )

    def magnitude_at(self, x: float, y: float, z: float) -> float:
        """Return ``|F(x, y, z)|``."""
        return self.value_at(x, y, z).magnitude()

    def unit_at(self, x: float, y: float, z: float) -> Vector3D:
        """Return the unit vector in the direction of ``F(x, y, z)``."""
        return self.value_at(x, y, z).unit()

    def sample(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        x_steps: int,
        y_steps: int,
        z_steps: int,
    ) -> list[VectorFieldPoint3D]:
        """Sample the field on an evenly spaced rectangular grid."""
        x_values = _linspace(
            _validate_bounds(x_bounds, "x_bounds"),
            x_steps,
            "x_steps",
        )
        y_values = _linspace(
            _validate_bounds(y_bounds, "y_bounds"),
            y_steps,
            "y_steps",
        )
        z_values = _linspace(
            _validate_bounds(z_bounds, "z_bounds"),
            z_steps,
            "z_steps",
        )
        return [
            self.point_at(x, y, z)
            for x in x_values
            for y in y_values
            for z in z_values
        ]

    def jacobian_matrix(
        self,
        x: float,
        y: float,
        z: float,
        h: float = 1e-5,
    ) -> tuple[tuple[float, float, float], ...]:
        """Approximate the matrix of first partial derivatives of ``F``."""
        return (
            _gradient_3d(self.p_component, x, y, z, h),
            _gradient_3d(self.q_component, x, y, z, h),
            _gradient_3d(self.r_component, x, y, z, h),
        )

    def divergence(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate ``div F = P_x + Q_y + R_z`` at ``(x, y, z)``."""
        return (
            _partial_x_3d(self.p_component, x, y, z, h)
            + _partial_y_3d(self.q_component, x, y, z, h)
            + _partial_z_3d(self.r_component, x, y, z, h)
        )

    def curl(self, x: float, y: float, z: float, h: float = 1e-5) -> Vector3D:
        """Approximate ``curl F`` at ``(x, y, z)``."""
        return Vector3D(
            _partial_y_3d(self.r_component, x, y, z, h)
            - _partial_z_3d(self.q_component, x, y, z, h),
            _partial_z_3d(self.p_component, x, y, z, h)
            - _partial_x_3d(self.r_component, x, y, z, h),
            _partial_x_3d(self.q_component, x, y, z, h)
            - _partial_y_3d(self.p_component, x, y, z, h),
        )

    def line_integral(
        self,
        x_of_t: Callable[[float], float],
        y_of_t: Callable[[float], float],
        z_of_t: Callable[[float], float],
        t_bounds: tuple[float, float],
        segments: int = 1000,
        h: float = 1e-5,
    ) -> float:
        """Approximate the work integral ``integral_C F dot dr``."""
        start, stop = t_bounds

        def integrand(t: float) -> float:
            velocity = Vector3D(
                _central_difference(x_of_t, t, h),
                _central_difference(y_of_t, t, h),
                _central_difference(z_of_t, t, h),
            )
            return self.value_at(x_of_t(t), y_of_t(t), z_of_t(t)).dot(velocity)

        return _trapezoid_rule(integrand, start, stop, segments)

    def is_conservative_at(
        self,
        x: float,
        y: float,
        z: float,
        tolerance: float = 1e-5,
        h: float = 1e-5,
    ) -> bool:
        """Return whether numerical evidence suggests ``F`` is conservative here."""
        _validate_positive(tolerance, "tolerance")
        return self.curl(x, y, z, h).magnitude() <= tolerance

    def is_conservative_on_box(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        x_steps: int = 3,
        y_steps: int = 3,
        z_steps: int = 3,
        tolerance: float = 1e-5,
        h: float = 1e-5,
    ) -> bool:
        """Check sampled curl values over a rectangular box."""
        _validate_positive(tolerance, "tolerance")
        x_values = _linspace(
            _validate_bounds(x_bounds, "x_bounds"),
            x_steps,
            "x_steps",
        )
        y_values = _linspace(
            _validate_bounds(y_bounds, "y_bounds"),
            y_steps,
            "y_steps",
        )
        z_values = _linspace(
            _validate_bounds(z_bounds, "z_bounds"),
            z_steps,
            "z_steps",
        )
        return all(
            self.is_conservative_at(x, y, z, tolerance, h)
            for x in x_values
            for y in y_values
            for z in z_values
        )

    def potential_difference(
        self,
        start: Point3D,
        end: Point3D,
        segments: int = 1000,
    ) -> float:
        """Approximate ``phi(end) - phi(start)`` along an axis-aligned path."""
        x_integral = _trapezoid_rule(
            lambda x: self.p_component(x, start.y, start.z),
            start.x,
            end.x,
            segments,
        )
        y_integral = _trapezoid_rule(
            lambda y: self.q_component(end.x, y, start.z),
            start.y,
            end.y,
            segments,
        )
        z_integral = _trapezoid_rule(
            lambda z: self.r_component(end.x, end.y, z),
            start.z,
            end.z,
            segments,
        )
        return x_integral + y_integral + z_integral

    def potential_at(
        self,
        x: float,
        y: float,
        z: float,
        base_point: Point3D | None = None,
        segments: int = 1000,
    ) -> float:
        """Approximate a potential value with ``phi(base_point) = 0``."""
        if base_point is None:
            base_point = Point3D(0.0, 0.0, 0.0)
        return self.potential_difference(base_point, Point3D(x, y, z), segments)


def make_vector_field_2d(
    p_component: NumberFunction2D,
    q_component: NumberFunction2D,
) -> VectorField2D:
    """Create a plane vector field ``F(x, y) = <P(x, y), Q(x, y)>``."""
    return VectorField2D(p_component=p_component, q_component=q_component)


def make_vector_field_3d(
    p_component: NumberFunction3D,
    q_component: NumberFunction3D,
    r_component: NumberFunction3D,
) -> VectorField3D:
    """Create a space vector field ``F(x, y, z) = <P, Q, R>``."""
    return VectorField3D(
        p_component=p_component,
        q_component=q_component,
        r_component=r_component,
    )


def vector_field_2d_table(points: Iterable[VectorFieldPoint2D]) -> str:
    """Format plane vector-field samples as a compact table."""
    rows = ["x | y | P | Q", "--|---|---|--"]
    for point in points:
        rows.append(
            f"{_format_number(point.point.x)} | {_format_number(point.point.y)} | "
            f"{_format_number(point.vector.x)} | {_format_number(point.vector.y)}"
        )
    return "\n".join(rows)


def vector_field_3d_table(points: Iterable[VectorFieldPoint3D]) -> str:
    """Format space vector-field samples as a compact table."""
    rows = ["x | y | z | P | Q | R", "--|---|---|---|---|--"]
    for point in points:
        rows.append(
            f"{_format_number(point.point.x)} | {_format_number(point.point.y)} | "
            f"{_format_number(point.point.z)} | {_format_number(point.vector.x)} | "
            f"{_format_number(point.vector.y)} | {_format_number(point.vector.z)}"
        )
    return "\n".join(rows)


def _validate_bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    lower, upper = bounds
    if lower > upper:
        raise ValueError(f"{name} must be ordered as (lower, upper).")
    return (lower, upper)


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _linspace(
    bounds: tuple[float, float],
    steps: int,
    name: str,
) -> tuple[float, ...]:
    if steps < 1:
        raise ValueError(f"{name} must be at least 1.")
    lower, upper = bounds
    step_size = (upper - lower) / steps
    return tuple(lower + index * step_size for index in range(steps + 1))


def _central_difference(
    function: Callable[[float], float],
    value: float,
    h: float,
) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (function(value + h) - function(value - h)) / (2 * h)


def _trapezoid_rule(
    function: Callable[[float], float],
    start: float,
    stop: float,
    segments: int,
) -> float:
    if segments < 1:
        raise ValueError("segments must be at least 1.")

    step = (stop - start) / segments
    total = 0.5 * (function(start) + function(stop))
    for index in range(1, segments):
        total += function(start + index * step)
    return total * step


def _partial_x_2d(function: NumberFunction2D, x: float, y: float, h: float) -> float:
    return _central_difference(lambda value: function(value, y), x, h)


def _partial_y_2d(function: NumberFunction2D, x: float, y: float, h: float) -> float:
    return _central_difference(lambda value: function(x, value), y, h)


def _partial_x_3d(
    function: NumberFunction3D,
    x: float,
    y: float,
    z: float,
    h: float,
) -> float:
    return _central_difference(lambda value: function(value, y, z), x, h)


def _partial_y_3d(
    function: NumberFunction3D,
    x: float,
    y: float,
    z: float,
    h: float,
) -> float:
    return _central_difference(lambda value: function(x, value, z), y, h)


def _partial_z_3d(
    function: NumberFunction3D,
    x: float,
    y: float,
    z: float,
    h: float,
) -> float:
    return _central_difference(lambda value: function(x, y, value), z, h)


def _gradient_3d(
    function: NumberFunction3D,
    x: float,
    y: float,
    z: float,
    h: float,
) -> tuple[float, float, float]:
    return (
        _partial_x_3d(function, x, y, z, h),
        _partial_y_3d(function, x, y, z, h),
        _partial_z_3d(function, x, y, z, h),
    )


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"
