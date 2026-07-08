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

    def greens_theorem_circulation_over_rectangle(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        x_segments: int = 100,
        y_segments: int = 100,
        h: float = 1e-5,
    ) -> float:
        """Approximate positive circulation using ``double integral curl_z dA``."""
        return _double_trapezoid_rule(
            lambda x, y: self.curl_z(x, y, h),
            x_bounds,
            y_bounds,
            x_segments,
            y_segments,
        )

    def greens_theorem_circulation_type_i(
        self,
        x_bounds: tuple[float, float],
        y_lower: Callable[[float], float],
        y_upper: Callable[[float], float],
        x_segments: int = 100,
        y_segments: int = 100,
        h: float = 1e-5,
    ) -> float:
        """Approximate positive circulation over ``g1(x) <= y <= g2(x)``."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        return _iterated_trapezoid_rule(
            x_bounds,
            y_lower,
            y_upper,
            lambda x, y: self.curl_z(x, y, h),
            x_segments,
            y_segments,
            "x_segments",
            "y_segments",
            "y_bounds",
        )

    def greens_theorem_circulation_type_ii(
        self,
        y_bounds: tuple[float, float],
        x_lower: Callable[[float], float],
        x_upper: Callable[[float], float],
        y_segments: int = 100,
        x_segments: int = 100,
        h: float = 1e-5,
    ) -> float:
        """Approximate positive circulation over ``h1(y) <= x <= h2(y)``."""
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        return _iterated_trapezoid_rule(
            y_bounds,
            x_lower,
            x_upper,
            lambda y, x: self.curl_z(x, y, h),
            y_segments,
            x_segments,
            "y_segments",
            "x_segments",
            "x_bounds",
        )

    def greens_theorem_flux_over_rectangle(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        x_segments: int = 100,
        y_segments: int = 100,
        h: float = 1e-5,
    ) -> float:
        """Approximate outward flux using ``double integral div F dA``."""
        return _double_trapezoid_rule(
            lambda x, y: self.divergence(x, y, h),
            x_bounds,
            y_bounds,
            x_segments,
            y_segments,
        )

    def greens_theorem_flux_type_i(
        self,
        x_bounds: tuple[float, float],
        y_lower: Callable[[float], float],
        y_upper: Callable[[float], float],
        x_segments: int = 100,
        y_segments: int = 100,
        h: float = 1e-5,
    ) -> float:
        """Approximate outward flux over ``g1(x) <= y <= g2(x)``."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        return _iterated_trapezoid_rule(
            x_bounds,
            y_lower,
            y_upper,
            lambda x, y: self.divergence(x, y, h),
            x_segments,
            y_segments,
            "x_segments",
            "y_segments",
            "y_bounds",
        )

    def greens_theorem_flux_type_ii(
        self,
        y_bounds: tuple[float, float],
        x_lower: Callable[[float], float],
        x_upper: Callable[[float], float],
        y_segments: int = 100,
        x_segments: int = 100,
        h: float = 1e-5,
    ) -> float:
        """Approximate outward flux over ``h1(y) <= x <= h2(y)``."""
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        return _iterated_trapezoid_rule(
            y_bounds,
            x_lower,
            x_upper,
            lambda y, x: self.divergence(x, y, h),
            y_segments,
            x_segments,
            "y_segments",
            "x_segments",
            "x_bounds",
        )

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

    def flux_integral_parametric(
        self,
        u_bounds: tuple[float, float],
        v_bounds: tuple[float, float],
        x_of_u_v: Callable[[float, float], float],
        y_of_u_v: Callable[[float, float], float],
        z_of_u_v: Callable[[float, float], float],
        u_segments: int = 100,
        v_segments: int = 100,
        h: float = 1e-5,
        reverse_orientation: bool = False,
    ) -> float:
        """Approximate oriented flux ``integral integral_S F dot n dS``."""
        u_bounds = _validate_bounds(u_bounds, "u_bounds")
        v_bounds = _validate_bounds(v_bounds, "v_bounds")
        _validate_segments(u_segments, "u_segments")
        _validate_segments(v_segments, "v_segments")

        def integrand(u: float, v: float) -> float:
            normal = _surface_normal(x_of_u_v, y_of_u_v, z_of_u_v, u, v, h)
            if reverse_orientation:
                normal = -normal
            return self.value_at(
                x_of_u_v(u, v),
                y_of_u_v(u, v),
                z_of_u_v(u, v),
            ).dot(normal)

        return _double_trapezoid_rule(
            integrand,
            u_bounds,
            v_bounds,
            u_segments,
            v_segments,
        )

    def flux_integral_over_graph(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_of_x_y: Callable[[float, float], float],
        x_segments: int = 100,
        y_segments: int = 100,
        h: float = 1e-5,
        orientation: str = "up",
    ) -> float:
        """Approximate flux over ``z = g(x, y)`` with up/down orientation."""
        if orientation not in ("up", "down"):
            raise ValueError("orientation must be 'up' or 'down'.")
        return self.flux_integral_parametric(
            x_bounds,
            y_bounds,
            lambda x, y: x,
            lambda x, y: y,
            z_of_x_y,
            x_segments,
            y_segments,
            h,
            reverse_orientation=orientation == "down",
        )

    def stokes_theorem_parametric(
        self,
        u_bounds: tuple[float, float],
        v_bounds: tuple[float, float],
        x_of_u_v: Callable[[float, float], float],
        y_of_u_v: Callable[[float, float], float],
        z_of_u_v: Callable[[float, float], float],
        u_segments: int = 100,
        v_segments: int = 100,
        h: float = 1e-5,
        reverse_orientation: bool = False,
    ) -> float:
        """Approximate circulation using ``integral integral_S curl(F) dot n dS``."""
        u_bounds = _validate_bounds(u_bounds, "u_bounds")
        v_bounds = _validate_bounds(v_bounds, "v_bounds")
        _validate_segments(u_segments, "u_segments")
        _validate_segments(v_segments, "v_segments")

        def integrand(u: float, v: float) -> float:
            x = x_of_u_v(u, v)
            y = y_of_u_v(u, v)
            z = z_of_u_v(u, v)
            normal = _surface_normal(x_of_u_v, y_of_u_v, z_of_u_v, u, v, h)
            if reverse_orientation:
                normal = -normal
            return self.curl(x, y, z, h).dot(normal)

        return _double_trapezoid_rule(
            integrand,
            u_bounds,
            v_bounds,
            u_segments,
            v_segments,
        )

    def stokes_theorem_over_graph(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_of_x_y: Callable[[float, float], float],
        x_segments: int = 100,
        y_segments: int = 100,
        h: float = 1e-5,
        orientation: str = "up",
    ) -> float:
        """Approximate Stokes circulation over ``z = g(x, y)``."""
        if orientation not in ("up", "down"):
            raise ValueError("orientation must be 'up' or 'down'.")
        return self.stokes_theorem_parametric(
            x_bounds,
            y_bounds,
            lambda x, y: x,
            lambda x, y: y,
            z_of_x_y,
            x_segments,
            y_segments,
            h,
            reverse_orientation=orientation == "down",
        )

    def divergence_theorem_over_box(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        x_segments: int = 30,
        y_segments: int = 30,
        z_segments: int = 30,
        h: float = 1e-5,
    ) -> float:
        """Approximate outward flux using ``triple integral_E div(F) dV``."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        z_bounds = _validate_bounds(z_bounds, "z_bounds")
        return _triple_trapezoid_rule(
            lambda x, y, z: self.divergence(x, y, z, h),
            x_bounds,
            y_bounds,
            z_bounds,
            x_segments,
            y_segments,
            z_segments,
        )

    def divergence_theorem_iterated(
        self,
        x_bounds: tuple[float, float],
        y_lower: Callable[[float], float],
        y_upper: Callable[[float], float],
        z_lower: Callable[[float, float], float],
        z_upper: Callable[[float, float], float],
        x_segments: int = 30,
        y_segments: int = 30,
        z_segments: int = 30,
        h: float = 1e-5,
    ) -> float:
        """Approximate outward flux over a solid using ``triple integral_E div(F) dV``."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        return _iterated_triple_trapezoid_rule(
            lambda x, y, z: self.divergence(x, y, z, h),
            x_bounds,
            y_lower,
            y_upper,
            z_lower,
            z_upper,
            x_segments,
            y_segments,
            z_segments,
        )

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


def _validate_segments(segments: int, name: str) -> None:
    if segments < 1:
        raise ValueError(f"{name} must be at least 1.")


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
    _validate_segments(segments, "segments")

    step = (stop - start) / segments
    total = 0.5 * (function(start) + function(stop))
    for index in range(1, segments):
        total += function(start + index * step)
    return total * step


def _double_trapezoid_rule(
    function: NumberFunction2D,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    x_segments: int,
    y_segments: int,
) -> float:
    x_bounds = _validate_bounds(x_bounds, "x_bounds")
    y_bounds = _validate_bounds(y_bounds, "y_bounds")
    _validate_segments(x_segments, "x_segments")
    _validate_segments(y_segments, "y_segments")

    x_start, x_stop = x_bounds
    y_start, y_stop = y_bounds
    x_step = (x_stop - x_start) / x_segments
    y_step = (y_stop - y_start) / y_segments
    total = 0.0

    for x_index in range(x_segments + 1):
        x = x_start + x_index * x_step
        x_weight = 0.5 if x_index in (0, x_segments) else 1.0
        for y_index in range(y_segments + 1):
            y = y_start + y_index * y_step
            y_weight = 0.5 if y_index in (0, y_segments) else 1.0
            total += x_weight * y_weight * function(x, y)

    return total * x_step * y_step


def _iterated_trapezoid_rule(
    outer_bounds: tuple[float, float],
    inner_lower: Callable[[float], float],
    inner_upper: Callable[[float], float],
    function: NumberFunction2D,
    outer_segments: int,
    inner_segments: int,
    outer_segments_name: str,
    inner_segments_name: str,
    inner_bounds_name: str,
) -> float:
    _validate_segments(outer_segments, outer_segments_name)
    _validate_segments(inner_segments, inner_segments_name)

    outer_start, outer_stop = outer_bounds
    outer_step = (outer_stop - outer_start) / outer_segments
    total = 0.0

    for outer_index in range(outer_segments + 1):
        outer_value = outer_start + outer_index * outer_step
        outer_weight = 0.5 if outer_index in (0, outer_segments) else 1.0
        inner_start = inner_lower(outer_value)
        inner_stop = inner_upper(outer_value)
        _validate_bounds((inner_start, inner_stop), inner_bounds_name)

        inner_step = (inner_stop - inner_start) / inner_segments
        inner_total = 0.0
        for inner_index in range(inner_segments + 1):
            inner_value = inner_start + inner_index * inner_step
            inner_weight = 0.5 if inner_index in (0, inner_segments) else 1.0
            inner_total += inner_weight * function(outer_value, inner_value)

        total += outer_weight * inner_total * inner_step

    return total * outer_step


def _triple_trapezoid_rule(
    function: NumberFunction3D,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    x_segments: int,
    y_segments: int,
    z_segments: int,
) -> float:
    _validate_segments(x_segments, "x_segments")
    _validate_segments(y_segments, "y_segments")
    _validate_segments(z_segments, "z_segments")

    x_start, x_stop = x_bounds
    y_start, y_stop = y_bounds
    z_start, z_stop = z_bounds
    x_step = (x_stop - x_start) / x_segments
    y_step = (y_stop - y_start) / y_segments
    z_step = (z_stop - z_start) / z_segments
    total = 0.0

    for x_index in range(x_segments + 1):
        x = x_start + x_index * x_step
        x_weight = 0.5 if x_index in (0, x_segments) else 1.0
        for y_index in range(y_segments + 1):
            y = y_start + y_index * y_step
            y_weight = 0.5 if y_index in (0, y_segments) else 1.0
            for z_index in range(z_segments + 1):
                z = z_start + z_index * z_step
                z_weight = 0.5 if z_index in (0, z_segments) else 1.0
                total += x_weight * y_weight * z_weight * function(x, y, z)

    return total * x_step * y_step * z_step


def _iterated_triple_trapezoid_rule(
    function: NumberFunction3D,
    x_bounds: tuple[float, float],
    y_lower: Callable[[float], float],
    y_upper: Callable[[float], float],
    z_lower: Callable[[float, float], float],
    z_upper: Callable[[float, float], float],
    x_segments: int,
    y_segments: int,
    z_segments: int,
) -> float:
    _validate_segments(x_segments, "x_segments")
    _validate_segments(y_segments, "y_segments")
    _validate_segments(z_segments, "z_segments")

    x_start, x_stop = x_bounds
    x_step = (x_stop - x_start) / x_segments
    total = 0.0

    for x_index in range(x_segments + 1):
        x = x_start + x_index * x_step
        x_weight = 0.5 if x_index in (0, x_segments) else 1.0
        y_start = y_lower(x)
        y_stop = y_upper(x)
        _validate_bounds((y_start, y_stop), "y_bounds")

        y_step = (y_stop - y_start) / y_segments
        y_total = 0.0
        for y_index in range(y_segments + 1):
            y = y_start + y_index * y_step
            y_weight = 0.5 if y_index in (0, y_segments) else 1.0
            z_start = z_lower(x, y)
            z_stop = z_upper(x, y)
            _validate_bounds((z_start, z_stop), "z_bounds")

            z_step = (z_stop - z_start) / z_segments
            z_total = 0.0
            for z_index in range(z_segments + 1):
                z = z_start + z_index * z_step
                z_weight = 0.5 if z_index in (0, z_segments) else 1.0
                z_total += z_weight * function(x, y, z)

            y_total += y_weight * z_total * z_step

        total += x_weight * y_total * y_step

    return total * x_step


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


def _parameter_partials(
    function: Callable[[float, float], float],
    u: float,
    v: float,
    h: float,
) -> Vector2D:
    return Vector2D(
        _central_difference(lambda value: function(value, v), u, h),
        _central_difference(lambda value: function(u, value), v, h),
    )


def _surface_normal(
    x_of_u_v: Callable[[float, float], float],
    y_of_u_v: Callable[[float, float], float],
    z_of_u_v: Callable[[float, float], float],
    u: float,
    v: float,
    h: float,
) -> Vector3D:
    x_partials = _parameter_partials(x_of_u_v, u, v, h)
    y_partials = _parameter_partials(y_of_u_v, u, v, h)
    z_partials = _parameter_partials(z_of_u_v, u, v, h)
    tangent_u = Vector3D(x_partials.x, y_partials.x, z_partials.x)
    tangent_v = Vector3D(x_partials.y, y_partials.y, z_partials.y)
    return tangent_u.cross(tangent_v)


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
