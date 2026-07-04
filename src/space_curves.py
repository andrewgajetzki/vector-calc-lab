"""Utilities for vector-valued functions and space curves."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from src.space_geometry import Line3D
from src.vectors import Point3D, Vector3D


NumberFunction = Callable[[float], float]


@dataclass(frozen=True)
class SpaceCurvePoint:
    """A point on a space curve."""

    t: float
    x: float
    y: float
    z: float

    def to_point(self) -> Point3D:
        """Return the point as rectangular coordinates."""
        return Point3D(self.x, self.y, self.z)

    def position_vector(self) -> Vector3D:
        """Return the position vector from the origin."""
        return Vector3D(self.x, self.y, self.z)

    def as_text(self) -> str:
        """Return a compact point form."""
        return (
            f"({_format_number(self.x)}, {_format_number(self.y)}, "
            f"{_format_number(self.z)})"
        )


@dataclass(frozen=True)
class SpaceCurve:
    """A vector-valued function ``r(t) = <x(t), y(t), z(t)>``."""

    x_of_t: NumberFunction
    y_of_t: NumberFunction
    z_of_t: NumberFunction

    def point_at(self, t: float) -> SpaceCurvePoint:
        """Evaluate the space curve at parameter value ``t``."""
        return SpaceCurvePoint(
            t=t,
            x=self.x_of_t(t),
            y=self.y_of_t(t),
            z=self.z_of_t(t),
        )

    def position_vector(self, t: float) -> Vector3D:
        """Return the position vector ``r(t)``."""
        return self.point_at(t).position_vector()

    def sample(self, start: float, stop: float, steps: int) -> list[SpaceCurvePoint]:
        """Return evenly spaced points from ``start`` to ``stop``, inclusive."""
        if steps < 1:
            raise ValueError("steps must be at least 1.")

        step_size = (stop - start) / steps
        return [self.point_at(start + index * step_size) for index in range(steps + 1)]

    def derivative(self, t: float, h: float = 1e-5) -> Vector3D:
        """Approximate ``r'(t)``."""
        return Vector3D(
            x=_central_difference(self.x_of_t, t, h),
            y=_central_difference(self.y_of_t, t, h),
            z=_central_difference(self.z_of_t, t, h),
        )

    def velocity(self, t: float, h: float = 1e-5) -> Vector3D:
        """Approximate velocity ``r'(t)``."""
        return self.derivative(t, h)

    def second_derivative(self, t: float, h: float = 1e-5) -> Vector3D:
        """Approximate ``r''(t)``."""
        return Vector3D(
            x=_second_central_difference(self.x_of_t, t, h),
            y=_second_central_difference(self.y_of_t, t, h),
            z=_second_central_difference(self.z_of_t, t, h),
        )

    def acceleration(self, t: float, h: float = 1e-5) -> Vector3D:
        """Approximate acceleration ``r''(t)``."""
        return self.second_derivative(t, h)

    def third_derivative(self, t: float, h: float = 1e-4) -> Vector3D:
        """Approximate ``r'''(t)``."""
        return Vector3D(
            x=_third_central_difference(self.x_of_t, t, h),
            y=_third_central_difference(self.y_of_t, t, h),
            z=_third_central_difference(self.z_of_t, t, h),
        )

    def speed(self, t: float, h: float = 1e-5) -> float:
        """Approximate speed ``|r'(t)|``."""
        return self.velocity(t, h).magnitude()

    def unit_tangent(self, t: float, h: float = 1e-5) -> Vector3D:
        """Approximate the unit tangent vector ``T(t)``."""
        velocity = self.velocity(t, h)
        if velocity.is_zero():
            raise ValueError("The unit tangent is undefined because velocity is zero.")
        return velocity.unit()

    def tangent_line(self, t: float, h: float = 1e-5) -> Line3D:
        """Return the tangent line at parameter value ``t``."""
        velocity = self.velocity(t, h)
        if velocity.is_zero():
            raise ValueError("The tangent line is undefined because velocity is zero.")
        return Line3D(point=self.point_at(t).to_point(), direction=velocity)

    def curvature(self, t: float, h: float = 1e-5) -> float:
        """Approximate curvature ``kappa = |r' x r''| / |r'|^3``."""
        velocity = self.velocity(t, h)
        acceleration = self.acceleration(t, h)
        speed = velocity.magnitude()
        if math.isclose(speed, 0.0, abs_tol=1e-12):
            raise ValueError("Curvature is undefined because velocity is zero.")
        return velocity.cross(acceleration).magnitude() / speed**3

    def unit_normal(self, t: float, h: float = 1e-5) -> Vector3D:
        """Approximate the principal unit normal vector ``N(t)``."""
        tangent_rate = _central_vector_difference(
            lambda value: self.unit_tangent(value, h),
            t,
            h,
        )
        if tangent_rate.is_zero():
            raise ValueError("The unit normal is undefined because dT/dt is zero.")
        return tangent_rate.unit()

    def binormal(self, t: float, h: float = 1e-5) -> Vector3D:
        """Approximate the binormal vector ``B(t) = T(t) x N(t)``."""
        return self.unit_tangent(t, h).cross(self.unit_normal(t, h)).unit()

    def torsion(self, t: float, h: float = 1e-4) -> float:
        """Approximate torsion ``tau`` from ``r'``, ``r''``, and ``r'''``."""
        velocity = self.velocity(t, h)
        acceleration = self.acceleration(t, h)
        jerk = self.third_derivative(t, h)
        cross = velocity.cross(acceleration)
        denominator = cross.dot(cross)
        if math.isclose(denominator, 0.0, abs_tol=1e-12):
            raise ValueError("Torsion is undefined because r' x r'' is zero.")
        return velocity.dot(acceleration.cross(jerk)) / denominator

    def tangential_acceleration(self, t: float, h: float = 1e-5) -> float:
        """Approximate tangential acceleration component ``a_T``."""
        velocity = self.velocity(t, h)
        speed = velocity.magnitude()
        if math.isclose(speed, 0.0, abs_tol=1e-12):
            raise ValueError("Tangential acceleration is undefined when speed is zero.")
        return velocity.dot(self.acceleration(t, h)) / speed

    def normal_acceleration(self, t: float, h: float = 1e-5) -> float:
        """Approximate normal acceleration component ``a_N``."""
        velocity = self.velocity(t, h)
        speed = velocity.magnitude()
        if math.isclose(speed, 0.0, abs_tol=1e-12):
            raise ValueError("Normal acceleration is undefined when speed is zero.")
        return velocity.cross(self.acceleration(t, h)).magnitude() / speed

    def arc_length(self, start: float, stop: float, segments: int = 1000) -> float:
        """Approximate arc length over ``start <= t <= stop`` with the trapezoid rule."""
        return abs(_trapezoid_rule(lambda t: self.speed(t), start, stop, segments))


def make_space_curve(
    x_of_t: NumberFunction,
    y_of_t: NumberFunction,
    z_of_t: NumberFunction,
) -> SpaceCurve:
    """Create a space curve from three one-variable component functions."""
    return SpaceCurve(x_of_t=x_of_t, y_of_t=y_of_t, z_of_t=z_of_t)


def make_vector_valued_function(
    x_of_t: NumberFunction,
    y_of_t: NumberFunction,
    z_of_t: NumberFunction,
) -> SpaceCurve:
    """Create a vector-valued function ``r(t) = <x(t), y(t), z(t)>``."""
    return make_space_curve(x_of_t, y_of_t, z_of_t)


def space_curve_points_table(points: Iterable[SpaceCurvePoint]) -> str:
    """Format space-curve points as a compact table."""
    rows = ["t | x(t) | y(t) | z(t)", "--|------|------|-----"]
    for point in points:
        rows.append(
            f"{_format_number(point.t)} | {_format_number(point.x)} | "
            f"{_format_number(point.y)} | {_format_number(point.z)}"
        )
    return "\n".join(rows)


def _central_difference(function: NumberFunction, t: float, h: float) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (function(t + h) - function(t - h)) / (2 * h)


def _second_central_difference(function: NumberFunction, t: float, h: float) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (function(t + h) - 2 * function(t) + function(t - h)) / h**2


def _third_central_difference(function: NumberFunction, t: float, h: float) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (
        function(t + 2 * h)
        - 2 * function(t + h)
        + 2 * function(t - h)
        - function(t - 2 * h)
    ) / (2 * h**3)


def _central_vector_difference(
    function: Callable[[float], Vector3D],
    t: float,
    h: float,
) -> Vector3D:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (function(t + h) - function(t - h)) / (2 * h)


def _trapezoid_rule(
    function: NumberFunction,
    start: float,
    stop: float,
    segments: int,
) -> float:
    if segments < 1:
        raise ValueError("segments must be at least 1.")

    step_size = (stop - start) / segments
    total = 0.5 * (function(start) + function(stop))
    for index in range(1, segments):
        total += function(start + index * step_size)
    return total * step_size


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"
