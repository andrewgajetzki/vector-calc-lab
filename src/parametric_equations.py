"""Utilities for parametric equations and parametric curves."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass


NumberFunction = Callable[[float], float]


@dataclass(frozen=True)
class ParametricPoint:
    """A point on a parametric curve."""

    t: float
    x: float
    y: float


@dataclass(frozen=True)
class Line:
    """A line attached to a point on a parametric curve."""

    point: ParametricPoint
    slope: float | None

    def as_text(self) -> str:
        if self.slope is None:
            return f"x = {_format_number(self.point.x)}"
        return (
            f"y - {_format_number(self.point.y)} = "
            f"{_format_number(self.slope)}(x - {_format_number(self.point.x)})"
        )


@dataclass(frozen=True)
class ParametricCurve:
    """A plane curve described by ``x = x(t)`` and ``y = y(t)``."""

    x_of_t: NumberFunction
    y_of_t: NumberFunction

    def point_at(self, t: float) -> ParametricPoint:
        """Evaluate the curve at parameter value ``t``."""
        return ParametricPoint(t=t, x=self.x_of_t(t), y=self.y_of_t(t))

    def sample(self, start: float, stop: float, steps: int) -> list[ParametricPoint]:
        """Return evenly spaced points from ``start`` to ``stop``, inclusive."""
        if steps < 1:
            raise ValueError("steps must be at least 1.")

        step_size = (stop - start) / steps
        return [self.point_at(start + index * step_size) for index in range(steps + 1)]

    def dx_dt(self, t: float, h: float = 1e-5) -> float:
        """Approximate ``dx/dt`` at ``t``."""
        return _central_difference(self.x_of_t, t, h)

    def dy_dt(self, t: float, h: float = 1e-5) -> float:
        """Approximate ``dy/dt`` at ``t``."""
        return _central_difference(self.y_of_t, t, h)

    def slope(self, t: float, h: float = 1e-5) -> float:
        """Approximate ``dy/dx = (dy/dt)/(dx/dt)`` at ``t``."""
        dx = self.dx_dt(t, h)
        if math.isclose(dx, 0.0, abs_tol=1e-12):
            raise ValueError("dy/dx is undefined because dx/dt is zero at this t.")
        return self.dy_dt(t, h) / dx

    def second_derivative(self, t: float, h: float = 1e-5) -> float:
        """Approximate ``d2y/dx2 = d/dt(dy/dx) / (dx/dt)`` at ``t``."""
        dx = self.dx_dt(t, h)
        if math.isclose(dx, 0.0, abs_tol=1e-12):
            raise ValueError("d2y/dx2 is undefined because dx/dt is zero at this t.")
        slope_rate = _central_difference(lambda value: self.slope(value, h), t, h)
        return slope_rate / dx

    def tangent_line(self, t: float, h: float = 1e-5) -> Line:
        """Return the tangent line at parameter value ``t``."""
        return Line(point=self.point_at(t), slope=self.slope(t, h))

    def normal_line(self, t: float, h: float = 1e-5) -> Line:
        """Return the normal line at parameter value ``t``."""
        tangent_slope = self.slope(t, h)
        if math.isclose(tangent_slope, 0.0, abs_tol=1e-12):
            return Line(point=self.point_at(t), slope=None)
        return Line(point=self.point_at(t), slope=-1 / tangent_slope)

    def concavity(self, t: float, h: float = 1e-5) -> str:
        """Classify concavity from the sign of ``d2y/dx2`` at ``t``."""
        value = self.second_derivative(t, h)
        if math.isclose(value, 0.0, abs_tol=1e-8):
            return "neither concave up nor concave down"
        if value > 0:
            return "concave up"
        return "concave down"

    def speed(self, t: float, h: float = 1e-5) -> float:
        """Approximate speed ``sqrt((dx/dt)^2 + (dy/dt)^2)`` at ``t``."""
        return math.hypot(self.dx_dt(t, h), self.dy_dt(t, h))

    def arc_length(self, start: float, stop: float, segments: int = 1000) -> float:
        """Approximate arc length over ``start <= t <= stop`` with the trapezoid rule."""
        return abs(_trapezoid_rule(self.speed, start, stop, segments))

    def signed_area_under_curve(self, start: float, stop: float, segments: int = 1000) -> float:
        """Approximate ``integral y dx`` over the parameter interval."""
        return _trapezoid_rule(
            lambda t: self.y_of_t(t) * self.dx_dt(t),
            start,
            stop,
            segments,
        )

    def surface_area_about_x_axis(
        self, start: float, stop: float, segments: int = 1000
    ) -> float:
        """Approximate surface area from rotating the curve about the x-axis."""
        return 2 * math.pi * abs(
            _trapezoid_rule(
                lambda t: abs(self.y_of_t(t)) * self.speed(t),
                start,
                stop,
                segments,
            )
        )

    def surface_area_about_y_axis(
        self, start: float, stop: float, segments: int = 1000
    ) -> float:
        """Approximate surface area from rotating the curve about the y-axis."""
        return 2 * math.pi * abs(
            _trapezoid_rule(
                lambda t: abs(self.x_of_t(t)) * self.speed(t),
                start,
                stop,
                segments,
            )
        )


def make_curve(x_of_t: NumberFunction, y_of_t: NumberFunction) -> ParametricCurve:
    """Create a parametric curve from two one-variable functions."""
    return ParametricCurve(x_of_t=x_of_t, y_of_t=y_of_t)


def points_table(points: Iterable[ParametricPoint]) -> str:
    """Format parametric points as a compact table."""
    rows = ["t | x(t) | y(t)", "--|------|-----"]
    for point in points:
        rows.append(
            f"{_format_number(point.t)} | {_format_number(point.x)} | {_format_number(point.y)}"
        )
    return "\n".join(rows)


def _central_difference(function: NumberFunction, t: float, h: float) -> float:
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
