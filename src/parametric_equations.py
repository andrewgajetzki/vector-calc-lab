"""Utilities for Chapter 1.1: parametric equations."""

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

    def speed(self, t: float, h: float = 1e-5) -> float:
        """Approximate speed ``sqrt((dx/dt)^2 + (dy/dt)^2)`` at ``t``."""
        return math.hypot(self.dx_dt(t, h), self.dy_dt(t, h))

    def arc_length(self, start: float, stop: float, segments: int = 1000) -> float:
        """Approximate arc length over ``start <= t <= stop`` with the trapezoid rule."""
        if segments < 1:
            raise ValueError("segments must be at least 1.")

        step_size = (stop - start) / segments
        total = 0.5 * (self.speed(start) + self.speed(stop))
        for index in range(1, segments):
            total += self.speed(start + index * step_size)
        return abs(total * step_size)


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


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"
