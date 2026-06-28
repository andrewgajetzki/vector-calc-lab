"""Utilities for polar coordinates and polar curves."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from src.parametric_equations import ParametricCurve


NumberFunction = Callable[[float], float]


@dataclass(frozen=True)
class CartesianPoint:
    """A point represented by rectangular coordinates."""

    x: float
    y: float

    def to_polar(self) -> "PolarPoint":
        """Convert ``(x, y)`` to polar coordinates using the principal angle."""
        return cartesian_to_polar(self.x, self.y)


@dataclass(frozen=True)
class PolarPoint:
    """A point represented by polar coordinates."""

    r: float
    theta: float

    def to_cartesian(self) -> CartesianPoint:
        """Convert ``(r, theta)`` to rectangular coordinates."""
        return polar_to_cartesian(self.r, self.theta)

    def equivalent(self, turns: int = 1) -> list["PolarPoint"]:
        """Return common equivalent polar representations of the same point."""
        if turns < 1:
            raise ValueError("turns must be at least 1.")

        points: list[PolarPoint] = []
        for k in range(-turns, turns + 1):
            points.append(PolarPoint(self.r, self.theta + 2 * math.pi * k))
            points.append(PolarPoint(-self.r, self.theta + (2 * k + 1) * math.pi))
        return points


@dataclass(frozen=True)
class PolarCurve:
    """A polar curve described by ``r = r(theta)``."""

    r_of_theta: NumberFunction

    def point_at(self, theta: float) -> PolarPoint:
        """Evaluate the polar curve at angle ``theta``."""
        return PolarPoint(r=self.r_of_theta(theta), theta=theta)

    def cartesian_point_at(self, theta: float) -> CartesianPoint:
        """Evaluate the polar curve and return the rectangular point."""
        return self.point_at(theta).to_cartesian()

    def dr_dtheta(self, theta: float, h: float = 1e-5) -> float:
        """Approximate ``dr/dtheta`` at angle ``theta``."""
        return _central_difference(self.r_of_theta, theta, h)

    def sample(self, start: float, stop: float, steps: int) -> list[PolarPoint]:
        """Return evenly spaced polar points from ``start`` to ``stop``, inclusive."""
        if steps < 1:
            raise ValueError("steps must be at least 1.")

        step_size = (stop - start) / steps
        return [self.point_at(start + index * step_size) for index in range(steps + 1)]

    def area(self, start: float, stop: float, segments: int = 1000) -> float:
        """Approximate the polar area swept over an angle interval."""
        return 0.5 * abs(
            _trapezoid_rule(
                lambda theta: self.r_of_theta(theta) ** 2,
                start,
                stop,
                segments,
            )
        )

    def arc_length(
        self,
        start: float,
        stop: float,
        segments: int = 1000,
        h: float = 1e-5,
    ) -> float:
        """Approximate polar arc length over an angle interval."""
        return abs(
            _trapezoid_rule(
                lambda theta: math.hypot(
                    self.r_of_theta(theta),
                    self.dr_dtheta(theta, h),
                ),
                start,
                stop,
                segments,
            )
        )

    def to_parametric_curve(self) -> ParametricCurve:
        """Return the equivalent parametric curve ``x(theta), y(theta)``."""
        return ParametricCurve(
            x_of_t=lambda theta: self.r_of_theta(theta) * math.cos(theta),
            y_of_t=lambda theta: self.r_of_theta(theta) * math.sin(theta),
        )


def make_polar_curve(r_of_theta: NumberFunction) -> PolarCurve:
    """Create a polar curve from a one-variable radius function."""
    return PolarCurve(r_of_theta=r_of_theta)


def polar_to_cartesian(r: float, theta: float) -> CartesianPoint:
    """Convert ``(r, theta)`` to ``(x, y)``."""
    return CartesianPoint(x=r * math.cos(theta), y=r * math.sin(theta))


def cartesian_to_polar(x: float, y: float) -> PolarPoint:
    """Convert ``(x, y)`` to ``(r, theta)`` with ``theta`` in ``[-pi, pi]``."""
    return PolarPoint(r=math.hypot(x, y), theta=math.atan2(y, x))


def normalize_angle(theta: float) -> float:
    """Normalize an angle in radians to ``[0, 2*pi)``."""
    return theta % (2 * math.pi)


def polar_points_table(points: Iterable[PolarPoint]) -> str:
    """Format polar points as a compact table."""
    rows = ["theta | r", "------|--"]
    for point in points:
        rows.append(f"{_format_number(point.theta)} | {_format_number(point.r)}")
    return "\n".join(rows)


def polar_to_cartesian_table(points: Iterable[PolarPoint]) -> str:
    """Format polar points with their rectangular coordinates."""
    rows = ["theta | r | x | y", "------|---|---|--"]
    for point in points:
        cartesian = point.to_cartesian()
        rows.append(
            f"{_format_number(point.theta)} | {_format_number(point.r)} | "
            f"{_format_number(cartesian.x)} | {_format_number(cartesian.y)}"
        )
    return "\n".join(rows)


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"


def _central_difference(function: NumberFunction, theta: float, h: float) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (function(theta + h) - function(theta - h)) / (2 * h)


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
