"""Utilities for differentiating scalar functions of several variables."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from src.space_geometry import Plane3D
from src.vectors import Point3D, Vector2D, Vector3D


NumberFunction2D = Callable[[float, float], float]
NumberFunction3D = Callable[[float, float, float], float]


@dataclass(frozen=True)
class GraphTangentPlane:
    """A tangent plane to a graph ``z = f(x, y)``."""

    point: Point3D
    x_slope: float
    y_slope: float

    def evaluate(self, x: float, y: float) -> float:
        """Evaluate the tangent-plane approximation at ``(x, y)``."""
        return (
            self.point.z
            + self.x_slope * (x - self.point.x)
            + self.y_slope * (y - self.point.y)
        )

    def as_text(self) -> str:
        """Return point-slope form for the tangent plane."""
        if _is_close(self.x_slope) and _is_close(self.y_slope):
            return f"z = {_format_number(self.point.z)}"
        terms = [
            (self.x_slope, _linear_variable("x", self.point.x)),
            (self.y_slope, _linear_variable("y", self.point.y)),
        ]
        return (
            f"z - {_format_number(self.point.z)} = "
            f"{_slope_terms(terms)}"
        )


@dataclass(frozen=True)
class ScalarFunction2D:
    """A scalar function ``f(x, y)``."""

    function: NumberFunction2D

    def value_at(self, x: float, y: float) -> float:
        """Evaluate ``f(x, y)``."""
        return self.function(x, y)

    def partial_x(self, x: float, y: float, h: float = 1e-5) -> float:
        """Approximate ``df/dx`` at ``(x, y)``."""
        return _central_difference(lambda value: self.function(value, y), x, h)

    def partial_y(self, x: float, y: float, h: float = 1e-5) -> float:
        """Approximate ``df/dy`` at ``(x, y)``."""
        return _central_difference(lambda value: self.function(x, value), y, h)

    def second_partial_xx(self, x: float, y: float, h: float = 1e-5) -> float:
        """Approximate ``d2f/dx2`` at ``(x, y)``."""
        return _second_central_difference(lambda value: self.function(value, y), x, h)

    def second_partial_yy(self, x: float, y: float, h: float = 1e-5) -> float:
        """Approximate ``d2f/dy2`` at ``(x, y)``."""
        return _second_central_difference(lambda value: self.function(x, value), y, h)

    def second_partial_xy(self, x: float, y: float, h: float = 1e-5) -> float:
        """Approximate the mixed partial ``d2f/dxdy`` at ``(x, y)``."""
        return _mixed_partial_2d(self.function, x, y, h)

    def second_partial_yx(self, x: float, y: float, h: float = 1e-5) -> float:
        """Approximate the mixed partial ``d2f/dydx`` at ``(x, y)``."""
        return self.second_partial_xy(x, y, h)

    def gradient(self, x: float, y: float, h: float = 1e-5) -> Vector2D:
        """Approximate the gradient vector ``grad f`` at ``(x, y)``."""
        return Vector2D(self.partial_x(x, y, h), self.partial_y(x, y, h))

    def directional_derivative(
        self,
        x: float,
        y: float,
        direction: Vector2D,
        h: float = 1e-5,
    ) -> float:
        """Approximate the directional derivative in a given direction."""
        if direction.is_zero():
            raise ValueError("Directional derivative needs a nonzero direction vector.")
        return self.gradient(x, y, h).dot(direction.unit())

    def tangent_plane(self, x: float, y: float, h: float = 1e-5) -> GraphTangentPlane:
        """Approximate the tangent plane to ``z = f(x, y)`` at ``(x, y)``."""
        return GraphTangentPlane(
            point=Point3D(x, y, self.function(x, y)),
            x_slope=self.partial_x(x, y, h),
            y_slope=self.partial_y(x, y, h),
        )

    def linear_approximation(
        self,
        x0: float,
        y0: float,
        x: float,
        y: float,
        h: float = 1e-5,
    ) -> float:
        """Approximate ``f(x, y)`` using the linearization at ``(x0, y0)``."""
        return self.tangent_plane(x0, y0, h).evaluate(x, y)

    def differential(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        h: float = 1e-5,
    ) -> float:
        """Approximate the differential ``df = f_x dx + f_y dy``."""
        gradient = self.gradient(x, y, h)
        return gradient.x * dx + gradient.y * dy

    def hessian(
        self,
        x: float,
        y: float,
        h: float = 1e-5,
    ) -> tuple[tuple[float, float], ...]:
        """Approximate the Hessian matrix at ``(x, y)``."""
        mixed = self.second_partial_xy(x, y, h)
        return (
            (self.second_partial_xx(x, y, h), mixed),
            (mixed, self.second_partial_yy(x, y, h)),
        )

    def classify_critical_point(self, x: float, y: float, h: float = 1e-4) -> str:
        """Classify a critical point using the second-derivative test."""
        f_xx, f_xy = self.hessian(x, y, h)[0]
        f_yy = self.second_partial_yy(x, y, h)
        determinant = f_xx * f_yy - f_xy**2

        if determinant > 1e-8 and f_xx > 0:
            return "local minimum"
        if determinant > 1e-8 and f_xx < 0:
            return "local maximum"
        if determinant < -1e-8:
            return "saddle point"
        return "inconclusive"


@dataclass(frozen=True)
class ScalarFunction3D:
    """A scalar function ``f(x, y, z)``."""

    function: NumberFunction3D

    def value_at(self, x: float, y: float, z: float) -> float:
        """Evaluate ``f(x, y, z)``."""
        return self.function(x, y, z)

    def partial_x(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate ``df/dx`` at ``(x, y, z)``."""
        return _central_difference(lambda value: self.function(value, y, z), x, h)

    def partial_y(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate ``df/dy`` at ``(x, y, z)``."""
        return _central_difference(lambda value: self.function(x, value, z), y, h)

    def partial_z(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate ``df/dz`` at ``(x, y, z)``."""
        return _central_difference(lambda value: self.function(x, y, value), z, h)

    def second_partial_xx(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate ``d2f/dx2`` at ``(x, y, z)``."""
        return _second_central_difference(lambda value: self.function(value, y, z), x, h)

    def second_partial_yy(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate ``d2f/dy2`` at ``(x, y, z)``."""
        return _second_central_difference(lambda value: self.function(x, value, z), y, h)

    def second_partial_zz(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate ``d2f/dz2`` at ``(x, y, z)``."""
        return _second_central_difference(lambda value: self.function(x, y, value), z, h)

    def second_partial_xy(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate the mixed partial ``d2f/dxdy`` at ``(x, y, z)``."""
        return _mixed_partial_3d(self.function, x, y, z, "x", "y", h)

    def second_partial_xz(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate the mixed partial ``d2f/dxdz`` at ``(x, y, z)``."""
        return _mixed_partial_3d(self.function, x, y, z, "x", "z", h)

    def second_partial_yz(self, x: float, y: float, z: float, h: float = 1e-5) -> float:
        """Approximate the mixed partial ``d2f/dydz`` at ``(x, y, z)``."""
        return _mixed_partial_3d(self.function, x, y, z, "y", "z", h)

    def gradient(self, x: float, y: float, z: float, h: float = 1e-5) -> Vector3D:
        """Approximate the gradient vector ``grad f`` at ``(x, y, z)``."""
        return Vector3D(
            self.partial_x(x, y, z, h),
            self.partial_y(x, y, z, h),
            self.partial_z(x, y, z, h),
        )

    def directional_derivative(
        self,
        x: float,
        y: float,
        z: float,
        direction: Vector3D,
        h: float = 1e-5,
    ) -> float:
        """Approximate the directional derivative in a given direction."""
        if direction.is_zero():
            raise ValueError("Directional derivative needs a nonzero direction vector.")
        return self.gradient(x, y, z, h).dot(direction.unit())

    def linear_approximation(
        self,
        x0: float,
        y0: float,
        z0: float,
        x: float,
        y: float,
        z: float,
        h: float = 1e-5,
    ) -> float:
        """Approximate ``f(x, y, z)`` using the linearization at ``(x0, y0, z0)``."""
        gradient = self.gradient(x0, y0, z0, h)
        return (
            self.function(x0, y0, z0)
            + gradient.x * (x - x0)
            + gradient.y * (y - y0)
            + gradient.z * (z - z0)
        )

    def differential(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        dz: float,
        h: float = 1e-5,
    ) -> float:
        """Approximate ``df = f_x dx + f_y dy + f_z dz``."""
        gradient = self.gradient(x, y, z, h)
        return gradient.x * dx + gradient.y * dy + gradient.z * dz

    def level_surface_tangent_plane(
        self,
        x: float,
        y: float,
        z: float,
        h: float = 1e-5,
    ) -> Plane3D:
        """Approximate the tangent plane to a level surface ``f(x, y, z) = c``."""
        gradient = self.gradient(x, y, z, h)
        if gradient.is_zero():
            raise ValueError("Level-surface tangent plane needs a nonzero gradient.")
        return Plane3D(point=Point3D(x, y, z), normal=gradient)

    def hessian(
        self,
        x: float,
        y: float,
        z: float,
        h: float = 1e-5,
    ) -> tuple[tuple[float, float, float], ...]:
        """Approximate the Hessian matrix at ``(x, y, z)``."""
        xy = self.second_partial_xy(x, y, z, h)
        xz = self.second_partial_xz(x, y, z, h)
        yz = self.second_partial_yz(x, y, z, h)
        return (
            (self.second_partial_xx(x, y, z, h), xy, xz),
            (xy, self.second_partial_yy(x, y, z, h), yz),
            (xz, yz, self.second_partial_zz(x, y, z, h)),
        )


def make_function_2d(function: NumberFunction2D) -> ScalarFunction2D:
    """Create a scalar function ``f(x, y)``."""
    return ScalarFunction2D(function=function)


def make_function_3d(function: NumberFunction3D) -> ScalarFunction3D:
    """Create a scalar function ``f(x, y, z)``."""
    return ScalarFunction3D(function=function)


def _central_difference(
    function: Callable[[float], float],
    value: float,
    h: float,
) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (function(value + h) - function(value - h)) / (2 * h)


def _second_central_difference(
    function: Callable[[float], float],
    value: float,
    h: float,
) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (function(value + h) - 2 * function(value) + function(value - h)) / h**2


def _mixed_partial_2d(function: NumberFunction2D, x: float, y: float, h: float) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")
    return (
        function(x + h, y + h)
        - function(x + h, y - h)
        - function(x - h, y + h)
        + function(x - h, y - h)
    ) / (4 * h**2)


def _mixed_partial_3d(
    function: NumberFunction3D,
    x: float,
    y: float,
    z: float,
    first_variable: str,
    second_variable: str,
    h: float,
) -> float:
    if h <= 0:
        raise ValueError("h must be positive.")

    def shifted(first_delta: float, second_delta: float) -> float:
        coordinates = {"x": x, "y": y, "z": z}
        coordinates[first_variable] += first_delta
        coordinates[second_variable] += second_delta
        return function(coordinates["x"], coordinates["y"], coordinates["z"])

    return (
        shifted(h, h)
        - shifted(h, -h)
        - shifted(-h, h)
        + shifted(-h, -h)
    ) / (4 * h**2)


def _linear_variable(variable: str, offset: float) -> str:
    if _is_close(offset):
        return variable
    if offset > 0:
        return f"({variable} - {_format_number(offset)})"
    return f"({variable} + {_format_number(abs(offset))})"


def _slope_term(coefficient: float, expression: str, is_first: bool) -> str:
    magnitude = abs(coefficient)
    if _is_close(magnitude, 1.0):
        term = expression
    else:
        term = f"{_format_number(magnitude)}{expression}"

    if is_first:
        return f"-{term}" if coefficient < 0 else term
    sign = "-" if coefficient < 0 else "+"
    return f" {sign} {term}"


def _slope_terms(terms: list[tuple[float, str]]) -> str:
    formatted_terms: list[str] = []
    for coefficient, expression in terms:
        if _is_close(coefficient):
            continue
        formatted_terms.append(_slope_term(coefficient, expression, not formatted_terms))
    return "".join(formatted_terms)


def _is_close(value: float, target: float = 0.0) -> bool:
    return math.isclose(value, target, abs_tol=1e-10)


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"
