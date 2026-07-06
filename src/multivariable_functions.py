"""Utilities for scalar functions of several variables."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from src.space_geometry import Plane3D
from src.vectors import Point3D, Vector2D, Vector3D


NumberFunction2D = Callable[[float, float], float]
NumberFunction3D = Callable[[float, float, float], float]


@dataclass(frozen=True)
class LimitEstimate:
    """A numerical estimate of a multivariable limit."""

    value: float
    converged: bool
    spread: float
    drift: float
    samples: tuple[tuple[float, ...], ...]

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        status = "converged" if self.converged else "not converged"
        return (
            f"Estimated limit: {_format_number(self.value)} "
            f"({status}; spread={_format_number(self.spread)}, "
            f"drift={_format_number(self.drift)})"
        )


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

    def chain_rule_derivative(
        self,
        x_of_t: Callable[[float], float],
        y_of_t: Callable[[float], float],
        t: float,
        h: float = 1e-5,
    ) -> float:
        """Approximate ``d/dt f(x(t), y(t))`` using the chain rule."""
        x = x_of_t(t)
        y = y_of_t(t)
        path_velocity = Vector2D(
            _central_difference(x_of_t, t, h),
            _central_difference(y_of_t, t, h),
        )
        return self.gradient(x, y, h).dot(path_velocity)

    def chain_rule_partials(
        self,
        x_of_u_v: Callable[[float, float], float],
        y_of_u_v: Callable[[float, float], float],
        u: float,
        v: float,
        h: float = 1e-5,
    ) -> Vector2D:
        """Approximate ``<dw/du, dw/dv>`` for ``w = f(x(u, v), y(u, v))``."""
        x = x_of_u_v(u, v)
        y = y_of_u_v(u, v)
        gradient = self.gradient(x, y, h)
        x_partials = _parameter_partials(x_of_u_v, u, v, h)
        y_partials = _parameter_partials(y_of_u_v, u, v, h)
        return Vector2D(
            gradient.x * x_partials.x + gradient.y * y_partials.x,
            gradient.x * x_partials.y + gradient.y * y_partials.y,
        )

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

    def limit_at(
        self,
        x: float,
        y: float,
        tolerance: float = 1e-4,
        radii: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    ) -> LimitEstimate:
        """Estimate ``lim f(x, y)`` as ``(x, y)`` approaches a point."""
        samples = []
        for radius in radii:
            radius_samples = []
            for direction_x, direction_y in _UNIT_DIRECTIONS_2D:
                radius_samples.append(
                    self.function(x + radius * direction_x, y + radius * direction_y)
                )
            samples.append(tuple(radius_samples))
        return _estimate_limit(tuple(samples), tolerance)

    def limit_along_path(
        self,
        x_of_t: Callable[[float], float],
        y_of_t: Callable[[float], float],
        t0: float = 0.0,
        tolerance: float = 1e-4,
        offsets: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        side: str = "both",
    ) -> LimitEstimate:
        """Estimate a limit along a parameterized path approaching ``t0``."""
        samples = []
        for offset in offsets:
            offset_samples = []
            for path_offset in _path_offsets(offset, side):
                t = t0 + path_offset
                offset_samples.append(self.function(x_of_t(t), y_of_t(t)))
            samples.append(tuple(offset_samples))
        return _estimate_limit(tuple(samples), tolerance)

    def is_continuous_at(
        self,
        x: float,
        y: float,
        tolerance: float = 1e-4,
    ) -> bool:
        """Return whether numerical evidence suggests continuity at ``(x, y)``."""
        try:
            function_value = self.function(x, y)
            estimate = self.limit_at(x, y, tolerance=tolerance)
        except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
            return False
        return estimate.converged and abs(function_value - estimate.value) <= tolerance


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

    def chain_rule_derivative(
        self,
        x_of_t: Callable[[float], float],
        y_of_t: Callable[[float], float],
        z_of_t: Callable[[float], float],
        t: float,
        h: float = 1e-5,
    ) -> float:
        """Approximate ``d/dt f(x(t), y(t), z(t))`` using the chain rule."""
        x = x_of_t(t)
        y = y_of_t(t)
        z = z_of_t(t)
        path_velocity = Vector3D(
            _central_difference(x_of_t, t, h),
            _central_difference(y_of_t, t, h),
            _central_difference(z_of_t, t, h),
        )
        return self.gradient(x, y, z, h).dot(path_velocity)

    def chain_rule_partials(
        self,
        x_of_u_v: Callable[[float, float], float],
        y_of_u_v: Callable[[float, float], float],
        z_of_u_v: Callable[[float, float], float],
        u: float,
        v: float,
        h: float = 1e-5,
    ) -> Vector2D:
        """Approximate ``<dw/du, dw/dv>`` for a two-parameter composition."""
        x = x_of_u_v(u, v)
        y = y_of_u_v(u, v)
        z = z_of_u_v(u, v)
        gradient = self.gradient(x, y, z, h)
        x_partials = _parameter_partials(x_of_u_v, u, v, h)
        y_partials = _parameter_partials(y_of_u_v, u, v, h)
        z_partials = _parameter_partials(z_of_u_v, u, v, h)
        return Vector2D(
            gradient.x * x_partials.x
            + gradient.y * y_partials.x
            + gradient.z * z_partials.x,
            gradient.x * x_partials.y
            + gradient.y * y_partials.y
            + gradient.z * z_partials.y,
        )

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

    def limit_at(
        self,
        x: float,
        y: float,
        z: float,
        tolerance: float = 1e-4,
        radii: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    ) -> LimitEstimate:
        """Estimate ``lim f(x, y, z)`` as ``(x, y, z)`` approaches a point."""
        samples = []
        for radius in radii:
            radius_samples = []
            for direction_x, direction_y, direction_z in _UNIT_DIRECTIONS_3D:
                radius_samples.append(
                    self.function(
                        x + radius * direction_x,
                        y + radius * direction_y,
                        z + radius * direction_z,
                    )
                )
            samples.append(tuple(radius_samples))
        return _estimate_limit(tuple(samples), tolerance)

    def limit_along_path(
        self,
        x_of_t: Callable[[float], float],
        y_of_t: Callable[[float], float],
        z_of_t: Callable[[float], float],
        t0: float = 0.0,
        tolerance: float = 1e-4,
        offsets: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        side: str = "both",
    ) -> LimitEstimate:
        """Estimate a limit along a parameterized path approaching ``t0``."""
        samples = []
        for offset in offsets:
            offset_samples = []
            for path_offset in _path_offsets(offset, side):
                t = t0 + path_offset
                offset_samples.append(self.function(x_of_t(t), y_of_t(t), z_of_t(t)))
            samples.append(tuple(offset_samples))
        return _estimate_limit(tuple(samples), tolerance)

    def is_continuous_at(
        self,
        x: float,
        y: float,
        z: float,
        tolerance: float = 1e-4,
    ) -> bool:
        """Return whether numerical evidence suggests continuity at ``(x, y, z)``."""
        try:
            function_value = self.function(x, y, z)
            estimate = self.limit_at(x, y, z, tolerance=tolerance)
        except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
            return False
        return estimate.converged and abs(function_value - estimate.value) <= tolerance


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


def _estimate_limit(
    samples: tuple[tuple[float, ...], ...],
    tolerance: float,
) -> LimitEstimate:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    finite_samples = []
    for sample_ring in samples:
        ring = tuple(value for value in sample_ring if math.isfinite(value))
        if not ring:
            raise ValueError("No finite samples were available for limit estimation.")
        finite_samples.append(ring)

    means = [_mean(sample_ring) for sample_ring in finite_samples]
    final_samples = finite_samples[-1]
    spread = max(final_samples) - min(final_samples)
    drift = abs(means[-1] - means[-2]) if len(means) > 1 else 0.0

    return LimitEstimate(
        value=means[-1],
        converged=spread <= tolerance and drift <= tolerance,
        spread=spread,
        drift=drift,
        samples=tuple(finite_samples),
    )


def _path_offsets(offset: float, side: str) -> tuple[float, ...]:
    if offset <= 0:
        raise ValueError("path offsets must be positive.")
    if side == "both":
        return (offset, -offset)
    if side == "positive":
        return (offset,)
    if side == "negative":
        return (-offset,)
    raise ValueError("side must be 'both', 'positive', or 'negative'.")


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


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


_SQRT2 = math.sqrt(2)
_SQRT3 = math.sqrt(3)

_UNIT_DIRECTIONS_2D = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1 / _SQRT2, 1 / _SQRT2),
    (1 / _SQRT2, -1 / _SQRT2),
    (-1 / _SQRT2, 1 / _SQRT2),
    (-1 / _SQRT2, -1 / _SQRT2),
)

_UNIT_DIRECTIONS_3D = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1 / _SQRT3, 1 / _SQRT3, 1 / _SQRT3),
    (1 / _SQRT3, 1 / _SQRT3, -1 / _SQRT3),
    (1 / _SQRT3, -1 / _SQRT3, 1 / _SQRT3),
    (1 / _SQRT3, -1 / _SQRT3, -1 / _SQRT3),
    (-1 / _SQRT3, 1 / _SQRT3, 1 / _SQRT3),
    (-1 / _SQRT3, 1 / _SQRT3, -1 / _SQRT3),
    (-1 / _SQRT3, -1 / _SQRT3, 1 / _SQRT3),
    (-1 / _SQRT3, -1 / _SQRT3, -1 / _SQRT3),
)
