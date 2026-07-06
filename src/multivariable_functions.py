"""Utilities for scalar functions of several variables."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from src.space_geometry import Plane3D
from src.vectors import Point2D, Point3D, Vector2D, Vector3D


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
class FunctionPoint2D:
    """A sampled point on ``z = f(x, y)``."""

    point: Point2D
    value: float

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"f({_format_number(self.point.x)}, {_format_number(self.point.y)}) = "
            f"{_format_number(self.value)}"
        )


@dataclass(frozen=True)
class FunctionPoint3D:
    """A sampled point for ``f(x, y, z)``."""

    point: Point3D
    value: float

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"f({_format_number(self.point.x)}, {_format_number(self.point.y)}, "
            f"{_format_number(self.point.z)}) = {_format_number(self.value)}"
        )


@dataclass(frozen=True)
class CriticalPoint2D:
    """A numerical critical point estimate for ``f(x, y)``."""

    point: Point2D
    value: float
    classification: str
    gradient_magnitude: float

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"{self.classification}: "
            f"f({_format_number(self.point.x)}, {_format_number(self.point.y)}) = "
            f"{_format_number(self.value)}"
        )


@dataclass(frozen=True)
class CriticalPoint3D:
    """A numerical critical point estimate for ``f(x, y, z)``."""

    point: Point3D
    value: float
    classification: str
    gradient_magnitude: float

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"{self.classification}: "
            f"f({_format_number(self.point.x)}, {_format_number(self.point.y)}, "
            f"{_format_number(self.point.z)}) = {_format_number(self.value)}"
        )


@dataclass(frozen=True)
class ExtremaEstimate2D:
    """Numerical absolute extrema estimates over a rectangular region."""

    minimum: FunctionPoint2D
    maximum: FunctionPoint2D
    candidates: tuple[FunctionPoint2D, ...]


@dataclass(frozen=True)
class ExtremaEstimate3D:
    """Numerical absolute extrema estimates over a rectangular box."""

    minimum: FunctionPoint3D
    maximum: FunctionPoint3D
    candidates: tuple[FunctionPoint3D, ...]


@dataclass(frozen=True)
class LagrangePoint2D:
    """A Lagrange-multiplier candidate for ``f(x, y)`` with one constraint."""

    point: Point2D
    value: float
    multiplier: float
    constraint_value: float

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"f({_format_number(self.point.x)}, {_format_number(self.point.y)}) = "
            f"{_format_number(self.value)}, lambda = "
            f"{_format_number(self.multiplier)}"
        )


@dataclass(frozen=True)
class LagrangePoint3D:
    """A Lagrange-multiplier candidate for ``f(x, y, z)`` with one constraint."""

    point: Point3D
    value: float
    multiplier: float
    constraint_value: float

    def as_text(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"f({_format_number(self.point.x)}, {_format_number(self.point.y)}, "
            f"{_format_number(self.point.z)}) = {_format_number(self.value)}, "
            f"lambda = {_format_number(self.multiplier)}"
        )


@dataclass(frozen=True)
class LagrangeExtrema2D:
    """Numerical constrained extrema for one constraint in the plane."""

    minimum: LagrangePoint2D
    maximum: LagrangePoint2D
    candidates: tuple[LagrangePoint2D, ...]


@dataclass(frozen=True)
class LagrangeExtrema3D:
    """Numerical constrained extrema for one constraint in space."""

    minimum: LagrangePoint3D
    maximum: LagrangePoint3D
    candidates: tuple[LagrangePoint3D, ...]


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

    def double_integral_over_rectangle(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        x_segments: int = 100,
        y_segments: int = 100,
    ) -> float:
        """Approximate ``integral integral f(x, y) dA`` over a rectangle."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        return _double_trapezoid_rule(
            self.function,
            x_bounds,
            y_bounds,
            x_segments,
            y_segments,
        )

    def double_integral_type_i(
        self,
        x_bounds: tuple[float, float],
        y_lower: Callable[[float], float],
        y_upper: Callable[[float], float],
        x_segments: int = 100,
        y_segments: int = 100,
    ) -> float:
        """Approximate ``integral_a^b integral_g1(x)^g2(x) f(x, y) dy dx``."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        return _iterated_trapezoid_rule(
            x_bounds,
            y_lower,
            y_upper,
            lambda x, y: self.function(x, y),
            x_segments,
            y_segments,
            "x_segments",
            "y_segments",
            "y_bounds",
        )

    def double_integral_type_ii(
        self,
        y_bounds: tuple[float, float],
        x_lower: Callable[[float], float],
        x_upper: Callable[[float], float],
        y_segments: int = 100,
        x_segments: int = 100,
    ) -> float:
        """Approximate ``integral_c^d integral_h1(y)^h2(y) f(x, y) dx dy``."""
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        return _iterated_trapezoid_rule(
            y_bounds,
            x_lower,
            x_upper,
            lambda y, x: self.function(x, y),
            y_segments,
            x_segments,
            "y_segments",
            "x_segments",
            "x_bounds",
        )

    def double_integral_polar(
        self,
        theta_bounds: tuple[float, float],
        r_lower: Callable[[float], float],
        r_upper: Callable[[float], float],
        theta_segments: int = 100,
        r_segments: int = 100,
    ) -> float:
        """Approximate ``integral integral f(r cos t, r sin t) r dr dt``."""
        theta_bounds = _validate_bounds(theta_bounds, "theta_bounds")
        return _polar_trapezoid_rule(
            lambda theta, radius: self.function(
                radius * math.cos(theta),
                radius * math.sin(theta),
            ),
            theta_bounds,
            r_lower,
            r_upper,
            theta_segments,
            r_segments,
        )

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

    def critical_point(self, x: float, y: float, h: float = 1e-4) -> CriticalPoint2D:
        """Return a critical-point estimate and second-derivative classification."""
        gradient = self.gradient(x, y, h)
        return CriticalPoint2D(
            point=Point2D(x, y),
            value=self.function(x, y),
            classification=self.classify_critical_point(x, y, h),
            gradient_magnitude=gradient.magnitude(),
        )

    def find_critical_points(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        grid_size: int = 7,
        tolerance: float = 1e-5,
        max_iterations: int = 30,
        h: float = 1e-5,
    ) -> tuple[CriticalPoint2D, ...]:
        """Search a rectangle for numerical critical points."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        points: list[CriticalPoint2D] = []

        for seed in _grid_2d(x_bounds, y_bounds, grid_size):
            try:
                x, y = _newton_solve(
                    lambda values: self.gradient(values[0], values[1], h).components(),
                    seed,
                    tolerance,
                    max_iterations,
                    h,
                )
            except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
                continue

            if not _within_bounds((x, y), (x_bounds, y_bounds), tolerance):
                continue

            gradient = self.gradient(x, y, h)
            if gradient.magnitude() > tolerance * 10:
                continue

            point = CriticalPoint2D(
                point=Point2D(x, y),
                value=self.function(x, y),
                classification=self.classify_critical_point(x, y),
                gradient_magnitude=gradient.magnitude(),
            )
            if not _has_nearby_point_2d(point.point, points, tolerance * 10):
                points.append(point)

        return tuple(sorted(points, key=lambda item: (item.point.x, item.point.y)))

    def absolute_extrema_on_rectangle(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        grid_size: int = 25,
        tolerance: float = 1e-5,
        h: float = 1e-5,
    ) -> ExtremaEstimate2D:
        """Estimate absolute extrema over a closed rectangular region."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        candidates = [
            FunctionPoint2D(Point2D(x, y), self.function(x, y))
            for x, y in _grid_2d(x_bounds, y_bounds, grid_size)
        ]
        candidates.extend(
            FunctionPoint2D(point.point, point.value)
            for point in self.find_critical_points(
                x_bounds,
                y_bounds,
                grid_size=max(3, min(grid_size, 9)),
                tolerance=tolerance,
                h=h,
            )
        )
        return ExtremaEstimate2D(
            minimum=min(candidates, key=lambda item: item.value),
            maximum=max(candidates, key=lambda item: item.value),
            candidates=tuple(candidates),
        )

    def lagrange_extrema(
        self,
        constraint: NumberFunction2D,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        grid_size: int = 7,
        tolerance: float = 1e-5,
        max_iterations: int = 40,
        h: float = 1e-5,
    ) -> LagrangeExtrema2D:
        """Estimate constrained extrema using one Lagrange multiplier."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        candidates: list[LagrangePoint2D] = []

        def system(values: tuple[float, ...]) -> tuple[float, ...]:
            x, y, multiplier = values
            function_gradient = self.gradient(x, y, h)
            constraint_gradient = _gradient_2d(constraint, x, y, h)
            return (
                function_gradient.x - multiplier * constraint_gradient.x,
                function_gradient.y - multiplier * constraint_gradient.y,
                constraint(x, y),
            )

        for x, y in _grid_2d(x_bounds, y_bounds, grid_size):
            for multiplier in (-1.0, 0.0, 1.0):
                try:
                    solution = _newton_solve(
                        system,
                        (x, y, multiplier),
                        tolerance,
                        max_iterations,
                        h,
                    )
                except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
                    continue

                candidate_x, candidate_y, candidate_multiplier = solution
                constraint_value = constraint(candidate_x, candidate_y)
                if not _within_bounds(
                    (candidate_x, candidate_y),
                    (x_bounds, y_bounds),
                    tolerance,
                ):
                    continue
                if abs(constraint_value) > tolerance * 10:
                    continue

                point = LagrangePoint2D(
                    point=Point2D(candidate_x, candidate_y),
                    value=self.function(candidate_x, candidate_y),
                    multiplier=candidate_multiplier,
                    constraint_value=constraint_value,
                )
                if not _has_nearby_lagrange_point_2d(point, candidates, tolerance * 10):
                    candidates.append(point)

        if not candidates:
            raise ValueError("No Lagrange extrema candidates were found.")

        return LagrangeExtrema2D(
            minimum=min(candidates, key=lambda item: item.value),
            maximum=max(candidates, key=lambda item: item.value),
            candidates=tuple(sorted(candidates, key=lambda item: item.value)),
        )

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

    def triple_integral_over_box(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        x_segments: int = 30,
        y_segments: int = 30,
        z_segments: int = 30,
    ) -> float:
        """Approximate ``integral integral integral f(x, y, z) dV`` over a box."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        z_bounds = _validate_bounds(z_bounds, "z_bounds")
        return _triple_trapezoid_rule(
            self.function,
            x_bounds,
            y_bounds,
            z_bounds,
            x_segments,
            y_segments,
            z_segments,
        )

    def triple_integral_iterated(
        self,
        x_bounds: tuple[float, float],
        y_lower: Callable[[float], float],
        y_upper: Callable[[float], float],
        z_lower: Callable[[float, float], float],
        z_upper: Callable[[float, float], float],
        x_segments: int = 30,
        y_segments: int = 30,
        z_segments: int = 30,
    ) -> float:
        """Approximate a nested ``dz dy dx`` triple integral."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        return _iterated_triple_trapezoid_rule(
            self.function,
            x_bounds,
            y_lower,
            y_upper,
            z_lower,
            z_upper,
            x_segments,
            y_segments,
            z_segments,
        )

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

    def classify_critical_point(
        self,
        x: float,
        y: float,
        z: float,
        h: float = 1e-4,
    ) -> str:
        """Classify a critical point using Hessian eigenvalue signs."""
        eigenvalues = _symmetric_eigenvalues(self.hessian(x, y, z, h))
        if all(value > 1e-8 for value in eigenvalues):
            return "local minimum"
        if all(value < -1e-8 for value in eigenvalues):
            return "local maximum"
        if any(value > 1e-8 for value in eigenvalues) and any(
            value < -1e-8 for value in eigenvalues
        ):
            return "saddle point"
        return "inconclusive"

    def critical_point(
        self,
        x: float,
        y: float,
        z: float,
        h: float = 1e-4,
    ) -> CriticalPoint3D:
        """Return a critical-point estimate and Hessian classification."""
        gradient = self.gradient(x, y, z, h)
        return CriticalPoint3D(
            point=Point3D(x, y, z),
            value=self.function(x, y, z),
            classification=self.classify_critical_point(x, y, z, h),
            gradient_magnitude=gradient.magnitude(),
        )

    def find_critical_points(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        grid_size: int = 5,
        tolerance: float = 1e-5,
        max_iterations: int = 30,
        h: float = 1e-5,
    ) -> tuple[CriticalPoint3D, ...]:
        """Search a rectangular box for numerical critical points."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        z_bounds = _validate_bounds(z_bounds, "z_bounds")
        points: list[CriticalPoint3D] = []

        for seed in _grid_3d(x_bounds, y_bounds, z_bounds, grid_size):
            try:
                x, y, z = _newton_solve(
                    lambda values: self.gradient(
                        values[0],
                        values[1],
                        values[2],
                        h,
                    ).components(),
                    seed,
                    tolerance,
                    max_iterations,
                    h,
                )
            except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
                continue

            if not _within_bounds(
                (x, y, z),
                (x_bounds, y_bounds, z_bounds),
                tolerance,
            ):
                continue

            gradient = self.gradient(x, y, z, h)
            if gradient.magnitude() > tolerance * 10:
                continue

            point = CriticalPoint3D(
                point=Point3D(x, y, z),
                value=self.function(x, y, z),
                classification=self.classify_critical_point(x, y, z),
                gradient_magnitude=gradient.magnitude(),
            )
            if not _has_nearby_point_3d(point.point, points, tolerance * 10):
                points.append(point)

        return tuple(
            sorted(points, key=lambda item: (item.point.x, item.point.y, item.point.z))
        )

    def absolute_extrema_on_box(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        grid_size: int = 9,
        tolerance: float = 1e-5,
        h: float = 1e-5,
    ) -> ExtremaEstimate3D:
        """Estimate absolute extrema over a closed rectangular box."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        z_bounds = _validate_bounds(z_bounds, "z_bounds")
        candidates = [
            FunctionPoint3D(Point3D(x, y, z), self.function(x, y, z))
            for x, y, z in _grid_3d(x_bounds, y_bounds, z_bounds, grid_size)
        ]
        candidates.extend(
            FunctionPoint3D(point.point, point.value)
            for point in self.find_critical_points(
                x_bounds,
                y_bounds,
                z_bounds,
                grid_size=max(3, min(grid_size, 7)),
                tolerance=tolerance,
                h=h,
            )
        )
        return ExtremaEstimate3D(
            minimum=min(candidates, key=lambda item: item.value),
            maximum=max(candidates, key=lambda item: item.value),
            candidates=tuple(candidates),
        )

    def lagrange_extrema(
        self,
        constraint: NumberFunction3D,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        grid_size: int = 5,
        tolerance: float = 1e-5,
        max_iterations: int = 40,
        h: float = 1e-5,
    ) -> LagrangeExtrema3D:
        """Estimate constrained extrema using one Lagrange multiplier."""
        x_bounds = _validate_bounds(x_bounds, "x_bounds")
        y_bounds = _validate_bounds(y_bounds, "y_bounds")
        z_bounds = _validate_bounds(z_bounds, "z_bounds")
        candidates: list[LagrangePoint3D] = []

        def system(values: tuple[float, ...]) -> tuple[float, ...]:
            x, y, z, multiplier = values
            function_gradient = self.gradient(x, y, z, h)
            constraint_gradient = _gradient_3d(constraint, x, y, z, h)
            return (
                function_gradient.x - multiplier * constraint_gradient.x,
                function_gradient.y - multiplier * constraint_gradient.y,
                function_gradient.z - multiplier * constraint_gradient.z,
                constraint(x, y, z),
            )

        for x, y, z in _grid_3d(x_bounds, y_bounds, z_bounds, grid_size):
            for multiplier in (-1.0, 0.0, 1.0):
                try:
                    solution = _newton_solve(
                        system,
                        (x, y, z, multiplier),
                        tolerance,
                        max_iterations,
                        h,
                    )
                except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
                    continue

                candidate_x, candidate_y, candidate_z, candidate_multiplier = (
                    solution
                )
                constraint_value = constraint(candidate_x, candidate_y, candidate_z)
                if not _within_bounds(
                    (candidate_x, candidate_y, candidate_z),
                    (x_bounds, y_bounds, z_bounds),
                    tolerance,
                ):
                    continue
                if abs(constraint_value) > tolerance * 10:
                    continue

                point = LagrangePoint3D(
                    point=Point3D(candidate_x, candidate_y, candidate_z),
                    value=self.function(candidate_x, candidate_y, candidate_z),
                    multiplier=candidate_multiplier,
                    constraint_value=constraint_value,
                )
                if not _has_nearby_lagrange_point_3d(
                    point,
                    candidates,
                    tolerance * 10,
                ):
                    candidates.append(point)

        if not candidates:
            raise ValueError("No Lagrange extrema candidates were found.")

        return LagrangeExtrema3D(
            minimum=min(candidates, key=lambda item: item.value),
            maximum=max(candidates, key=lambda item: item.value),
            candidates=tuple(sorted(candidates, key=lambda item: item.value)),
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


def _gradient_2d(function: NumberFunction2D, x: float, y: float, h: float) -> Vector2D:
    return Vector2D(
        _central_difference(lambda value: function(value, y), x, h),
        _central_difference(lambda value: function(x, value), y, h),
    )


def _gradient_3d(
    function: NumberFunction3D,
    x: float,
    y: float,
    z: float,
    h: float,
) -> Vector3D:
    return Vector3D(
        _central_difference(lambda value: function(value, y, z), x, h),
        _central_difference(lambda value: function(x, value, z), y, h),
        _central_difference(lambda value: function(x, y, value), z, h),
    )


def _validate_bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    lower, upper = bounds
    if lower > upper:
        raise ValueError(f"{name} must be ordered as (lower, upper).")
    return (lower, upper)


def _grid_2d(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    grid_size: int,
) -> tuple[tuple[float, float], ...]:
    x_values = _linspace(x_bounds, grid_size)
    y_values = _linspace(y_bounds, grid_size)
    return tuple((x, y) for x in x_values for y in y_values)


def _grid_3d(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    grid_size: int,
) -> tuple[tuple[float, float, float], ...]:
    x_values = _linspace(x_bounds, grid_size)
    y_values = _linspace(y_bounds, grid_size)
    z_values = _linspace(z_bounds, grid_size)
    return tuple((x, y, z) for x in x_values for y in y_values for z in z_values)


def _linspace(bounds: tuple[float, float], count: int) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("grid_size must be at least 2.")
    lower, upper = bounds
    step = (upper - lower) / (count - 1)
    return tuple(lower + index * step for index in range(count))


def _double_trapezoid_rule(
    function: NumberFunction2D,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    x_segments: int,
    y_segments: int,
) -> float:
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


def _polar_trapezoid_rule(
    function: NumberFunction2D,
    theta_bounds: tuple[float, float],
    r_lower: Callable[[float], float],
    r_upper: Callable[[float], float],
    theta_segments: int,
    r_segments: int,
) -> float:
    _validate_segments(theta_segments, "theta_segments")
    _validate_segments(r_segments, "r_segments")

    theta_start, theta_stop = theta_bounds
    theta_step = (theta_stop - theta_start) / theta_segments
    total = 0.0

    for theta_index in range(theta_segments + 1):
        theta = theta_start + theta_index * theta_step
        theta_weight = 0.5 if theta_index in (0, theta_segments) else 1.0
        r_start = r_lower(theta)
        r_stop = r_upper(theta)
        _validate_bounds((r_start, r_stop), "r_bounds")
        if r_start < 0 or r_stop < 0:
            raise ValueError("r_bounds must be nonnegative for polar integration.")

        r_step = (r_stop - r_start) / r_segments
        r_total = 0.0
        for r_index in range(r_segments + 1):
            radius = r_start + r_index * r_step
            r_weight = 0.5 if r_index in (0, r_segments) else 1.0
            r_total += r_weight * function(theta, radius) * radius

        total += theta_weight * r_total * r_step

    return total * theta_step


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


def _validate_segments(segments: int, name: str) -> None:
    if segments < 1:
        raise ValueError(f"{name} must be at least 1.")


def _within_bounds(
    coordinates: tuple[float, ...],
    bounds: tuple[tuple[float, float], ...],
    tolerance: float,
) -> bool:
    return all(
        lower - tolerance <= coordinate <= upper + tolerance
        for coordinate, (lower, upper) in zip(coordinates, bounds)
    )


def _newton_solve(
    system: Callable[[tuple[float, ...]], tuple[float, ...]],
    seed: tuple[float, ...],
    tolerance: float,
    max_iterations: int,
    h: float,
) -> tuple[float, ...]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    point = tuple(seed)
    for _ in range(max_iterations):
        values = system(point)
        if not _finite_values(values):
            raise ValueError("Newton iteration produced non-finite values.")
        if _vector_norm(values) <= tolerance:
            return point

        jacobian = _jacobian(system, point, h)
        delta = _solve_linear_system(jacobian, values)
        point = tuple(coordinate - change for coordinate, change in zip(point, delta))
        if not _finite_values(point):
            raise ValueError("Newton iteration produced non-finite coordinates.")
        if _vector_norm(delta) <= tolerance:
            return point

    values = system(point)
    if _finite_values(values) and _vector_norm(values) <= tolerance * 10:
        return point
    raise ValueError("Newton iteration did not converge.")


def _jacobian(
    system: Callable[[tuple[float, ...]], tuple[float, ...]],
    point: tuple[float, ...],
    h: float,
) -> tuple[tuple[float, ...], ...]:
    if h <= 0:
        raise ValueError("h must be positive.")

    columns = []
    for variable_index in range(len(point)):
        forward = list(point)
        backward = list(point)
        forward[variable_index] += h
        backward[variable_index] -= h
        forward_values = system(tuple(forward))
        backward_values = system(tuple(backward))
        columns.append(
            tuple(
                (forward_value - backward_value) / (2 * h)
                for forward_value, backward_value in zip(forward_values, backward_values)
            )
        )

    return tuple(
        tuple(columns[column][row] for column in range(len(point)))
        for row in range(len(point))
    )


def _solve_linear_system(
    matrix: tuple[tuple[float, ...], ...],
    rhs: tuple[float, ...],
) -> tuple[float, ...]:
    size = len(rhs)
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot_row = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot_row][column]) <= 1e-12:
            raise ValueError("Linear system is singular.")
        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )

        for row in range(column + 1, size):
            factor = augmented[row][column] / augmented[column][column]
            for entry in range(column, size + 1):
                augmented[row][entry] -= factor * augmented[column][entry]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        known = sum(
            augmented[row][column] * solution[column]
            for column in range(row + 1, size)
        )
        solution[row] = (augmented[row][size] - known) / augmented[row][row]
    return tuple(solution)


def _symmetric_eigenvalues(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float]:
    values = [list(row) for row in matrix]

    for _ in range(40):
        pivot_row, pivot_column = max(
            ((0, 1), (0, 2), (1, 2)),
            key=lambda indexes: abs(values[indexes[0]][indexes[1]]),
        )
        off_diagonal = values[pivot_row][pivot_column]
        if abs(off_diagonal) <= 1e-12:
            break

        tau = (
            values[pivot_column][pivot_column] - values[pivot_row][pivot_row]
        ) / (2 * off_diagonal)
        sign = 1.0 if tau >= 0 else -1.0
        tangent = sign / (abs(tau) + math.sqrt(1 + tau**2))
        cosine = 1 / math.sqrt(1 + tangent**2)
        sine = tangent * cosine

        row_value = values[pivot_row][pivot_row]
        column_value = values[pivot_column][pivot_column]
        values[pivot_row][pivot_row] = (
            cosine**2 * row_value
            - 2 * sine * cosine * off_diagonal
            + sine**2 * column_value
        )
        values[pivot_column][pivot_column] = (
            sine**2 * row_value
            + 2 * sine * cosine * off_diagonal
            + cosine**2 * column_value
        )
        values[pivot_row][pivot_column] = 0.0
        values[pivot_column][pivot_row] = 0.0

        for index in range(3):
            if index in (pivot_row, pivot_column):
                continue
            row_entry = values[index][pivot_row]
            column_entry = values[index][pivot_column]
            values[index][pivot_row] = values[pivot_row][index] = (
                cosine * row_entry - sine * column_entry
            )
            values[index][pivot_column] = values[pivot_column][index] = (
                sine * row_entry + cosine * column_entry
            )

    return (values[0][0], values[1][1], values[2][2])


def _finite_values(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def _vector_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(value**2 for value in values))


def _has_nearby_point_2d(
    point: Point2D,
    existing_points: list[CriticalPoint2D],
    tolerance: float,
) -> bool:
    return any(
        _distance_2d(point, existing.point) <= tolerance
        for existing in existing_points
    )


def _has_nearby_point_3d(
    point: Point3D,
    existing_points: list[CriticalPoint3D],
    tolerance: float,
) -> bool:
    return any(
        _distance_3d(point, existing.point) <= tolerance
        for existing in existing_points
    )


def _has_nearby_lagrange_point_2d(
    point: LagrangePoint2D,
    existing_points: list[LagrangePoint2D],
    tolerance: float,
) -> bool:
    return any(
        _distance_2d(point.point, existing.point) <= tolerance
        for existing in existing_points
    )


def _has_nearby_lagrange_point_3d(
    point: LagrangePoint3D,
    existing_points: list[LagrangePoint3D],
    tolerance: float,
) -> bool:
    return any(
        _distance_3d(point.point, existing.point) <= tolerance
        for existing in existing_points
    )


def _distance_2d(first: Point2D, second: Point2D) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _distance_3d(first: Point3D, second: Point3D) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
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
