"""Utilities for vectors in the plane and in space."""

from __future__ import annotations

import math
from dataclasses import dataclass


_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Point2D:
    """A point in the rectangular plane."""

    x: float
    y: float

    def vector_to(self, other: "Point2D") -> "Vector2D":
        """Return the vector from this point to another point."""
        return Vector2D(x=other.x - self.x, y=other.y - self.y)

    def distance_to(self, other: "Point2D") -> float:
        """Return the distance from this point to another point."""
        return self.vector_to(other).magnitude()


@dataclass(frozen=True)
class Point3D:
    """A point in rectangular space."""

    x: float
    y: float
    z: float

    def vector_to(self, other: "Point3D") -> "Vector3D":
        """Return the vector from this point to another point."""
        return Vector3D(x=other.x - self.x, y=other.y - self.y, z=other.z - self.z)

    def distance_to(self, other: "Point3D") -> float:
        """Return the distance from this point to another point."""
        return self.vector_to(other).magnitude()


@dataclass(frozen=True)
class Vector2D:
    """A vector in the plane."""

    x: float
    y: float

    @classmethod
    def from_points(cls, start: Point2D, end: Point2D) -> "Vector2D":
        """Create the vector from ``start`` to ``end``."""
        return start.vector_to(end)

    @classmethod
    def from_polar(cls, magnitude: float, angle: float) -> "Vector2D":
        """Create a plane vector from magnitude and direction angle."""
        return cls(x=magnitude * math.cos(angle), y=magnitude * math.sin(angle))

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __neg__(self) -> "Vector2D":
        return Vector2D(-self.x, -self.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Vector2D":
        if _is_close(scalar):
            raise ValueError("Cannot divide a vector by zero.")
        return Vector2D(self.x / scalar, self.y / scalar)

    def components(self) -> tuple[float, float]:
        """Return vector components as ``(x, y)``."""
        return (self.x, self.y)

    def magnitude(self) -> float:
        """Return vector length ``sqrt(x^2 + y^2)``."""
        return math.hypot(self.x, self.y)

    def is_zero(self, tolerance: float = _TOLERANCE) -> bool:
        """Return whether this vector is approximately the zero vector."""
        return self.magnitude() <= tolerance

    def direction_angle(self) -> float:
        """Return the direction angle in radians, using ``atan2(y, x)``."""
        if self.is_zero():
            raise ValueError("The zero vector does not have a direction angle.")
        return math.atan2(self.y, self.x)

    def to_polar(self) -> tuple[float, float]:
        """Return ``(magnitude, direction_angle)``."""
        return (self.magnitude(), self.direction_angle())

    def unit(self) -> "Vector2D":
        """Return the unit vector in the same direction."""
        magnitude = self.magnitude()
        if _is_close(magnitude):
            raise ValueError("The zero vector does not have a unit vector.")
        return self / magnitude

    def dot(self, other: "Vector2D") -> float:
        """Return the dot product with another plane vector."""
        return self.x * other.x + self.y * other.y

    def cross_z(self, other: "Vector2D") -> float:
        """Return the z-component of the 2D cross product."""
        return self.x * other.y - self.y * other.x

    def determinant(self, other: "Vector2D") -> float:
        """Return the 2D determinant formed by this vector and another vector."""
        return self.cross_z(other)

    def angle_with(self, other: "Vector2D") -> float:
        """Return the smaller angle between two vectors in radians."""
        denominator = self.magnitude() * other.magnitude()
        if _is_close(denominator):
            raise ValueError("The angle with the zero vector is undefined.")
        cosine = _clamp(self.dot(other) / denominator, -1.0, 1.0)
        return math.acos(cosine)

    def scalar_projection_onto(self, other: "Vector2D") -> float:
        """Return the signed scalar projection of this vector onto another vector."""
        magnitude = other.magnitude()
        if _is_close(magnitude):
            raise ValueError("Cannot project onto the zero vector.")
        return self.dot(other) / magnitude

    def projection_onto(self, other: "Vector2D") -> "Vector2D":
        """Return the vector projection of this vector onto another vector."""
        denominator = other.dot(other)
        if _is_close(denominator):
            raise ValueError("Cannot project onto the zero vector.")
        return other * (self.dot(other) / denominator)

    def perpendicular_left(self) -> "Vector2D":
        """Return this vector rotated 90 degrees counterclockwise."""
        return Vector2D(-self.y, self.x)

    def perpendicular_right(self) -> "Vector2D":
        """Return this vector rotated 90 degrees clockwise."""
        return Vector2D(self.y, -self.x)

    def is_parallel_to(
        self,
        other: "Vector2D",
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether two plane vectors are approximately parallel."""
        return _is_close(self.cross_z(other), tolerance=tolerance)

    def is_orthogonal_to(
        self,
        other: "Vector2D",
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether two plane vectors are approximately orthogonal."""
        return _is_close(self.dot(other), tolerance=tolerance)

    def parallelogram_area_with(self, other: "Vector2D") -> float:
        """Return the area of the parallelogram spanned by two plane vectors."""
        return abs(self.cross_z(other))

    def triangle_area_with(self, other: "Vector2D") -> float:
        """Return the area of the triangle spanned by two plane vectors."""
        return 0.5 * self.parallelogram_area_with(other)

    def as_text(self) -> str:
        """Return a compact component form."""
        return f"<{_format_number(self.x)}, {_format_number(self.y)}>"


@dataclass(frozen=True)
class Vector3D:
    """A vector in rectangular space."""

    x: float
    y: float
    z: float

    @classmethod
    def from_points(cls, start: Point3D, end: Point3D) -> "Vector3D":
        """Create the vector from ``start`` to ``end``."""
        return start.vector_to(end)

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vector3D":
        return Vector3D(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> "Vector3D":
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3D":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Vector3D":
        if _is_close(scalar):
            raise ValueError("Cannot divide a vector by zero.")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def components(self) -> tuple[float, float, float]:
        """Return vector components as ``(x, y, z)``."""
        return (self.x, self.y, self.z)

    def magnitude(self) -> float:
        """Return vector length ``sqrt(x^2 + y^2 + z^2)``."""
        return math.sqrt(self.dot(self))

    def is_zero(self, tolerance: float = _TOLERANCE) -> bool:
        """Return whether this vector is approximately the zero vector."""
        return self.magnitude() <= tolerance

    def unit(self) -> "Vector3D":
        """Return the unit vector in the same direction."""
        magnitude = self.magnitude()
        if _is_close(magnitude):
            raise ValueError("The zero vector does not have a unit vector.")
        return self / magnitude

    def dot(self, other: "Vector3D") -> float:
        """Return the dot product with another space vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3D") -> "Vector3D":
        """Return the cross product with another space vector."""
        return Vector3D(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    def angle_with(self, other: "Vector3D") -> float:
        """Return the smaller angle between two vectors in radians."""
        denominator = self.magnitude() * other.magnitude()
        if _is_close(denominator):
            raise ValueError("The angle with the zero vector is undefined.")
        cosine = _clamp(self.dot(other) / denominator, -1.0, 1.0)
        return math.acos(cosine)

    def scalar_projection_onto(self, other: "Vector3D") -> float:
        """Return the signed scalar projection of this vector onto another vector."""
        magnitude = other.magnitude()
        if _is_close(magnitude):
            raise ValueError("Cannot project onto the zero vector.")
        return self.dot(other) / magnitude

    def projection_onto(self, other: "Vector3D") -> "Vector3D":
        """Return the vector projection of this vector onto another vector."""
        denominator = other.dot(other)
        if _is_close(denominator):
            raise ValueError("Cannot project onto the zero vector.")
        return other * (self.dot(other) / denominator)

    def is_parallel_to(
        self,
        other: "Vector3D",
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether two space vectors are approximately parallel."""
        return self.cross(other).magnitude() <= tolerance

    def is_orthogonal_to(
        self,
        other: "Vector3D",
        tolerance: float = _TOLERANCE,
    ) -> bool:
        """Return whether two space vectors are approximately orthogonal."""
        return _is_close(self.dot(other), tolerance=tolerance)

    def parallelogram_area_with(self, other: "Vector3D") -> float:
        """Return the area of the parallelogram spanned by two space vectors."""
        return self.cross(other).magnitude()

    def triangle_area_with(self, other: "Vector3D") -> float:
        """Return the area of the triangle spanned by two space vectors."""
        return 0.5 * self.parallelogram_area_with(other)

    def as_text(self) -> str:
        """Return a compact component form."""
        return (
            f"<{_format_number(self.x)}, {_format_number(self.y)}, "
            f"{_format_number(self.z)}>"
        )


def make_vector2d(x: float, y: float) -> Vector2D:
    """Create a vector in the plane."""
    return Vector2D(x=x, y=y)


def make_vector3d(x: float, y: float, z: float) -> Vector3D:
    """Create a vector in rectangular space."""
    return Vector3D(x=x, y=y, z=z)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _is_close(
    value: float,
    target: float = 0.0,
    tolerance: float = _TOLERANCE,
) -> bool:
    return math.isclose(value, target, abs_tol=tolerance)


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"
