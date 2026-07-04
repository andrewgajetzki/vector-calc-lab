"""Utilities for cylindrical and spherical coordinates."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from src.vectors import Point3D


@dataclass(frozen=True)
class CylindricalPoint:
    """A point represented by cylindrical coordinates ``(r, theta, z)``."""

    r: float
    theta: float
    z: float

    def to_rectangular(self) -> Point3D:
        """Convert ``(r, theta, z)`` to rectangular coordinates."""
        return cylindrical_to_rectangular(self.r, self.theta, self.z)

    def to_spherical(self) -> "SphericalPoint":
        """Convert cylindrical coordinates to spherical coordinates."""
        return cylindrical_to_spherical(self.r, self.theta, self.z)

    def volume_element_factor(self) -> float:
        """Return the cylindrical Jacobian factor for volume integrals."""
        return abs(self.r)

    def as_text(self) -> str:
        """Return a compact coordinate form."""
        return (
            f"({_format_number(self.r)}, {_format_number(self.theta)}, "
            f"{_format_number(self.z)})"
        )


@dataclass(frozen=True)
class SphericalPoint:
    """A point represented by spherical coordinates ``(rho, theta, phi)``."""

    rho: float
    theta: float
    phi: float

    def to_rectangular(self) -> Point3D:
        """Convert ``(rho, theta, phi)`` to rectangular coordinates."""
        return spherical_to_rectangular(self.rho, self.theta, self.phi)

    def to_cylindrical(self) -> CylindricalPoint:
        """Convert spherical coordinates to cylindrical coordinates."""
        return spherical_to_cylindrical(self.rho, self.theta, self.phi)

    def volume_element_factor(self) -> float:
        """Return the spherical Jacobian factor for volume integrals."""
        return abs(self.rho**2 * math.sin(self.phi))

    def as_text(self) -> str:
        """Return a compact coordinate form."""
        return (
            f"({_format_number(self.rho)}, {_format_number(self.theta)}, "
            f"{_format_number(self.phi)})"
        )


def cylindrical_to_rectangular(r: float, theta: float, z: float) -> Point3D:
    """Convert cylindrical ``(r, theta, z)`` to rectangular ``(x, y, z)``."""
    return Point3D(x=r * math.cos(theta), y=r * math.sin(theta), z=z)


def rectangular_to_cylindrical(x: float, y: float, z: float) -> CylindricalPoint:
    """Convert rectangular ``(x, y, z)`` to cylindrical coordinates."""
    return CylindricalPoint(r=math.hypot(x, y), theta=math.atan2(y, x), z=z)


def spherical_to_rectangular(rho: float, theta: float, phi: float) -> Point3D:
    """Convert spherical ``(rho, theta, phi)`` to rectangular coordinates."""
    xy_radius = rho * math.sin(phi)
    return Point3D(
        x=xy_radius * math.cos(theta),
        y=xy_radius * math.sin(theta),
        z=rho * math.cos(phi),
    )


def rectangular_to_spherical(x: float, y: float, z: float) -> SphericalPoint:
    """Convert rectangular coordinates to spherical coordinates."""
    rho = math.sqrt(x**2 + y**2 + z**2)
    if math.isclose(rho, 0.0, abs_tol=1e-12):
        return SphericalPoint(rho=0.0, theta=0.0, phi=0.0)
    return SphericalPoint(
        rho=rho,
        theta=math.atan2(y, x),
        phi=math.acos(_clamp(z / rho, -1.0, 1.0)),
    )


def cylindrical_to_spherical(r: float, theta: float, z: float) -> SphericalPoint:
    """Convert cylindrical coordinates to spherical coordinates."""
    rho = math.hypot(r, z)
    if math.isclose(rho, 0.0, abs_tol=1e-12):
        return SphericalPoint(rho=0.0, theta=theta, phi=0.0)
    return SphericalPoint(rho=rho, theta=theta, phi=math.atan2(r, z))


def spherical_to_cylindrical(rho: float, theta: float, phi: float) -> CylindricalPoint:
    """Convert spherical coordinates to cylindrical coordinates."""
    return CylindricalPoint(
        r=rho * math.sin(phi),
        theta=theta,
        z=rho * math.cos(phi),
    )


def normalize_angle(theta: float) -> float:
    """Normalize an angle in radians to ``[0, 2*pi)``."""
    return theta % (2 * math.pi)


def cylindrical_points_table(points: Iterable[CylindricalPoint]) -> str:
    """Format cylindrical points as a compact table."""
    rows = ["r | theta | z", "--|-------|--"]
    for point in points:
        rows.append(
            f"{_format_number(point.r)} | {_format_number(point.theta)} | "
            f"{_format_number(point.z)}"
        )
    return "\n".join(rows)


def spherical_points_table(points: Iterable[SphericalPoint]) -> str:
    """Format spherical points as a compact table."""
    rows = ["rho | theta | phi", "----|-------|----"]
    for point in points:
        rows.append(
            f"{_format_number(point.rho)} | {_format_number(point.theta)} | "
            f"{_format_number(point.phi)}"
        )
    return "\n".join(rows)


def cylindrical_to_rectangular_table(points: Iterable[CylindricalPoint]) -> str:
    """Format cylindrical points with their rectangular coordinates."""
    rows = ["r | theta | z_cyl | x | y | z_rect", "--|-------|-------|---|---|-------"]
    for point in points:
        rectangular = point.to_rectangular()
        rows.append(
            f"{_format_number(point.r)} | {_format_number(point.theta)} | "
            f"{_format_number(point.z)} | {_format_number(rectangular.x)} | "
            f"{_format_number(rectangular.y)} | {_format_number(rectangular.z)}"
        )
    return "\n".join(rows)


def spherical_to_rectangular_table(points: Iterable[SphericalPoint]) -> str:
    """Format spherical points with their rectangular coordinates."""
    rows = ["rho | theta | phi | x | y | z", "----|-------|-----|---|---|--"]
    for point in points:
        rectangular = point.to_rectangular()
        rows.append(
            f"{_format_number(point.rho)} | {_format_number(point.theta)} | "
            f"{_format_number(point.phi)} | {_format_number(rectangular.x)} | "
            f"{_format_number(rectangular.y)} | {_format_number(rectangular.z)}"
        )
    return "\n".join(rows)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(round(value))
    return f"{value:.6g}"
