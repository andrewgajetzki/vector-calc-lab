"""Solve homogeneous second-order linear ODEs with constant coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ODESolution:
    """Human-readable parts of a general ODE solution."""

    characteristic_equation: str
    root_type: str
    roots: str
    general_solution: str

    def as_text(self) -> str:
        return "\n".join(
            [
                self.characteristic_equation,
                self.roots,
                self.general_solution,
            ]
        )


def classify_roots(a: float, b: float, c: float) -> str:
    """Classify roots of ``a*r^2 + b*r + c = 0``."""
    _validate_second_order(a)
    discriminant = b**2 - 4 * a * c

    if discriminant > 0:
        return "two distinct real roots"
    if discriminant == 0:
        return "repeated real root"
    return "complex conjugate roots"


def solve_homogeneous_second_order(a: float, b: float, c: float) -> ODESolution:
    """Solve ``a*y'' + b*y' + c*y = 0`` using the characteristic equation."""
    _validate_second_order(a)

    root_type = classify_roots(a, b, c)
    characteristic_equation = f"Characteristic equation: {_format_characteristic_equation(a, b, c)} = 0"

    if root_type == "two distinct real roots":
        sqrt_discriminant = math.sqrt(b**2 - 4 * a * c)
        r1 = (-b + sqrt_discriminant) / (2 * a)
        r2 = (-b - sqrt_discriminant) / (2 * a)
        roots = f"Roots: r = {_format_number(r1)}, {_format_number(r2)}"
        general_solution = (
            "General solution: y = "
            f"C1*e^({_format_exponent(r1)}) + C2*e^({_format_exponent(r2)})"
        )
    elif root_type == "repeated real root":
        r = -b / (2 * a)
        roots = f"Roots: r = {_format_number(r)}"
        general_solution = (
            "General solution: y = "
            f"(C1 + C2*x)*e^({_format_exponent(r)})"
        )
    else:
        alpha = -b / (2 * a)
        beta = math.sqrt(4 * a * c - b**2) / (2 * a)
        roots = (
            "Roots: r = "
            f"{_format_complex_root(alpha, beta)}, {_format_complex_root(alpha, -beta)}"
        )
        general_solution = (
            "General solution: y = "
            f"e^({_format_exponent(alpha)})*(C1*cos({_format_number(beta)}x) "
            f"+ C2*sin({_format_number(beta)}x))"
        )

    return ODESolution(
        characteristic_equation=characteristic_equation,
        root_type=root_type,
        roots=roots,
        general_solution=general_solution,
    )


def _validate_second_order(a: float) -> None:
    if a == 0:
        raise ValueError("Coefficient 'a' must be nonzero for a second-order ODE.")


def _format_characteristic_equation(a: float, b: float, c: float) -> str:
    terms = [_format_power_term(a, "r^2"), _format_power_term(b, "r"), _format_constant_term(c)]
    equation = "".join(term for term in terms if term)
    return equation.removeprefix(" + ").removeprefix(" ")


def _format_power_term(coefficient: float, variable: str) -> str:
    if coefficient == 0:
        return ""
    sign = " + " if coefficient > 0 else " - "
    magnitude = abs(coefficient)
    if magnitude == 1:
        return f"{sign}{variable}"
    return f"{sign}{_format_number(magnitude)}{variable}"


def _format_constant_term(coefficient: float) -> str:
    if coefficient == 0:
        return ""
    sign = " + " if coefficient > 0 else " - "
    return f"{sign}{_format_number(abs(coefficient))}"


def _format_complex_root(real_part: float, imaginary_part: float) -> str:
    sign = "+" if imaginary_part >= 0 else "-"
    return f"{_format_number(real_part)} {sign} {_format_number(abs(imaginary_part))}i"


def _format_exponent(coefficient: float) -> str:
    if coefficient == 0:
        return "0"
    if coefficient == 1:
        return "x"
    if coefficient == -1:
        return "-x"
    return f"{_format_number(coefficient)}x"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"