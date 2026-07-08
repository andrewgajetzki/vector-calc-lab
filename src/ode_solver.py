"""Solve second-order linear ODEs with constant coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearForcingTerm:
    """A supported right-hand side for a constant-coefficient linear ODE."""

    kind: str
    coefficient: float = 0.0
    exponent: float = 0.0
    cosine_coefficient: float = 0.0
    sine_coefficient: float = 0.0
    frequency: float = 0.0

    def as_text(self) -> str:
        """Return the forcing term as a compact expression in ``x``."""
        if self.kind == "constant":
            return _format_number(self.coefficient)
        if self.kind == "exponential":
            return _format_exponential_particular(
                self.coefficient,
                self.exponent,
                power=0,
            )
        if self.kind == "sinusoidal":
            return _format_sinusoidal_expression(
                self.cosine_coefficient,
                self.sine_coefficient,
                self.frequency,
            )
        raise ValueError(f"Unsupported forcing term kind: {self.kind}.")

    def is_zero(self) -> bool:
        """Return whether this forcing contributes no right-hand side."""
        if self.kind in ("constant", "exponential"):
            return _is_close(self.coefficient)
        if self.kind == "sinusoidal":
            return _is_close(self.cosine_coefficient) and _is_close(
                self.sine_coefficient
            )
        return False


@dataclass(frozen=True)
class ODESolution:
    """Human-readable parts of a general ODE solution."""

    characteristic_equation: str
    root_type: str
    roots: str
    general_solution: str
    equation: str | None = None
    complementary_solution: str | None = None
    particular_solution: str | None = None

    def as_text(self) -> str:
        parts = []
        if self.equation is not None:
            parts.append(self.equation)
        parts.extend([self.characteristic_equation, self.roots])
        if self.complementary_solution is not None:
            parts.append(self.complementary_solution)
        if self.particular_solution is not None:
            parts.append(self.particular_solution)
        parts.append(self.general_solution)
        return "\n".join(parts)


def constant_forcing(value: float) -> LinearForcingTerm:
    """Create a constant forcing term ``g(x) = value``."""
    return LinearForcingTerm(kind="constant", coefficient=value)


def exponential_forcing(coefficient: float, exponent: float) -> LinearForcingTerm:
    """Create an exponential forcing term ``g(x) = coefficient*e^(exponent*x)``."""
    return LinearForcingTerm(
        kind="exponential",
        coefficient=coefficient,
        exponent=exponent,
    )


def sinusoidal_forcing(
    cosine_coefficient: float = 1.0,
    sine_coefficient: float = 0.0,
    frequency: float = 1.0,
) -> LinearForcingTerm:
    """Create ``g(x) = A*cos(omega*x) + B*sin(omega*x)``."""
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    return LinearForcingTerm(
        kind="sinusoidal",
        cosine_coefficient=cosine_coefficient,
        sine_coefficient=sine_coefficient,
        frequency=frequency,
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

    root_type, roots, homogeneous_expression = _homogeneous_solution_parts(a, b, c)
    characteristic_equation = (
        f"Characteristic equation: {_format_characteristic_equation(a, b, c)} = 0"
    )
    general_solution = f"General solution: y = {homogeneous_expression}"

    return ODESolution(
        characteristic_equation=characteristic_equation,
        root_type=root_type,
        roots=roots,
        general_solution=general_solution,
    )


def solve_linear_second_order(
    a: float,
    b: float,
    c: float,
    forcing: LinearForcingTerm | float | None = None,
) -> ODESolution:
    """Solve ``a*y'' + b*y' + c*y = g(x)`` for supported forcing terms."""
    _validate_second_order(a)
    forcing_term = _coerce_forcing(forcing)
    if forcing_term.is_zero():
        return solve_homogeneous_second_order(a, b, c)

    root_type, roots, homogeneous_expression = _homogeneous_solution_parts(a, b, c)
    particular_expression = _particular_solution(a, b, c, forcing_term)
    characteristic_equation = (
        f"Characteristic equation: {_format_characteristic_equation(a, b, c)} = 0"
    )
    equation = (
        f"Equation: {_format_linear_ode_left(a, b, c)} = {forcing_term.as_text()}"
    )
    complementary_solution = f"Complementary solution: y_c = {homogeneous_expression}"
    particular_solution = f"Particular solution: y_p = {particular_expression}"
    general_solution = (
        "General solution: y = "
        f"{_combine_solution_expressions(homogeneous_expression, particular_expression)}"
    )

    return ODESolution(
        characteristic_equation=characteristic_equation,
        root_type=root_type,
        roots=roots,
        general_solution=general_solution,
        equation=equation,
        complementary_solution=complementary_solution,
        particular_solution=particular_solution,
    )


def _homogeneous_solution_parts(
    a: float,
    b: float,
    c: float,
) -> tuple[str, str, str]:
    root_type = classify_roots(a, b, c)

    if root_type == "two distinct real roots":
        sqrt_discriminant = math.sqrt(b**2 - 4 * a * c)
        r1 = (-b + sqrt_discriminant) / (2 * a)
        r2 = (-b - sqrt_discriminant) / (2 * a)
        roots = f"Roots: r = {_format_number(r1)}, {_format_number(r2)}"
        homogeneous_expression = (
            f"C1*e^({_format_exponent(r1)}) + C2*e^({_format_exponent(r2)})"
        )
    elif root_type == "repeated real root":
        r = -b / (2 * a)
        roots = f"Roots: r = {_format_number(r)}"
        homogeneous_expression = f"(C1 + C2*x)*e^({_format_exponent(r)})"
    else:
        alpha = -b / (2 * a)
        beta = math.sqrt(4 * a * c - b**2) / (2 * a)
        roots = (
            "Roots: r = "
            f"{_format_complex_root(alpha, beta)}, {_format_complex_root(alpha, -beta)}"
        )
        trig = (
            f"C1*cos({_format_trig_argument(beta)}) + "
            f"C2*sin({_format_trig_argument(beta)})"
        )
        if _is_close(alpha):
            homogeneous_expression = trig
        else:
            homogeneous_expression = f"e^({_format_exponent(alpha)})*({trig})"

    return root_type, roots, homogeneous_expression


def _coerce_forcing(forcing: LinearForcingTerm | float | None) -> LinearForcingTerm:
    if forcing is None:
        return constant_forcing(0)
    if isinstance(forcing, LinearForcingTerm):
        return forcing
    return constant_forcing(float(forcing))


def _particular_solution(
    a: float,
    b: float,
    c: float,
    forcing: LinearForcingTerm,
) -> str:
    if forcing.kind == "constant":
        return _particular_exponential(a, b, c, forcing.coefficient, exponent=0)
    if forcing.kind == "exponential":
        return _particular_exponential(a, b, c, forcing.coefficient, forcing.exponent)
    if forcing.kind == "sinusoidal":
        return _particular_sinusoidal(
            a,
            b,
            c,
            forcing.cosine_coefficient,
            forcing.sine_coefficient,
            forcing.frequency,
        )
    raise ValueError(f"Unsupported forcing term kind: {forcing.kind}.")


def _particular_exponential(
    a: float,
    b: float,
    c: float,
    coefficient: float,
    exponent: float,
) -> str:
    polynomial_value = a * exponent**2 + b * exponent + c
    polynomial_derivative = 2 * a * exponent + b
    if not _is_close(polynomial_value):
        return _format_exponential_particular(
            coefficient / polynomial_value,
            exponent,
            power=0,
        )
    if not _is_close(polynomial_derivative):
        return _format_exponential_particular(
            coefficient / polynomial_derivative,
            exponent,
            power=1,
        )
    return _format_exponential_particular(coefficient / (2 * a), exponent, power=2)


def _particular_sinusoidal(
    a: float,
    b: float,
    c: float,
    cosine_coefficient: float,
    sine_coefficient: float,
    frequency: float,
) -> str:
    _validate_positive_frequency(frequency)
    d_value = c - a * frequency**2
    coupling = b * frequency
    determinant = d_value**2 + coupling**2
    if not _is_close(determinant):
        cosine_part = (
            d_value * cosine_coefficient - coupling * sine_coefficient
        ) / determinant
        sine_part = (
            coupling * cosine_coefficient + d_value * sine_coefficient
        ) / determinant
        return _format_sinusoidal_expression(cosine_part, sine_part, frequency)

    cosine_part = -sine_coefficient / (2 * a * frequency)
    sine_part = cosine_coefficient / (2 * a * frequency)
    return _format_sinusoidal_expression(
        cosine_part,
        sine_part,
        frequency,
        multiply_by_x=True,
    )


def _validate_second_order(a: float) -> None:
    if a == 0:
        raise ValueError("Coefficient 'a' must be nonzero for a second-order ODE.")


def _validate_positive_frequency(frequency: float) -> None:
    if frequency <= 0:
        raise ValueError("frequency must be positive.")


def _format_characteristic_equation(a: float, b: float, c: float) -> str:
    terms = [
        _format_power_term(a, "r^2"),
        _format_power_term(b, "r"),
        _format_constant_term(c),
    ]
    equation = "".join(term for term in terms if term)
    return equation.removeprefix(" + ").removeprefix(" ")


def _format_linear_ode_left(a: float, b: float, c: float) -> str:
    terms = [
        _format_power_term(a, "y''"),
        _format_power_term(b, "y'"),
        _format_power_term(c, "y"),
    ]
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


def _format_exponential_particular(
    coefficient: float,
    exponent: float,
    power: int,
) -> str:
    factors = []
    if power == 1:
        factors.append("x")
    elif power == 2:
        factors.append("x^2")
    if not _is_close(exponent):
        factors.append(f"e^({_format_exponent(exponent)})")
    if not factors:
        return _format_number(coefficient)
    return _format_scaled_factor(coefficient, "*".join(factors))


def _format_sinusoidal_expression(
    cosine_coefficient: float,
    sine_coefficient: float,
    frequency: float,
    multiply_by_x: bool = False,
) -> str:
    argument = _format_trig_argument(frequency)
    expression = _join_terms(
        (
            (cosine_coefficient, f"cos({argument})"),
            (sine_coefficient, f"sin({argument})"),
        )
    )
    if multiply_by_x:
        return f"x*({expression})"
    return expression


def _format_trig_argument(frequency: float) -> str:
    if _is_close(frequency - 1):
        return "x"
    return f"{_format_number(frequency)}x"


def _format_scaled_factor(coefficient: float, factor: str) -> str:
    if _is_close(coefficient):
        return "0"
    if _is_close(coefficient - 1):
        return factor
    if _is_close(coefficient + 1):
        return f"-{factor}"
    return f"{_format_number(coefficient)}*{factor}"


def _join_terms(terms: tuple[tuple[float, str], ...]) -> str:
    pieces = []
    for coefficient, factor in terms:
        if _is_close(coefficient):
            continue
        term = _format_scaled_factor(coefficient, factor)
        if not pieces:
            pieces.append(term)
        elif term.startswith("-"):
            pieces.append(f" - {term[1:]}")
        else:
            pieces.append(f" + {term}")
    return "".join(pieces) if pieces else "0"


def _combine_solution_expressions(
    homogeneous_expression: str,
    particular_expression: str,
) -> str:
    if particular_expression == "0":
        return homogeneous_expression
    if particular_expression.startswith("-"):
        return f"{homogeneous_expression} - {particular_expression[1:]}"
    return f"{homogeneous_expression} + {particular_expression}"


def _is_close(value: float) -> bool:
    return math.isclose(value, 0.0, abs_tol=1e-10)


def _format_number(value: float) -> str:
    if _is_close(value):
        return "0"
    nearest_integer = round(value)
    if math.isclose(value, nearest_integer, abs_tol=1e-10):
        return str(int(nearest_integer))
    return f"{value:g}"
