"""Study helpers for Calculus 3 topics."""

from __future__ import annotations

from src.parametric_equations import make_curve, points_table


def run_chapter_1_1_demo() -> None:
    """Show the Chapter 1.1 parametric-equation tools on a sample curve."""
    curve = make_curve(
        x_of_t=lambda t: t**2 - 1,
        y_of_t=lambda t: 2 * t + 3,
    )

    print("Chapter 1.1: Parametric Equations")
    print()
    print("Example curve: x = t^2 - 1, y = 2t + 3")
    print()
    print(points_table(curve.sample(start=-2, stop=2, steps=4)))
    print()
    print(f"Slope dy/dx at t = 2: {curve.slope(2):.6g}")
    print(f"Speed at t = 2: {curve.speed(2):.6g}")
    print(f"Approximate arc length from t = -2 to t = 2: {curve.arc_length(-2, 2):.6g}")


if __name__ == "__main__":
    run_chapter_1_1_demo()
