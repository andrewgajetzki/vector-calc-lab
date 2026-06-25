# calculus-3-tools

A small Python project for studying Calculus 3 topics.

## Chapter 1.1: Parametric Equations

The parametric-equation tools work with curves of the form:

```text
x = x(t), y = y(t)
```

They can evaluate points, build a point table, estimate `dy/dx`, estimate `d2y/dx2`,
compute speed, and approximate arc length.

```python
from src.parametric_equations import make_curve, points_table

curve = make_curve(lambda t: t**2 - 1, lambda t: 2 * t + 3)

print(points_table(curve.sample(-2, 2, steps=4)))
print(curve.slope(2))
print(curve.arc_length(-2, 2))
```

You can also run the built-in demo:

```bash
python main.py
```

## Homogeneous Second-Order Linear ODEs

The ODE solver works with equations of the form:

```text
a*y'' + b*y' + c*y = 0
```

It builds the characteristic equation, classifies the roots, and returns a human-readable general solution.

## Example

```python
from src.ode_solver import solve_homogeneous_second_order

solution = solve_homogeneous_second_order(1, 3, 2)
print(solution.as_text())
```

Output:

```text
Characteristic equation: r^2 + 3r + 2 = 0
Roots: r = -1, -2
General solution: y = C1*e^(-x) + C2*e^(-2x)
```

## Run tests

```bash
pytest
```
