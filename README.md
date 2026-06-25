# vector-calc-lab

A small Python project for studying vector calculus topics.

## Parametric Equations

The parametric-equation tools work with curves of the form:

```text
x = x(t), y = y(t)
```

They can evaluate points, build a point table, and handle the usual calculus
operations for parametric curves.

```python
from src.parametric_equations import make_curve, points_table

curve = make_curve(lambda t: t**2 - 1, lambda t: 2 * t + 3)

print(points_table(curve.sample(-2, 2, steps=4)))
print(curve.dx_dt(2))
print(curve.dy_dt(2))
print(curve.slope(2))
print(curve.second_derivative(2))
print(curve.tangent_line(2).as_text())
print(curve.normal_line(2).as_text())
print(curve.concavity(2))
print(curve.speed(2))
print(curve.arc_length(-2, 2))
print(curve.signed_area_under_curve(-2, 2))
print(curve.surface_area_about_x_axis(0, 2))
print(curve.surface_area_about_y_axis(0, 2))
```

Formulas supported:

- `dy/dx = (dy/dt) / (dx/dt)`
- `d2y/dx2 = d/dt(dy/dx) / (dx/dt)`
- arc length `integral sqrt((dx/dt)^2 + (dy/dt)^2) dt`
- signed area `integral y(t) * dx/dt dt`
- surface area about the x-axis `2*pi*integral |y(t)| ds`
- surface area about the y-axis `2*pi*integral |x(t)| ds`

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

Example:

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
