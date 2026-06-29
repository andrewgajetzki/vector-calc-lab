# vector-calc-lab

A small Python project for studying vector calculus topics.

## Vectors in the Plane and Space

The vector tools work with vectors in the plane and in rectangular space.
Plane vectors have the richest API because they are the most common starting
point for this project.

```python
import math

from src.vectors import Point2D, Vector2D, make_vector2d, make_vector3d

start = Point2D(1, -2)
end = Point2D(4, 2)
displacement = start.vector_to(end)

u = make_vector2d(3, 4)
v = make_vector2d(-2, 1)

print(displacement.as_text())
print(u.magnitude())
print(u.direction_angle())
print(u.unit().as_text())
print(u.dot(v))
print(u.cross_z(v))
print(u.angle_with(v))
print(u.projection_onto(v).as_text())
print(u.parallelogram_area_with(v))
print(Vector2D.from_polar(2, math.pi / 2).as_text())

space_i = make_vector3d(1, 0, 0)
space_j = make_vector3d(0, 1, 0)
print(space_i.cross(space_j).as_text())
```

Formulas supported:

- magnitude in the plane `sqrt(x^2 + y^2)`
- magnitude in space `sqrt(x^2 + y^2 + z^2)`
- direction angle in the plane `atan2(y, x)`
- unit vectors `v / |v|`
- dot product and angle `cos(theta) = (u dot v) / (|u||v|)`
- 2D cross-product z-component `u_x*v_y - u_y*v_x`
- 3D cross product
- scalar and vector projections
- parallel and orthogonal checks
- parallelogram and triangle areas from vector products
- point-to-point vectors and distances

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

## Polar Coordinates

The polar-coordinate tools work with points and curves of the form:

```text
r = r(theta)
```

Angles are measured in radians. You can convert between polar and rectangular
coordinates, find equivalent polar representations, sample polar curves, and
turn a polar curve into an equivalent parametric curve.

```python
import math

from src.polar_coordinates import (
    PolarPoint,
    cartesian_to_polar,
    make_polar_curve,
    polar_to_cartesian,
    polar_to_cartesian_table,
)

print(polar_to_cartesian(2, math.pi / 2))
print(cartesian_to_polar(0, -3))
print(PolarPoint(2, math.pi / 6).equivalent())

curve = make_polar_curve(lambda theta: 1 + math.cos(theta))

print(polar_to_cartesian_table(curve.sample(0, math.pi, steps=4)))
print(curve.dr_dtheta(math.pi / 2))
print(curve.area(0, math.pi))
print(curve.arc_length(0, math.pi))
print(curve.to_parametric_curve().point_at(math.pi / 2))
```

Formulas supported:

- `x = r*cos(theta)`
- `y = r*sin(theta)`
- `r = sqrt(x^2 + y^2)`
- `theta = atan2(y, x)`
- `dr/dtheta` by central difference
- polar area `1/2*integral r(theta)^2 dtheta`
- polar arc length `integral sqrt(r(theta)^2 + (dr/dtheta)^2) dtheta`
- equivalent points `(r, theta + 2*pi*k)` and `(-r, theta + (2*k + 1)*pi)`

## Conic Sections

The conic-section tools work with implicit equations of the form:

```text
Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
```

They can classify general conics, evaluate the implicit equation at a point,
and compute standard-form features for axis-aligned conics.

```python
from src.conic_sections import classify_conic, make_conic

circle = make_conic(x_squared=1, y_squared=1, x=-4, y=6, constant=-12)
features = circle.standard_form()

print(circle.classification())
print(features.standard_form)
print(features.center)
print(features.radius)
print(circle.evaluate(2, 2))

ellipse_kind = classify_conic(9, 0, 4, -36, 24, -72)
print(ellipse_kind)

parabola = make_conic(x_squared=1, y=-4)
print(parabola.standard_form().as_text())
```

Formulas supported:

- general discriminant `B^2 - 4AC`
- degenerate conic detection from the symmetric conic matrix determinant
- circle, ellipse, parabola, and hyperbola classification
- axis-aligned standard forms by completing the square
- centers, radii, axes, vertices, foci, eccentricity, directrices, and asymptotes

Rotated conics with a nonzero `Bxy` term can be classified, but standard-form
feature extraction currently requires `B = 0`.

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
