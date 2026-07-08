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

## Functions of Several Variables

The multivariable-function tools work with scalar functions of two or three
variables:

```text
f(x, y)
f(x, y, z)
```

They approximate multivariable limits, continuity, partial derivatives,
gradients, directional derivatives, chain-rule derivatives, tangent planes, and
linear approximations. They also estimate double integrals over rectangular and
general Type I/Type II regions, triple integrals in rectangular, cylindrical,
and spherical coordinates, changes of variables with Jacobian determinants,
scalar line integrals, scalar surface integrals, centers of mass, moments of
inertia, plus local, absolute, and constrained extrema numerically.

```python
import math

from src.multivariable_functions import make_function_2d, make_function_3d
from src.vectors import Vector2D, Vector3D

surface = make_function_2d(lambda x, y: x**2 + 3 * x * y + y**2)
unit_lamina = make_function_2d(lambda x, y: 1)

print(surface.value_at(1, 2))
print(surface.partial_x(1, 2))
print(surface.partial_y(1, 2))
print(surface.second_partial_xx(1, 2))
print(surface.second_partial_xy(1, 2))
print(surface.gradient(1, 2).as_text())
print(surface.directional_derivative(1, 2, Vector2D(3, 4)))
print(surface.chain_rule_derivative(lambda t: t**2, lambda t: t + 1, 2))
print(
    surface.chain_rule_partials(
        lambda u, v: u + v**2,
        lambda u, v: u * v,
        2,
        3,
    ).as_text()
)
print(surface.tangent_plane(1, 2).as_text())
print(surface.linear_approximation(1, 2, 1.1, 1.9))
print(surface.differential(1, 2, dx=0.1, dy=-0.1))
print(surface.double_integral_over_rectangle((0, 1), (0, 1)))
print(surface.double_integral_type_i((0, 1), lambda x: 0, lambda x: x))
print(surface.double_integral_type_ii((0, 1), lambda y: y, lambda y: 1))
print(surface.double_integral_polar((0, math.pi / 2), lambda theta: 0, lambda theta: 1))
print(surface.line_integral(lambda t: t, lambda t: 2 * t, (0, 1)))
print(
    unit_lamina.double_integral_change_of_variables(
        (0, 1),
        (0, 1),
        lambda u, v: 2 * u,
        lambda u, v: 3 * v,
        jacobian=lambda u, v: 6,
    )
)
print(
    unit_lamina.mass_properties_polar(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 1,
    ).as_text()
)

extrema_surface = make_function_2d(lambda x, y: (x - 1)**2 + (y + 2)**2)
print(extrema_surface.find_critical_points((-3, 3), (-4, 2))[0].as_text())
print(
    extrema_surface.absolute_extrema_on_rectangle(
        (-1, 3),
        (-3, 1),
    ).minimum.as_text()
)

constrained = make_function_2d(lambda x, y: x + y)
print(
    constrained.lagrange_extrema(
        lambda x, y: x**2 + y**2 - 1,
        (-1.2, 1.2),
        (-1.2, 1.2),
    ).maximum.as_text()
)

origin_surface = make_function_2d(lambda x, y: x**2 + y**2)
print(origin_surface.limit_at(0, 0).as_text())
print(origin_surface.is_continuous_at(0, 0))

path_dependent = make_function_2d(
    lambda x, y: 0 if x == 0 and y == 0 else (x * y) / (x**2 + y**2)
)
print(path_dependent.limit_at(0, 0).as_text())
print(path_dependent.limit_along_path(lambda t: t, lambda t: t).as_text())

volume_function = make_function_3d(lambda x, y, z: x**2 + y * z + z**3)
unit_density = make_function_3d(lambda x, y, z: 1)

print(volume_function.partial_z(1, 2, 3))
print(volume_function.gradient(1, 2, 3).as_text())
print(volume_function.directional_derivative(1, 2, 3, Vector3D(0, 0, 2)))
print(
    volume_function.chain_rule_derivative(
        lambda t: t,
        lambda t: t**2,
        lambda t: t + 1,
        2,
    )
)
print(volume_function.linear_approximation(1, 2, 3, 1.1, 1.9, 3.05))
print(volume_function.limit_at(1, 2, 3).as_text())
print(volume_function.is_continuous_at(1, 2, 3))
print(
    unit_density.surface_integral_over_graph(
        (0, 1),
        (0, 1),
        lambda x, y: x + y,
    )
)
print(volume_function.triple_integral_over_box((0, 1), (0, 1), (0, 1)))
print(
    unit_density.triple_integral_change_of_variables(
        (0, 1),
        (0, 1),
        (0, 1),
        lambda u, v, w: 2 * u,
        lambda u, v, w: 3 * v,
        lambda u, v, w: 4 * w,
        jacobian=lambda u, v, w: 24,
    )
)
print(
    volume_function.triple_integral_iterated(
        (0, 1),
        lambda x: 0,
        lambda x: x,
        lambda x, y: 0,
        lambda x, y: y,
    )
)
print(
    unit_density.triple_integral_cylindrical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: 1,
        lambda theta, r: 0,
        lambda theta, r: 2,
    )
)
print(
    unit_density.triple_integral_spherical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: math.pi,
        lambda theta, phi: 0,
        lambda theta, phi: 1,
    )
)
print(
    unit_density.mass_properties_spherical(
        (0, 2 * math.pi),
        lambda theta: 0,
        lambda theta: math.pi,
        lambda theta, phi: 0,
        lambda theta, phi: 1,
    ).as_text()
)

space_extrema = make_function_3d(lambda x, y, z: x**2 + y**2 + z**2)
print(space_extrema.classify_critical_point(0, 0, 0))
print(
    space_extrema.absolute_extrema_on_box(
        (-1, 1),
        (-1, 1),
        (-1, 1),
    ).minimum.as_text()
)

level_surface = make_function_3d(lambda x, y, z: x**2 + y**2 + z**2)
print(level_surface.level_surface_tangent_plane(1, 2, 2).scalar_equation())
```

Formulas supported:

- numerical multivariable limit estimates by radial sampling
- path-based limit estimates
- numerical continuity checks at a point
- first partial derivatives `fx`, `fy`, and `fz`
- second partial derivatives and Hessian matrices
- gradient vectors `grad f`
- directional derivatives `grad f dot u`
- chain rule for one-parameter and two-parameter substitutions
- tangent planes to graphs `z = f(x, y)`
- tangent planes to level surfaces `F(x, y, z) = c`
- linear approximation and differentials
- scalar line integrals `integral_C f ds`
- scalar surface integrals over parametric surfaces and graph surfaces
- double integrals over rectangular, Type I/Type II, and polar regions
- triple integrals over rectangular boxes, nested `dz dy dx` regions,
  cylindrical coordinates, and spherical coordinates
- change of variables for double and triple integrals with supplied or
  numerical Jacobian determinants
- centers of mass and moments of inertia for laminas and solids
- second-derivative critical point test for `f(x, y)`
- Hessian eigenvalue critical point classification for `f(x, y, z)`
- numerical critical point searches over rectangles and boxes
- absolute extrema estimates over rectangles and boxes
- Lagrange-multiplier extrema for one equality constraint

Limit, continuity, integration, critical point, and extrema searches provide
numerical evidence, not symbolic proofs.

## Vector Fields

The vector-field tools work with plane and space vector fields:

```text
F(x, y) = <P(x, y), Q(x, y)>
F(x, y, z) = <P(x, y, z), Q(x, y, z), R(x, y, z)>
```

They evaluate and sample vector fields, compute magnitudes and unit directions,
and approximate the derivative quantities used at the start of vector calculus.
They also approximate oriented line integrals, surface flux integrals, check for
conservative fields, estimate potential differences, and apply Green's theorem
over rectangular, Type I, and Type II plane regions plus Stokes' theorem over
oriented surfaces in space and the divergence theorem over solids.

```python
from src.vectors import Point2D, Point3D
from src.vector_fields import make_vector_field_2d, make_vector_field_3d

plane_field = make_vector_field_2d(
    lambda x, y: x * y,
    lambda x, y: x**2 - y,
)

print(plane_field.value_at(2, 3).as_text())
print(plane_field.magnitude_at(2, 3))
print(plane_field.unit_at(2, 3).as_text())
print(plane_field.jacobian_matrix(2, 3))
print(plane_field.divergence(2, 3))
print(plane_field.curl_z(2, 3))
print(plane_field.line_integral(lambda t: t, lambda t: t, (0, 1)))

circulation_field = make_vector_field_2d(lambda x, y: -y, lambda x, y: x)
source_field = make_vector_field_2d(lambda x, y: x, lambda x, y: y)
print(circulation_field.greens_theorem_circulation_over_rectangle((0, 2), (0, 3)))
print(source_field.greens_theorem_flux_over_rectangle((0, 2), (0, 3)))

plane_conservative = make_vector_field_2d(
    lambda x, y: 2 * x * y,
    lambda x, y: x**2 + 3 * y**2,
)
print(plane_conservative.is_conservative_on_rectangle((-1, 1), (-1, 1)))
print(plane_conservative.potential_difference(Point2D(0, 0), Point2D(2, 1)))

space_field = make_vector_field_3d(
    lambda x, y, z: x * y,
    lambda x, y, z: y * z,
    lambda x, y, z: z * x,
)

print(space_field.value_at(2, 3, 4).as_text())
print(space_field.jacobian_matrix(2, 3, 4))
print(space_field.divergence(2, 3, 4))
print(space_field.curl(2, 3, 4).as_text())
print(space_field.line_integral(lambda t: t, lambda t: t, lambda t: t, (0, 1)))

unit_flux = make_vector_field_3d(
    lambda x, y, z: 0,
    lambda x, y, z: 0,
    lambda x, y, z: 1,
)
print(unit_flux.flux_integral_over_graph((0, 1), (0, 1), lambda x, y: x + y))

stokes_field = make_vector_field_3d(
    lambda x, y, z: -0.5 * y,
    lambda x, y, z: 0.5 * x,
    lambda x, y, z: 0,
)
print(stokes_field.stokes_theorem_over_graph((0, 1), (0, 1), lambda x, y: x + y))

radial_field = make_vector_field_3d(
    lambda x, y, z: x,
    lambda x, y, z: y,
    lambda x, y, z: z,
)
print(radial_field.divergence_theorem_over_box((0, 1), (0, 1), (0, 1)))

space_conservative = make_vector_field_3d(
    lambda x, y, z: y + z,
    lambda x, y, z: x + z,
    lambda x, y, z: x + y,
)
print(space_conservative.is_conservative_on_box((-1, 1), (-1, 1), (-1, 1)))
print(space_conservative.potential_difference(Point3D(0, 0, 0), Point3D(1, 2, 3)))
```

Formulas supported:

- field evaluation in the plane and in space
- rectangular grid sampling
- vector-field magnitudes and unit directions
- Jacobian matrices of component partial derivatives
- plane divergence `P_x + Q_y`
- plane scalar curl `Q_x - P_y`
- space divergence `P_x + Q_y + R_z`
- space curl `<R_y - Q_z, P_z - R_x, Q_x - P_y>`
- oriented line integrals `integral_C F dot dr`
- oriented flux integrals over parametric surfaces and graph surfaces
- Green's theorem circulation form `double integral_R (Q_x - P_y) dA`
- Green's theorem flux form `double integral_R (P_x + Q_y) dA`
- Stokes' theorem circulation form `double integral_S curl(F) dot n dS`
- divergence theorem flux form `triple integral_E div(F) dV`
- conservative-field checks by sampled curl values
- potential differences for conservative fields

## Vector-Valued Functions and Space Curves

The space-curve tools work with vector-valued functions of the form:

```text
r(t) = <x(t), y(t), z(t)>
```

They can evaluate and sample space curves, approximate derivatives, build
tangent lines, and compute the usual motion and Frenet-frame quantities.

```python
import math

from src.space_curves import make_space_curve, space_curve_points_table

curve = make_space_curve(math.cos, math.sin, lambda t: t)

print(space_curve_points_table(curve.sample(0, math.pi / 2, steps=2)))
print(curve.point_at(0).as_text())
print(curve.position_vector(0).as_text())
print(curve.velocity(0).as_text())
print(curve.acceleration(0).as_text())
print(curve.speed(0))
print(curve.unit_tangent(0).as_text())
print(curve.unit_normal(0).as_text())
print(curve.binormal(0).as_text())
print(curve.curvature(0))
print(curve.torsion(0))
print(curve.arc_length(0, math.pi))
print(curve.tangent_line(0).parametric_equations())
```

Formulas supported:

- velocity `r'(t)`
- acceleration `r''(t)`
- speed `|r'(t)|`
- arc length `integral |r'(t)| dt`
- unit tangent `T(t) = r'(t) / |r'(t)|`
- curvature `kappa = |r'(t) x r''(t)| / |r'(t)|^3`
- principal unit normal `N(t) = T'(t) / |T'(t)|`
- binormal `B(t) = T(t) x N(t)`
- torsion `tau = ((r'(t) x r''(t)) dot r'''(t)) / |r'(t) x r''(t)|^2`
- tangential and normal acceleration components

## Lines and Planes in Space

The space-geometry tools work with lines and planes in three dimensions using
the existing `Point3D` and `Vector3D` types.

```python
import math

from src.space_geometry import (
    line_from_points,
    make_line,
    make_plane,
    plane_from_scalar_equation,
)
from src.vectors import Point3D, make_vector3d

line = line_from_points(Point3D(1, 2, -1), Point3D(3, 1, 2))

print(line.vector_equation())
print(line.parametric_equations())
print(line.symmetric_equations())
print(line.point_at(2))

plane = make_plane(Point3D(1, 2, -1), make_vector3d(2, -1, 3))

print(plane.scalar_equation())
print(plane.point_normal_form())
print(plane.distance_to_point(Point3D(1, 2, 0)))

xy_plane = plane_from_scalar_equation(0, 0, 1, 0)
crossing_line = make_line(Point3D(1, 1, 1), make_vector3d(0, 0, -1))

print(crossing_line.intersection_with_plane(xy_plane))
print(plane.angle_with_line(crossing_line))
print(math.degrees(plane.angle_with_line(crossing_line)))
```

Formulas supported:

- line vector equation `r(t) = r0 + t*v`
- line parametric equations `x = x0 + at`, `y = y0 + bt`, `z = z0 + ct`
- line symmetric equations when direction components are nonzero
- plane scalar equation `Ax + By + Cz = D`
- plane point-normal form `n dot ((x, y, z) - p0) = 0`
- planes from three noncollinear points
- point-to-line distance `|(p - p0) x v| / |v|`
- point-to-plane distance `|Ax + By + Cz - D| / sqrt(A^2 + B^2 + C^2)`
- angles between lines, planes, and line-plane pairs
- line-plane intersections and plane-plane intersection lines

## Quadric Surfaces

The quadric-surface tools work with second-degree equations in three variables:

```text
Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
```

They can evaluate implicit equations, detect rotated quadrics from mixed terms,
and compute standard-form features for axis-aligned quadric surfaces.

```python
from src.quadric_surfaces import classify_quadric, make_quadric

sphere = make_quadric(
    x_squared=1,
    y_squared=1,
    z_squared=1,
    x=-2,
    y=4,
    z=-6,
    constant=10,
)

print(sphere.classification())
print(sphere.standard_form().as_text())
print(sphere.evaluate(1, -2, 5))

ellipsoid_kind = classify_quadric(36, 16, 9, constant=-144)
print(ellipsoid_kind)

paraboloid = make_quadric(x_squared=1 / 4, y_squared=1 / 9, z=-1)
print(paraboloid.standard_form().as_text())
```

Surface types supported:

- sphere and ellipsoid
- hyperboloid of one sheet
- hyperboloid of two sheets
- elliptic cone
- elliptic paraboloid
- hyperbolic paraboloid
- circular, elliptic, hyperbolic, and parabolic cylinders

Standard-form extraction completes the square for axis-aligned quadrics.
Rotated quadrics with nonzero `Dxy`, `Exz`, or `Fyz` terms are detected, but
their rotated standard forms are not expanded yet.

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

## Cylindrical and Spherical Coordinates

The space-coordinate tools convert between rectangular, cylindrical, and
spherical coordinates. Spherical coordinates use the usual Calculus 3
convention:

```text
rho = distance from the origin
theta = angle in the xy-plane from the positive x-axis
phi = angle down from the positive z-axis
```

```python
import math

from src.space_coordinates import (
    CylindricalPoint,
    SphericalPoint,
    cylindrical_to_rectangular,
    rectangular_to_cylindrical,
    rectangular_to_spherical,
    spherical_to_rectangular,
)

cylindrical = CylindricalPoint(2, math.pi / 2, 3)
spherical = SphericalPoint(4, math.pi / 2, math.pi / 3)

print(cylindrical.to_rectangular())
print(cylindrical.to_spherical())
print(cylindrical.volume_element_factor())

print(spherical.to_rectangular())
print(spherical.to_cylindrical())
print(spherical.volume_element_factor())

print(cylindrical_to_rectangular(2, math.pi / 2, 3))
print(spherical_to_rectangular(4, math.pi / 2, math.pi / 3))
print(rectangular_to_cylindrical(0, -3, 4))
print(rectangular_to_spherical(0, 0, 3))
```

Formulas supported:

- cylindrical to rectangular `x = r*cos(theta)`, `y = r*sin(theta)`, `z = z`
- rectangular to cylindrical `r = sqrt(x^2 + y^2)`, `theta = atan2(y, x)`
- spherical to rectangular `x = rho*sin(phi)*cos(theta)`, `y = rho*sin(phi)*sin(theta)`, `z = rho*cos(phi)`
- rectangular to spherical `rho = sqrt(x^2 + y^2 + z^2)`, `phi = acos(z/rho)`
- cylindrical volume element `dV = r dr dtheta dz`
- spherical volume element `dV = rho^2*sin(phi) drho dtheta dphi`

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
