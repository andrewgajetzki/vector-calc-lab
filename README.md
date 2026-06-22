# calculus-3-tools

A small Python project for studying homogeneous second-order linear differential equations with constant coefficients.

The solver works with equations of the form:

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