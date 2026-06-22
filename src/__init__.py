"""Tools for studying second-order linear differential equations."""

from .ode_solver import classify_roots, solve_homogeneous_second_order

__all__ = ["classify_roots", "solve_homogeneous_second_order"]