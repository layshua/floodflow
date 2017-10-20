from math import sqrt

from .riemann_solver import RiemannSolverSWE1D


class RiemannSolverSWE1DExact(RiemannSolverSWE1D):
    """Solves the exact Riemann Problem for the one-
    dimensional Shallow Water Equations.

    See Toro (2001) for details.
    """

    def __init__(
        self, left_prim_state, right_prim_state,
        nr_iters, gravity=9.81,
    ):
        self.left_prim_state = left_prim_state
        self.right_prim_state = right_prim_state
        self.nr_iters = nr_iters
        self.gravity = gravity

        # Left state helpers (density, velocity, celerity)
        self.dl = self.left_prim_state.density
        self.ul = self.left_prim_state.velocity
        self.cl = sqrt(self.gravity*self.dl)

        # Right state helpers (density, velocity, celerity)
        self.dr = self.right_prim_state.density
        self.ur = self.right_prim_state.velocity
        self.cr = sqrt(self.gravity*self.dr)

        # Is this a dry bed case?
        self.d_critical = (self.ur - self.ul) - 2.0 * (self.cl + self.cr)
        if (self.dl <= 0.0 or self.dr <= 0.0 or self.d_critical >= 0.0):
            self.dry_bed = True
        else:
            self.dry_bed = False

    def _start_newton_raphson(self):
        """
        Determine the initial value for the Newton-Raphson
        iteration. Utilises the Two-Rarefaction Riemann Solver (TRRS)
        and the Two-Shock Riemann Solver (TSRS) adaptively.

        See Toro (2001) for details.
        """
        d_min = min(self.dl, self.dr)

        # Use the TRRS solution as the initial value
        ds = (1.0 / self.gravity) * (
            0.5 * (self.cl + self.cr) - 0.25 * (self.ur - self.ul)
        )**2

        if (ds <= d_min):
            # Use the TSRS approximation as initial value
            return ds
        else:
            # Use TSRS solution as the initial value with
            # ds computed from the TRRS estimate
            gel = sqrt(
                0.5 * self.gravity * (ds + self.dl) / (self.ds * self.dl)
            )
            ger = sqrt(
                0.5 * self.gravity * (ds + self.dr) / (self.ds * self.dr)
            )
            ds = (
                gel * self.dl + ger * self.dr - (self.ur - self.ul)
            ) / (gel + ger)
            return ds

    def _solve_wet_bed(self):
        """
        """
        d0 = self._start_newton_raphson()
        for i in range(0, self.nr_iters):
            # TODO: Fill this in
            pass

    def solve(self, wavespeed=None):
        """
        """
        pass
