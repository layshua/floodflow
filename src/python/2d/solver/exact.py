from math import sqrt

from .riemann_solver import RiemannSolverSWE1D


class RiemannSolverSWE1DExact(RiemannSolverSWE1D):
    """Solves the exact Riemann Problem for the one-
    dimensional Shallow Water Equations.

    See Toro (2001) for details.
    """

    def __init__(
        self, left_prim_state, right_prim_state,
        nr_iters, gravity=9.81, tol=1e-6
    ):
        self.left_prim_state = left_prim_state
        self.right_prim_state = right_prim_state
        self.nr_iters = nr_iters
        self.gravity = gravity
        self.tol = tol

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

    def _geofun(self, d, dk, ck):
        """
        To evaluate the functions FL and FR, as well
        as their derivatives in the iterative Riemann
        solver for the case of the wet-bed.
        """
        if (d <= dk):
            # Wave is a rarefaction (or depression)
            c = sqrt(self.gravity * d)
            f = 2.0 * (c - ck)
            fd = self.gravity / c
        else:
            # Wave is a shock wave (or bore)
            ges = sqrt(0.5 * self.gravity * (d + dk) / (d * dk))
            f = (d - dk)*ges
            fd = ges - 0.25 * self.gravity * (d - dk) / (ges * d * d)
        return f, fd

    def _solve_wet_bed(self):
        """
        Solve the Riemann problem exactly for the case
        of a wet-bed.
        """
        d0 = ds = self._start_newton_raphson()
        for i in range(0, self.nr_iters):
            fl, fld = self._geofun(ds, self.dl, self.cl)
            fr, frd = self._geofun(ds, self.dr, self.cr)
            ds = ds - (fl + fr + self.ur - self.ul) / (fld + frd)
            cha = abs(ds - d0) / (0.5 * (ds + d0))
            if cha <= self.tol:
                break
            if (ds < 0.0):
                ds = self.tol
            d0 = ds

        # Converged solution for depth DS in Star Region.
        # Compute velocity 'us' in Star Region
        us = 0.5 * (ul + ur) + 0.5 * (fr - fl)
        cs = sqrt(self.gravity * ds)
        return ds, us, cs

    def _sample_wet(self, d, u, s, ds, us, cs):
        """
        Sample the solution through the wave structure
        at a particular time for the wet-bed case
        """
        pass

    def solve(self, wavespeed=None):
        """
        """
        pass
