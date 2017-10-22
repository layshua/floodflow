from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd

#from .riemann_solver import RiemannSolverSWE1D

def plot_solution():
    csv = pd.read_csv(
        "exact.csv"
    )         
    fig = plt.figure(figsize=(12,3))
    fig.patch.set_facecolor('white')

    # Plot the basin/water height
    ax1 = fig.add_subplot(121, ylabel='Water height, h')
    ax1.plot(csv["x"], csv["h"], '-', color="brown")
    plt.xlim(-25.0, 25.0)
    plt.ylim(-0.05, 1.1)

    # Plot the x direction hu discharge
    ax2 = fig.add_subplot(122, ylabel='Water velocity, u')
    ax2.plot(csv["x"], csv["u"], '-', color="black")
    plt.xlim(-25.0, 25.0)
    plt.ylim(-5.5, 5.5)
    plt.show()


class RiemannSolverSWE1DExact(object):
    """Solves the exact Riemann Problem for the one-
    dimensional Shallow Water Equations.

    See Toro (2001) for details.
    """

    def __init__(
        self, left_prim_state, 
        right_prim_state,
        chalen, gate, time_out,
        nr_iters=50, gravity=9.8,
        tol=1e-6, mcells=500
    ):
        self.left_prim_state = left_prim_state
        self.right_prim_state = right_prim_state
        self.chalen = chalen
        self.gate = gate
        self.time_out = time_out
        self.nr_iters = nr_iters
        self.gravity = gravity
        self.tol = tol
        self.mcells = mcells
        self.d = [0.0]*self.mcells
        self.u = [0.0]*self.mcells
        self.xcoord = [0.0]*self.mcells

        # Left state helpers (height, velocity, celerity)
        self.dl = self.left_prim_state["height"]
        self.ul = self.left_prim_state["velocity"]
        self.cl = sqrt(self.gravity*self.dl)

        # Right state helpers (height, velocity, celerity)
        self.dr = self.right_prim_state["height"]
        self.ur = self.right_prim_state["velocity"]
        self.cr = sqrt(self.gravity*self.dr)

        # Is this a dry bed case?
        d_critical = (self.ur - self.ul) - 2.0 * (self.cl + self.cr)
        if (self.dl <= 0.0 or self.dr <= 0.0 or d_critical >= 0.0):
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
                0.5 * self.gravity * (ds + self.dl) / (ds * self.dl)
            )
            ger = sqrt(
                0.5 * self.gravity * (ds + self.dr) / (ds * self.dr)
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
        us = 0.5 * (self.ul + self.ur) + 0.5 * (fr - fl)
        cs = sqrt(self.gravity * ds)

        for i in range(0, self.mcells):
            xcoord = float(i) * self.chalen / float(self.mcells) - self.gate
            s = xcoord / self.time_out
            self.xcoord[i] = xcoord
            # Sample solution throughout wave structure at time time_out
            self.d[i], self.u[i] = self._sample_wet(s, ds, us, cs)

    def _sample_wet(self, s, ds, us, cs):
        """
        Sample the solution through the wave structure
        at a particular time for the wet-bed case
        """
        if (s <= us):
            # Sample left wave
            if (ds >= self.dl):
                # Left shock
                ql = sqrt(
                    (ds + self.dl) * ds / (2.0 * self.dl * self.dl)
                )
                sl = self.ul - self.cl * ql
                if (s <= sl):
                    # Sample point lies to the left of the shock
                    d = self.dl
                    u = self.ul
                else:
                    # Sample point lies to the right of the shock
                    d = ds
                    u = us 
            else:
                # Left rarefaction
                shl = self.ul - self.cl
                if (s <= shl):
                    # Sample point lies to the right of the rarefaction
                    d = self.dl
                    u = self.ul
                else:
                    stl = us - cs
                    if (s <= stl):
                        # Sample point lies inside the rarefaction
                        u = (self.ul + 2.0 * self.cl + 2.0 * s) / 3.0
                        c = (self.ul + 2.0 * self.cl - s) / 3.0
                        d = c * c / self.gravity
                    else:
                        # Sample point lies inside the STAR region
                        d = ds
                        u = us
        else:
            # Sample right wave
            if (ds >= self.dr):
                # Right shock
                qr = sqrt((ds + self.dr) * ds / (2.0 * self.dr * self.dr))
                sr = self.ur + self.cr * qr
                if (s >= sr):
                    # Sample point lies to the right of the shock
                    d = self.dr
                    u = self.ur
                else:
                    # Sample point lies to the left of the shock
                    d = ds
                    u = us
            else:
                # Right rarefaction
                shr = self.ur + self.cr
                if (s >= shr):
                    # Sample point lies to the right of the rarefaction
                    d = self.dr
                    u = self.ur
                else:
                    strr = us + cs
                    if (s >= strr):
                        # Sample point lies inside the rarefaction
                        u = (self.ur - 2.0 * self.cr + 2.0 * s) / 3.0
                        c = (-self.ur + 2.0 * self.cr + s) / 3.0
                        d = c * c / self.gravity
                    else:
                        # Sample point lies in the STAR region
                        d = ds
                        u = us
        return d, u

    def solve(self):
        """
        """
        if self.dry_bed:
            print("This is a dry bed test! Not implemented!")
        else:
            self._solve_wet_bed()
        outfile = open("exact.csv", "w")
        outfile.write("x,h,u\n")
        for i in range(0, self.mcells):
            outfile.write(
                "%s,%s,%s\n" % (self.xcoord[i], self.d[i], self.u[i])
            )
        outfile.close()


if __name__ == "__main__":
    left_prim_state = {
        "height": 1.0,
        "velocity": -5.0
    }

    right_prim_state = {
        "height": 1.0,
        "velocity": 5.0
    }

    rse = RiemannSolverSWE1DExact(
        left_prim_state, right_prim_state,
        chalen=50.0, gate=25.0, time_out=2.5,
        nr_iters=50, gravity=9.8,
        tol=1e-6, mcells=500
    )
    rse.solve()

    plot_solution()
