from math import sqrt
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_solution(test_idx, path, y_axis):
    exact_csv = pd.read_csv(os.path.join(path, "exact_%s.csv" % test_idx))  
    god_csv = pd.read_csv(os.path.join(path, "god_%s.csv" % test_idx))
    
    fig = plt.figure(figsize=(16,6))
    fig.patch.set_facecolor('white')

    # Plot the basin/water height
    ax1 = fig.add_subplot(121, ylabel='Water height, h')
    ax1.plot(exact_csv["x"], exact_csv["h"], '-', color="black")
    ax1.plot(god_csv["x"], god_csv["h"], 'o', color="blue", ms=4, mfc='none')
    plt.xlim(0.0, 50.0)
    plt.ylim(y_axis["h"]["min"], y_axis["h"]["max"])

    # Plot the x direction hu discharge
    ax2 = fig.add_subplot(122, ylabel='Water velocity, u')
    ax2.plot(exact_csv["x"], exact_csv["u"], '-', color="black")
    ax2.plot(god_csv["x"], god_csv["u"], 'o', color="blue", ms=4, mfc='none')
    plt.xlim(0, 50.0)
    plt.ylim(y_axis["u"]["min"], y_axis["u"]["max"])
    plt.savefig(os.path.join(path, "exact_%s.png" % test_idx))


class RiemannSolverSWE1DExact(object):
    """Solves the exact Riemann Problem for the one-
    dimensional Shallow Water Equations. This code
    is a direct Python implementation of the algorithm
    found in Toro (2001). See that text for details.

    Parameters
    ----------
    hul : dict
        Left-state height (h) and velocity (u)
    hur : dict
        Right-state height (h) and velocity (u)
    chalen : float
        Channel length in metres
    gate : float
        Instantaneous "gate" position in metres
    time_out : float
        The time at which to output the sampled wave-structure
    nr_iters : int, optional
        The number of iterations on the Newton Raphson procedure
    gravity : float, optional
        Acceleration due to gravity in ms^{-2}
    tol : float, optional
        Tolerance value to use as an exit condition in the
        Newton-Raphson procedure
    cells : int, optional
        Number of cells to use in the x-direction
    """

    def __init__(
        self, hul, hur,
        chalen, gate, time_out,
        nr_iters=50, gravity=9.8,
        tol=1e-6, cells=500
    ):
        # Initialise the solver flow values and parameters
        self.hul = hul
        self.hur = hur
        self.chalen = chalen
        self.gate = gate
        self.time_out = time_out
        self.nr_iters = nr_iters
        self.gravity = gravity
        self.tol = tol
        self.cells = cells

        # Create the height, velocity and coordinate arrays
        self.d = np.zeros(self.cells)
        self.u = np.zeros(self.cells)
        self.x = np.zeros(self.cells)

        # Left state helpers (height, velocity, celerity)
        self.dl = self.hul["height"]
        self.ul = self.hul["velocity"]
        self.cl = sqrt(self.gravity*self.dl)

        # Right state helpers (height, velocity, celerity)
        self.dr = self.hur["height"]
        self.ur = self.hur["velocity"]
        self.cr = sqrt(self.gravity*self.dr)

        # Is this a dry bed case?
        d_critical = (self.ur - self.ul) - 2.0 * (self.cl + self.cr)
        if (self.dl <= 0.0 or self.dr <= 0.0 or d_critical >= 0.0):
            self.dry_bed = True
        else:
            self.dry_bed = False

    def _start_newton_raphson(self, dl, dr, ul, ur, cl, cr, g):
        """
        Determine the initial value for the Newton-Raphson
        iteration. Utilises the Two-Rarefaction Riemann Solver (TRRS)
        and the Two-Shock Riemann Solver (TSRS) adaptively.

        See Toro (2001) for details.
        """
        d_min = min(dl, dr)

        # Use the TRRS solution as the initial value
        ds = (1.0 / g) * (0.5 * (cl + cr) - 0.25 * (ur - ul))**2

        if (ds <= d_min):
            # Use the TSRS approximation as initial value
            return ds
        else:
            # Use TSRS solution as the initial value with
            # ds computed from the TRRS estimate
            gel = sqrt(0.5 * g * (ds + dl) / (ds * dl))
            ger = sqrt(0.5 * g * (ds + dr) / (ds * dr))
            ds = (gel * dl + ger * dr - (ur - ul)) / (gel + ger)
            return ds

    def _geofun(self, g, d, dk, ck):
        """
        Evaluate the functions FL and FR, as well
        as their derivatives in the iterative Riemann
        solver for the case of the wet-bed.
        """
        if (d <= dk):
            # Wave is a rarefaction (or depression)
            c = sqrt(g * d)
            f = 2.0 * (c - ck)
            fd = g / c
        else:
            # Wave is a shock wave (or bore)
            ges = sqrt(0.5 * g * (d + dk) / (d * dk))
            f = (d - dk)*ges
            fd = ges - 0.25 * g * (d - dk) / (ges * d * d)
        return f, fd

    def _solve_wet_bed(self, dl, dr, ul, ur, cl, cr, g):
        """
        Solve the Riemann problem exactly for the case
        of a wet-bed.
        """
        d0 = ds = self._start_newton_raphson(
            dl, dr, ul, ur, cl, cr, g
        )
        for i in range(0, self.nr_iters):
            fl, fld = self._geofun(g, ds, dl, cl)
            fr, frd = self._geofun(g, ds, dr, cr)
            ds = ds - (fl + fr + ur - ul) / (fld + frd)
            cha = abs(ds - d0) / (0.5 * (ds + d0))
            if cha <= self.tol:
                break
            if (ds < 0.0):
                ds = self.tol
            d0 = ds

        # Converged solution for depth DS in Star Region.
        # Compute velocity 'us' in Star Region
        us = 0.5 * (ul + ur) + 0.5 * (fr - fl)
        cs = sqrt(g * ds)

        for i in range(0, self.cells):
            xcoord = float(i) * self.chalen / \
                float(self.cells) - self.gate
            if abs(self.time_out) < 1e-6:
                s = 0.0
            else:
                s = xcoord / self.time_out
            self.x[i] = xcoord + self.gate
            # Sample solution throughout wave
            # structure at time time_out
            self.d[i], self.u[i] = self._sample_wet(
                dl, dr, ul, ur, cl, cr, g,
                s, ds, us, cs
            )

    def _solve_dry_bed(self, dl, dr, ul, ur, cl, cr, g):
        """
        Compute the exact solution in the case in which
        a portion of dry bed is present.
        """
        for i in range(0, self.cells):
            xcoord = float(i) * self.chalen / \
                float(self.cells) - self.gate
            s = xcoord / self.time_out
            if (dl <= 0.0):
                # Left state is dry
                d, u = self._sample_left_dry_state(
                    dl, dr, ul, ur, cl, cr, g, s
                )
            else:
                if (dr <= 0.0):
                    # Right state is dry
                    d, u = self._sample_right_dry_state(
                        dl, dr, ul, ur, cl, cr, g, s
                    )
                else:
                    # Middle state is dry
                    d, u = self._sample_middle_dry_state(
                        dl, dr, ul, ur, cl, cr, g, s
                    )
            self.x[i] = xcoord + self.gate
            self.d[i] = d
            self.u[i] = u

    def _sample_left_dry_state(
        self, dl, dr, ul, ur, cl, cr, g, s
    ):
        """
        Sample the solution through the wave structure
        at time 'time_out', for the case in which the
        left state is dry. Solution consists of a single
        right rarefaction.
        """
        shr = ur + cr
        if (s >= shr):
            # Sampling point lies to the right
            # of the rarefaction
            d = dr
            u = ur
        else:
            strr = ur - 2.0 * cr
            if (s >= strr):
                # Sampling point lies inside the rarefaction
                u = (ur - 2.0 * cr + 2.0 * s) / 3.0
                c = (-ur + 2.0 * cr + s) / 3.0
                d = c * c / g
            else:
                # Sampling point lies in dry-bed state
                d = dl
                u = ul
        return d, u

    def _sample_middle_dry_state(
        self, dl, dr, ul, ur, cl, cr, g, s
    ):
        """
        Sample the solution through the wave structure
        at time 'time_out', for the case in which the
        middle state is dry. Solution consists of a left
        and a right rarefaction with a dry portion in the
        middle.
        """
        # Compute the wave speeds
        shl = ul - cl
        ssl = ul + 2.0 * cl
        ssr = ur - 2.0 * cr
        shr = ur + cr

        if (s <= shl):
            # Sampling point lies to the left
            # of the left rarefaction
            return dl, ul

        if (s > shl and s <= ssl):
            # Sampling point lies inside the
            # left rarefaction
            u = (ul + 2.0 * cl + 2.0 * s) / 3.0
            c = (ul + 2.0 * cl - s) / 3.0
            d = c * c / g
            return d, u

        if (s > ssl and s <= ssr):
            # Sampling point lies inside the middle
            # dry bed region
            return 0.0, 0.0

        if (s > ssr and s <= shr):
            # Sampling point lies inside the
            # right rarefaction
            u = (ur - 2.0 * cr + 2.0 * s) / 3.0
            c = (-ur + 2.0 * cr + s) / 3.0
            d = c * c / g
            return d, u

        if (s > shr):
            # Sampling point lies to the right
            # of the right rarefaction
            return dr, ur

    def _sample_right_dry_state(
        self, dl, dr, ul, ur, cl, cr, g, s
    ):
        """
        Sample the solution through the wave structure
        at time 'time_out', for the case in which the
        right state is dry. Solution consists of a
        single left rarefaction.
        """
        shl = ul - cl
        if (s <= shl):
            # Sampling point lies to the left of the rarefaction
            d = dl
            u = ul
        else:
            stl = ul + 2.0 * cl
            if (s <= stl):
                # Sampling point lies inside the rarefaction
                u = (ul + 2.0 * cl + 2.0 * s) / 3.0
                c = (ul + 2.0 * cl - s) / 3.0
                d = c * c / g
            else:
                # Sampling point lies in right dry-bed state
                d = dr
                u = ur
        return d, u

    def _sample_wet(
        self, dl, dr, ul, ur, cl, cr, g,
        s, ds, us, cs
    ):
        """
        Sample the solution through the wave structure
        at a particular time for the wet-bed case.
        """
        if (s <= us):
            # Sample left wave
            if (ds >= dl):
                # Left shock
                ql = sqrt(
                    (ds + dl) * ds / (2.0 * dl * dl)
                )
                sl = ul - cl * ql
                if (s <= sl):
                    # Sample point lies to the left of the shock
                    d = dl
                    u = ul
                else:
                    # Sample point lies to the right of the shock
                    d = ds
                    u = us 
            else:
                # Left rarefaction
                shl = ul - cl
                if (s <= shl):
                    # Sample point lies to the right of the rarefaction
                    d = dl
                    u = ul
                else:
                    stl = us - cs
                    if (s <= stl):
                        # Sample point lies inside the rarefaction
                        u = (ul + 2.0 * cl + 2.0 * s) / 3.0
                        c = (ul + 2.0 * cl - s) / 3.0
                        d = c * c / g
                    else:
                        # Sample point lies inside the STAR region
                        d = ds
                        u = us
        else:
            # Sample right wave
            if (ds >= dr):
                # Right shock
                qr = sqrt((ds + dr) * ds / (2.0 * dr * dr))
                sr = ur + cr * qr
                if (s >= sr):
                    # Sample point lies to the right of the shock
                    d = dr
                    u = ur
                else:
                    # Sample point lies to the left of the shock
                    d = ds
                    u = us
            else:
                # Right rarefaction
                shr = ur + cr
                if (s >= shr):
                    # Sample point lies to the right of the rarefaction
                    d = dr
                    u = ur
                else:
                    strr = us + cs
                    if (s >= strr):
                        # Sample point lies inside the rarefaction
                        u = (ur - 2.0 * cr + 2.0 * s) / 3.0
                        c = (-ur + 2.0 * cr + s) / 3.0
                        d = c * c / g
                    else:
                        # Sample point lies in the STAR region
                        d = ds
                        u = us
        return d, u

    def solve(self, test_idx, path):
        """
        Check whether this is a wet/dry bed situation and
        set the appropriate heights/velocities at the
        desired sampling points. Then output the sampled
        values to disk.
        """
        # Reassign variables for readability in formulae
        dl = self.dl
        dr = self.dr
        ul = self.ul
        ur = self.ur
        cl = self.cl
        cr = self.cr
        g = self.gravity

        # Determine if this is wet or dry case
        if self.dry_bed:
            self._solve_dry_bed(dl, dr, ul, ur, cl, cr, g)
        else:
            self._solve_wet_bed(dl, dr, ul, ur, cl, cr, g)

        # Output the test case results to disk
        outfile = open(os.path.join(path, "exact_%s.csv" % test_idx), "w")
        outfile.write("x,h,u\n")
        for i in range(0, self.cells):
            outfile.write(
                "%s,%s,%s\n" % (
                    self.x[i], self.d[i], self.u[i]
                )
            )
        outfile.close()


if __name__ == "__main__":
    path = "/home/mhallsmoore/sites/floodflow/out/"

    tests = {
        1: {
            "hul": {"height": 1.0, "velocity": 2.5},
            "hur": {"height": 0.1, "velocity": 0.0},
            "chalen": 50.0,
            "gate": 10.0,
            "time_out": 7.0,
            "y_axis": {
                "h": {"min": 0.0, "max": 1.1},
                "u": {"min": -0.5, "max": 4.5}
            }
        },
        2: {
            "hul": {"height": 1.0, "velocity": -5.0},
            "hur": {"height": 1.0, "velocity": 5.0},
            "chalen": 50.0,
            "gate": 25.0,
            "time_out": 2.5,
            "y_axis": {
                "h": {"min": 0.0, "max": 1.05},
                "u": {"min": -5.5, "max": 5.5}
            }
        },
        3: {
            "hul": {"height": 1.0, "velocity": 0.0},
            "hur": {"height": 0.0, "velocity": 0.0},
            "chalen": 50.0,
            "gate": 20.0,
            "time_out": 4.0,
            "y_axis": {
                "h": {"min": -0.05, "max": 1.05},
                "u": {"min": -0.5, "max": 7.0}
            }
        },
        4: {
            "hul": {"height": 0.0, "velocity": 0.0},
            "hur": {"height": 1.0, "velocity": 0.0},
            "chalen": 50.0,
            "gate": 30.0,
            "time_out": 4.0,
            "y_axis": {
                "h": {"min": -0.05, "max": 1.05},
                "u": {"min": -7.0, "max": 0.5}
            }
        },
        5: {
            "hul": {"height": 0.1, "velocity": -3.0},
            "hur": {"height": 0.1, "velocity": 3.0},
            "chalen": 50.0,
            "gate": 25.0,
            "time_out": 5.0,
            "y_axis": {
                "h": {"min": -0.015, "max": 0.12},
                "u": {"min": -3.5, "max": 3.5}
            }
        },
    }

    for i in range(1, 6):
        rse = RiemannSolverSWE1DExact(
            tests[i]["hul"], tests[i]["hur"],
            tests[i]["chalen"], tests[i]["gate"],
            tests[i]["time_out"],
            nr_iters=50, gravity=9.8,
            tol=1e-6, cells=500
        )
        rse.solve(i, path)
        plot_solution(i, path, tests[i]["y_axis"])
