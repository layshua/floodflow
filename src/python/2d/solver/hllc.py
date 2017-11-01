from math import sqrt

from riemann_solver import RiemannSolver2D


DOMAIN_DIR_N = 0
DOMAIN_DIR_S = 1
DOMAIN_DIR_E = 2
DOMAIN_DIR_W = 3


class RiemannSolver2DHLLC(RiemannSolver2D):
    def __init__(
        self, direction, left, right,
        gravity, depth_tol
    ):
        self.direction = direction
        self.left = left
        self.right = right
        self.gravity = gravity
        self.depth_tol = depth_tol
        self.dir_vec = self._set_dir_vec()

        # Set the velocity values appropriately based on water depth
        if self.left["z"] < depth_tol:
            self.left["u"] = 0.0
            self.left["v"] = 0.0
        else:
            self.left["u"] = self.left["qx"] / self.left["h"]
            self.left["v"] = self.left["qy"] / self.left["h"]
        if self.right["z"] < depth_tol:
            self.right["u"] = 0.0
            self.right["v"] = 0.0
        else:
            self.right["u"] = self.right["qx"] / self.right["h"]
            self.right["v"] = self.right["qy"] / self.right["h"]

    def _set_dir_vec(self):
        """
        Set the direction vector appropriately depending
        upon whether this is a north-south facing Riemann
        Problem, or a west-east facing one.
        """
        if (self.direction == DOMAIN_DIR_N or self.direction == DOMAIN_DIR_S):
            # North-south faces
            return (0.0, 1.0)
        else:
            # West-east faces
            return (1.0, 0.0)

    def _dry_flux(self, zl, zr, zbl, g):
        """
        Calculate the flux if both sides are dry.
        """
        calc = 0.5 * g * (
            ((zl + zr) / 2.0) * ((zl + zr) / 2.0) - zbl * (zl + zr)
        )
        return (
            0.0,
            self.dir_vec[0] * calc,
            self.dir_vec[1] * calc,
        )

    def flux(self):
        """
        Calculate the flux and return it for the
        HLLC Riemann Solver.
        """
        zl = self.left["z"]
        hl = self.left["h"]
        qxl = self.left["qx"]
        qyl = self.left["qy"]
        ul = self.left["u"]
        vl = self.left["v"]
        zbl = self.left["zb"]

        zr = self.right["z"]
        hr = self.right["h"]
        qxr = self.right["qx"]
        qyr = self.right["qy"]
        ur = self.right["u"]
        vr = self.right["v"]
        zbr = self.right["zb"]

        g = self.gravity
        depth_tol = self.depth_tol

        # If both sides are (very nearly or fully) dry
        if (hl < depth_tol and hr < depth_tol):
            return self._dry_flux(zl, zr, zbl, g)

        # If either one of the sides is dry, proceed
        dvel = (
            self.dir_vec[0] * ul + self.dir_vec[1] * vl,
            self.dir_vec[0] * ur + self.dir_vec[1] * vr
        )
        ddis = (
            self.dir_vec[0] * qxl + self.dir_vec[1] * qyl,
            self.dir_vec[0] * qxr + self.dir_vec[1] * qyr
        )
        da = (sqrt(g * hl), sqrt(g * hr))

        # Calculate wave speed estimates
        a_avg = (da[0] + da[1]) / 2.0
        h_star = ((a_avg + (dvel[0] - dvel[1]) / 4.0)**2) / g
        u_star = (dvel[0] + dvel[1]) / 2.0 + da[0] - da[1]
        a_star = sqrt(g * h_star)

        if hl < depth_tol:
            sl = dvel[1] - 2.0 * da[1]
        else:
            if ((dvel[0] - da[0]) > (u_star - a_star)):
                sl = u_star - a_star
            else:
                sl = dvel[0] - da[0]

        if hr < depth_tol:
            sr = dvel[0] + 2 * da[0]
        else:
            if ((dvel[1] + da[1]) < (u_star + a_star)):
                sr = u_star + a_star
            else:
                sr = dvel[1] + da[1]

        sm = (
            sl * hr * (dvel[1] - sr) - sr * hl * (dvel[0] - sl)
        ) / (
            hr * (dvel[1] - sr) - hl * (dvel[0] - sl)
        )

        # Calculate the left and right fluxes
        fluxl = (
            ddis[0],
            dvel[0] * qxl + self.dir_vec[0] * 0.5 * g * (zl * zl - 2.0 * zbl * zl),
            dvel[0] * qyl + self.dir_vec[1] * 0.5 * g * (zl * zl - 2.0 * zbl * zl),
        )
        fluxr = (
            ddis[1],  # TODO: Possible bug below here with zbl possibly being zbr!!
            dvel[1] * qxr + self.dir_vec[0] * 0.5 * g * (zr * zr - 2.0 * zbl * zr),
            dvel[1] * qyr + self.dir_vec[1] * 0.5 * g * (zr * zr - 2.0 * zbl * zr)
        )

        # Choose the correct fluxes based on the speeds
        bleft = (sl >= 0.0)
        bmiddle1 = (sl < 0.0 and sr >= 0.0 and sm >= 0.0)
        bmiddle2 = (sl < 0.0 and sr >= 0.0 and not bmiddle1)
        bright = (not bleft and not bmiddle1 and not bmiddle2)
        if bleft:
            return fluxl
        if bright:
            return fluxr

        fml = self.dir_vec[0] * fluxl[1] + self.dir_vec[1] * fluxl[2]
        fmr = self.dir_vec[0] * fluxr[1] + self.dir_vec[1] * fluxr[2]
        f1m = (sr * fluxl[0] - sl * fluxr[0] + sl * sr * (zr - zl)) / (sr - sl)
        f2m = (sr * fml - sl * fmr + sl * sr * (ddis[1] - ddis[0])) / (sr - sl)

        if (bmiddle1):
            return (
                f1m,
                self.dir_vec[0] * f2m + self.dir_vec[1] * f1m * ul,
                self.dir_vec[0] * f1m * vl + self.dir_vec[1] * f2m
            )
        if (bmiddle2):
            return (
                f1m,
                self.dir_vec[0] * f2m + self.dir_vec[1] * f1m * ur,
                self.dir_vec[0] * f1m * vr + self.dir_vec[1] * f2m
            )
