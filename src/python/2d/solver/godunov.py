import os

import numpy as np

from hllc import RiemannSolver2DHLLC


VERY_SMALL = 1e-10
#VERY_SMALL = 1e-5
QUITE_SMALL = VERY_SMALL * 10.0
DOMAIN_DIR_N = 0
DOMAIN_DIR_S = 1
DOMAIN_DIR_E = 2
DOMAIN_DIR_W = 3


def reconstruct_interface(
    left, bed_left, right, bed_right, direction
):
    ucStop = 0
    d_left = left["z"] - bed_left
    d_right = right["z"] - bed_right

    # Create reconstructed interfaces
    rec_left = {
        "z": left["z"],
        "h": d_left,
        "qx": left["qx"],
        "qy": left["qy"],
        "u": 0.0 if d_left < VERY_SMALL else left["qx"] / d_left,
        "v": 0.0 if d_left < VERY_SMALL else left["qy"] / d_left,
        "zb": bed_left
    }
    rec_right = {
        "z": right["z"],
        "h": d_right,
        "qx": right["qx"],
        "qy": right["qy"],
        "u": 0.0 if d_right < VERY_SMALL else right["qx"] / d_right,
        "v": 0.0 if d_right < VERY_SMALL else right["qy"] / d_right,
        "zb": bed_right
    }

    # Maximum bed elevation
    d_bed_max = rec_left["zb"] if rec_left["zb"] > rec_right["zb"] else rec_right["zb"]
    if direction < DOMAIN_DIR_S:
        d_shift_vert = d_bed_max - left["z"]
    else:
        d_shift_vert = d_bed_max - right["z"]
    if (d_shift_vert < 0.0):
        d_shift_vert = 0.0

    # Depth adjustments
    rec_left["h"] = (left["z"] - d_bed_max) if (left["z"] - d_bed_max > 0.0) else 0.0
    rec_left["z"] = rec_left["h"] + d_bed_max
    rec_left["qx"] = rec_left["h"] * rec_left["u"]
    rec_left["qy"] = rec_left["h"] * rec_left["v"]

    rec_right["h"] = (right["z"] - d_bed_max) if (right["z"] - d_bed_max > 0.0) else 0.0
    rec_right["z"] = rec_right["h"] + d_bed_max
    rec_right["qx"] = rec_right["h"] * rec_right["u"]
    rec_right["qy"] = rec_right["h"] * rec_right["v"]

    # Prevent draining from a dry cell
    if direction == DOMAIN_DIR_N:
        if rec_left["h"] <= VERY_SMALL and left["qy"] > 0.0:
            ucStop += 1
        if rec_right["h"] <= VERY_SMALL and rec_left["v"] < 0.0:
            ucStop += 1
            rec_left["v"] = 0.0
        if rec_left["h"] <= VERY_SMALL and rec_right["v"] > 0.0:
            ucStop += 1
            rec_right["v"] = 0.0

    elif direction == DOMAIN_DIR_S:
        if rec_right["h"] <= VERY_SMALL and right["qy"] < 0.0:
            ucStop += 1
        if rec_right["h"] <= VERY_SMALL and rec_left["v"] < 0.0:
            ucStop += 1
            rec_left["v"] = 0.0
        if rec_left["h"] <= VERY_SMALL and rec_right["v"] > 0.0:
            ucStop += 1
            rec_right["v"] = 0.0

    elif direction == DOMAIN_DIR_E:
        if rec_left["h"] <= VERY_SMALL and left["qx"] > 0.0:
            ucStop += 1
        if rec_right["h"] <= VERY_SMALL and rec_left["u"] < 0.0:
            ucStop += 1
            rec_left["u"] = 0.0
        if rec_left["h"] <= VERY_SMALL and rec_right["u"] > 0.0:
            ucStop += 1
            rec_right["u"] = 0.0

    elif direction == DOMAIN_DIR_W:
        if rec_right["h"] <= VERY_SMALL and right["qx"] < 0.0:
            ucStop += 1
        if rec_right["h"] <= VERY_SMALL and rec_left["u"] < 0.0:
            ucStop += 1
            rec_left["u"] = 0.0
        if rec_left["h"] <= VERY_SMALL and rec_right["u"] > 0.0:
            ucStop += 1
            rec_right["u"] = 0.0

    # Local modification of bed level (and FSL to maintain depth)
    rec_left["zb"] = d_bed_max - d_shift_vert
    rec_right["zb"] = d_bed_max - d_shift_vert
    rec_left["z"] -= d_shift_vert
    rec_right["z"] -= d_shift_vert
    return ucStop, rec_left, rec_right


def create_test_cases(U, bcells, hul, hur, chalen, gate):
    """
    Populate the initial data for
    the specified test case
    """
    gate_cell = int(bcells / chalen * gate)
    
    for i in range(0, gate_cell):
        U[i, 0] = hul["height"]
        U[i, 1] = hul["velocity"] * hul["height"]
    for i in range(gate_cell, bcells):
        U[i, 0] = hur["height"]
        U[i, 1] = hur["velocity"] * hur["height"]


def calc_time_step(cfl, dx, bcells, U, grav):
    """
    Calculates the maximum wavespeeds and thus the timestep
    via an enforced CFL condition.
    """
    max_speed = -1.0
    for i in range(1, bcells-1):
        h = U[i, 0]
        if h < VERY_SMALL:
            u = 0.0
        else:
            u = U[i, 1] / h
        c = np.sqrt(grav * h)
        max_speed = max(max_speed, abs(u)+c)
    dt = cfl*dx/max_speed  # CFL condition
    return dt


def update_solution(U, fluxes, dt, dx, bcells, grav, direction=2):
    """
    Updates the solution of the equation 
    via the Godunov procedure.
    """
    # Create fluxes
    for i in range(0, bcells-1):
        left = {
            "z": U[i, 0],
            "h": U[i, 0],
            "qx": U[i, 1],
            "qy": 0.0,
            "u": 0.0,
            "v": 0.0,
            "zb": 0.0
        }

        right = {
            "z": U[i+1, 0],
            "h": U[i+1, 0],
            "qx": U[i+1, 1],
            "qy": 0.0,
            "u": 0.0,
            "v": 0.0,
            "zb": 0.0
        }

        rec_left = left
        rec_right = right

        ucStop, rec_left, rec_right = reconstruct_interface(
            left, 0.0, right, 0.0, direction
        )

        hllc_rs = RiemannSolver2DHLLC(
            direction, rec_left, rec_right, grav, VERY_SMALL
        )       
        flux = hllc_rs.flux()
        fluxes[i] = np.array([flux[0], flux[1]])

    # Update solution
    for i in range(1, bcells-1):
        delta = (1.0 / dx) * (fluxes[i-1]-fluxes[i])
        if (
            (delta[0] > 0.0 and delta[0] < VERY_SMALL) or 
            (delta[0] < 0.0 and delta[0] > -VERY_SMALL)
        ):
            delta[0] = 0.0
        if (
            (delta[1] > 0.0 and delta[1] < VERY_SMALL) or 
            (delta[1] < 0.0 and delta[1] > -VERY_SMALL)
        ):
            delta[1] = 0.0
        U[i] = U[i] + dt * delta

    # TODO: Add RK4 here!!!

    # BCs
    U[0] = U[1]
    U[bcells-1] = U[bcells-2]


if __name__ == "__main__":
    chalen = 50.0
    grav = 9.8
    cells = 500
    bcells = cells + 2
    dx = chalen/cells
    cfl = 0.9

    test_cases = [
        {
            "hul": {"height": 1.0, "velocity": 2.5},
            "hur": {"height": 0.1, "velocity": 0.0},
            "gate": 10.0,
            "time_out": 7.0,
        },
        {
            "hul": {"height": 1.0, "velocity": -5.0},
            "hur": {"height": 1.0, "velocity": 5.0},
            "gate": 25.0,
            "time_out": 2.5,
        },
        {
            "hul": {"height": 1.0, "velocity": 0.0},
            "hur": {"height": 0.0, "velocity": 0.0},
            "gate": 20.0,
            "time_out": 4.0,
        },
        {
            "hul": {"height": 0.0, "velocity": 0.0},
            "hur": {"height": 1.0, "velocity": 0.0},
            "gate": 30.0,
            "time_out": 4.0,
        },
        {
            "hul": {"height": 0.1, "velocity": -3.0},
            "hur": {"height": 0.1, "velocity": 3.0},
            "gate": 25.0,
            "time_out": 5.0,
        }
    ]

    for i, test in enumerate(test_cases):
        t = 0.0
        tf = test["time_out"]
        nsteps = 0

        U = np.zeros((bcells,2))
        fluxes = np.zeros((bcells,2))
        create_test_cases(
            U, bcells,
            test["hul"], test["hur"], 
            chalen, test["gate"]
        )
        
        for n in range(1, 100000):
            if (t==tf): break
            dt = calc_time_step(cfl, dx, bcells, U, grav)
            if (t+dt > tf):
                dt = tf - t
            update_solution(U, fluxes, dt, dx, bcells, grav)
            t += dt
            nsteps += 1
            print(t, dt)
        
        path = "/home/mhallsmoore/sites/floodflow/out/"
        out_csv = open(os.path.join(path, "god_%s.csv" % (i+1)), "w")
        out_csv.write("x,h,u\n")
        for i, elem in enumerate(U):
            print("i, elem", i, elem)
            if elem[0] < VERY_SMALL:
                out_csv.write("%s,%s,%s\n" % (i*dx, 0.0, 0.0))
            else:
                out_csv.write("%s,%s,%s\n" % (i*dx, elem[0], elem[1]/elem[0]))
