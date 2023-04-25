import collections
import math
import sys

import freegs
import freegs.boundary as boundary
import freegs.equilibrium as equilibrium
import freegs.jtor as jtor
import freegs.picard as picard
import matplotlib.pyplot as plt
import numpy as np
from freegs.coil import Coil
from freegs.machine import Machine, Wall
from spdm.data.Collection import Collection
from spdm.utils.logger import logger

sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


def solve_eq(eq, profiles, constrain):
    return {}


def solve_trans(profiles, eq):
    return {}


if __name__ == "__main__":
    db = Collection("east+mdsplus:///home/salmon/public_data/~t/", default_tree_name="efit_east")
    entry = db.open(shot=55555).entry
    COIL = collections.namedtuple("COIL", "label r z current turns")

    vessel_inner_points = np.array([entry.wall.description_2d.vessel.annular.outline_inner.r.__value__(),
                                    entry.wall.description_2d.vessel.annular.outline_inner.z.__value__()]).transpose([1, 0])

    vessel_outer_points = np.array([entry.wall.description_2d.vessel.annular.outline_outer.r.__value__(),
                                    entry.wall.description_2d.vessel.annular.outline_outer.z.__value__()]).transpose([1, 0])

    limiter_points = np.array([entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                               entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__()]).transpose([1, 0])

    itime = 40000
    coils = []
    for coil in entry.pf_active.coil:
        rect = coil.element[0].geometry.rectangle.__value__()
        coils.append((coil.name.__value__(), Coil(
            rect.r+rect.width/2, rect.z+rect.height/2,
            current=coil.current.data.__value__()[itime],
            turns=int(coil.element[0].turns_with_sign)
        )))

    wall = Wall(entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__())

    Rdim = entry.equilibrium.time_slice[10].profiles_2d.grid.dim1.__value__()
    Zdim = entry.equilibrium.time_slice[10].profiles_2d.grid.dim2.__value__()

    lfcs_r = entry.equilibrium.time_slice[10].boundary.outline.r.__value__()[:, 0]
    lfcs_z = entry.equilibrium.time_slice[10].boundary.outline.z.__value__()[:, 0]

    rmin = min(Rdim)
    rmax = max(Rdim)
    zmin = min(Zdim)
    zmax = max(Zdim)

    eq = equilibrium.Equilibrium(tokamak=Machine(coils, wall),
                                 Rmin=rmin, Rmax=rmax,
                                 Zmin=zmin, Zmax=zmax,
                                 nx=129, ny=129,
                                 boundary=boundary.freeBoundaryHagenow)

    # profiles = jtor.ConstrainPaxisIp(1e3,  # Plasma pressure on axis [Pascals]
    #                             1e6,  # Plasma current [Amps]
    #                             1.0)  # fvac = R*Bt

    isoflux = []

    eq_constrain = freegs.control.constrain(isoflux=isoflux)

    profiles = jtor.ProfilesPprimeFfprime(pprime, ffprime, 1.0)

    pprime = None
    ffprime = None

    diff_psi = 1e5

    conv_max_iter = 1000

    time_start = 0.0    # [s]
    time_stop = 1.0    # [s]
    time_step = 0.10    # [s]

    time = time_start

    while time < time_stop:
        logger.debug(f" START TIME LOOP  {time}")

        for conv_iter in range(conv_max_iter):
            logger.debug(f"CONVERGENCE LOOP  {conv_iter}")

            ###################################################
            # EQUILIBRIUM
            logger.debug(f" EQUILIBRIUM ")

            eq_next = solve_eq(eq, profiles, eq_constrain)

            # STRSPORT

            # SOURCE

            # TRANSPORT EQ

            profiles_next = solve_trans(eq_next, profiles)

            # check convergence
            # diff_psi = math.sqrt(sum((eq.psi-psi_prev)**2)/sum(eq.psi**2))
            if check_convergence(profiles, profiles_next):

                break
            elif conv_iter >= conv_max_iter-1:
                raise RuntimeError(f"Too much convergence iter!")
            else:
                profiles = profiles_next
                eq=eq_next

        logger.debug(f" STOP CONVERGENCE LOOP  ")

        time += time_step

    fg = plt.figure()

    plt.gca().add_patch(plt.Polygon(limiter_points, fill=False, closed=True))
    plt.gca().add_patch(plt.Polygon(vessel_outer_points, fill=False, closed=True))
    plt.gca().add_patch(plt.Polygon(vessel_inner_points, fill=False, closed=True))

    plt.gca().add_patch(plt.Polygon(np.array([lfcs_r, lfcs_z]).transpose(
        [1, 0]), fill=False, closed=True, linestyle="dashed", color="red"))

    for coil in entry.pf_active.coil:
        rect = coil.element[0].geometry.rectangle.__value__()
        plt.gca().add_patch(plt.Rectangle((rect.r-rect.width/2.0, rect.z-rect.height/2.0), float(rect.width), float(rect.height), fill=False))

    # plt.contour(
    #     entry.equilibrium.time_slice[10].profiles_2d.grid.dim1.__value__(),
    #     entry.equilibrium.time_slice[10].profiles_2d.grid.dim2.__value__(),
    #     entry.equilibrium.time_slice[10].profiles_2d.psi.__value__(),
    #     linewidths=0.5,cmap="gray")

    plt.contour(np.linspace(rmin, rmax, 129), np.linspace(zmin, zmax, 129),
                eq.psi().transpose(-1, 0), levels=25, linewidths=0.5)

    plt.axis('scaled')
    plt.show()
