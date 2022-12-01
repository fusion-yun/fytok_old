import os
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.logger import logger
from spdm.data.File import File
from spdm.data.Mapping import Mapping
from fytok.transport.Equilibrium import Equilibrium
from fytok.device.PFActive import PFActive
from fytok.device.Wall import Wall
from fytok.device.Magnetics import Magnetics
from spdm import open_entry

if __name__ == '__main__':

    # entry = Mapping("/home/salmon/workspace/fytok_data/mapping")\
    #     .map(File("/home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300",
    #               format="mdsplus").read(), source_schema="EAST")
    entry = open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east#38300",
                       mapping_path="/home/salmon/workspace/fytok_data/mapping")

    magnetics = Magnetics(entry.get(["magnetics"]))
    pf_active = PFActive(entry.get(["pf_active"]))

    print(magnetics.bpol_probe[2].field(1.2))

    fig = plt.figure()
    axis = fig.gca()

    time_slice = 50

    desc = entry.get(["equilibrium", "time_slice", time_slice]).dump()

    desc["time"] = 1.2345

    desc["vacuum_toroidal_field"] = {
        "b0": entry.get(["equilibrium", "vacuum_toroidal_field", "b0"])[time_slice],
        "r0": entry.get(["equilibrium", "vacuum_toroidal_field", "r0"])[time_slice],
    }

    eq = Equilibrium(desc)
    psi_norm = np.linspace(0.01, 0.995, 16)
    logger.debug(eq.time)
    logger.debug(eq.global_quantities.ip)
    logger.debug(eq.profiles_1d.dvolume_dpsi(psi_norm))


    eq.plot(axis, contour=np.linspace(0, 5, 50))

    pf_active = PFActive(entry.get(["pf_active"]))

    for coil in pf_active.coil:
        logger.debug(coil.element[0].geometry.rectangle)
        logger.debug(coil.current.data[100])

    pf_active.plot(axis)

    wall = Wall(entry.get(["wall"]))

    wall.plot(axis)

    axis.set_aspect('equal')
    axis.axis('scaled')
    axis.set_xlabel(r"Major radius $R$ [m]")
    axis.set_ylabel(r"Height $Z$ [m]")

    fig.savefig("/home/salmon/workspace/output/tokamak.png", transparent=True)
