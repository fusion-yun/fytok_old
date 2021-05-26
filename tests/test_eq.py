import math
from scipy import special
from spdm.util.numlib import np
import matplotlib.pyplot as plt
import pprint
import collections
import sys

sys.path.append("/home/salmon/workspace/fy_trans/python")
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")
sys.path.append("/home/salmon/workspace/freegs/")

from spdm.data import Collection
from spdm.util.logger import logger

from freegs.machine import Machine,Wall
import freegs.equilibrium as equilibrium
import freegs.jtor as jtor
import freegs.picard as picard
import freegs.boundary as boundary
from  freegs.coil import Coil



if __name__ == '__main__':

    db = Collection("EAST:///home/salmon/public_data/~t/", default_tree_name="efit_east")
    entry = db.open(shot=55555).entry
    itime=40000
    coils = []
    for coil in entry.pf_active.coil:
        rect = coil.element[0].geometry.rectangle.__real_value__()
        Ip = coil.current.data.__value__()[itime] * (coil.element[0].turns_with_sign*1)
        coils.append((coil.name.__value__(), Coil(rect["r"], rect["z"], Ip)))

    wall = Wall(entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__())

    Rdim=entry.equilibrium.time_slice[10].profiles_2d.grid.dim1.__value__()
    Zdim=entry.equilibrium.time_slice[10].profiles_2d.grid.dim2.__value__()


    EASTTokamak = Machine(coils, wall)

    profiles = jtor.ConstrainPaxisIp(1e3,  # Plasma pressure on axis [Pascals]
                                1e5,  # Plasma current [Amps]
                                1.0)  # fvac = R*Bt

    eq = equilibrium.Equilibrium(tokamak=EASTTokamak,
                                 Rmin=min(Rdim), Rmax=max(Rdim),
                                 Zmin=min(Zdim), Zmax=max(Zdim),
                                 nx=len(Rdim), ny=len(Zdim),
                                 boundary=boundary.fixedBoundary)

    pprint.pprint(Rdim.shape)
    plt.contour(Rdim,   Zdim,    eq.psi,    linewidths=0.5)
    plt.savefig("/home/salmon/workspace/output/test_eq.png")
