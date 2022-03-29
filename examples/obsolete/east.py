import sys

sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")
sys.path.append("/home/salmon/workspace/freegs/")

import collections
import math

import freegs
import freegs.boundary as boundary
import freegs.equilibrium as equilibrium
import freegs.jtor as jtor
import freegs.picard as picard
import matplotlib.pyplot as plt
import numpy as np
from freegs.coil import Coil
from freegs.machine import Machine, Wall
from scipy import special
from spdm.data.Collection import Collection
from spdm.logger import logger

if __name__ == "__main__":

    db = Collection("east+mdsplus:///home/salmon/public_data/~t/",default_tree_name="efit_east")
    entry = db.open(shot=55555).entry

    vessel_inner_points= np.array([entry.wall.description_2d.vessel.annular.outline_inner.r.__value__(),    
                                            entry.wall.description_2d.vessel.annular.outline_inner.z.__value__()]).transpose([1,0]) 

    vessel_outer_points= np.array([entry.wall.description_2d.vessel.annular.outline_outer.r.__value__(),    
                                            entry.wall.description_2d.vessel.annular.outline_outer.z.__value__()]).transpose([1,0])  

    limiter_points =  np.array([entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                                    entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__()]).transpose([1,0]) 


    itime=40000
    coils = []
    for coil in entry.pf_active.coil:
        rect = coil.element[0].geometry.rectangle.__value__()
        coils.append((coil.name.__value__(), Coil(
        rect.r+rect.width/2, rect.z+rect.height/2,
        current= coil.current.data.__value__()[itime] ,
        turns=int(coil.element[0].turns_with_sign)
        )))

    wall = Wall(entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__())

    Rdim=entry.equilibrium.time_slice[10].profiles_2d.grid.dim1.__value__()
    Zdim=entry.equilibrium.time_slice[10].profiles_2d.grid.dim2.__value__()

    lfcs_r=entry.equilibrium.time_slice[10].boundary.outline.r.__value__()[:,0]
    lfcs_z=entry.equilibrium.time_slice[10].boundary.outline.z.__value__()[:,0]

    EASTTokamak = Machine(coils, wall)

    profiles = jtor.ConstrainPaxisIp(1e3,  # Plasma pressure on axis [Pascals]
                                1e6,  # Plasma current [Amps]
                                1.0)  # fvac = R*Bt

    psivals = [ (R, Z, 0.0) for R, Z in zip(lfcs_r,lfcs_z)]
    # psivals = [ (R, Z, 0.0) for R, Z in zip(entry.equilibrium.time_slice[10].boundary.outline.r.__value__(), 
    #             entry.equilibrium.time_slice[10].boundary.outline.z.__value__()) ]

    constrain = freegs.control.constrain(psivals=psivals)

    rmin=min(Rdim)
    rmax=max(Rdim)
    zmin=min(Zdim)
    zmax=max(Zdim)

    eq = equilibrium.Equilibrium(tokamak=EASTTokamak,
                                    Rmin=rmin, Rmax=rmax,
                                    Zmin=zmin, Zmax=zmax,
                                    nx=129, ny=129,
                                    boundary=boundary.freeBoundaryHagenow)

    freegs.solve(eq, profiles,constrain, psi_bndry=0.0,show=True)                                  
