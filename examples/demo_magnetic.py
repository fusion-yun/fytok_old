import sys
import collections
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import math


sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


if __name__ == "__main__":
    from spdm.util.logger import logger
    from spdm.data.Collection import Collection
    from fytok.Tokamak import Tokamak
    from spdm.util.logger import logger
    from spdm.util.Profiles import Profile
    from spdm.data.Entry import open_entry
    from fytok.Plot import plot_profiles
    from spdm.util.AttributeTree import _next_
    import freegs

    tok = Tokamak(open_entry("east+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east", shot=55555, time_slice=20))

    control_line = [

        ((1.99, 0),    (2.40, 0)),     # isoflux01
        ((1.93, 0.15),    (2.24, 0.52)),   # isoflux03
        ((1.72, 0.24),    (1.35, 0.45)),   # isoflux04
        ((1.66, 0),    (1.35, 0)),     # isoflux06
        ((1.72, -0.24),   (1.35, -0.45)),    # isoflux08
        ((1.93, -0.15),    (2.24, -0.52)),   # isoflux09
    ]

    r = 0.2
    pts = [((1.0-r)*b[0] + r*e[0], (1.0-r)*b[1] + r*e[1]) for b, e in control_line]

    isoflux = [
        (*pts[0], *pts[3]),
        (*pts[1], *pts[4]),
        (*pts[2], *pts[5]),

    ]

    for r0, z0, r1, z1 in isoflux:
        plt.plot([r0, r1], [z0, z1])

    # xpoints = [(1.63, -0.83), (1.63, 0.83)]

    # lfcs_r = tok.equilibrium.boundary.outline.r
    # lfcs_z = tok.equilibrium.boundary.outline.z

    # psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    # profiles = freegs.jtor.ConstrainPaxisIp(1e3,  # Plasma pressure on axis [Pascals]
    #                                         1e6,  # Plasma current [Amps]
    #                                         1.0)  # fvac = R*Bt

    # constrain = freegs.control.constrain( isoflux=isoflux,
    #     # psivals=psivals, xpoints=xpoints
    #     )

    # eq_wall = freegs.machine.Wall(tok.wall.limiter.outline.r,
    #                                     tok.wall.limiter.outline.z)

    # eq_coils = []

    # for coil in tok.pf_active.coil:
    #     t_coil = freegs.machine.Coil(
    #         coil.r+coil.width/2,
    #         coil.z+coil.height/2,
    #         turns=coil.turns)
    #     eq_coils.append((coil.name, t_coil))

    # machine = freegs.machine.Machine(eq_coils, wall=eq_wall)

    # eq = freegs.Equilibrium(
    #     tokamak=machine,
    #     Rmin=min(tok.equilibrium.profiles_2d.grid.dim1), Rmax=max(tok.equilibrium.profiles_2d.grid.dim1),
    #     Zmin=min(tok.equilibrium.profiles_2d.grid.dim2), Zmax=max(tok.equilibrium.profiles_2d.grid.dim2),
    #     nx=len(tok.equilibrium.profiles_2d.grid.dim1), ny=len(tok.equilibrium.profiles_2d.grid.dim2),
    #     # psi=psi,
    #     current=tok.equilibrium.profiles_2d.global_quantities.ip,
    #     boundary=freegs.boundary.freeBoundaryHagenow)

    # logger.debug(eq)

    # freegs.solve(eq, profiles, constrain, show=True)
    R = tok.equilibrium.profiles_2d.r
    Z = tok.equilibrium.profiles_2d.z

    tok.equilibrium.update(
        constraints={
            # "psivals": psivals,
            # "xpoints": xpoints,
            "isoflux": isoflux
        })

    tok.wall.plot()
    tok.pf_active.plot()

    # psi = self.profiles_2d.psi
    psi = tok.equilibrium.profiles_2d.psi

    
    # psi = (psi - self.global_quantities.psi_axis) / \
    #     (self.global_quantities.psi_boundary - self.global_quantities.psi_axis)

    # if type(levels) is int:
    #     levels = np.linspace(-2, 2,  levels)

    plt.contour(R[1:-1, 1:-1], Z[1:-1, 1:-1], psi[1:-1, 1:-1], levels=20,  linewidths=0.2)
    # plt.plot(lfcs_r, lfcs_z)
    # tok.equilibrium.plot(oxpoints=False, boundary=False)
    plt.gca().set_aspect('equal')

    plt.savefig("../output/magnetic.svg")
