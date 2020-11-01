from freegs.machine import Machine, Wall
from freegs.coil import Coil
import freegs.picard as picard
import freegs.jtor as jtor
import freegs.equilibrium as equilibrium
import freegs.boundary as boundary
import freegs


class FyTok:
    def __init__(self, *args, nx=None, ny=None, R0=None, **kwargs):
        self._nx = nx or 129
        self._ny = ny or 129
        self._R0 = R0 or 1.0  # R0*B0

    def load_machine(self, entry,  itime=0.0, **kwargs):
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

        # lfcs_r = entry.equilibrium.time_slice[10].boundary.outline.r.__value__()[:, 0]
        # lfcs_z = entry.equilibrium.time_slice[10].boundary.outline.z.__value__()[:, 0]

        rmin = min(Rdim)
        rmax = max(Rdim)
        zmin = min(Zdim)
        zmax = max(Zdim)

        self._eq = equilibrium.Equilibrium(tokamak=Machine(coils, wall),
                                           Rmin=rmin, Rmax=rmax,
                                           Zmin=zmin, Zmax=zmax,
                                           nx=self._nx, ny=self._ny,
                                           boundary=boundary.freeBoundaryHagenow)

    def constrain(self, *args, isoflux=[], **kwargs):
        self._constrain = freegs.control.constrain(*args, isoflux=isoflux, **kwargs)

    def update_eq(self, pprime, ffprime, B0=1.0, **kwargs):
        freegs.solve(self._eq, jtor.ProfilesPprimeFfprime(pprime, ffprime, self._R0*B0), self._constrain)
