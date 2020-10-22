from enum import IntFlag

import fenics
import mshr
import numpy as np


class EqSolver:

    class MeshTag(IntFlag):
        BOX = 0
        LIMTER = 1
        VACUUM = 2
        WALL = 3
        COIL = 10

    def __init__(self, *args, **kwargs):
        pass
        self._geometry = {}
        self._domain = None
        self._mesh = None

    @property
    def mesh(self):
        return self._mesh

    @property
    def geometry(self):
        return self._geometry

    @property
    def domain(self):
        return self._domain

    def load_geometry(self, entry):
        vessel_inner_points = np.array([entry.wall.description_2d.vessel.annular.outline_inner.r.__value__(),
                                        entry.wall.description_2d.vessel.annular.outline_inner.z.__value__()]).transpose([1, 0])
        vessel_inner = mshr.Polygon([fenics.Point((p[0], p[1])) for p in vessel_inner_points])

        vessel_outer_points = np.array([entry.wall.description_2d.vessel.annular.outline_outer.r.__value__(),
                                        entry.wall.description_2d.vessel.annular.outline_outer.z.__value__()]).transpose([1, 0])
        vessel_outer = mshr.Polygon([fenics.Point((p[0], p[1])) for p in vessel_outer_points])

        limiter_points = np.array([entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                                   entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__()]).transpose([1, 0])

        limiter = mshr.Polygon([fenics.Point(p[0], p[1]) for p in limiter_points])

        rpos = []
        zpos = []
        idx = 0
        for coil in entry.pf_active.coil:
            rect = coil.element[0].geometry.rectangle.__value__()
            rpos.append(rect.r-rect.width/2.0)
            rpos.append(rect.r+rect.width/2.0)
            zpos.append(rect.z-rect.height/2.0)
            zpos.append(rect.z+rect.height/2.0)
            self._geometry[EqSolver.MeshTag.COIL+idx] = (mshr.Rectangle(fenics.Point(rect.r-rect.width/2.0, rect.z-rect.height/2.0),
                                                                        fenics.Point(rect.r+rect.width/2.0, rect.z+rect.height/2.0)))
            idx = idx+1

        rmin = min(rpos)
        rmax = max(rpos)
        zmin = min(zpos)
        zmax = max(zpos)

        self._geometry[EqSolver.MeshTag.BOX] = mshr.Rectangle(fenics.Point(rmin-0.1*(rmax-rmin), zmin-0.1*(zmax-zmin)),
                                                              fenics.Point(rmax+0.1*(rmax-rmin), zmax+0.1*(zmax-zmin)))
        self._geometry[EqSolver.MeshTag.LIMTER] = limiter
        self._geometry[EqSolver.MeshTag.VACUUM] = vessel_inner-limiter
        self._geometry[EqSolver.MeshTag.WALL] = vessel_outer-vessel_inner

    def create_mesh(self, resolution=32,  **kwargs):
        self._domain = self._geometry[EqSolver.MeshTag.BOX]
        for tag, geo in self._geometry.items():
            if tag == EqSolver.MeshTag.BOX:
                continue
            self._domain.set_subdomain(tag, geo)

        self._mesh = mshr.generate_mesh(self._domain, resolution=resolution, **kwargs)
