from __future__ import annotations

import numpy as np
from fytok._imas.lastest.equilibrium import (_T_equilibrium,
                                             _T_equilibrium_time_slice,
                                             _T_equilibrium_profiles_1d,
                                             _T_equilibrium_profiles_2d,
                                             _T_equilibrium_coordinate_system,
                                             _T_equilibrium_global_quantities,
                                             _T_equilibrium_constraints,
                                             _T_equilibrium_boundary,
                                             _T_equilibrium_boundary_separatrix)
from spdm.data.Function import Function
from spdm.data.Node import Node
from spdm.data.sp_property import SpDict, sp_property
from spdm.utils.logger import logger
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Entry import deep_reduce
from spdm.utils.tags import _not_found_
from .CoreProfiles import CoreProfiles

from .PFActive import PFActive
from .Wall import Wall


class EquilibriumTimeSlice(_T_equilibrium_time_slice):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._R0 = self.get("../vacuum_toroidal_field/r0")
        self._B0 = self.get("../vacuum_toroidal_field/b0")(self.time)

    CoordinateSystem = _T_equilibrium_coordinate_system
    Profiles1d = _T_equilibrium_profiles_1d
    Profiles2d = _T_equilibrium_profiles_2d
    GlobalQuantities = _T_equilibrium_global_quantities
    Boundary = _T_equilibrium_boundary
    BoundarySeparatrix = _T_equilibrium_boundary_separatrix
    Constraints = _T_equilibrium_constraints


class Equilibrium(_T_equilibrium):
    r"""
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

        Reference:
            - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.

        COCOS  11
        ```{text}
            Top view
                     ***************
                    *               *
                   *   ***********   *
                  *   *           *   *
                 *   *             *   *
                 *   *             *   *
             Ip  v   *             *   ^  \phi
                 *   *    Z o--->R *   *
                 *   *             *   *
                 *   *             *   *
                 *   *     Bpol    *   *
                  *   *     o     *   *
                   *   ***********   *
                    *               *
                     ***************
                       Bpol x
                    Poloidal view
                ^Z
                |
                |       ************
                |      *            *
                |     *         ^    *
                |     *   \rho /     *
                |     *       /      *
                +-----*------X-------*---->R
                |     *  Ip, \phi   *
                |     *              *
                |      *            *
                |       *****<******
                |       Bpol,\theta
                |
                    Cylindrical coordinate      : $(R,\phi,Z)$
            Poloidal plane coordinate   : $(\rho,\theta,\phi)$
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    TimeSlice = EquilibriumTimeSlice
    time_slice: TimeSeriesAoS[TimeSlice] = sp_property(coordinate1="time", type="dynamic")

    _plugin_registry = {}

    def update(self, *args,
               core_profiles_1d: CoreProfiles.Profiles1d = None,
               pf_active: PFActive = None,
               wall: Wall = None, **kwargs) -> TimeSlice:
        """
            update the last time slice, base on profiles_2d[-1].psi
        """
        logger.debug(f"Update Equlibrium at time={self.time_slice.current.time}")
        super().update()
        #  core_profile_1d=core_profile_1d, pf_active=pf_active, wall=wall,
        return self.time_slice.update(*args, **kwargs)

    def advance(self, *args, time: float = 0.0,
                core_profile_1d: CoreProfiles.Profiles1d = None,
                pf_active: PFActive = None,
                wall: Wall = None, **kwargs) -> Equilibrium.TimeSlice:
        # core_profile_1d=core_profile_1d, pf_active=pf_active, wall=wall,
        super().advance(time=time)
        return self.time_slice.advance(*args, time=time, **kwargs)

    def plot(self, axis=None, *args,
             scalar_field={},
             vector_field={},
             boundary=True,
             separatrix=True,
             contours=16,
             oxpoints=True,
             time_slice=-1,
             **kwargs):
        """
            plot o-point,x-point,lcfs,separatrix and contour of psi
        """

        import matplotlib.pyplot as plt

        if axis is None:
            axis = plt.gca()

        if len(self.time_slice) == 0 or time_slice >= len(self.time_slice):
            logger.error(f"Time slice {time_slice} is out range {len(self.time_slice)} !")
            return axis

        eq = self.time_slice[time_slice]

        if oxpoints is not False:

            axis.plot(eq.global_quantities.magnetic_axis.r,
                      eq.global_quantities.magnetic_axis.z,
                      'g+',
                      linewidth=0.5,
                      #   markersize=2,
                      label="Magnetic axis")

        if eq.boundary_separatrix.type == 1:
            p = eq.boundary_separatrix.geometric_axis
            axis.plot(p.r, p.z, 'rx')
            axis.text(p.r, p.z, 'x',
                      horizontalalignment='center',
                      verticalalignment='center')
            axis.plot([], [], 'rx', label="X-Point")

            for idx, p in eq.boundary_secondary_separatrix.x_point:
                axis.plot(p.r, p.z, 'rx')
                axis.text(p.r, p.z, f'x{idx}',
                          horizontalalignment='center',
                          verticalalignment='center')

            for idx, p in eq.boundary_secondary_separatrix.strike_point:
                axis.plot(p.r, p.z, 'rx')
                axis.text(p.r, p.z, f's{idx}',
                          horizontalalignment='center',
                          verticalalignment='center')

        if boundary and eq.boundary is not None:

            boundary_points = np.vstack([eq.boundary.outline.r.__array__(),
                                         eq.boundary.outline.z.__array__()]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='solid',
                                       linewidth=0.5, fill=False, closed=True))

            axis.plot([], [], 'g-', label="Boundary")

        if separatrix and eq.boundary_separatrix.outline.r is not None:

            separatrix_outline = np.vstack([eq.boundary_separatrix.outline.r.__array__(),
                                            eq.boundary_separatrix.outline.z.__array__()]).T

            axis.add_patch(plt.Polygon(separatrix_outline, color='r', linestyle='dashed',
                                       linewidth=0.5, fill=False, closed=False))
            axis.plot([], [], 'r--', label="Separatrix")

            for idx, p in enumerate(eq.boundary_separatrix.x_point):
                axis.plot(p.r, p.z, 'rx')
                axis.text(p.r, p.z, idx,
                          horizontalalignment='center',
                          verticalalignment='center')

        if contours:
            if contours is True:
                contours = 16
            profiles_2d = eq.profiles_2d[time_slice]

            axis.contour(profiles_2d.r.__array__(),
                         profiles_2d.z.__array__(),
                         profiles_2d.psi.__array__(), linewidths=0.2, levels=contours)

        for s, opts in scalar_field.items():
            if s == "psirz":
                self.coordinate_system._psirz.plot(axis, **opts)
            else:
                sf = getattr(profiles_2d, s, None)

                if isinstance(sf,  Function):
                    sf = sf(profiles_2d.r, profiles_2d.z)

                if isinstance(sf, np.ndarray):
                    axis.contour(profiles_2d.r, profiles_2d.z, sf, linewidths=0.2, **opts)
                else:
                    logger.error(f"Can not find field {sf} {type(sf)}!")

        for u, v, opts in vector_field.items():
            uf = profiles_2d[u]
            vf = profiles_2d[v]
            axis.streamplot(profiles_2d.grid.dim1, profiles_2d.grid.dim2, vf, uf, **opts)

        return axis
