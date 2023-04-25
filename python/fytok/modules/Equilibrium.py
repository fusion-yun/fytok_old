
import numpy as np
from _imas.equilibrium import _T_equilibrium

from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_

# from .PFActive import PFActive
# from .Wall import Wall


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

    def update(self,  *args,
               #    wall: Wall = _undefined_,
               #    pf_active: PFActive = _undefined_,
               #    core_profiles=_undefined_,
               **kwargs):
        super().update(*args, **kwargs)

    def plot(self, axis=None, *args,
             scalar_field={},
             vector_field={},
             boundary=True,
             separatrix=True,
             contours=16,
             oxpoints=True,
             **kwargs):
        """
            plot o-point,x-point,lcfs,separatrix and contour of psi
        """

        import matplotlib.pyplot as plt

        if axis is None:
            axis = plt.gca()

        # R = self.profiles_2d.r
        # Z = self.profiles_2d.z
        # psi = self.profiles_2d.psi(R, Z)
        eq = self.time_slice[-1]

        # axis.contour(R[1:-1, 1:-1], Z[1:-1, 1:-1], psi[1:-1, 1:-1], levels=levels, linewidths=0.2)
        if oxpoints is not False:

            axis.plot(eq.global_quantities.magnetic_axis.r,
                      eq.global_quantities.magnetic_axis.z,
                      'g+',
                      linewidth=0.5,
                      #   markersize=2,
                      label="Magnetic axis")

            # if len(x_point) > 0:
            #     for idx, p in enumerate(x_point):
            #         axis.plot(p.r, p.z, 'rx')
            #         axis.text(p.r, p.z, idx,
            #                   horizontalalignment='center',
            #                   verticalalignment='center')

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

        if boundary and eq.boundary is not _not_found_:

            boundary_points = np.vstack([eq.boundary.outline.r,
                                         eq.boundary.outline.z]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='solid',
                                       linewidth=0.5, fill=False, closed=True))

            axis.plot([], [], 'g-', label="Boundary")

        if separatrix and eq.boundary_separatrix.outline.r is not _not_found_:
            # r0 = self._entry.get("boundary_separatrix.outline.r", None)
            # z0 = self._entry.get("boundary_separatrix.outline.z", None)
            # if r0 is not None and z0 is not None:
            #     axis.add_patch(plt.Polygon(np.vstack([r0, z0]).T, color='b', linestyle=':',
            #                                linewidth=1.0, fill=False, closed=True))

            boundary_points = np.vstack([eq.boundary_separatrix.outline.r,
                                         eq.boundary_separatrix.outline.z]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                       linewidth=0.5, fill=False, closed=False))
            axis.plot([], [], 'r--', label="Separatrix")

        if contours and eq.profiles_2d[-1].grid_type is not _not_found_:
            if contours is True:
                contours = 16
            
            if eq.profiles_2d[-1].grid_type.name == "rectangular":
                dim1 = eq.profiles_2d[-1].grid.dim1
                dim2 = eq.profiles_2d[-1].grid.dim2
                axis.contour(dim1, dim2, eq.profiles_2d[-1].psi, levels=contours)
            elif eq.profiles_2d[-1].r is not _not_found_:
                axis.contour(eq.profiles_2d[-1].r, eq.profiles_2d[-1].z, eq.profiles_2d[-1].psi, levels=contours)

            # if isinstance(contour, int):
            #     c_list = range(0, self.coordinate_system.mesh.shape[0], int(
            #         self.coordinate_system.mesh.shape[0]/contour+0.5))
            # elif isinstance(contour, collections.abc.Sequcence):
            #     c_list = contour
            # for idx in c_list:
            #     ax0 = self.coordinate_system.mesh.axis(idx, axis=0)

            #     if ax0.xy.shape[1] == 1:
            #         axis.add_patch(plt.Circle(ax0.xy[:, 0], radius=0.05, fill=False,color="b", linewidth=0.2))
            #     else:
            #         axis.add_patch(plt.Polygon(ax0.xy, fill=False, closed=True, color="b", linewidth=0.2))

        for s, opts in scalar_field.items():
            if s == "psirz":
                self.coordinate_system._psirz.plot(axis, **opts)
            else:
                sf = getattr(eq.profiles_2d[0], s, None)

                if isinstance(sf, (Field, Function)):
                    sf = sf(eq.profiles_2d[0].r, eq.profiles_2d[0].z)

                if isinstance(sf, np.ndarray):
                    axis.contour(eq.profiles_2d[0].r, eq.profiles_2d[0].z, sf, **opts)
                else:
                    logger.error(f"Can not find field {sf} {type(sf)}!")

        for u, v, opts in vector_field.items():
            uf = eq.profiles_2d[0][u]
            vf = eq.profiles_2d[0][v]
            axis.streamplot(eq.profiles_2d[0].grid.dim1,
                            eq.profiles_2d[0].grid.dim2,
                            vf, uf, **opts)

        return axis
