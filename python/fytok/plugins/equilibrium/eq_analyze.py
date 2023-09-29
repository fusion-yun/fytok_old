import collections
import collections.abc
import functools
import typing
from dataclasses import dataclass

import numpy as np
import scipy.constants
from fytok._imas.lastest.equilibrium import \
    _T_equilibrium_global_quantities_magnetic_axis
from fytok._imas.lastest.utilities import _T_identifier_dynamic_aos3
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.logger import logger
from fytok.utils.utilities import CurveRZ, RZTuple, RZTuple_
from spdm.data.AoS import AoS
from spdm.data.Expression import Expression, Variable
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.HTree import List
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject, GeoObjectSet
from spdm.geometry.Point import Point
from spdm.mesh.Mesh import Mesh
from spdm.mesh.mesh_curvilinear import CurvilinearMesh
from spdm.numlib.contours import find_countours
from spdm.numlib.optimize import minimize_filter
from spdm.utils.constants import *
from spdm.utils.tags import _not_found_
from spdm.utils.tree_utils import merge_tree_recursive
from spdm.utils.typing import (ArrayLike, ArrayType, NumericType, array_type,
                               scalar_type)

_R = Variable(0, "R")
_Z = Variable(1, "Z")


@dataclass
class OXPoint:
    r: float
    z: float
    psi: float


TOLERANCE = 1.0e-6


# fmt:off
COCOS_TABLE = [
    # e_Bp ,    $\sigma_{Bp}$,    $\sigma_{R\varphi\Z}$,  $\sigma_{\rho\theta\varphi}$
    None,                                                                                    # 0
    (0,         +1,                  +1,                       +1                   ),       # 1
    (0,         +1,                  -1,                       +1                   ),       # 2
    (0,         -1,                  +1,                       -1                   ),       # 3
    (0,         -1,                  -1,                       -1                   ),       # 4
    (0,         +1,                  +1,                       -1                   ),       # 5
    (0,         +1,                  -1,                       -1                   ),       # 6
    (0,         -1,                  +1,                       +1                   ),       # 7
    (0,         -1,                  -1,                       +1                   ),       # 8
    None,                                                                                    # 9
    None,                                                                                    # 10
    (1,         +1,                  +1,                       +1                   ),       # 11
    (1,         +1,                  -1,                       +1                   ),       # 12
    (1,         -1,                  +1,                       -1                   ),       # 13
    (1,         -1,                  -1,                       -1                   ),       # 14
    (1,         +1,                  +1,                       -1                   ),       # 15
    (1,         +1,                  -1,                       -1                   ),       # 16
    (1,         -1,                  +1,                       +1                   ),       # 17
    (1,         -1,                  -1,                       +1                   ),       # 18
]
# fmt:on


class FyEquilibriumCoordinateSystem(Equilibrium.TimeSlice.CoordinateSystem):
    r"""
        Flux surface coordinate system on a square grid of flux and poloidal angle
        默认采用磁面坐标

        $$
            V^{\prime}\left(\rho\right)=\frac{\partial V}{\partial\rho}=2\pi\int_{0}^{2\pi}\sqrt{g}d\theta=2\pi\oint\frac{R}{\left|\nabla\rho\right|}dl
        $$

        $$
            \left\langle\alpha\right\rangle\equiv\frac{2\pi}{V^{\prime}}\int_{0}^{2\pi}\alpha\sqrt{g}d\theta=\frac{2\pi}{V^{\prime}}\varoint\alpha\frac{R}{\left|\nabla\rho\right|}dl
        $$

        Magnetic Flux Coordinates
        psi         :                     ,  flux function , $B \cdot \nabla \psi=0$ need not to be the poloidal flux funcion $\Psi$
        theta       : 0 <= theta   < 2*pi ,  poloidal angle
        phi         : 0 <= phi     < 2*pi ,  toroidal angle
    """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        # logger.debug(f"Create MagneticCoordSystem.")

        self._B0 = self._parent._B0  # super().get("b0", )   # magnetic field on magnetic axis
        self._R0 = self._parent._R0  # super().get("r0", , type_hint=float)   # major radius of magnetic axis
        self._Ip = self._parent.global_quantities.ip  # super().get("ip", , type_hint=float)  # plasma current

        self._fpol = self._parent.profiles_1d.f  # poloidal current function

        self._s_B0 = np.sign(self._B0)

        self._s_Ip = np.sign(self._Ip)

        self._e_Bp,  self._s_Bp, self._s_RpZ, self._s_rtp = COCOS_TABLE[self.cocos]

        self._s_eBp_2PI = 1.0 if self._e_Bp == 0 else TWOPI

        # logger.debug(f"COCOS={self.cocos}")

    @sp_property
    def cocos(self) -> int:
        cocos_flag = super().get("cocos", _not_found_, type_hint=int)

        if cocos_flag is not _not_found_:
            return cocos_flag
        return 5

    @functools.cached_property
    def _psirz(self) -> Field:
        psirz = super().get("psirz", _not_found_, force=True, type_hint=np.ndarray)

        if isinstance(psirz, np.ndarray):
            dim1 = super().get("grid/dim1", _not_found_)
            dim2 = super().get("grid/dim2", _not_found_)
            grid_type = super().get("grid/type", "rectangular")

            if not isinstance(dim1, np.ndarray) or not isinstance(dim2, np.ndarray):
                raise RuntimeError(f"Can not create grid!")

            psirz = Field(psirz, dim1, dim2, mesh_type=grid_type, name="psirz")
        elif psirz is _not_found_:
            psirz = self._parent.profiles_2d[0].psi
        else:
            logger.warning(f"Ignore {type(psirz)}. Using ../profiles_2d[0].psi ")

        if not isinstance(psirz, Field):
            raise RuntimeError(f"Can not get psirz! {type(psirz)}")

        return psirz

    @functools.cached_property
    def critical_points(self) -> typing.Tuple[typing.Sequence[OXPoint], typing.Sequence[OXPoint]]:

        opoints = []

        xpoints = []

        psi = self._psirz

        R, Z = psi.mesh.points

        Bp2 = (psi.pd(0, 1)**2 + psi.pd(1, 0)**2)/(_R**2)

        D = psi.pd(2, 0) * psi.pd(0, 2) - psi.pd(1, 1)**2

        for r, x_z in minimize_filter(Bp2, R, Z):

            p = OXPoint(r, x_z, psi(r, x_z))

            if D(r, x_z) < 0.0:  # saddle/X-point
                xpoints.append(p)
            else:  # extremum/ O-point
                opoints.append(p)

        Rmid, Zmid = self._psirz.mesh.geometry.bbox.origin + \
            self._psirz.mesh.geometry.bbox.dimensions*0.5

        opoints.sort(key=lambda x: (x.r - Rmid)**2 + (x.z - Zmid)**2)

        # TODO:

        o_psi = opoints[0].psi
        o_r = opoints[0].r
        o_z = opoints[0].z

        # remove illegal x-points . learn from freegs
        # check psi should be monotonic from o-point to x-point

        x_points = []
        s_points = []
        for xp in xpoints:
            length = 20

            psiline = psi(np.linspace(o_r, xp.r, length),
                          np.linspace(o_z, xp.z, length))

            if len(np.unique(psiline[1:] > psiline[:-1])) != 1:
                s_points.append(xp)
            else:
                x_points.append(xp)

        xpoints = x_points

        xpoints.sort(key=lambda x: (x.psi - o_psi)**2)

        return opoints, xpoints

    @sp_property
    def grid_type(self) -> _T_identifier_dynamic_aos3:
        desc = super().grid_type
        if desc.name is None or desc.name is _not_found_:
            desc = {"name": "rectangular", "index": 1, "description": "default"}
        return desc

    @sp_property[Mesh]
    def grid(self) -> Mesh:
        psi_norm = super().grid.dim1

        if psi_norm is _not_found_:
            psi_norm = self._parent.code.parameters.get("psi_norm", 128)

        if isinstance(psi_norm, np.ndarray) and psi_norm.ndim == 0:
            psi_norm_boundary = self._parent.boundary.psi_norm
            psi_norm = np.linspace(1.0-psi_norm_boundary, psi_norm_boundary, int(psi_norm), endpoint=True)
            logger.debug(f"Set psi_norm ({psi_norm[0]},{psi_norm[-1]}) {len(psi_norm)}")

        if isinstance(psi_norm, int):
            psi_norm = np.linspace(0.0, 0.995, psi_norm)  # 0.995 lcfs

        elif not isinstance(psi_norm, np.ndarray):
            raise ValueError(f"Can not create grid! psi_norm={psi_norm}")

        elif np.isclose(psi_norm[0], 0.0) and np.isclose(psi_norm[-1], 1.0):
            logger.warning(
                f"Singular values are caused when psi_norm takes values of 0.0 or 1.0.! {psi_norm[0]} {psi_norm[-1]}")

        theta = super().grid.dim2

        if isinstance(theta, int):
            theta = np.linspace(0, TWOPI, theta, endpoint=False)
        elif isinstance(theta, np.ndarray) and theta.ndim == 0:
            theta = np.linspace(0, TWOPI, int(theta), endpoint=False)

        if not (isinstance(theta, np.ndarray) and theta.ndim == 1):
            raise ValueError(f"Can not create grid! theta={theta}")

        surfs = GeoObjectSet([surf for _, surf in self.find_surfaces_by_psi_norm(psi_norm, o_point=True)])

        return CurvilinearMesh(psi_norm, theta, geometry=surfs, cycles=[False, TWOPI])

    @property
    def psi_norm(self) -> ArrayType: return self.grid.dim1

    @property
    def psi(self) -> ArrayType:
        return self.psi_norm * (self.psi_boundary-self.psi_magnetic_axis) + self.psi_magnetic_axis

    def psirz(self, r: NumericType, z: NumericType) -> NumericType:
        return self._psirz(r, z)

    @functools.cached_property
    def magnetic_axis(self) -> typing.Tuple[float, float]:
        o_points = self.critical_points[0]
        return o_points[0].r, o_points[0].z

    @property
    def psi_magnetic_axis(self) -> float: return self.critical_points[0][0].psi

    @property
    def psi_boundary(self) -> float: return self.critical_points[1][0].psi

    @sp_property
    def r(self) -> Field: return Field(self.grid.points[0], grid=self.grid)

    @sp_property
    def z(self) -> Field: return Field(self.grid.points[1], grid=self.grid)

    @sp_property
    def jacobian(self) -> Field:
        raise NotImplementedError(f"")

    @sp_property
    def tensor_covariant(self) -> Field:
        raise NotImplementedError(f"")

    @sp_property
    def tensor_contravariant(self) -> Field:
        raise NotImplementedError(f"")

    def find_surfaces(self, psi:  float | ArrayType | typing.Sequence[float], o_point: OXPoint = True) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:
        """
            if o_point is not None:
                only return  closed surface  enclosed o-point
                if closed surface does not exists, return None
                number of surface == len(psi)
            else:
                do not guarantee the number of surface == len(psi)
                return all surface ,
        """

        if o_point is None or o_point is False:
            for psi_val, surfs in find_countours(self._psirz, values=psi):
                for surf in surfs:
                    if isinstance(surf, GeoObject):
                        surf.set_coordinates("r", "z")
                    yield psi_val, surf

        else:
            # x_point = None
            if o_point is True:
                opts, xpts = self.critical_points
                if len(opts) == 0:
                    raise RuntimeError(f"O-point is not defined!")
                o_point = opts[0]
                if len(xpts) > 0:
                    x_point = xpts[0]

            if isinstance(psi, float):
                psi = [psi]

            current_psi = np.nan
            current_count = 0
            for psi_val, surfs in find_countours(self._psirz, values=psi):
                # 累计相同 level 的 surface个数
                # 如果累计的 surface 个数大于1，说明存在磁岛
                # 如果累计的 surface 个数等于0，说明该 level 对应的 surface 不存在
                # 如果累计的 surface 个数等于1，说明该 level 对应的 surface 存在且唯一

                count = 0
                for surf in surfs:
                    count += 1
                    if surf is None and np.isclose(psi_val, o_point.psi):
                        yield psi_val, Point(o_point.r, o_point.z)
                    elif isinstance(surf, Point) and all(np.isclose(surf.points, [o_point.r, o_point.z])):
                        yield psi_val, surf  # raise RuntimeError(f"Can not find surface psi={level}")
                    elif isinstance(surf, Curve) and surf.is_closed and surf.enclose(o_point.r, o_point.z):
                        # theta_0 = np.arctan2(x_point.r-o_point.r, x_point.z-o_point.z)
                        # theta = ((np.arctan2(_R-o_point.r, _Z-o_point.z)-theta_0)+TWOPI) % TWOPI
                        # surf = surf.remesh(theta)
                        surf.set_coordinates("r", "z")
                        yield psi_val, surf
                    else:
                        count -= 1
                if count <= 0:
                    if np.isclose(psi_val, o_point.psi):
                        yield psi_val, Point(o_point.r, o_point.z)
                    else:
                        logger.warning(f"{psi_val} {o_point.psi}")
                        yield psi_val, None
                elif current_count > 1:
                    raise RuntimeError(f"Something wrong! Get {current_count} closed surfaces for psi={current_psi}")

                # theta = np.arctan2(surf[:, 0]-o_point.r, surf[:, 1]-o_point.z)
                # logger.debug((max(theta)-min(theta))/TWOPI)
                # if 1.0 - (max(theta)-min(theta))/TWOPI > 2.0/len(theta):  # open or do not contain o-point
                #     current_count -= 1
                #     continue

                # is_closed = False

                # if np.isclose((theta[0]-theta[-1]) % TWOPI, 0.0):
                #     # 封闭曲线
                #     theta = theta[:-1]
                #     surf = surf[:-1]
                #     # is_closed = True
                # else:  # boundary separatrix
                #     if x_point is None:
                #         raise RuntimeError(f"No X-point ")
                #     # logger.warning(f"The magnetic surface average is not well defined on the separatrix!")
                #     xpt = np.asarray([x_point.r, x_point.z], dtype=float)
                #     b = surf[1:]
                #     a = surf[:-1]
                #     d = b-a
                #     d2 = d[:, 0]**2+d[:, 1]**2
                #     p = xpt-a

                #     c = (p[:, 0]*d[:, 0]+p[:, 1]*d[:, 1])/d2
                #     s = (p[:, 0]*d[:, 1]-p[:, 1]*d[:, 0])/d2
                #     idx = np.flatnonzero(np.logical_and(c >= 0, c**2+s**2 < 1))

                #     if len(idx) == 2:

                #         idx0 = idx[0]
                #         idx1 = idx[1]

                #         theta_x = np.arctan2(xpt[0]-o_point.r, xpt[1]-o_point.z)

                #         surf = np.vstack([[xpt], surf[idx0:idx1]])
                #         theta = np.hstack([theta_x, theta[idx0:idx1]])
                #     else:
                #         raise RuntimeError(f"Can not get closed boundary {o_point}, {x_point} {idx} !")

                # # theta must be strictly increased
                # p_min = np.argmin(theta)
                # p_max = np.argmax(theta)

                # if p_min > 0:
                #     if p_min == p_max+1:
                #         theta = np.roll(theta, -p_min)
                #         surf = np.roll(surf, -p_min, axis=0)
                #     elif p_min == p_max-1:
                #         theta = np.flip(np.roll(theta, -p_min-1))
                #         surf = np.flip(np.roll(surf, -p_min-1, axis=0), axis=0)
                #     else:
                #         raise ValueError(f"Can not convert 'u' to be strictly increased!")
                #     theta = np.hstack([theta, [theta[0]+TWOPI]])
                #     theta = (theta-theta.min())/(theta.max()-theta.min())
                #     surf = np.vstack([surf, surf[:1]])

                # if surf.shape[0] == 0:
                #     logger.warning(f"{level},{o_point.psi},{(max(theta),min(theta))}")

                # elif surf.shape[0] == 1:
                #     yield level, Point(surf[0][0], surf[0][1])
                # else:
                #     yield level, Curve(surf, theta, is_closed=is_closed)

    def find_surfaces_by_psi_norm(self, psi_norm: float | ArrayType | typing.Sequence[float], *args,   **kwargs) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:

        psi_magnetic_axis = self.psi_magnetic_axis

        psi_boundary = self.psi_boundary

        if isinstance(psi_norm, (collections.abc.Sequence, np.ndarray)):
            psi = np.asarray(psi_norm, dtype=float)*(psi_boundary-psi_magnetic_axis)+psi_magnetic_axis
            yield from self.find_surfaces(psi, *args,  **kwargs)
        elif isinstance(psi_norm, collections.abc.Generator):
            for psi_n in psi_norm:
                yield from self.find_surfaces(psi_n*(psi_boundary-psi_magnetic_axis)+psi_magnetic_axis, *args,  **kwargs)

    @dataclass
    class ShapeProperty:
        psi: float | np.ndarray
        Rmin: float | np.ndarray
        Zmin: float | np.ndarray
        Rmax: float | np.ndarray
        Zmax: float | np.ndarray
        Rzmin: float | np.ndarray
        Rzmax: float | np.ndarray
        r_inboard: float | np.ndarray
        r_outboard: float | np.ndarray

    def shape_property(self, psi: typing.Union[float, typing.Sequence[float]] = None) -> ShapeProperty:
        def shape_box(s: GeoObject):
            if isinstance(s, Point):
                r, z = s.points
                rmin = r
                rmax = r
                zmin = z
                zmax = z
                r_inboard = r
                r_outboard = r
                rzmin = r
                rzmax = r
            elif isinstance(s, GeoObject):
                r, z = s.points
                (rmin, zmin) = s.bbox.origin
                (rmax, zmax) = s.bbox.origin + s.bbox.dimensions
                rzmin = r[np.argmin(z)]
                rzmax = r[np.argmax(z)]
                r_inboard = s.coordinates(0.5)[0]
                r_outboard = s.coordinates(0)[0]
            else:
                raise TypeError(f"Invalid type {type(s)}")
            return rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard

        if psi is None:
            psi = self.psi
        elif not isinstance(psi, (np.ndarray, collections.abc.MutableSequence)):
            psi = [psi]

        sbox = np.asarray([[p, *shape_box(s)] for p, s in self.find_surfaces(psi, o_point=True)], dtype=float)

        if sbox.shape[0] == 1:
            psi, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox[0]
        else:
            psi, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox.T
        if np.isscalar(psi):
            return FyEquilibriumCoordinateSystem.ShapeProperty(
                psi, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard)
        else:
            return FyEquilibriumCoordinateSystem.ShapeProperty(
                psi,
                Function(rmin,      psi,   name="rmin"),
                Function(zmin,      psi,   name="zmin"),
                Function(rmax,      psi,   name="rmax"),
                Function(zmax,      psi,   name="zmax"),
                Function(rzmin,     psi,   name="rzmin"),
                Function(rzmax,     psi,   name="rzmax"),
                Function(r_inboard, psi,   name="r_inboard"),
                Function(r_outboard, psi,  name="r_outboard"),
            )

    #################################
    # fields
    @property
    def Bpol2(self) -> Expression: return self.b_field_r**2+self.b_field_z**2
    r""" $B_{pol}= \left|\nabla \psi \right|/2 \pi R $ """

    @property
    def Bpol(self) -> Expression: return np.sqrt(self.b_field_r**2+self.b_field_z**2)
    r""" $B_{pol}= \left|\nabla \psi \right|/2 \pi R $ """

    @property
    def b_field_r(self) -> Expression:
        """ COCOS Eq.19 [O. Sauter and S.Yu. Medvedev, Computer Physics Communications 184 (2013) 293] """
        return self._psirz.pd(0, 1) / _R * (self._s_RpZ * self._s_Bp / self._s_eBp_2PI)

    @property
    def b_field_z(self) -> Expression:
        return -self._psirz.pd(1, 0) / _R * (self._s_RpZ * self._s_Bp / self._s_eBp_2PI)

    @property
    def b_field_tor(self) -> Expression: return self._fpol(self._psirz) / _R

    @property
    def B2(self) -> Expression: return (self.b_field_r**2 + self.b_field_z**2 + self.b_field_tor ** 2)

    @property
    def grad_psi2(self) -> Expression: return self._psirz.pd(1, 0)**2+self._psirz.pd(0, 1)**2

    @property
    def grad_psi(self) -> Expression: return np.sqrt(self.grad_psi2)

    @property
    def ddpsi(self) -> Expression:
        return np.sqrt(self._psirz.pd(2, 0) * self._psirz.pd(0, 2) + self._psirz.pd(1, 1)**2)

    @functools.cached_property
    def dvolume_dpsi(self) -> Function:
        return Function(*self._surface_integral(1.0), name="dvolume_dpsi")

    ###############################
    # surface integral

    def _surface_integral(self, func: Expression, psi: NumericType = None) -> typing.Tuple[ArrayLike, ArrayLike]:
        r"""
            $ V^{\prime} =  2 \pi  \int{ R / \left|\nabla \psi \right| * dl }$
            $ V^{\prime}(psi)= 2 \pi  \int{ dl * R / \left|\nabla \psi \right|}$
        """

        # r0, z0 = self.magnetic_axis

        if psi is None or psi is self.psi:
            if np.isclose(self.psi[0], self.psi_magnetic_axis):
                surfs_list = zip(self.psi[1:], self.grid.geometry[1:])
            else:
                surfs_list = zip(self.psi, self.grid.geometry)
        else:
            if isinstance(psi, scalar_type):
                psi = [psi]
            elif np.isclose(psi[0], self.psi_magnetic_axis):
                psi = psi[1:]
            surfs_list = self.find_surfaces(psi, o_point=True)

        psi = []
        res = []
        for p, surf in surfs_list:
            if isinstance(surf, Curve):
                v = surf.integral(func/self.Bpol)
            elif isinstance(surf, Point):  # o-point
                # R, Z = surf.points
                # v = (func(R, Z) if callable(func) else func) * self.ddpsi(R, Z)
                # logger.debug((R, Z, v))
                # # v *= r**2/self.ddpsi(r, z)
                # logger.warning(f"Found point pos={surf.points}  psi={p}")
                v = np.nan
            else:
                continue
                logger.warning(f"Found an island at psi={p} pos={surf}")
                v = np.nan
            res.append(v)
            psi.append(p)

        if len(psi) > 1:
            res = np.asarray(res, dtype=float)
            psi = np.asarray(psi, dtype=float)
            return res, psi
        else:
            return res[0], psi[0]

        # return np.asarray([(axis.integral(func/self.Bpol) if not np.isclose(p, 0) else func(r0, z0) * c0) for p, axis in surfs_list], dtype=float)

    def surface_integral(self, func: Expression, psi: NumericType = None) -> Function | float:

        value, psi = self._surface_integral(func, psi)

        if np.isscalar(psi):
            return value
        else:
            return Function(value, psi, name=f"surface_integral({str(func)})")

    def surface_average(self, func: Expression, *xargs) -> Expression | ArrayLike:
        r"""
            $\left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}$
        """
        return self.surface_integral(func, *xargs)/self.dvolume_dpsi(*xargs)


class FyEquilibriumGlobalQuantities(Equilibrium.TimeSlice.GlobalQuantities):
    @property
    def _coord(self) -> FyEquilibriumCoordinateSystem: return self._parent.coordinate_system

    # beta_pol  :float =  sp_property(type="dynamic",units="-")

    # beta_tor  :float =  sp_property(type="dynamic",units="-")

    # beta_normal  :float =  sp_property(type="dynamic",units="-")

    # ip  :float =  sp_property(type="dynamic",units="A",cocos_label_transformation="ip_like",cocos_transformation_expression=".sigma_ip_eff",cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.ip")

    # li_3  :float =  sp_property(type="dynamic",units="-")

    # volume  :float =  sp_property(type="dynamic",units="m^3")

    # area  :float =  sp_property(type="dynamic",units="m^2")

    # surface  :float =  sp_property(type="dynamic",units="m^2")

    # length_pol  :float =  sp_property(type="dynamic",units="m")

    @sp_property
    def psi_axis(self) -> float: return self._coord.psi_magnetic_axis  # sp_property(type="dynamic",units="Wb")

    @sp_property
    def psi_boundary(self) -> float: return self._coord.psi_boundary  # sp_property(type="dynamic",units="Wb")

    @sp_property
    def magnetic_axis(self) -> _T_equilibrium_global_quantities_magnetic_axis:
        """Magnetic axis position and toroidal field	structure"""
        return {
            "r":  self._coord.magnetic_axis[0],
            "z":  self._coord.magnetic_axis[1],
            "b_field_tor": np.nan  # FIXME: b_field_tor
        }

        # magnetic_axis  :_T_equilibrium_global_quantities_magnetic_axis =  sp_property()

        # current_centre  :_T_equilibrium_global_quantities_current_centre =  sp_property()

        # q_axis  :float =  sp_property(type="dynamic",units="-",cocos_label_transformation="q_like",cocos_transformation_expression=".fact_q",cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.q_axis")

        # q_95  :float =  sp_property(type="dynamic",units="-",cocos_label_transformation="q_like",cocos_transformation_expression=".fact_q",cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.q_95")

        # q_min  :_T_equilibrium_global_quantities_qmin =  sp_property()

        # energy_mhd  :float =  sp_property(type="dynamic",units="J")

        # psi_external_average  :float =  sp_property(type="dynamic",units="Wb",cocos_label_transformation="psi_like",cocos_transformation_expression=".fact_psi",cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.psi_external_average")

        # v_external  :float =  sp_property(type="dynamic",units="V",cocos_label_transformation="ip_like",cocos_transformation_expression=".sigma_ip_eff",cocos_leaf_name_aos_indices=["core_profiles.global_quantities.v_loop","equilibrium.time_slice{i}.global_quantities.v_external"],introduced_after_version="3.37.2")

        # plasma_inductance  :float =  sp_property(type="dynamic",units="H")

        # plasma_resistance  :float =  sp_property(type="dynamic",units="ohm",introduced_after_version="3.37.2")


class FyEquilibriumProfiles1D(Equilibrium.TimeSlice.Profiles1D):

    @property
    def _coord(self) -> Equilibrium.TimeSlice.CoordinateSystem:
        return self._parent.coordinate_system

    ###############################
    # 1-D
    @property
    def psi_norm(self) -> ArrayType: return self._coord.psi_norm

    @property
    def psi(self) -> ArrayType: return self._coord.psi

    @sp_property
    def phi(self) -> Function: return self.dphi_dpsi.antiderivative()
    r"""  $\Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi$    """

    @sp_property(coordinate1="../psi")
    def dphi_dpsi(self) -> Function: return self.f * self._coord.surface_integral(1.0/(_R**2))
    # return self.f * self.gm1 * self.dvolume_dpsi / TWOPI

    @property
    def fpol(self) -> Function: return np.sqrt(2.0*self.f_df_dpsi.antiderivative()+(self._R0*self._B0)**2)

    dpressure_dpsi: Function = sp_property(extrapolate='zeros')

    f_df_dpsi: Function = sp_property(extrapolate='zeros')

    @property
    def ffprime(self) -> Function: return self.f_df_dpsi

    @property
    def pprime(self) -> Function: return self.dpressure_dpsi

    @sp_property
    def j_tor(self) -> Function:
        return self.plasma_current.pd() / (self._coord.psi_boundary - self._coord.psi_magnetic_axis)/self.dvolume_dpsi * self._coord.r0

    @sp_property
    def j_parallel(self) -> Function:
        fvac = self._coord._fvac
        d = np.asarray(function_like(np.asarray(self.volume),
                                     np.asarray(fvac*self.plasma_current/self.fpol)).pd())
        return self._coord._R0*(self.fpol / fvac)**2 * d

    @sp_property
    def q(self) -> Function:
        return self.dphi_dpsi * (self._coord._s_Bp * self._coord._s_rtp * self._coord._s_eBp_2PI/TWOPI)

    @sp_property
    def magnetic_shear(self) -> Function:
        return self.rho_tor * self.q.pd()/self.q*self.dpsi_drho_tor

    @sp_property
    def rho_tor(self) -> Function: return np.sqrt(self.phi / (PI*self._coord._B0))

    @sp_property
    def rho_tor_norm(self) -> Function:
        return np.sqrt(self.phi/self.phi(self._parent.boundary.psi))

    @sp_property
    def drho_tor_dpsi(self) -> Function: return 1.0/self.dpsi_drho_tor
    r"""
        $\frac{d\rho_{tor}}{d\psi} =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                    =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                    =\frac{q}{2\pi B_{0}\rho_{tor}}
        $
    """

    @sp_property
    def dpsi_drho_tor(self) -> Function: return (self._coord._s_B0)*self._coord._B0*self.rho_tor/self.q

    @sp_property
    def volume(self) -> Function: return self.dvolume_dpsi.antiderivative()

    @sp_property
    def dvolume_dpsi(self) -> Function: return self._coord.dvolume_dpsi

    @sp_property
    def dvolume_drho_tor(self) -> Function:
        return (self._coord._s_B0*self._coord._s_eBp_2PI*self._coord._B0) * self.dvolume_dpsi*self.dpsi_drho_tor
    # return self._coord._s_Ip * TWOPI * self.rho_tor / \
    #     (self.gm1)/(self._coord._R0*self._coord._B0/self.fpol)/self._coord._R0

    @sp_property
    def area(self) -> Function: return self.darea_dpsi.antiderivative()

    @sp_property
    def darea_dpsi(self) -> Function:
        logger.warning(f"FIXME: just a simple approximation! ")
        return self.dvolume_dpsi/(TWOPI*self._coord._R0)

    @sp_property
    def darea_drho_tor(self) -> Function: return self.darea_dpsi*self.dpsi_drho_tor

    @sp_property
    def surface(self) -> Function: return self.dvolume_drho_tor*self.gm7

    @sp_property
    def dphi_dvolume(self) -> Function: return self.fpol * self.gm1

    @sp_property
    def gm1(self) -> Function: return self._coord.surface_average(1.0/(_R**2))

    @sp_property
    def gm2(self) -> Function:
        return self._coord.surface_average(self._coord.grad_psi2/(_R**2)) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm3(self) -> Function:
        return self._coord.surface_average(self._coord.grad_psi2) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm4(self) -> Function: return self._coord.surface_average(1.0/self._coord.B2)

    @sp_property
    def gm5(self) -> Function: return self._coord.surface_average(self._coord.B2)

    @sp_property
    def gm6(self) -> Function:
        return self._coord.surface_average(self._coord.grad_psi2 / self._coord.B2) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm7(self) -> Function:
        return self._coord.surface_average(np.sqrt(self._coord.grad_psi2)) / self.dpsi_drho_tor

    @sp_property
    def gm8(self) -> Function: return self._coord.surface_average(_R)

    @sp_property
    def gm9(self) -> Function: return self._coord.surface_average(1.0 / _R)

    @sp_property
    def plasma_current(self) -> Function:
        return self.gm2 * self.dvolume_drho_tor / self.dpsi_drho_tor/scipy.constants.mu_0

    @sp_property
    def dpsi_drho_tor_norm(self) -> Function: return self.dpsi_drho_tor*self.rho_tor[-1]

    @functools.cached_property
    def _shape_property(self) -> FyEquilibriumCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi)

    @sp_property
    def geometric_axis(self) -> RZTuple_:
        return {"r": (self._shape_property.Rmin+self._shape_property.Rmax)*0.5,
                "z": (self._shape_property.Zmin+self._shape_property.Zmax)*0.5}

    @sp_property
    def minor_radius(self) -> Function:
        return (self._shape_property.Rmax - self._shape_property.Rmin)*0.5

    @sp_property
    def r_inboard(self) -> Function:
        return self._shape_property.r_inboard

    @sp_property
    def r_outboard(self) -> Function: return self._shape_property.r_outboard

    @sp_property
    def elongation(self) -> Function:
        return (self._shape_property.Zmax - self._shape_property.Zmin)/(self._shape_property.Rmax - self._shape_property.Rmin)

    @sp_property
    def elongation_upper(self) -> Function:
        return (self._shape_property.Zmax-(self._shape_property.Zmax+self._shape_property.Zmin)*0.5)/(self._shape_property.Rmax-self._shape_property.Rmin)

    @sp_property
    def elongation_lower(self) -> Function:
        return ((self._shape_property.Zmax+self._shape_property.Zmin)*0.5-self._shape_property.Zmin)/(self._shape_property.Rmax-self._shape_property.Rmin)

    @sp_property(coordinate1="../psi")
    def triangularity(self) -> Function:
        return (self._shape_property.Rzmax-self._shape_property.Rzmin)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def triangularity_upper(self) -> Function:
        return ((self._shape_property.Rmax+self._shape_property.Rmin)*0.5 - self._shape_property.Rzmax)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def triangularity_lower(self) -> Function:
        return ((self._shape_property.Rmax+self._shape_property.Rmin)*0.5 - self._shape_property.Rzmin)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def trapped_fraction(self) -> Function:
        """Trapped particle fraction[-]
            Tokamak 3ed, 14.10
        """
        epsilon = self.rho_tor(self.psi)/self._coord._R0
        return 1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon))


class FyEquilibriumProfiles2D(Equilibrium.TimeSlice.Profiles2D):

    @property
    def _coord(self) -> FyEquilibriumCoordinateSystem: return self._parent.coordinate_system

    @property
    def _global_quantities(self) -> FyEquilibriumGlobalQuantities: return self._parent.global_quantities

    @property
    def _profiles_1d(self) -> FyEquilibriumProfiles1D: return self._parent.profiles_1d

    @sp_property[Mesh]
    def grid(self) -> Mesh:
        dim1 = super().grid.dim1
        dim2 = super().grid.dim2
        mesh_type = super().grid_type
        return Mesh(dim1, dim2, mesh_type=mesh_type)

    @sp_property
    def r(self) -> Field: return Field(self.grid.points[0], mesh=self.grid)

    @sp_property
    def z(self) -> Field: return Field(self.grid.points[1], mesh=self.grid)

    # @sp_property[Field]
    # def psi(self) -> Field: return super().psi

    @property
    def psi_norm(self) -> Field:
        return (super().psi-self._coord.psi_magetic_axis)/(self._coord.psi_boundary - self._coord.psi_magetic_axis)

    @sp_property
    def phi(self) -> Field: return super().phi

    @sp_property
    def theta(self) -> Field: return super().theta

    @sp_property
    def j_tor(self) -> Field:
        return _R*self._profiles_1d.pprime(self.psi) + self._profiles_1d.ffprime(self.psi)/(_R*scipy.constants.mu_0)
        # return super().j_tor  # return self._profiles_1d.j_tor(self.psi)

    @sp_property
    def j_parallel(self) -> Field: return super().j_parallel  # return self._profiles_1d.j_parallel(self.psi)

    @sp_property
    def b_field_r(self) -> Field:
        """ COCOS Eq.19 [O. Sauter and S.Yu. Medvedev, Computer Physics Communications 184 (2013) 293] """
        return self.psi.pd(0, 1) / _R * (self._coord._s_RpZ * self._coord._s_Bp / self._coord._s_eBp_2PI)

    @sp_property
    def b_field_z(self) -> Field:
        return -self.psi.pd(1, 0) / _R * (self._coord._s_RpZ * self._coord._s_Bp / self._coord._s_eBp_2PI)

    @sp_property
    def b_field_tor(self) -> Field: return self._coord.fpol(self.psi) / _R


class FyEquilibriumBoundary(Equilibrium.TimeSlice.Boundary):
    @property
    def _coord(self) -> FyEquilibriumCoordinateSystem: return self._parent.coordinate_system

    @sp_property[Curve](coordinates="r z")
    def outline(self) -> Curve:
        _, surf = next(self._coord.find_surfaces(self.psi, o_point=True))
        return surf

    psi_norm: float = sp_property(default_value=0.999)

    @sp_property
    def psi(self) -> float:
        return self.psi_norm*(self._coord.psi_boundary-self._coord.psi_magnetic_axis) + self._coord.psi_magnetic_axis

    @sp_property
    def phi(self) -> float: return np.nan  # raise NotImplementedError(f"{self.__class__.__name__}.phi")

    @sp_property
    def rho(self) -> float: return np.sqrt(self.phi/(scipy.constants.pi * self._coord._B0))

    @functools.cached_property
    def _shape_property(self) -> FyEquilibriumCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi)

    @sp_property
    def geometric_axis(self) -> RZTuple:
        return {"r": (self._shape_property.Rmin+self._shape_property.Rmax)*0.5,
                "z": (self._shape_property.Zmin+self._shape_property.Zmax)*0.5}

    @sp_property
    def minor_radius(self) -> float: return (self._shape_property.Rmax - self._shape_property.Rmin)*0.5

    @sp_property
    def elongation(self) -> float:
        return (self._shape_property.Zmax - self._shape_property.Zmin)/(self._shape_property.Rmax - self._shape_property.Rmin)

    @sp_property
    def elongation_upper(self) -> float:
        return (self._shape_property.Zmax-(self._shape_property.Zmax+self._shape_property.Zmin)*0.5)/(self._shape_property.Rmax-self._shape_property.Rmin)

    @sp_property
    def elongation_lower(self) -> float:
        return ((self._shape_property.Zmax+self._shape_property.Zmin)*0.5-self._shape_property.Zmin)/(self._shape_property.Rmax-self._shape_property.Rmin)

    @sp_property(coordinate1="../psi")
    def triangularity(self) -> float:
        return (self._shape_property.Rzmax-self._shape_property.Rzmin)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def triangularity_upper(self) -> float:
        return ((self._shape_property.Rmax+self._shape_property.Rmin)*0.5 - self._shape_property.Rzmax)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def triangularity_lower(self) -> float:
        return ((self._shape_property.Rmax+self._shape_property.Rmin)*0.5 - self._shape_property.Rzmin)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def x_point(self) -> List[OXPoint]:
        _, x_pt = self._coord.critical_points
        return x_pt

    @sp_property
    def strike_point(self) -> List[OXPoint]:
        return

    @sp_property
    def active_limiter_point(self) -> List[RZTuple]: return NotImplemented


class FyEquilibriumBoundarySeparatrix(Equilibrium.TimeSlice.BoundarySeparatrix):

    @property
    def _coord(self) -> Equilibrium.TimeSlice.CoordinateSystem: return self._parent.coordinate_system

    @sp_property[Curve](coordinates="r z")
    def outline(self) -> Curve:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surfaces(self.psi, o_point=None))

        return surf
        # return {"r": points[..., 0], "z": points[..., 1]}

    @sp_property
    def magnetic_axis(self) -> float: return self._coord.psi_magnetic_axis

    @sp_property
    def psi(self) -> float: return self._coord.psi_boundary

    @sp_property
    def x_point(self) -> List[RZTuple]:
        _, x = self._coord.critical_points
        return List[RZTuple]([{"r": v.r, "z": v.z} for v in x[:]])

    @sp_property
    def strike_point(self) -> List[RZTuple]: raise NotImplementedError("TODO: strike_point")


class FyEquilibriumTimeSlice(Equilibrium.TimeSlice):

    profiles_1d: FyEquilibriumProfiles1D = sp_property()

    profiles_2d: AoS[FyEquilibriumProfiles2D] = sp_property()
    """ 定义多个 profiles_2d, type==0 对应  Total fields """

    global_quantities: FyEquilibriumGlobalQuantities = sp_property()

    boundary: FyEquilibriumBoundary = sp_property()

    boundary_separatrix: FyEquilibriumBoundarySeparatrix = sp_property()

    coordinate_system: FyEquilibriumCoordinateSystem = sp_property()

    def __geometry__(self, view_point="RZ", **kwargs) -> GeoObject:
        """
            plot o-point,x-point,lcfs,separatrix and contour of psi
        """

        geo = {}
        styles = {}

        match view_point.lower():
            case "rz":
                if self.profiles_2d[0].psi._cache is not _not_found_:

                    o_points, x_points = self.coordinate_system.critical_points

                    geo["o_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(o_points)]
                    geo["x_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(x_points)]

                    geo["boundary"] = [surf for _, surf in
                                       self.coordinate_system.find_surfaces(self.boundary.psi, o_point=True)]

                    geo["boundary_separatrix"] = [surf for _, surf in
                                                  self.coordinate_system.find_surfaces(self.boundary_separatrix.psi, o_point=False)]

                    geo["psi"] = self.profiles_2d[0].psi

                styles["o_points"] = {"$matplotlib": {"color": 'red',   'marker': '.', "linewidths": 0.5}}
                styles["x_points"] = {"$matplotlib": {"color": 'blue',  'marker': 'x', "linewidths": 0.5}}
                styles["boundary"] = {"$matplotlib": {"color": 'blue', 'linestyle': 'dotted', 'linewidth': 0.5}}
                styles["boundary_separatrix"] = {"$matplotlib": {
                    "color": 'red', "linestyle": 'dashed', 'linewidth': 0.25}}
                styles["psi"] = {"$matplotlib": {"levels": 40, "cmap": "jet"}}

        styles = merge_tree_recursive(styles, kwargs)

        return geo, styles


@Equilibrium.register(["eq_analyze"])
class FyEqAnalyze(Equilibrium):
    """
    Magnetic surface analyze 磁面分析工具
    =============================
    input:
        - vacuum_toroidal_field.b0, vacuum_toroidal_field.r0
        - fpol, Diamagnetic function (F=R B_Phi)
        - profiles_2d.psi (RZ 2D)

    output：
        - 识别 O,X point
        - 识别 Separatrix, boundary
        - Surface average

    """

    TimeSlice = FyEquilibriumTimeSlice

    time_slice: TimeSeriesAoS[FyEquilibriumTimeSlice] = sp_property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, ** kwargs)
