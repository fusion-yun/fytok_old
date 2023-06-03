import collections
import collections.abc
import typing
from dataclasses import dataclass
from enum import Enum
from math import isclose
import functools
import numpy as np
from fytok._imas.lastest.equilibrium import (
    _T_equilibrium_boundary, _T_equilibrium_boundary_separatrix,
    _T_equilibrium_coordinate_system, _T_equilibrium_global_quantities,
    _T_equilibrium_global_quantities_magnetic_axis, _T_equilibrium_profiles_1d,
    _T_equilibrium_profiles_2d, _T_equilibrium_time_slice,
    _T_equilibrium_profiles_1d_rz1d_dynamic_aos)
from fytok._imas.lastest.utilities import _T_identifier_dynamic_aos3
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.Utilities import RZTuple, RZTuple1D, RZTuple_
from scipy import constants
from spdm.data.Dict import Dict
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.Expression import Expression,  Variable
from spdm.data.List import List,AoS
from spdm.data.Node import Node
from spdm.data.Function import Function
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.geometry.CubicSplineCurve import CubicSplineCurve
from spdm.geometry.GeoObject import GeoObject, GeoObjectSet
from spdm.geometry.Point import Point
from spdm.geometry.Curve import Curve
from spdm.mesh.CurvilinearMesh import CurvilinearMesh
from spdm.mesh.Mesh import Mesh
from spdm.mesh.RectilinearMesh import RectilinearMesh
from spdm.numlib.contours import find_countours
from spdm.numlib.optimize import find_critical_points
from spdm.numlib.interpolate import interpolate
from spdm.utils.logger import logger
from spdm.utils.misc import convert_to_named_tuple
from spdm.utils.tags import _not_found_
from spdm.utils.typing import ArrayType, NumericType, scalar_type, ArrayLike

_R = Variable(0, "R")
_Z = Variable(1, "Z")

TOLERANCE = 1.0e-6
EPS = np.finfo(float).eps

TWOPI = 2.0*constants.pi


@dataclass
class OXPoint:
    r: float
    z: float
    psi: float


TOLERANCE = 1.0e-6

EPS = np.finfo(float).eps
PI = constants.pi
TWOPI = 2.0*PI



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


class EquilibriumCoordinateSystem(_T_equilibrium_coordinate_system):
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
        logger.debug(f"Create MagneticCoordSystem.")

        self._B0 = super().get("b0", self._parent._B0)   # magnetic field on magnetic axis
        self._R0 = super().get("r0", self._parent._R0)   # major radius of magnetic axis
        self._Ip = super().get("ip", self._parent.global_quantities.ip)  # plasma current

        self._fpol = self._parent.profiles_1d.f  # poloidal current function

        self._s_B0 = np.sign(self._B0)

        self._s_Ip = np.sign(self._Ip)

        self._e_Bp,  self._s_Bp, self._s_RpZ, self._s_rtp = COCOS_TABLE[self.cocos]

        self._s_eBp_2PI = 1.0 if self._e_Bp == 0 else TWOPI

        logger.debug(f"COCOS={self.cocos}")

    @sp_property
    def cocos(self) -> int:
        cocos_flag = super().get("cocos", _not_found_)

        if cocos_flag is not _not_found_:
            return cocos_flag
        return 5

    @functools.cached_property
    def _psirz(self) -> Field[float]:
        psirz = super().get("psirz", _not_found_)

        if isinstance(psirz, np.ndarray):
            dim1 = super().get("grid/dim1", _not_found_)
            dim2 = super().get("grid/dim2", _not_found_)
            grid_type = super().get("grid/type", "rectangular")

            if not isinstance(dim1, np.ndarray) or not isinstance(dim2, np.ndarray):
                raise RuntimeError(f"Can not create grid!")

            psirz = Field(psirz, dim1, dim2, name="psirz")
        elif psirz is _not_found_:
            psirz = self._parent.profiles_2d[0].psi
        else:
            logger.warning(f"Ignore {type(psirz)}. Using ../profiles_2d[0].psi ")

        if not isinstance(psirz, Field):
            raise RuntimeError(f"Can not get psirz!")

        return psirz

    @functools.cached_property
    def critical_points(self) -> typing.Tuple[typing.Sequence[OXPoint], typing.Sequence[OXPoint]]:

        opoints = []
        xpoints = []

        for r, z, psi, D in find_critical_points(self._psirz):
            p = OXPoint(r, z, psi)

            if D < 0.0:  # saddle/X-point
                xpoints.append(p)
            else:  # extremum/ O-point
                opoints.append(p)

        # wall = getattr(self._parent._parent, "wall", None)
        # if wall is not None:
        #     xpoints = [p for p in xpoints if wall.in_limiter(p.r, p.z)]

        if not opoints:
            raise RuntimeError(f"Can not find o-point!")
        else:

            bbox = self._psirz.mesh.geometry.bbox
            Rmid = (bbox[0][0] + bbox[1][0])/2.0
            Zmid = (bbox[0][1] + bbox[1][1])/2.0

            opoints.sort(key=lambda x: (x.r - Rmid)**2 + (x.z - Zmid)**2)

            o_r = opoints[0].r
            o_z = opoints[0].z
            # TOOD: NEED　IMPROVMENT!!
            xpoints.sort(key=lambda x: (x.r - o_r)**2 + (x.z - o_z)**2)
            # psi_magnetic_axis = opoints[0].psi
            # xpoints.sort(key=lambda x: (x.psi - psi_magnetic_axis)**2)

        return opoints, xpoints

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

    @functools.cached_property
    def psi_bc(self) -> typing.Tuple[float, float]:

        o, x = self.critical_points

        if len(o) == 0:
            raise RuntimeError(f"Can not find o-point")
        psi_magnetic_axis = o[0].psi

        if len(x) == 0:
            raise RuntimeError(f"Can not find x-point")
        psi_boundary = x[0].psi

        return psi_magnetic_axis, psi_boundary

    @property
    def psi_magnetic_axis(self) -> float: return self.psi_bc[0]

    @property
    def psi_boundary(self) -> float: return self.psi_bc[1]

    @sp_property
    def grid_type(self) -> _T_identifier_dynamic_aos3:
        desc = super().grid_type
        if desc.name is None or desc.name is _not_found_:
            desc = {"name": "rectangular", "index": 1, "description": "default"}
        return desc

    @sp_property[Mesh]
    def grid(self) -> Mesh:
        psi_norm = super().grid.dim1

        if isinstance(psi_norm, np.ndarray) and psi_norm.ndim == 1:
            pass
        elif isinstance(psi_norm, np.ndarray) and psi_norm.ndim == 0:
            psi_norm_boundary = self._parent.boundary.psi_norm
            psi_norm = np.linspace(0, psi_norm_boundary, int(psi_norm), endpoint=True)
        elif isinstance(psi_norm, collections.abc.Sequence) and len(psi_norm) == 3:
            psi_norm = np.linspace(psi_norm[0], psi_norm[1], psi_norm[2], endpoint=True)
        else:
            raise ValueError(f"Can not create grid! psi_norm={psi_norm}")

        theta = super().grid.dim2
        if isinstance(theta, np.ndarray) and theta.ndim == 1:
            pass
        elif isinstance(theta, np.ndarray) and theta.ndim == 0:
            theta = np.linspace(0, TWOPI, int(theta), endpoint=False)
        elif isinstance(theta, collections.abc.Sequence) and len(theta) == 3:
            theta = np.linspace(theta[0], theta[1], int(theta[2]), endpoint=False)
        else:
            raise ValueError(f"Can not create grid! theta={theta}")

        surfs = GeoObjectSet([surf for _, surf in self.find_surface_by_psi_norm(psi_norm, o_point=True)])

        return CurvilinearMesh(psi_norm, theta, geometry=surfs, cycles=[False, TWOPI])

    @sp_property
    def r(self) -> Field[float]: return Field(self.grid.points[..., 0], grid=self.grid)

    @sp_property
    def z(self) -> Field[float]: return Field(self.grid.points[..., 1], grid=self.grid)

    @sp_property
    def jacobian(self) -> Field[float]:
        raise NotImplementedError(f"")

    @sp_property
    def tensor_covariant(self) -> Field[float]:
        raise NotImplementedError(f"")

    @sp_property
    def tensor_contravariant(self) -> Field[float]:
        raise NotImplementedError(f"")

    def find_surface(self, psi:  float | ArrayType | typing.Sequence[float], o_point: OXPoint = True) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:
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
            for level, points in find_countours(self._psirz, levels=psi):
                yield level, CubicSplineCurve(points)
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
            for level, points in find_countours(self._psirz, levels=psi):
                # 累计相同 level 的 surface个数
                # 如果累计的 surface 个数大于1，说明存在磁岛
                # 如果累计的 surface 个数等于0，说明该 level 对应的 surface 不存在
                # 如果累计的 surface 个数等于1，说明该 level 对应的 surface 存在且唯一
                if points is None:
                    if np.isclose(level, o_point.psi):
                        yield level, Point(o_point.r, o_point.z)
                    else:
                        yield level, None  # raise RuntimeError(f"Can not find surface psi={level}")
                    continue

                if current_psi is np.nan:
                    current_psi = level
                    current_count = 1
                elif np.isclose(level, current_psi):
                    current_count += 1
                else:
                    if current_count < 1:
                        yield current_psi, None
                    elif current_count > 1:
                        raise RuntimeError(f"find magnetic island! get {current_count} surface for psi={current_psi}")
                    current_count = 0
                    current_psi = np.nan

                theta = np.arctan2(points[:, 0]-o_point.r, points[:, 1]-o_point.z)

                if 1.0 - (max(theta)-min(theta))/TWOPI > 2.0/len(theta):  # open or do not contain o-point
                    current_count -= 1
                    continue

                if np.isclose((theta[0]-theta[-1]) % TWOPI, 0.0):
                    theta = theta[:-1]
                    points = points[:-1]
                else:  # boundary separatrix
                    if x_point is None:
                        raise RuntimeError(f"No X-point ")
                    # logger.warning(f"The magnetic surface average is not well defined on the separatrix!")
                    xpt = np.asarray([x_point.r, x_point.z], dtype=float)
                    b = points[1:]
                    a = points[:-1]
                    d = b-a
                    d2 = d[:, 0]**2+d[:, 1]**2
                    p = xpt-a

                    c = (p[:, 0]*d[:, 0]+p[:, 1]*d[:, 1])/d2
                    s = (p[:, 0]*d[:, 1]-p[:, 1]*d[:, 0])/d2
                    idx = np.flatnonzero(np.logical_and(c >= 0, c**2+s**2 < 1))

                    if len(idx) != 2:
                        raise NotImplementedError()

                    idx0 = idx[0]
                    idx1 = idx[1]

                    theta_x = np.arctan2(xpt[0]-o_point.r, xpt[1]-o_point.z)

                    points = np.vstack([[xpt], points[idx0:idx1]])
                    theta = np.hstack([theta_x, theta[idx0:idx1]])

                # theta must be strictly increased
                p_min = np.argmin(theta)
                p_max = np.argmax(theta)

                if p_min > 0:
                    if p_min == p_max+1:
                        theta = np.roll(theta, -p_min)
                        points = np.roll(points, -p_min, axis=0)
                    elif p_min == p_max-1:
                        theta = np.flip(np.roll(theta, -p_min-1))
                        points = np.flip(np.roll(points, -p_min-1, axis=0), axis=0)
                    else:
                        raise ValueError(f"Can not convert 'u' to be strictly increased!")
                    theta = np.hstack([theta, [theta[0]+TWOPI]])
                    theta = (theta-theta.min())/(theta.max()-theta.min())
                    points = np.vstack([points, points[:1]])

                if points.shape[0] == 0:
                    logger.warning(f"{level},{o_point.psi},{(max(theta),min(theta))}")

                elif points.shape[0] == 1:
                    yield level, Point(points[0][0], points[0][1])
                else:
                    yield level, CubicSplineCurve(points, theta)

    def find_surface_by_psi_norm(self, psi_norm: float | ArrayType | typing.Sequence[float], *args,   **kwargs) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:

        psi_magnetic_axis, psi_boundary = self.psi_bc

        if isinstance(psi_norm, (collections.abc.Sequence, np.ndarray)):
            psi = np.asarray(psi_norm, dtype=float)*(psi_boundary-psi_magnetic_axis)+psi_magnetic_axis
            yield from self.find_surface(psi, *args,  **kwargs)
        elif isinstance(psi_norm, collections.abc.Generator):
            for psi_n in psi_norm:
                yield from self.find_surface(psi_n*(psi_boundary-psi_magnetic_axis)+psi_magnetic_axis, *args,  **kwargs)

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
            r, z = s.points
            # r = rz[..., 0]
            # z = rz[..., 1]
            if isinstance(s, Point):
                rmin = r
                rmax = r
                zmin = z
                zmax = z
                r_inboard = r
                r_outboard = r
                rzmin = r
                rzmax = r
            else:
                (rmin, zmin), (rmax, zmax) = s.bbox
                rzmin = r[np.argmin(z)]
                rzmax = r[np.argmax(z)]
                r_inboard = s.coordinates(0.5)[0]
                r_outboard = s.coordinates(0)[0]
            return rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard

        if psi is None:
            psi = self.psi
        elif not isinstance(psi, (np.ndarray, collections.abc.MutableSequence)):
            psi = [psi]

        sbox = np.asarray([[p, *shape_box(s)] for p, s in self.find_surface(psi)], dtype=float)
        logger.debug(sbox.shape)
        if sbox.shape[0] == 1:
            psi, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox[0]
        else:
            psi, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox.T
        if np.isscalar(psi):
            return EquilibriumCoordinateSystem.ShapeProperty(
                psi, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard)
        else:
            return EquilibriumCoordinateSystem.ShapeProperty(
                psi,
                Function(rmin,      psi,   name="rmin"),
                Function(zmin,      psi,   name="zmin"),
                Function(rmax,      psi,   name="rmax"),
                Function(zmax,      psi,   name="zmax"),
                Function(rzmin,     psi,   name="rzmin"),
                Function(rzmax,     psi,   name="rzmax"),
                Function(r_inboard, psi,   name="r_inboard"),
                Function(r_outboard, psi,   name="r_outboard"),
            )

    #################################
    # fields

    @property
    def Bpol(self) -> Expression[float]: return np.sqrt(self.b_field_r**2+self.b_field_z**2)
    r""" $B_{pol}= \left|\nabla \psi \right|/2 \pi R $ """

    @property
    def b_field_r(self) -> Expression[float]:
        """ COCOS Eq.19 [O. Sauter and S.Yu. Medvedev, Computer Physics Communications 184 (2013) 293] """
        return self._psirz.pd(0, 1) / _R * (self._s_RpZ * self._s_Bp / self._s_eBp_2PI)

    @property
    def b_field_z(self) -> Expression[float]:
        return -self._psirz.pd(1, 0) / _R * (self._s_RpZ * self._s_Bp / self._s_eBp_2PI)

    @property
    def b_field_tor(self) -> Expression[float]: return self._fpol(self._psirz) / _R

    @property
    def B2(self) -> Expression[float]: return (self.b_field_r**2 + self.b_field_z**2 + self.b_field_tor ** 2)

    @property
    def grad_psi2(self) -> Expression[float]: return self._psirz.pd(1, 0)**2+self._psirz.pd(0, 1)**2

    @property
    def grad_psi(self) -> Expression[float]: return np.sqrt(self.grad_psi2)

    @property
    def ddpsi(self) -> Expression[float]:
        return np.sqrt(self._psirz.pd(2, 0) * self._psirz.pd(0, 2) + self._psirz.pd(1, 1)**2)

    @functools.cached_property
    def dvolume_dpsi(self) -> Function[float]: return Function(*self._surface_integral(1.0), name="dvolume_dpsi")

    ###############################
    # surface integral

    def _surface_integral(self, func: Expression, psi: NumericType = None) -> typing.Tuple[ArrayLike, ArrayLike]:
        r"""
            $ V^{\prime} =  2 \pi  \int{ R / \left|\nabla \psi \right| * dl }$
            $ V^{\prime}(psi)= 2 \pi  \int{ dl * R / \left|\nabla \psi \right|}$
        """

        # r0, z0 = self.magnetic_axis
        psi_axis, psi_boundary = self.psi_bc

        # ddpsi = self.ddpsi(r0, z0)

        # c0 = r0**2/ddpsi

        if psi is None or psi is self.psi:
            surfs_list = zip(self.psi, self.grid.geometry)
        else:
            if isinstance(psi, scalar_type):
                psi = [psi]
            surfs_list = self.find_surface(psi, o_point=True)

        # f_Bpol = Field(func/self.Bpol, mesh=self.grid, name="f_Bpol").compile()

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

    def surface_integral(self, func: Expression, * psi: NumericType) -> Expression | ArrayLike:

        if not np.isscalar(psi):
            return Function(*self._surface_integral(func, *psi), name=f"surface_integral({str(func)})")
        else:
            value, psi = self._surface_integral(func, *psi)
            # if isinstance(value, np.ndarray) and np.any(np.isnan(value)):
            #     # 若存在nan，则通过Function（插值）消除
            #     value = Function(value, psi)(psi)
            return value

    def surface_average(self, func: Expression, *psi: NumericType) -> Expression | ArrayLike:
        r"""
            $\left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}$
        """
        return self.surface_integral(func,  *psi)/self.dvolume_dpsi(*psi)


class EquilibriumGlobalQuantities(_T_equilibrium_global_quantities):
    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

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
    def psi_magnetic_axis(self) -> float: return self._coord.psi_magnetic_axis  # sp_property(type="dynamic",units="Wb")

    @sp_property
    def psi_boundary(self) -> float: return self._coord.psi_boundary  # sp_property(type="dynamic",units="Wb")

    @sp_property
    def magnetic_axis(self) -> _T_equilibrium_global_quantities_magnetic_axis:
        """Magnetic axis position and toroidal field	structure"""
        return {
            "r":  self._coord.magnetic_axis[0],
            "z":  self._coord.magnetic_axis[1],
            "b_field_tor": NotImplemented  # FIXME: b_field_tor
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


class EquilibriumFunctions1D(_T_equilibrium_profiles_1d):

    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

    ###############################
    # 1-D
    @property
    def psi_norm(self) -> ArrayType: return self._coord.psi_norm

    @property
    def psi(self) -> ArrayType: return self._coord.psi

    @sp_property
    def phi(self) -> Function[float]: return self.dphi_dpsi.antiderivative()
    r"""  $\Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi$    """

    @sp_property(coordinate1="../psi")
    def dphi_dpsi(self) -> Function[float]:
        return self.f * self._coord.surface_integral(1.0/(_R**2))
        # return self.f * self.gm1 * self.dvolume_dpsi / TWOPI

    @property
    def fpol(self) -> Function[float]: return super().f

    @sp_property
    def dpressure_dpsi(self) -> Function[float]: return self.pressure.derivative()

    @sp_property
    def f_df_dpsi(self) -> Function[float]: return self.f * self.f.pd()

    @property
    def pprime(self) -> Function[float]: return self.dpressure_dpsi

    @sp_property
    def j_tor(self) -> Function[float]:
        return self.plasma_current.pd() / (self._coord.psi_boundary - self._coord.psi_magnetic_axis)/self.dvolume_dpsi * self._coord.r0

    @sp_property
    def j_parallel(self) -> Function[float]:
        fvac = self._coord._fvac
        d = np.asarray(function_like(np.asarray(self.volume),
                                     np.asarray(fvac*self.plasma_current/self.fpol)).pd())
        return self._coord._R0*(self.fpol / fvac)**2 * d

    @sp_property
    def q(self) -> Function[float]:
        return self.dphi_dpsi * (self._coord._s_Bp * self._coord._s_rtp * self._coord._s_eBp_2PI/TWOPI)

    @sp_property
    def magnetic_shear(self) -> Function[float]:
        # return self.rho_tor/self.q * function_like(self.q(self.psi), self.rho_tor(self.psi)).pd()
        return self.rho_tor * self.q.pd()/self.q*self.dpsi_drho_tor

    @sp_property
    def rho_tor(self) -> Function[float]: return np.sqrt(self.phi / (PI*self._coord._B0))

    @sp_property
    def rho_tor_norm(self) -> Function[float]: return np.sqrt(self.phi/self.phi(self._parent.boundary.psi))

    @sp_property
    def drho_tor_dpsi(self) -> Function[float]: return 1.0/self.dpsi_drho_tor
    r"""
        $\frac{d\rho_{tor}}{d\psi} =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                    =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                    =\frac{q}{2\pi B_{0}\rho_{tor}}
        $
    """

    @sp_property
    def dpsi_drho_tor(self) -> Function[float]: return (self._coord._s_B0)*self._coord._B0*self.rho_tor/self.q

    @sp_property
    def volume(self) -> Function[float]: return self.dvolume_dpsi.antiderivative()

    @sp_property
    def dvolume_dpsi(self) -> Function[float]: return self._coord.dvolume_dpsi

    @sp_property
    def dvolume_drho_tor(self) -> Function[float]:
        return (self._coord._s_B0*self._coord._s_eBp_2PI*self._coord._B0) * self.dvolume_dpsi*self.dpsi_drho_tor
    # return self._coord._s_Ip * TWOPI * self.rho_tor / \
    #     (self.gm1)/(self._coord._R0*self._coord._B0/self.fpol)/self._coord._R0

    @sp_property
    def area(self) -> Function[float]: return self.darea_dpsi.antiderivative()

    @sp_property
    def darea_dpsi(self) -> Function[float]:
        logger.warning(f"FIXME: just a simple approximation! ")
        return self.dvolume_dpsi/(TWOPI*self._coord._R0)

    @sp_property
    def darea_drho_tor(self) -> Function[float]: return self.darea_dpsi*self.dpsi_drho_tor

    @sp_property
    def surface(self) -> Function[float]: return self.dvolume_drho_tor*self.gm7

    @sp_property
    def dphi_dvolume(self) -> Function[float]: return self.fpol * self.gm1

    @sp_property
    def gm1(self) -> Function[float]: return self._coord.surface_average(1.0/(_R**2))

    @sp_property
    def gm2_(self) -> Function[float]: return self._coord.surface_average(self._coord.grad_psi2/(_R**2))

    @sp_property
    def gm2(self) -> Function[float]:
        return self._coord.surface_average(self._coord.grad_psi2/(_R**2)) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm3(self) -> Function[float]:
        return self._coord.surface_average(self._coord.grad_psi2) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm4(self) -> Function[float]: return self._coord.surface_average(1.0/self._coord.B2)

    @sp_property
    def gm5(self) -> Function[float]: return self._coord.surface_average(self._coord.B2)

    @sp_property
    def gm6(self) -> Function[float]:
        return self._coord.surface_average(self._coord.grad_psi2 / self._coord.B2) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm7(self) -> Function[float]:
        return self._coord.surface_average(np.sqrt(self._coord.grad_psi2)) / self.dpsi_drho_tor

    @sp_property
    def gm8(self) -> Function[float]: return self._coord.surface_average(_R)

    @sp_property
    def gm9(self) -> Function[float]: return self._coord.surface_average(1.0 / _R)

    @sp_property
    def plasma_current(self) -> Function[float]:
        return self.gm2 * self.dvolume_drho_tor / self.dpsi_drho_tor/constants.mu_0

    @sp_property
    def dpsi_drho_tor_norm(self) -> Function[float]: return self.dpsi_drho_tor*self.rho_tor[-1]

    @functools.cached_property
    def _shape_property(self) -> EquilibriumCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi)

    @sp_property
    def geometric_axis(self) -> RZTuple_:
        return {"r": (self._shape_property.Rmin+self._shape_property.Rmax)*0.5,
                "z": (self._shape_property.Zmin+self._shape_property.Zmax)*0.5}

    @sp_property
    def minor_radius(self) -> Function[float]:
        return (self._shape_property.Rmax - self._shape_property.Rmin)*0.5,

    @sp_property
    def r_inboard(self) -> Function[float]: return self._shape_property.r_inboard

    @sp_property
    def r_outboard(self) -> Function[float]: return self._shape_property.r_outboard

    @sp_property
    def elongation(self) -> Function[float]:
        return (self._shape_property.Zmax - self._shape_property.Zmin)/(self._shape_property.Rmax - self._shape_property.Rmin)

    @sp_property
    def elongation_upper(self) -> Function[float]:
        return (self._shape_property.Zmax-(self._shape_property.Zmax+self._shape_property.Zmin)*0.5)/(self._shape_property.Rmax-self._shape_property.Rmin),

    @sp_property
    def elongation_lower(self) -> Function[float]:
        return ((self._shape_property.Zmax+self._shape_property.Zmin)*0.5-self._shape_property.Zmin)/(self._shape_property.Rmax-self._shape_property.Rmin),

    @sp_property(coordinate1="../psi")
    def triangularity(self) -> Function[float]:
        return (self._shape_property.Rzmax-self._shape_property.Rzmin)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def triangularity_upper(self) -> Function[float]:
        return ((self._shape_property.Rmax+self._shape_property.Rmin)*0.5 - self._shape_property.Rzmax)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def triangularity_lower(self) -> Function[float]:
        return ((self._shape_property.Rmax+self._shape_property.Rmin)*0.5 - self._shape_property.Rzmin)/(self._shape_property.Rmax - self._shape_property.Rmin)*2

    @sp_property
    def trapped_fraction(self, value) -> Function[float]:
        """Trapped particle fraction[-]
            Tokamak 3ed, 14.10
        """
        if value is _not_found_:
            epsilon = self.rho_tor/self._coord._R0
            value = np.asarray(1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon)))
        return value


class EquilibriumFunctions2D(_T_equilibrium_profiles_2d):

    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

    @property
    def _global_quantities(self) -> _T_equilibrium_global_quantities: return self._parent.global_quantities

    @property
    def _profiles_1d(self) -> _T_equilibrium_profiles_1d: return self._parent.profiles_1d

    @sp_property
    def grid(self) -> Mesh: return Mesh(super().grid.dim1, super().grid.dim2,  type=super().grid_type)

    @sp_property
    def r(self) -> Field[float]: return Field(self.grid.points[0], mesh=self.grid)

    @sp_property
    def z(self) -> Field[float]: return Field(self.grid.points[1], mesh=self.grid)

    @sp_property
    def psi(self) -> Field[float]: return super().psi

    @property
    def psi_norm(self) -> Field[float]:
        psi_magetic_axis, psi_boundary = self._coord.psi_bc
        return (super().psi-psi_magetic_axis)/(psi_boundary - psi_magetic_axis)

    @sp_property
    def phi(self) -> Field[float]: return super().phi

    @sp_property
    def theta(self) -> Field[float]: return super().theta

    @sp_property
    def j_tor(self) -> Field[float]: return super().j_tor  # return self._profiles_1d.j_tor(self.psi)

    @sp_property
    def j_parallel(self) -> Field[float]: return super().j_parallel  # return self._profiles_1d.j_parallel(self.psi)

    @sp_property
    def b_field_r(self) -> Field[float]:
        """ COCOS Eq.19 [O. Sauter and S.Yu. Medvedev, Computer Physics Communications 184 (2013) 293] """
        return self.psi.pd(0, 1) / _R * (self._coord._s_RpZ * self._coord._s_Bp / self._coord._s_eBp_2PI)

    @sp_property
    def b_field_z(self) -> Field[float]:
        return -self.psi.pd(1, 0) / _R * (self._coord._s_RpZ * self._coord._s_Bp / self._coord._s_eBp_2PI)

    @sp_property
    def b_field_tor(self) -> Field[float]:
        return self._coord.fpol(self.psi) / _R


class EquilibriumBoundary(_T_equilibrium_boundary):
    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

    @sp_property
    def outline(self) -> RZTuple1D:
        _, surf = next(self._coord.find_surface(self.psi, o_point=True))
        points = surf.xyz()
        return {"r": points[..., 0], "z": points[..., 1]}

    psi_norm: float = sp_property(default_value=0.999)

    @sp_property
    def psi(self) -> float:
        return self.psi_norm*(self._coord.psi_boundary-self._coord.psi_magnetic_axis) + self._coord.psi_magnetic_axis

    @sp_property
    def phi(self) -> float: raise NotImplementedError(f"{self.__class__.__name__}.phi")

    @sp_property
    def rho(self) -> float: return np.sqrt(self.phi/(constants.pi * self._coord._B0))

    @functools.cached_property
    def _shape_property(self) -> EquilibriumCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi)

    @sp_property
    def geometric_axis(self) -> RZTuple: return self._shape_property.geometric_axis

    @sp_property
    def minor_radius(self) -> float: return self._shape_property.minor_radius

    @sp_property
    def elongation(self) -> float: return self._shape_property.elongation

    @sp_property
    def elongation_upper(self) -> float: return self._shape_property.elongation_upper

    @sp_property
    def elongation_lower(self) -> float: return self._shape_property.elongation_lower

    @sp_property
    def triangularity(self) -> float: return self._shape_property.triangularity

    @sp_property
    def triangularity_upper(self) -> float: return self._shape_property.triangularity_upper

    @sp_property
    def triangularity_lower(self) -> float: return self._shape_property.triangularity_lower

    @sp_property
    def x_point(self) -> List[RZTuple]:
        _, xpt = self._coord.critical_points
        return xpt

    @sp_property
    def strike_point(self) -> List[RZTuple]: return NotImplemented

    @sp_property
    def active_limiter_point(self) -> List[RZTuple]: return NotImplemented


class EquilibriumBoundarySeparatrix(_T_equilibrium_boundary_separatrix):

    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

    @sp_property
    def outline(self) -> RZTuple1D:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface(self.psi, o_point=None))
        points = surf.xyz()
        return {"r": points[..., 0], "z": points[..., 1]}

    @sp_property
    def magnetic_axis(self) -> float: return self._coord.psi_magnetic_axis

    @sp_property
    def psi(self) -> float: return self._coord.psi_boundary

    @sp_property
    def x_point(self) -> List[RZTuple]:
        _, x = self._coord.critical_points
        return List[RZTuple]([{"r": v.r, "z": v.z} for v in x[:]])

    @sp_property
    def strike_point(self) -> List[RZTuple]:
        raise NotImplementedError("TODO:")


class EquilibriumTimeSlice(_T_equilibrium_time_slice):

    @property
    def _R0(self) -> float: return self._parent.vacuum_toroidal_field.r0

    @property
    def _B0(self) -> float: return self._parent.vacuum_toroidal_field.b0(self.time)

    profiles_1d: EquilibriumFunctions1D = sp_property()

    profiles_2d: AoS[EquilibriumFunctions2D] = sp_property()
    """ FIXME: 定义多个 profiles_2d, type==0 对应  Total fields """

    global_quantities: EquilibriumGlobalQuantities = sp_property()

    boundary: EquilibriumBoundary = sp_property()

    boundary_separatrix: EquilibriumBoundarySeparatrix = sp_property()

    coordinate_system: EquilibriumCoordinateSystem = sp_property()


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

    TimeSlice = EquilibriumTimeSlice

    time_slice: TimeSeriesAoS[EquilibriumTimeSlice] = sp_property()

    def __init__(self, *args, **kwargs):
        code = {**kwargs.get("code", {}), "name": "fy_equilibrium", "version": "0.0.1", "commit": "-dirty"}
        super().__init__(*args, **{**kwargs, "code": code})


__SP_EXPORT__ = FyEqAnalyze
