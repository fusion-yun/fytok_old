import collections
import collections.abc
import functools
import typing
from dataclasses import dataclass
import numpy as np
import scipy.constants


from spdm.data.Expression import Expression, Variable
from spdm.data.Field import Field
from spdm.data.Expression import Expression
from spdm.data.HTree import List
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject, GeoObjectSet
from spdm.geometry.Point import Point
from spdm.mesh.Mesh import Mesh
from spdm.mesh.mesh_curvilinear import CurvilinearMesh
from spdm.numlib.contours import find_countours
from spdm.numlib.optimize import minimize_filter
from spdm.utils.tags import _not_found_
from spdm.utils.typing import ArrayLike, NumericType, array_type, scalar_type, as_array

from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.logger import logger
from fytok.modules.Utilities import *


PI = scipy.constants.pi

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


@sp_tree
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._B0 = super().get("../vacuum_toroidal_field/b0", np.nan)
        self._R0 = super().get("../vacuum_toroidal_field/r0", np.nan)
        self._Ip = super().get("../global_quantities/ip", np.nan)

        self._s_B0 = np.sign(self._B0)
        self._s_Ip = np.sign(self._Ip)

        self._e_Bp, self._s_Bp, self._s_RpZ, self._s_rtp = COCOS_TABLE[5]

        self._s_eBp_2PI = 1.0 if self._e_Bp == 0 else (2.0 * scipy.constants.pi)

    cocos: int = 5

    @property
    def _root(self) -> Equilibrium.TimeSlice:
        return self._parent

    @functools.cached_property
    def _fpol(self) -> Expression:
        ffprime = as_array(self._root.profiles_1d.f_df_dpsi)

        psi = as_array(self._root.profiles_1d.psi)

        if psi is None or psi is _not_found_:
            psi = np.linspace(0, 1, len(ffprime)) * (self.psi_boundary - self.psi_axis) + self.psi_axis

        return np.sqrt(2.0 * Function(psi, ffprime).I + (self._B0 * self._R0) ** 2)

    @functools.cached_property
    def _psirz(self) -> Field:
        psirz = super().get("psirz", _not_found_, _force=True, _type_hint=np.ndarray)

        if isinstance(psirz, np.ndarray):
            dim1 = super().get("grid/dim1", _not_found_)
            dim2 = super().get("grid/dim2", _not_found_)
            grid_type = super().get("grid/type", "rectangular")

            if not isinstance(dim1, np.ndarray) or not isinstance(dim2, np.ndarray):
                raise RuntimeError(f"Can not create grid!")

            psirz = Field(dim1, dim2, psirz, mesh_type=grid_type, name="psirz")
        elif psirz is _not_found_ or psirz is None:
            psirz = self._parent.profiles_2d.psi
        else:
            logger.warning(f"Ignore {type(psirz)}. Using ../profiles_2d.psi ")

        if not isinstance(psirz, Field):
            raise RuntimeError(f"Can not get psirz! {type(psirz)}")

        return psirz

    @functools.cached_property
    def critical_points(self) -> typing.Tuple[typing.Sequence[OXPoint], typing.Sequence[OXPoint]]:
        opoints = []

        xpoints = []

        psi: Expression = self._psirz

        R, Z = psi.mesh.points

        Bp2 = (psi.pd(0, 1) ** 2 + psi.pd(1, 0) ** 2) / (_R**2)

        D = psi.pd(2, 0) * psi.pd(0, 2) - psi.pd(1, 1) ** 2

        for r, x_z in minimize_filter(Bp2, R, Z):
            p = OXPoint(r, x_z, psi(r, x_z))

            if D(r, x_z) < 0.0:  # saddle/X-point
                xpoints.append(p)
            else:  # extremum/ O-point
                opoints.append(p)

        Rmid, Zmid = self._psirz.mesh.geometry.bbox.origin + self._psirz.mesh.geometry.bbox.dimensions * 0.5

        opoints.sort(key=lambda x: (x.r - Rmid) ** 2 + (x.z - Zmid) ** 2)

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

            psiline = psi(np.linspace(o_r, xp.r, length), np.linspace(o_z, xp.z, length))

            if len(np.unique(psiline[1:] > psiline[:-1])) != 1:
                s_points.append(xp)
            else:
                x_points.append(xp)

        xpoints = x_points

        xpoints.sort(key=lambda x: (x.psi - o_psi) ** 2)

        if len(opoints) == 0 or len(xpoints) == 0:
            raise RuntimeError(f"Can not find O-point or X-point! {opoints} {xpoints}")

        return opoints, xpoints

    @sp_property
    def grid_type(self) -> Identifier:
        desc = super().grid_type
        if desc.name is None or desc.name is _not_found_:
            desc = {"name": "rectangular", "index": 1, "description": "default"}
        return desc

    @sp_property
    def grid(self) -> Mesh:
        psi_norm = super().get("dim1", _not_found_, _type_hint=array_type)

        if psi_norm is _not_found_:
            psi_norm = self.get("../../code/parameters/psi_norm", 128)

        if isinstance(psi_norm, np.ndarray) and psi_norm.ndim == 0:
            psi_norm_boundary = self._parent.boundary.psi_norm
            psi_norm = np.linspace(1.0 - psi_norm_boundary, psi_norm_boundary, int(psi_norm), endpoint=True)
            logger.debug(f"Set psi_norm ({psi_norm[0]},{psi_norm[-1]}) {len(psi_norm)}")

        if isinstance(psi_norm, int):
            psi_norm = np.linspace(0.0, 0.995, psi_norm)  # 0.995 lcfs

        elif not isinstance(psi_norm, np.ndarray):
            raise ValueError(f"Can not create grid! psi_norm={psi_norm}")

        elif np.isclose(psi_norm[0], 0.0) and np.isclose(psi_norm[-1], 1.0):
            logger.warning(
                f"Singular values are caused when psi_norm takes values of 0.0 or 1.0.! {psi_norm[0]} {psi_norm[-1]}"
            )

        theta = super().get("dim2", _not_found_, _type_hint=array_type)

        if theta is _not_found_:
            theta = self.get("../../code/parameters/theta", 64)

        if isinstance(theta, int):
            theta = np.linspace(0, 2.0 * scipy.constants.pi, theta, endpoint=False)
        elif isinstance(theta, np.ndarray) and theta.ndim == 0:
            theta = np.linspace(0, 2.0 * scipy.constants.pi, int(theta), endpoint=False)

        if not (isinstance(theta, np.ndarray) and theta.ndim == 1):
            raise ValueError(f"Can not create grid! theta={theta}")

        surfs = GeoObjectSet([surf for _, surf in self.find_surfaces_by_psi_norm(psi_norm, o_point=True)])

        return CurvilinearMesh(psi_norm, theta, geometry=surfs, cycles=[False, 2.0 * scipy.constants.pi])

    def psirz(self, r: NumericType, z: NumericType) -> array_type:
        return self._psirz(r, z)

    @property
    def psi_axis(self) -> float:
        return self.critical_points[0][0].psi

    @property
    def psi_boundary(self) -> float:
        return self.critical_points[1][0].psi

    @property
    def psi_norm(self) -> array_type:
        return self.grid.dim1

    # @functools.cached_property
    # def psi(self) -> array_type:
    #     return self.psi_norm * (self.psi_boundary-self.psi_axis) + self.psi_axis

    # @property
    # def phi(self) -> array_type: return Function(self.psi,self.dphi_dpsi).I(self.psi)

    # @property
    # def dphi_dpsi(self) -> array_type: return self._fpol * self.surface_integral(1.0/(_R**2), self.psi)

    # @property
    # def rho_tor(self) -> array_type: return np.sqrt(np.abs(self.phi / (PI*self._B0)))

    # @property
    # def rho_tor_norm(self) -> array_type: return np.sqrt(self.phi/self.phi[-1])

    # @property
    # def rho_tor_boundary(self) -> float: return self.rho_tor[-1]

    # @functools.cached_property
    # def magnetic_axis(self) -> typing.Tuple[float, float]:
    #     o_points = self.critical_points[0]
    #     return o_points[0].r, o_points[0].z

    @sp_property
    def r(self) -> Expression:
        return Field(self.grid.points[0], mesh=self.grid)

    @sp_property
    def z(self) -> Expression:
        return Field(self.grid.points[1], mesh=self.grid)

    @sp_property
    def jacobian(self) -> Expression:
        raise NotImplementedError(f"")

    @sp_property
    def tensor_covariant(self) -> Expression:
        raise NotImplementedError(f"")

    @sp_property
    def tensor_contravariant(self) -> Expression:
        raise NotImplementedError(f"")

    def find_surfaces(
        self, psi: float | array_type | typing.Sequence[float], o_point: OXPoint | bool = True
    ) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:
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
                        # theta = ((np.arctan2(_R-o_point.r, _Z-o_point.z)-theta_0)+2.0*scipy.constants.pi) % (2.0*scipy.constants.pi)
                        # surf = surf.remesh(theta)
                        surf.set_coordinates("r", "z")
                        yield psi_val, surf
                    else:
                        count -= 1
                if count <= 0:
                    if np.isclose(psi_val, o_point.psi):
                        yield psi_val, Point(o_point.r, o_point.z)
                    else:
                        # logger.warning(f"{psi_val} {o_point.psi}")
                        yield psi_val, None
                elif current_count > 1:
                    raise RuntimeError(f"Something wrong! Get {current_count} closed surfaces for psi={current_psi}")

                # theta = np.arctan2(surf[:, 0]-o_point.r, surf[:, 1]-o_point.z)
                # logger.debug((max(theta)-min(theta))/(2.0*scipy.constants.pi))
                # if 1.0 - (max(theta)-min(theta))/(2.0*scipy.constants.pi) > 2.0/len(theta):  # open or do not contain o-point
                #     current_count -= 1
                #     continue

                # is_closed = False

                # if np.isclose((theta[0]-theta[-1]) % (2.0*scipy.constants.pi), 0.0):
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
                #     theta = np.hstack([theta, [theta[0]+(2.0*scipy.constants.pi)]])
                #     theta = (theta-theta.min())/(theta.max()-theta.min())
                #     surf = np.vstack([surf, surf[:1]])

                # if surf.shape[0] == 0:
                #     logger.warning(f"{level},{o_point.psi},{(max(theta),min(theta))}")

                # elif surf.shape[0] == 1:
                #     yield level, Point(surf[0][0], surf[0][1])
                # else:
                #     yield level, Curve(surf, theta, is_closed=is_closed)

    def find_surfaces_by_psi_norm(
        self, psi_norm: float | array_type | typing.Sequence[float], *args, **kwargs
    ) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:
        psi_axis = self.psi_axis

        psi_boundary = self.psi_boundary

        if isinstance(psi_norm, (collections.abc.Sequence, np.ndarray)):
            psi = np.asarray(psi_norm, dtype=float) * (psi_boundary - psi_axis) + psi_axis
            yield from self.find_surfaces(psi, *args, **kwargs)
        elif isinstance(psi_norm, collections.abc.Generator):
            for psi_n in psi_norm:
                yield from self.find_surfaces(psi_n * (psi_boundary - psi_axis) + psi_axis, *args, **kwargs)

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
                psi, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard
            )
        else:
            return FyEquilibriumCoordinateSystem.ShapeProperty(
                psi,
                Function(psi, rmin, name="rmin"),
                Function(psi, zmin, name="zmin"),
                Function(psi, rmax, name="rmax"),
                Function(psi, zmax, name="zmax"),
                Function(psi, rzmin, name="rzmin"),
                Function(psi, rzmax, name="rzmax"),
                Function(psi, r_inboard, name="r_inboard"),
                Function(psi, r_outboard, name="r_outboard"),
            )

    #################################
    # fields
    @property
    def Bpol2(self) -> Expression:
        return self.b_field_r**2 + self.b_field_z**2

    r""" $B_{pol}= \left|\nabla \psi \right|/2 \pi R $ """

    @property
    def Bpol(self) -> Expression:
        return np.sqrt(self.b_field_r**2 + self.b_field_z**2)

    r""" $B_{pol}= \left|\nabla \psi \right|/2 \pi R $ """

    @property
    def b_field_r(self) -> Expression:
        """COCOS Eq.19 [O. Sauter and S.Yu. Medvedev, Computer Physics Communications 184 (2013) 293]"""
        return self._psirz.pd(0, 1) / _R * (self._s_RpZ * self._s_Bp / self._s_eBp_2PI)

    @property
    def b_field_z(self) -> Expression:
        return -self._psirz.pd(1, 0) / _R * (self._s_RpZ * self._s_Bp / self._s_eBp_2PI)

    @property
    def b_field_tor(self) -> Expression:
        return self._fpol(self._psirz) / _R

    @property
    def B2(self) -> Expression:
        return self.b_field_r**2 + self.b_field_z**2 + self.b_field_tor**2

    @property
    def grad_psi2(self) -> Expression:
        return self._psirz.pd(1, 0) ** 2 + self._psirz.pd(0, 1) ** 2

    @property
    def grad_psi(self) -> Expression:
        return np.sqrt(self.grad_psi2)

    @property
    def ddpsi(self) -> Expression:
        return np.sqrt(self._psirz.pd(2, 0) * self._psirz.pd(0, 2) + self._psirz.pd(1, 1) ** 2)

    @functools.cached_property
    def dvolume_dpsi(self) -> Expression:
        return Expression(*self._surface_integral(1.0), name="dvolume_dpsi", label=r"\frac{d volume}{d\psi}")

    ###############################
    # surface integral

    def _surface_integral(self, func: Expression, psi: NumericType = None) -> typing.Tuple[ArrayLike, ArrayLike]:
        r"""
        $ V^{\prime} =  2 \pi  \int{ R / \left|\nabla \psi \right| * dl }$
        $ V^{\prime}(psi)= 2 \pi  \int{ dl * R / \left|\nabla \psi \right|}$
        """

        # r0, z0 = self.magnetic_axis

        if psi is None:
            psi = self.grid.dim1 * (self.psi_boundary - self.psi_axis) + self.psi_axis
            surfs_list = zip(psi, self.grid.geometry)
        else:
            if isinstance(psi, scalar_type):
                psi = [psi]
            surfs_list = self.find_surfaces(psi, o_point=True)

        psi = []
        res = []
        for p, surf in surfs_list:
            if isinstance(surf, Curve):
                v = surf.integral(func / self.Bpol)
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
            return psi, res
        else:
            return psi[0], res[0]

        # return np.asarray([(axis.integral(func/self.Bpol) if not np.isclose(p, 0) else func(r0, z0) * c0) for p, axis in surfs_list], dtype=float)

    def surface_integral(self, func: Expression, psi: NumericType = None) -> Expression | float:
        psi, value = self._surface_integral(func, psi)

        if np.isscalar(psi):
            return value
        else:
            return Expression(
                psi,
                value,
                name=f"surface_integral({func.__label__})",
                label=rf"<{func.__repr__()}>",
            )

    def surface_average(self, func: Expression, *xargs) -> Expression | ArrayLike:
        r"""
        $\left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}$
        """
        return self.surface_integral(func, *xargs) / self.dvolume_dpsi(*xargs)


class FyEquilibriumGlobalQuantities(Equilibrium.TimeSlice.GlobalQuantities):
    @property
    def _coord(self) -> FyEquilibriumCoordinateSystem:
        return self._parent.coordinate_system

    @sp_property
    def psi_axis(self) -> float:
        return self._coord.critical_points[0][0].psi

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord.critical_points[1][0].psi

    @sp_property
    def magnetic_axis(self) -> Equilibrium.TimeSlice.GlobalQuantities.MagneticAxis:
        """Magnetic axis position and toroidal field"""
        opoint = self._coord.critical_points[0][0]
        r = opoint.r
        z = opoint.z
        return {"r": r, "z": z, "b_field_tor": self._coord.b_field_tor(r, z)}


@sp_tree
class FyEquilibriumProfiles1D(Equilibrium.TimeSlice.Profiles1D):
    @property
    def _coord(self) -> FyEquilibriumCoordinateSystem:
        return self._parent.coordinate_system

    psi_norm: array_type

    psi: Expression

    f_df_dpsi: Expression

    ffprime: Expression = sp_property(alias="f_df_dpsi")

    @sp_property
    def fpol(self) -> Expression:
        return np.sqrt(
            2.0 * self.f_df_dpsi.I + (self._root.vacuum_toroidal_field.r0 * self._root.vacuum_toroidal_field.b0) ** 2
        )

    @sp_property
    def dphi_dpsi(self) -> Expression:
        return self.fpol * self._coord.surface_integral(1.0 / (_R**2))

    @sp_property
    def q(self) -> Expression:
        return self.dphi_dpsi * (
            self._coord._s_Bp * self._coord._s_rtp * self._coord._s_eBp_2PI / (2.0 * scipy.constants.pi)
        )

    @sp_property(label=r"\phi")
    def phi(self) -> Expression:
        r"""$\Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi$"""
        return self.dphi_dpsi.I

    @sp_property
    def magnetic_shear(self) -> Expression:
        return self.rho_tor * self.q.d() / self.q * self.dpsi_drho_tor

    @sp_property(label=r"\rho_{tor}")
    def rho_tor(self) -> Expression:
        return np.sqrt(self.phi / np.abs(PI * self._root.vacuum_toroidal_field.b0))

    @sp_property(label=r"\bar{\rho}_{tor}")
    def rho_tor_norm(self) -> Expression:
        return np.sqrt(self.phi / self.phi(self._root.global_quantities.psi_boundary))

    @sp_property
    def drho_tor_dpsi(self) -> Expression:
        r"""
        $\frac{d\rho_{tor}}{d\psi} 
            =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
            =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
            =\frac{q}{2\pi B_{0}\rho_{tor}}
        $
        """
        return 1.0 / self.dpsi_drho_tor

    @sp_property
    def dpsi_drho_tor(self) -> Expression:
        return np.abs(self._root.vacuum_toroidal_field.b0) * self.rho_tor / self.q

    @sp_property
    def volume(self) -> Expression:
        return self.dvolume_dpsi.I

    @sp_property
    def dvolume_dpsi(self) -> Expression:
        return self._coord.dvolume_dpsi

    @sp_property(label=r"\frac{dV}{d\rho_{tor}}")
    def dvolume_drho_tor(self) -> Expression:
        return np.abs(
            self._coord._s_eBp_2PI * self._root.vacuum_toroidal_field.b0 * self.dvolume_dpsi * self.dpsi_drho_tor
        )

    @sp_property
    def area(self) -> Expression:
        return self.darea_dpsi.I

    @sp_property
    def darea_dpsi(self) -> Expression:
        """FIXME: just a simple approximation!"""
        return self.dvolume_dpsi / ((2.0 * scipy.constants.pi) * self._coord._R0)

    @sp_property
    def darea_drho_tor(self) -> Expression:
        return self.darea_dpsi * self.dpsi_drho_tor

    @sp_property
    def surface(self) -> Expression:
        return self.dvolume_drho_tor * self.gm7

    @sp_property
    def dphi_dvolume(self) -> Expression:
        return self.fpol * self.gm1

    @sp_property
    def gm1(self) -> Expression:
        return self._coord.surface_average(1.0 / (_R**2))

    @sp_property
    def gm2(self) -> Expression:
        return self._coord.surface_average(self._coord.grad_psi2 / (_R**2)) / (self.dpsi_drho_tor**2)

    @sp_property
    def gm3(self) -> Expression:
        return self._coord.surface_average(self._coord.grad_psi2) / (self.dpsi_drho_tor**2)

    @sp_property
    def gm4(self) -> Expression:
        return self._coord.surface_average(1.0 / self._coord.B2)

    @sp_property
    def gm5(self) -> Expression:
        return self._coord.surface_average(self._coord.B2)

    @sp_property
    def gm6(self) -> Expression:
        return self._coord.surface_average(self._coord.grad_psi2 / self._coord.B2) / (self.dpsi_drho_tor**2)

    @sp_property
    def gm7(self) -> Expression:
        return self._coord.surface_average(np.sqrt(self._coord.grad_psi2)) / self.dpsi_drho_tor

    @sp_property
    def gm8(self) -> Expression:
        return self._coord.surface_average(_R)

    @sp_property
    def gm9(self) -> Expression:
        return self._coord.surface_average(1.0 / _R)

    @sp_property
    def dpsi_drho_tor_norm(self) -> Expression:
        return self.dpsi_drho_tor * self.rho_tor(self._root.global_quantities.psi_boundary)

    # 描述磁面形状
    @functools.cached_property
    def _shape_property(self) -> FyEquilibriumCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi)

    @sp_property
    def minor_radius(self) -> Expression:
        return (self._shape_property.Rmax - self._shape_property.Rmin) * 0.5

    @sp_property
    def major_radius(self) -> Expression:
        return (self._shape_property.Rmax + self._shape_property.Rmin) * 0.5

    @sp_property
    def magnetic_z(self) -> Expression:
        return (self._shape_property.Zmax + self._shape_property.Zmin) * 0.5

    @sp_property
    def r_inboard(self) -> Expression:
        return self._shape_property.r_inboard

    @sp_property
    def r_outboard(self) -> Expression:
        return self._shape_property.r_outboard

    @sp_property
    def elongation(self) -> Expression:
        return (self._shape_property.Zmax - self._shape_property.Zmin) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def elongation_upper(self) -> Expression:
        return (self._shape_property.Zmax - (self._shape_property.Zmax + self._shape_property.Zmin) * 0.5) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def elongation_lower(self) -> Expression:
        return ((self._shape_property.Zmax + self._shape_property.Zmin) * 0.5 - self._shape_property.Zmin) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def triangularity_upper(self) -> Expression:
        return (
            ((self._shape_property.Rmax - self._shape_property.Rmin) * 0.5 - self._shape_property.Rzmax)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def triangularity_lower(self) -> Expression:
        return (
            ((self._shape_property.Rmax + self._shape_property.Rmin) * 0.5 - self._shape_property.Rzmin)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def triangularity(self) -> Expression:
        psi = self.grid.psi[1:]
        res = (
            (self._shape_property.Rzmax(psi) - self._shape_property.Rzmin(psi))
            / (self._shape_property.Rmax(psi) - self._shape_property.Rmin(psi))
            * 2
        )
        return Function(psi, res, name="triangularity")

    @sp_property
    def squareness(self) -> Expression:
        return 0.0

    @sp_property
    def trapped_fraction(self) -> Expression:
        """Trapped particle fraction[-]
        Tokamak 3ed, 14.10
        """
        epsilon = self.rho_tor(self.psi) / self._coord._R0
        return 1.0 - (1 - epsilon) ** 2 / np.sqrt(1.0 - epsilon**2) / (1 + 1.46 * np.sqrt(epsilon))

    # 关于 plasma current和 压强，  pprime & J
    dpressure_dpsi: Expression

    pprime: Expression = sp_property(alias="dpressure_dpsi")

    @sp_property
    def plasma_current(self) -> Expression:
        return self.gm2 * self.dvolume_drho_tor / self.dpsi_drho_tor / scipy.constants.mu_0

    @sp_property
    def j_tor(self) -> Expression:
        return self.plasma_current.d() / self.dvolume_dpsi * self._root.vacuum_toroidal_field.r0

    @sp_property
    def j_parallel(self) -> Expression:
        return self._coord.surface_average(dot(self._coord.j, self._coord.B) / np.sqrt(self._coord.B2))

        # fvac = self._coord._fvac
        # d = np.asarray(function_like(np.asarray(self.volume),
        #                              np.asarray(fvac*self.plasma_current/self.fpol)).pd())
        # return self._coord._R0*(self.fpol / fvac)**2 * d


@sp_tree(mesh="grid")
class FyEquilibriumProfiles2D(Equilibrium.TimeSlice.Profiles2D):
    @property
    def _coord(self) -> FyEquilibriumCoordinateSystem:
        return self._parent.coordinate_system

    @property
    def _profiles_1d(self) -> FyEquilibriumProfiles1D:
        return self._root.profiles_1d

    @sp_property
    def grid(self) -> Mesh:
        dim1 = super().grid.dim1
        dim2 = super().grid.dim2
        mesh_type = super().grid_type.name
        return Mesh(dim1, dim2, type=mesh_type)

    @sp_property
    def r(self) -> array_type:
        return self.grid.points[0]

    @sp_property
    def z(self) -> array_type:
        return self.grid.points[1]

    psi: Field

    @property
    def psi_norm(self) -> Expression:
        return (super().psi - self._coord.psi_axis) / (self._coord.psi_boundary - self._coord.psi_axis)

    @sp_property
    def phi(self) -> Expression:
        return super().phi

    @sp_property
    def theta(self) -> Expression:
        return super().theta

    @sp_property
    def j_tor(self) -> Expression:
        return _R * self._profiles_1d.pprime(self.psi) + self._profiles_1d.ffprime(self.psi) / (
            _R * scipy.constants.mu_0
        )
        # return super().j_tor  # return self._profiles_1d.j_tor(self.psi)

    @sp_property
    def j_parallel(self) -> Expression:
        return super().j_parallel  # return self._profiles_1d.j_parallel(self.psi)

    @sp_property
    def b_field_r(self) -> Expression:
        """COCOS Eq.19 [O. Sauter and S.Yu. Medvedev, Computer Physics Communications 184 (2013) 293]"""
        return self.psi.pd(0, 1) / _R * (self._coord._s_RpZ * self._coord._s_Bp / self._coord._s_eBp_2PI)

    @sp_property
    def b_field_z(self) -> Expression:
        return -self.psi.pd(1, 0) / _R * (self._coord._s_RpZ * self._coord._s_Bp / self._coord._s_eBp_2PI)

    @sp_property
    def b_field_tor(self) -> Expression:
        return self._root.profiles_1d.f(self.psi) / _R


class FyEquilibriumBoundary(Equilibrium.TimeSlice.Boundary):
    @property
    def _coord(self) -> FyEquilibriumCoordinateSystem:
        return self._parent.coordinate_system

    @sp_property(coordinates="r z")
    def outline(self) -> Curve:
        _, surf = next(self._coord.find_surfaces(self.psi, o_point=True))
        return surf

    psi_norm: float = sp_property(default_value=0.999)

    @sp_property
    def psi(self) -> float:
        g: Equilibrium.TimeSlice.GlobalQuantities = self._parent.global_quantities
        return self.psi_norm * (g.psi_boundary - g.psi_axis) + g.psi_axis

    @sp_property
    def phi(self) -> float:
        return np.nan  # raise NotImplementedError(f"{self.__class__.__name__}.phi")

    @sp_property
    def rho(self) -> float:
        return np.sqrt(self.phi / (scipy.constants.pi * self._coord._B0))

    @functools.cached_property
    def _shape_property(self) -> FyEquilibriumCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi)

    @sp_property
    def geometric_axis(self) -> Point:
        return {
            "r": (self._shape_property.Rmin + self._shape_property.Rmax) * 0.5,
            "z": (self._shape_property.Zmin + self._shape_property.Zmax) * 0.5,
        }

    @sp_property
    def minor_radius(self) -> float:
        return (self._shape_property.Rmax - self._shape_property.Rmin) * 0.5

    @sp_property
    def elongation(self) -> float:
        return (self._shape_property.Zmax - self._shape_property.Zmin) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def elongation_upper(self) -> float:
        return (self._shape_property.Zmax - (self._shape_property.Zmax + self._shape_property.Zmin) * 0.5) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def elongation_lower(self) -> float:
        return ((self._shape_property.Zmax + self._shape_property.Zmin) * 0.5 - self._shape_property.Zmin) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property(coordinate1="../psi")
    def triangularity(self) -> float:
        return (
            (self._shape_property.Rzmax - self._shape_property.Rzmin)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def triangularity_upper(self) -> float:
        return (
            ((self._shape_property.Rmax + self._shape_property.Rmin) * 0.5 - self._shape_property.Rzmax)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def triangularity_lower(self) -> float:
        return (
            ((self._shape_property.Rmax + self._shape_property.Rmin) * 0.5 - self._shape_property.Rzmin)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def x_point(self) -> List[OXPoint]:
        _, x_pt = self._coord.critical_points
        return x_pt

    @sp_property
    def strike_point(self) -> List[OXPoint]:
        return

    @sp_property
    def active_limiter_point(self) -> List[Point]:
        return NotImplemented


class FyEquilibriumBoundarySeparatrix(Equilibrium.TimeSlice.BoundarySeparatrix):
    @property
    def _coord(self) -> FyEquilibriumCoordinateSystem:
        return self._parent.coordinate_system  # type:ignore

    @sp_property(coordinates="r z")
    def outline(self) -> Curve:
        """RZ outline of the plasma boundary"""
        _, surf = next(self._coord.find_surfaces(self.psi, o_point=True))
        if surf is None:
            return _not_found_
        else:
            return surf

    @sp_property
    def magnetic_axis(self) -> float:
        return self._coord.psi_axis

    @sp_property
    def psi(self) -> float:
        return self._coord.psi_boundary

    @sp_property
    def x_point(self) -> List[OXPoint]:
        _, x = self._coord.critical_points
        return List[OXPoint]([(v.r, v.z, v.psi) for v in x[:]])

    @sp_property
    def strike_point(self) -> List[Point]:
        raise NotImplementedError("TODO: strike_point")


@sp_tree
class FyEquilibriumTimeSlice(Equilibrium.TimeSlice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    profiles_1d: FyEquilibriumProfiles1D

    profiles_2d: FyEquilibriumProfiles2D

    global_quantities: FyEquilibriumGlobalQuantities

    boundary: FyEquilibriumBoundary

    boundary_separatrix: FyEquilibriumBoundarySeparatrix

    coordinate_system: FyEquilibriumCoordinateSystem


@sp_tree
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

    code: Code = {"name": "fy_eq", "copyright": "fytok"}

    TimeSlice = FyEquilibriumTimeSlice

    time_slice: TimeSeriesAoS[FyEquilibriumTimeSlice]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


Equilibrium.register(["fy_eq"], FyEqAnalyze)
