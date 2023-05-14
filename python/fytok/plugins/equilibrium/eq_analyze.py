import collections
import collections.abc
import typing
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import isclose

import numpy as np
from fytok._imas.lastest.equilibrium import (
    _T_equilibrium_boundary, _T_equilibrium_boundary_separatrix,
    _T_equilibrium_coordinate_system, _T_equilibrium_global_quantities,
    _T_equilibrium_global_quantities_magnetic_axis, _T_equilibrium_profiles_1d,
    _T_equilibrium_profiles_2d, _T_equilibrium_time_slice)
from fytok._imas.lastest.utilities import _T_identifier_dynamic_aos3
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.Utilities import RZTuple, RZTuple1D
from scipy import constants
from spdm.data.Dict import Dict
from spdm.data.Field import Field
from spdm.data.Function import Function, _0, _1, _2, function_like
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.Profile import Profile
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.geometry.CubicSplineCurve import CubicSplineCurve
from spdm.geometry.GeoObject import GeoObject, GeoObjectSet
from spdm.geometry.Point import Point
from spdm.mesh.CurvilinearMesh import CurvilinearMesh
from spdm.mesh.Mesh import Mesh
from spdm.mesh.RectilinearMesh import RectilinearMesh
from spdm.numlib.contours import find_countours
from spdm.numlib.optimize import find_critical_points
from spdm.utils.logger import logger
from spdm.utils.misc import convert_to_named_tuple
from spdm.utils.tags import _not_found_
from spdm.utils.typing import ArrayType, NumericType

_R = _0
_Z = _1

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

TWOPI = 2.0*constants.pi

COCOS_INDEX = 11

COCOS_TABLE = [
    # e_Bp ,    $\sigma_{Bp}$,    $\sigma_{R\varphi\Z}$
    None,                             # 0
    (1,         +1,             +1),  # 1
    (1,         +1,             -1),  # 2
    (1,         -1,             +1),  # 3
    (1,         -1,             -1),  # 4
    (1,         +1,             +1),  # 5
    (1,         +1,             -1),  # 6
    (1,         -1,             +1),  # 7
    (1,         -1,             -1),  # 8
    None,                             # 9
    None,                             # 10
    (TWOPI,     +1,             +1),  # 11
    (TWOPI,     +1,             -1),  # 12
    (TWOPI,     -1,             +1),  # 13
    (TWOPI,     -1,             -1),  # 14
    (TWOPI,     +1,             +1),  # 15
    (TWOPI,     +1,             -1),  # 16
    (TWOPI,     -1,             +1),  # 17
    (TWOPI,     -1,             -1),  # 18
]


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

        self._s_Bp = np.sign(self._B0)
        self._s_2PI = TWOPI

    # @property
    # def cocos(self) -> int: return self._cocos

    # @cached_property
    # def cocos_flag(self) -> int: return 1 if self.psi_boundary > self.psi_magnetic_axis else -1

    @cached_property
    def _psirz(self) -> Field[float]:
        psirz = super().get("psirz", _not_found_)

        if isinstance(psirz, np.ndarray):
            dim1 = super().get("grid/dim1", _not_found_)
            dim2 = super().get("grid/dim2", _not_found_)
            grid_type = super().get("grid/type", "rectangular")

            if not isinstance(dim1, np.ndarray) or not isinstance(dim2, np.ndarray):
                raise RuntimeError(f"Can not create grid!")

            psirz = Field(psirz, dim1, dim2)
        elif psirz is _not_found_:
            psirz = self._parent.profiles_2d[0].psi
        else:
            logger.warning(f"Ignore {type(psirz)}. Using ../profiles_2d[0].psi ")

        if not isinstance(psirz, Field):
            raise RuntimeError(f"Can not get psirz!")

        return psirz

    @cached_property
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

            bbox = self._psirz.bbox
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
    def psi_norm(self) -> ArrayType: return self.mesh.dim1

    @cached_property
    def psi(self) -> ArrayType: return self.psi_norm * self.psi_delta + self.psi_magnetic_axis

    def psirz(self, r: NumericType, z: NumericType, grid=False, **kwargs) -> NumericType:
        return self._psirz(r, z, grid=grid, **kwargs)

    @cached_property
    def magnetic_axis(self) -> typing.Tuple[float, float]:
        o_points = self.critical_points[0]
        return o_points[0].r, o_points[0].z

    @cached_property
    def psi_bc(self) -> typing.Tuple[float, float]:
        o, x = self.critical_points
        psi_magnetic_axis = o[0].psi
        if len(x) == 0:
            raise RuntimeError(f"Can not find x-point")
        psi_boundary = x[0].psi
        return psi_magnetic_axis, psi_boundary

    @cached_property
    def psi_delta(self) -> float: return self.psi_bc[1]-self.psi_bc[0]

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

    @sp_property
    def grid(self) -> Mesh:
        psi_norm = super().mesh.dim1
        if isinstance(psi_norm, np.ndarray) and psi_norm.ndim == 1:
            pass
        elif isinstance(psi_norm, np.ndarray) and psi_norm.ndim == 0:
            psi_norm_boundary = self._parent.boundary.psi_norm
            psi_norm = np.linspace(0, psi_norm_boundary, int(psi_norm), endpoint=True)
        elif isinstance(psi_norm, collections.abc.Sequence) and len(psi_norm) == 3:
            psi_norm = np.linspace(psi_norm[0], psi_norm[1], psi_norm[2], endpoint=True)
        else:
            raise ValueError(f"Can not create grid! psi_norm={psi_norm}")

        theta = super().mesh.dim2
        if isinstance(theta, np.ndarray) and theta.ndim == 1:
            pass
        elif isinstance(theta, np.ndarray) and theta.ndim == 0:
            theta = np.linspace(0, TWOPI, int(theta), endpoint=False)
        elif isinstance(theta, collections.abc.Sequence) and len(theta) == 3:
            theta = np.linspace(theta[0], theta[1], int(theta[2]), endpoint=False)
        else:
            raise ValueError(f"Can not create grid! theta={theta}")

        surfs = GeoObjectSet([surf for _, surf in self.find_surface_by_psi_norm(psi_norm, o_point=True)])

        return CurvilinearMesh(psi_norm, theta, geometry=surfs, cycle=[False, TWOPI])

    @sp_property
    def r(self) -> Field[float]: return Field(self.mesh.points[0], grid=self.grid)

    @sp_property
    def z(self) -> Field[float]: return Field(self.mesh.points[1], grid=self.grid)

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
        # R, Z = self._psirz.mesh.points
        # F = np.asarray(self._psirz)
        # if not isinstance(psi, (collections.abc.Sequence, np.ndarray)):
        #     psi = [psi]
        # psi = np.asarray(psi, dtype=float)

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

            for level, points in find_countours(self._psirz, levels=psi):

                theta = np.arctan2(points[:, 0]-o_point.r, points[:, 1]-o_point.z)

                if 1.0 - (max(theta)-min(theta))/TWOPI > 2.0/len(theta):  # open or do not contain o-point
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

                # if surf is None:
                #     if np.isclose(level, o_point.psi):
                #         yield level, Point(o_point.r, o_point.z)
                #     else:
                #         raise RuntimeError(f"{level},{o_point.psi},{(max(theta),min(theta))}")

    def find_surface_by_psi_norm(self, psi_norm: float | ArrayType | typing.Sequence[float], *args,   **kwargs) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:

        psi_magnetic_axis, psi_boundary = self.psi_bc

        if isinstance(psi_norm, (collections.abc.Sequence, np.ndarray)):
            yield from self.find_surface(np.asarray(psi_norm, dtype=float)*(psi_boundary-psi_magnetic_axis)+psi_magnetic_axis, *args,  **kwargs)
        elif isinstance(psi_norm, collections.abc.Generator):
            for psi in psi_norm:
                yield from self.find_surface_by_psi_norm(psi*(psi_boundary-psi_magnetic_axis)+psi_magnetic_axis, *args,  **kwargs)

    ###############################

    @dataclass
    class ShapeProperty:
        # RZ position of the geometric axis of the magnetic surfaces (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the surface)
        geometric_axis: RZTuple
        # Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary)[m]
        minor_radius: np.ndarray  # (rmax - rmin)*0.5,
        # Elongation of the plasma boundary. [-]
        elongation: np.ndarray  # (zmax-zmin)/(rmax-rmin),
        # Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]
        elongation_upper: np.ndarray  # (zmax-(zmax+zmin)*0.5)/(rmax-rmin),
        # longation(lower half w.r.t. geometric axis) of the plasma boundary. [-]
        elongation_lower: np.ndarray  # ((zmax+zmin)*0.5-zmin)/(rmax-rmin),
        # Triangularity of the plasma boundary. [-]
        triangularity: np.ndarray  # (rzmax-rzmin)/(rmax - rmin)*2,
        # Upper triangularity of the plasma boundary. [-]
        triangularity_upper: np.ndarray  # ((rmax+rmin)*0.5 - rzmax)/(rmax - rmin)*2,
        # Lower triangularity of the plasma boundary. [-]
        triangularity_lower: np.ndarray  # ((rmax+rmin)*0.5 - rzmin)/(rmax - rmin)*2,
        # Radial coordinate(major radius) on the inboard side of the magnetic axis[m]
        r_inboard: np.ndarray  # r_inboard,
        # Radial coordinate(major radius) on the outboard side of the magnetic axis[m]
        r_outboard: np.ndarray  # r_outboard,

    def shape_property(self, psi_norm: typing.Union[float, typing.Sequence[float]] = None) -> ShapeProperty:
        def shape_box(s: GeoObject):
            r, z = s.xyz
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
                (rmin, rmax), (zmin, zmax) = s.bbox
                rzmin = r[np.argmin(z)]
                rzmax = r[np.argmax(z)]
                r_inboard = s.points(0.5)[0]
                r_outboard = s.points(0)[0]
            return rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard

        if psi_norm is None:
            psi_norm = self.psi_norm
        elif not isinstance(psi_norm, (np.ndarray, collections.abc.MutableSequence)):
            psi_norm = [psi_norm]

        sbox = np.asarray([[*shape_box(s)] for _, s in self.find_surface_by_psi_norm(psi_norm)], dtype=float)

        if sbox.shape[0] == 1:
            rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox[0]
        else:
            rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox.T
        if isinstance(rmax, np.ndarray) and np.isclose(rmax[0], rmin[0]):
            return EquilibriumCoordinateSystem.ShapeProperty(
                {
                    # RZ position of the geometric axis of the magnetic surfaces (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the surface)
                    "rz": RZTuple1D({"r": (rmin+rmax)*0.5, "z": (zmin+zmax)*0.5, }),
                    # Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary)[m]
                    "minor_radius": (rmax - rmin)*0.5,  #
                    # Elongation of the plasma boundary. [-]
                    "elongation":  np.hstack([(zmax[1]-zmin[1])/(rmax[1]-rmin[1]), (zmax[1:]-zmin[1:])/(rmax[1:]-rmin[1:])]),
                    # Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]
                    "elongation_upper":  np.hstack([0, (zmax[1:]-(zmax[1:]+zmin[1:])*0.5)/(rmax[1:]-rmin[1:])]),
                    # longation(lower half w.r.t. geometric axis) of the plasma boundary. [-]
                    "elongation_lower": np.hstack([0, ((zmax[1:]+zmin[1:])*0.5-zmin[1:])/(rmax[1:]-rmin[1:])]),
                    # Triangularity of the plasma boundary. [-]
                    "triangularity": np.hstack([0, (rzmax[1:]-rzmin[1:])/(rmax[1:] - rmin[1:])*2]),
                    # Upper triangularity of the plasma boundary. [-]
                    "triangularity_upper": np.hstack([0, ((rmax[1:]+rmin[1:])*0.5 - rzmax[1:])/(rmax[1:] - rmin[1:])*2]),
                    # Lower triangularity of the plasma boundary. [-]
                    "triangularity_lower": np.hstack([0, ((rmax[1:]+rmin[1:])*0.5 - rzmin[1:])/(rmax[1:] - rmin[1:])*2]),
                    # Radial coordinate(major radius) on the inboard side of the magnetic axis[m]
                    "r_inboard": r_inboard,  #
                    # Radial coordinate(major radius) on the outboard side of the magnetic axis[m]
                    "r_outboard": r_outboard,  #
                }
            )
        else:
            return EquilibriumCoordinateSystem.ShapeProperty(
                # RZ position of the geometric axis of the magnetic surfaces (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the surface)
                RZTuple({"r": (rmin+rmax)*0.5, "z": (zmin+zmax)*0.5}),
                # Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary)[m]
                (rmax - rmin)*0.5,  # "minor_radius":
                # Elongation of the plasma boundary. [-]
                (zmax-zmin)/(rmax-rmin),  # "elongation":
                # Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]
                (zmax-(zmax+zmin)*0.5)/(rmax-rmin),  # "elongation_upper":
                # longation(lower half w.r.t. geometric axis) of the plasma boundary. [-]
                ((zmax+zmin)*0.5-zmin)/(rmax-rmin),  # elongation_lower":
                # Triangularity of the plasma boundary. [-]
                (rzmax-rzmin)/(rmax - rmin)*2,  # "triangularity":
                # Upper triangularity of the plasma boundary. [-]
                ((rmax+rmin)*0.5 - rzmax)/(rmax - rmin)*2,  # "triangularity_upper":
                # Lower triangularity of the plasma boundary. [-]
                ((rmax+rmin)*0.5 - rzmin)/(rmax - rmin)*2,  # "triangularity_lower":
                # Radial coordinate(major radius) on the inboard side of the magnetic axis[m]
                r_inboard,  # "r_inboard":
                # Radial coordinate(major radius) on the outboard side of the magnetic axis[m]
                r_outboard,  # "r_outboard":
            )

    #################################
    # fields

    @property
    def Bpol(self) -> Field[float]: return self.grad_psi / _R / (TWOPI)
    r""" $B_{pol}= \left|\nabla \psi \right|/2 \pi R $ """

    @property
    def B2(self) -> Field[float]: return (self.b_field_r ** 2+self.b_field_z ** 2 + self.b_field_tor ** 2)

    @property
    def grad_psi2(self) -> Field[float]: return self._psirz.pd(1, 0)**2+self._psirz.pd(0, 1)**2

    @property
    def grad_psi(self) -> Field[float]: return np.sqrt(self.grad_psi2)

    @property
    def ddpsi(self) -> Field[float]: return np.sqrt(self._psirz.pd(2, 0)
                                                    * self._psirz.pd(0, 2) + self._psirz.pd(1, 1)**2)

    @cached_property
    def dvolume_dpsi(self) -> np.ndarray: return self._surface_integral()
    ###############################
    # surface integral

    def _surface_integral(self, func: Function = None, psi_norm=None) -> np.ndarray:
        r"""
            $ V^{\prime} =  2 \pi  \int{ R / \left|\nabla \psi \right| * dl }$
            $ V^{\prime}(psi)= 2 \pi  \int{ dl * R / \left|\nabla \psi \right|}$
        """
        r0, z0 = self.magnetic_axis

        ddpsi = self.ddpsi(r0, z0)

        c0 = TWOPI*r0**2/ddpsi

        if psi_norm is None:
            surfs_list = self.mesh.axis_iter()
        else:
            surfs_list = [*self.find_surface_by_psi_norm(psi_norm, o_point=True)]

        if func is None:
            func = 1.0

        return np.asarray([(axis.integral(func/self.Bpol) if not np.isclose(p, 0) else func(r0, z0) * c0) for p, axis in surfs_list], dtype=float)

    def surface_average(self,  func,  psi: float | typing.Sequence[float] = None, extrapolate_left=False, ** kwargs) -> np.ndarray:
        r"""
            $\left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}$
        """
        if psi is None:
            psi = self.psi
        res = self._surface_integral(func, psi)/self.dvolume_dpsi

        if isinstance(psi, np.ndarray) and extrapolate_left:
            res[0] = res[1]+(res[1]-res[2])/(psi[1]-psi[2])*(psi[0]-psi[1])

        return res


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


class EquilibriumProfiles1D(_T_equilibrium_profiles_1d):

    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

    ###############################
    # 1-D

    @property
    def psi_norm(self) -> ArrayType: return self._coord.psi_norm

    @sp_property
    def psi(self) -> ArrayType: return self._coord.psi

    @sp_property
    def fpol(self) -> Profile[float]: return super().f

    @sp_property
    def f_df_dpsi(self) -> Profile[float]: return self.f * self.f.pd()

    # @sp_property
    # def pprime(self) -> Profile[float]:
    #     """Diamagnetic function (F=R B_Phi)  [T.m]."""
    #     return self._pprime(self.psi_norm)

    @sp_property
    def dphi_dpsi(self) -> Profile[float]: return self.f * self.gm1 * self.dvolume_dpsi / TWOPI

    @sp_property
    def q(self) -> Profile[float]: return self.dphi_dpsi * self._coord._s_Bp * self._coord._s_2PI

    @sp_property
    def magnetic_shear(self) -> Profile[float]: return self.rho_tor/self.q * \
        function_like(self.q, self.rho_tor).pd()(self.rho_tor)

    @sp_property
    def phi(self) -> Profile[float]:
        r"""  $\Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi$    """
        return function_like(self.dphi_dpsi, self.psi).antiderivative(self.psi)

    @sp_property
    def rho_tor(self) -> Profile[float]: return np.sqrt(self.phi/(constants.pi * self._parent._B0))

    @sp_property
    def rho_tor_norm(self) -> Profile[float]: return np.sqrt(self.phi/self._parent.boundary.phi)

    @sp_property
    def volume(self) -> Profile[float]: return function_like(self.dvolume_dpsi, self.psi).antiderivative(self.psi)

    @sp_property
    def surface(self) -> Profile[float]: return self.dvolume_drho_tor*self.gm7

    @sp_property
    def dvolume_drho_tor(self) -> Profile[float]: return (TWOPI**2) * self.rho_tor / \
        (self.gm1)/(self._coord._R0*self._coord._B0/self.fpol)/self._coord._R0

    @sp_property
    def drho_tor_dpsi(self) -> Profile[float]: return 1.0/self.dpsi_drho_tor
    r"""
        $\frac{d\rho_{tor}}{d\psi} =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                    =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                    =\frac{q}{2\pi B_{0}\rho_{tor}}
        $
    """

    @sp_property
    def dpsi_drho_tor(self) -> Profile[float]: return (self._coord._s_Bp)*self._coord._B0*self.rho_tor/self.q

    @sp_property
    def dphi_dvolume(self) -> Profile[float]: return self.fpol * self.gm1

    @sp_property
    def gm1(self) -> Profile[float]: return self._coord.surface_average(1.0/(_R**2))

    @sp_property
    def gm2_(self) -> Profile[float]: return self._coord.surface_average(self._coord.grad_psi2/(_R**2))

    @sp_property
    def gm2(self) -> Profile[float]:
        return self._coord.surface_average(self._coord.grad_psi2/(_R**2), extrapolate_left=True) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm3(self) -> Profile[float]:
        return self._coord.surface_average(self._coord.grad_psi2, extrapolate_left=True) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm4(self) -> Profile[float]: return self._coord.surface_average(1.0/self._coord.B2)

    @sp_property
    def gm5(self) -> Profile[float]: return self._coord.surface_average(self._coord.B2)

    @sp_property
    def gm6(self) -> Profile[float]:
        return self._coord.surface_average(self._coord.grad_psi2 / self._coord.B2, extrapolate_left=True) / (self.dpsi_drho_tor ** 2)

    @sp_property
    def gm7(self) -> Profile[float]:
        return self._coord.surface_average(np.sqrt(self._coord.grad_psi2), extrapolate_left=True) / self.dpsi_drho_tor

    @sp_property
    def gm8(self) -> Profile[float]: return self._coord.surface_average(_R)

    @sp_property
    def gm9(self) -> Profile[float]: return self._coord.surface_average(1.0 / _R)

    @sp_property
    def plasma_current(self) -> Profile[float]: return self.gm2 * \
        self.dvolume_drho_tor / self.dpsi_drho_tor/constants.mu_0

    @sp_property
    def j_tor(self) -> Profile[float]:
        return self.plasma_current.pd() / (self._coord.psi_boundary - self._coord.psi_magnetic_axis)/self.dvolume_dpsi * self._coord.r0

    @sp_property
    def j_parallel(self) -> Profile[float]:
        fvac = self._coord._fvac
        d = np.asarray(function_like(np.asarray(self.volume),
                                     np.asarray(fvac*self.plasma_current/self.fpol)).pd())
        return self._coord.r0*(self.fpol / fvac)**2 * d

    @sp_property
    def dpsi_drho_tor_norm(self) -> Profile[float]: return self.dpsi_drho_tor*self.rho_tor[-1]

    @cached_property
    def _shape_property(self) -> EquilibriumCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi_norm)

    @sp_property
    def geometric_axis(self) -> RZTuple:
        return {"r": self._shape_property.geometric_axis.r,
                "z": self._shape_property.geometric_axis.z}

    @sp_property
    def minor_radius(self) -> Profile[float]: return self._shape_property.minor_radius

    @sp_property
    def r_inboard(self) -> Profile[float]: return self._shape_property.r_inboard

    @sp_property
    def r_outboard(self) -> Profile[float]: return self._shape_property.r_outboard

    @sp_property
    def elongation(self) -> Profile[float]: return self._shape_property.elongation

    @sp_property
    def triangularity(self) -> Profile[float]: return self._shape_property.triangularity

    @sp_property
    def triangularity_upper(self) -> Profile[float]: return self._shape_property.triangularity_upper

    @sp_property
    def triangularity_lower(self) -> Profile[float]: return self._shape_property.triangularity_lower

    @sp_property
    def trapped_fraction(self, value) -> Profile[float]:
        """Trapped particle fraction[-]
            Tokamak 3ed, 14.10
        """
        if value is _not_found_:
            epsilon = self.rho_tor/self._coord.r0
            value = np.asarray(1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon)))
        return value


class EquilibriumProfiles2D(_T_equilibrium_profiles_2d):

    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

    @property
    def _global_quantities(self) -> _T_equilibrium_global_quantities: return self._parent.global_quantities

    @property
    def _profiles_1d(self) -> _T_equilibrium_profiles_1d: return self._parent.profiles_1d

    @sp_property
    def grid(self) -> Mesh:
        return Mesh(super().grid.dim1, super().grid.dim2, volume_element=super().grid.volume_element, type=super().grid_type)

    @sp_property
    def r(self) -> Field[float]: return Field(self.mesh.points[0], grid=self.grid)

    @sp_property
    def z(self) -> Field[float]: return Field(self.mesh.points[1], grid=self.grid)

    @sp_property
    def psi(self) -> Field[float]: return super().psi

    @property
    def psi_norm(self) -> Field[float]:
        psi_magetic_axis, psi_boundary = self._coord.psi_bc
        return (super().psi-psi_magetic_axis)/(psi_boundary - psi_magetic_axis)

    @sp_property
    def phi(self) -> Field[float]: return self._coord.apply_psifunc(self._profiles_1d.phi)

    @sp_property
    def theta(self) -> Field[float]: return self._coord.apply_psifunc(self._profiles_1d.phi)

    @sp_property
    def j_tor(self) -> Field[float]: return self._coord.apply_psifunc(self._profiles_1d.j_tor)

    @sp_property
    def j_parallel(self) -> Field[float]: return self._coord.apply_psifunc(self._profiles_1d.j_parallel)

    @sp_property
    def b_field_r(self) -> Field[float]: return self.psi.pd(0, 1) / _R / TWOPI

    @sp_property
    def b_field_z(self) -> Field[float]: return -self.psi.pd(1, 0) / _R / TWOPI

    @sp_property
    def b_field_tor(self) -> Field[float]: return self._profiles_1d.f(self.psi_norm) / _R


class EquilibriumBoundary(_T_equilibrium_boundary):
    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

    @sp_property
    def outline(self) -> RZTuple1D:
        _, surf = next(self._coord.find_surface(self._coord.psi_bc[1], o_point=True))
        return {"r": surf.xyz[0], "z": surf.xyz[1]}

    psi_norm: float = sp_property(default_value=0.999)

    @sp_property
    def psi(self) -> float: return self.psi_norm*self._coord.psi_delta + self._coord.psi_magnetic_axis

    @sp_property
    def phi(self) -> float: raise NotImplementedError(f"{self.__class__.__name__}.phi")

    @sp_property
    def rho(self) -> float: return np.sqrt(self.phi_boundary/(constants.pi * self._coord._B0))

    @cached_property
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
    def strike_point(self) -> List[RZTuple]:
        return NotImplemented

    @sp_property
    def active_limiter_point(self) -> List[RZTuple]:
        return NotImplemented


class EquilibriumBoundarySeparatrix(_T_equilibrium_boundary_separatrix):

    @property
    def _coord(self) -> EquilibriumCoordinateSystem: return self._parent.coordinate_system

    @sp_property
    def outline(self) -> RZTuple1D:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface(self.psi, o_point=None))
        return {"r": surf.xyz[0], "z": surf.xyz[1]}

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

    @cached_property
    def _B0(self) -> float: return 1.0  # self._parent.vacuum_toroidal_field.b0(self.time)

    profiles_1d: EquilibriumProfiles1D = sp_property()

    profiles_2d: List[EquilibriumProfiles2D] = sp_property()
    """ FIXME: 定义多个 profiles_2d，与profiles_1d, global_quantities如何保持一致？ 这里会有歧义    """

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
