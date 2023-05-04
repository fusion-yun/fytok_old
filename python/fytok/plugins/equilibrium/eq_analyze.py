import collections
import collections.abc
import typing
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import isclose

import numpy as np
from  fytok._imas.equilibrium import (_T_equilibrium_boundary,
                               _T_equilibrium_boundary_separatrix,
                               _T_equilibrium_global_quantities,
                               _T_equilibrium_global_quantities_magnetic_axis,
                               _T_equilibrium_profiles_1d,
                               _T_equilibrium_profiles_2d,
                               _T_equilibrium_time_slice)
from scipy import constants
from spdm.data.Dict import Dict
from spdm.data.Function import Function, function_like
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.Profile import Profile
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.geometry.CubicSplineCurve import CubicSplineCurve
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Point import Point
from spdm.grid.CurvilinearMesh import CurvilinearMesh
from spdm.grid.RectilinearMesh import RectilinearMesh
from spdm.grid.Grid import Grid
from spdm.numlib.contours import find_countours
from spdm.numlib.optimize import find_critical_points
from spdm.utils.logger import logger
from spdm.utils.misc import convert_to_named_tuple
from spdm.utils.tags import _not_found_

from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.Utilities import RZTuple, RZTuple1D

TOLERANCE = 1.0e-6
EPS = np.finfo(float).eps

TWOPI = 2.0*constants.pi

_T=typing.TypeVar("_T")

@dataclass
class OXPoint:
    r: float
    z: float
    psi: float


def extrapolate_left(x, d):
    if hasattr(d, "__array__"):
        d = d.__array__()

    d[0] = d[1]+(d[1]-d[2])/(x[1]-x[2])*(x[0]-x[1])
    return d


class MagneticSurfaceAnalyze(Dict[Node]):
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

    def __init__(self,  *args,
                 psirz: Function,
                 B0: float,
                 R0: float,
                 Ip: float,
                 fpol:     np.ndarray,
                 psi_norm: typing.Union[int, np.ndarray] = 128,
                 theta:    typing.Union[int, np.ndarray] = 32,
                 grid_type=13,
                 **kwargs):
        """
            Initialize FluxSurface
        """
        super().__init__(*args, **kwargs)
        self._grid_type = grid_type

        self._psirz = psirz

        self._Ip = Ip
        self._B0 = B0
        self._R0 = R0
        self._fvac = self._B0*self._R0

        # @TODO: COCOS transformation ?
        self._cocos = self.get("cocos", 11)
        self._s_Bp = np.sign(self._B0)
        self._s_Ip = np.sign(self._Ip)
        self._s_2PI = 1.0/(constants.pi*2.0)  # 1.0/(TWOPI ** (1-e_Bp))

        if isinstance(theta, int):
            self._theta = np.linspace(0, TWOPI, theta)
        elif isinstance(theta, np.ndarray):
            self._theta = theta
            if not isclose(self._theta[0], self._theta[-1]):
                self._theta.append(self._theta[0])
        else:
            raise RuntimeError(f"theta grid is not defined!")

        if isinstance(psi_norm, int):
            self._psi_norm = np.linspace(0.0, 1.0, psi_norm)
        elif isinstance(psi_norm, np.ndarray):
            self._psi_norm = psi_norm
        else:
            raise RuntimeError(f"psi_norm grid is not defined!")

        if isinstance(fpol, Function):
            self._fpol = fpol
        elif isinstance(fpol, np.ndarray) and len(fpol) == len(self._psi_norm):
            self._fpol = function_like(self._psi_norm, fpol)
        else:
            raise RuntimeError(f"fpol is not defined!")

        logger.debug(f"Create MagneticCoordSystem: type index={self._grid_type} primary='psi'  ")

    @property
    def r0(self) -> float:
        return self._R0

    @property
    def b0(self) -> float:
        return self._B0

    @property
    def grid_type_index(self) -> int:
        return self._grid_type

    @property
    def cocos(self) -> int:
        logger.warning("NOT IMPLEMENTED!")
        return self._cocos

    @cached_property
    def cocos_flag(self) -> int:
        return 1 if self.psi_boundary > self.psi_magnetic_axis else -1

    @property
    def psi_norm(self) -> np.ndarray:
        return self._psi_norm

    @cached_property
    def critical_points(self) -> typing.Tuple[typing.Sequence[OXPoint], typing.Sequence[OXPoint]]:
        opoints = []
        xpoints = []

        for r, z, psi, D in find_critical_points(self._psirz, *self._psirz.grid.bbox, tolerance=self._psirz.grid.dx):
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

            bbox = self._psirz.grid.bbox
            Rmid = (bbox[0] + bbox[2])/2.0
            Zmid = (bbox[1] + bbox[3])/2.0

            opoints.sort(key=lambda x: (x.r - Rmid)**2 + (x.z - Zmid)**2)

            o_r = opoints[0].r
            o_z = opoints[0].z
            # TOOD: NEED　IMPROVMENT!!
            xpoints.sort(key=lambda x: (x.r - o_r)**2 + (x.z - o_z)**2)
            # psi_magnetic_axis = opoints[0].psi
            # xpoints.sort(key=lambda x: (x.psi - psi_magnetic_axis)**2)

        return opoints, xpoints

    def find_surface(self, psi: typing.Union[float, typing.Sequence] = None, o_point: OXPoint = True) -> typing.Iterator[typing.Tuple[float, GeoObject]]:
        """
            if o_point is not None:
                only return  closed surface  enclosed o-point
                if closed surface does not exists, return None
                number of surface == len(psi)
            else:
                do not guarantee the number of surface == len(psi)
                return all surface ,
        """

        x_point = None
        if o_point is True:
            opts, xpts = self.critical_points
            if len(opts) == 0:
                raise RuntimeError(f"O-point is not defined!")
            o_point = opts[0]
            if len(xpts) > 0:
                x_point = xpts[0]

        R, Z = self._psirz.grid.xy

        F = np.asarray(self._psirz)

        if not isinstance(psi, (collections.abc.Sequence, np.ndarray)):
            psi = [psi]

        psi = np.asarray(psi, dtype=float)

        if o_point is None or o_point is False:
            for level, col in find_countours(F, R, Z, levels=psi):
                for points in col:
                    yield level, CubicSplineCurve(points)
        else:
            for level, col in find_countours(F, R, Z, levels=psi):
                surf = None
                for points in col:
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

                        self._grid = [theta]
                        self._points = points
                    surf = CubicSplineCurve(points, [theta])

                    yield level, surf
                    break

                if surf is None:
                    if np.isclose(level, o_point.psi):
                        yield level, Point(o_point.r, o_point.z)
                    else:
                        raise RuntimeError(f"{level},{o_point.psi},{(max(theta),min(theta))}")

    def find_surface_by_psi_norm(self, psi_norm: typing.Union[float, typing.Sequence], *args,   **kwargs) -> typing.Iterator[typing.Tuple[float, GeoObject]]:
        yield from self.find_surface(np.asarray(psi_norm, dtype=float)*(self.psi_boundary-self.psi_magnetic_axis)+self.psi_magnetic_axis, *args,  **kwargs)

    ###############################
    # 0-D

    @cached_property
    def magnetic_axis(self):
        o, _ = self.critical_points
        if not o:
            raise RuntimeError(f"Can not find magnetic axis")

        return {
            "r": o[0].r,
            "z": o[0].z,
            "b_field_tor": NotImplemented
        }

    @cached_property
    def psi_magnetic_axis(self) -> float:
        """Poloidal flux at the magnetic axis  [Wb]."""
        o, _ = self.critical_points
        return o[0].psi

    @cached_property
    def psi_boundary(self) -> float:
        """Poloidal flux at the selected plasma boundary  [Wb]."""
        _, x = self.critical_points
        if len(x) > 0:
            return x[0].psi
        else:
            raise ValueError(f"No x-point")

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
            return MagneticSurfaceAnalyze.ShapeProperty(
                # RZ position of the geometric axis of the magnetic surfaces (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the surface)
                RZTuple1D({"r": function_like(psi_norm, (rmin+rmax)*0.5),
                           "z": function_like(psi_norm, (zmin+zmax)*0.5)}),
                # Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary)[m]
                function_like(psi_norm, (rmax - rmin)*0.5),  # "minor_radius":
                # Elongation of the plasma boundary. [-]
                # "elongation":
                function_like(psi_norm, np.hstack(
                    [(zmax[1]-zmin[1])/(rmax[1]-rmin[1]), (zmax[1:]-zmin[1:])/(rmax[1:]-rmin[1:])])),
                # Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]
                # "elongation_upper":
                function_like(psi_norm, np.hstack([0, (zmax[1:]-(zmax[1:]+zmin[1:])*0.5)/(rmax[1:]-rmin[1:])])),
                # longation(lower half w.r.t. geometric axis) of the plasma boundary. [-]
                # elongation_lower":
                function_like(psi_norm, np.hstack([0, ((zmax[1:]+zmin[1:])*0.5-zmin[1:])/(rmax[1:]-rmin[1:])])),
                # Triangularity of the plasma boundary. [-]
                # "triangularity":
                function_like(psi_norm, np.hstack([0, (rzmax[1:]-rzmin[1:])/(rmax[1:] - rmin[1:])*2])),
                # Upper triangularity of the plasma boundary. [-]
                # "triangularity_upper":
                function_like(psi_norm, np.hstack([0, ((rmax[1:]+rmin[1:])*0.5 - rzmax[1:])/(rmax[1:] - rmin[1:])*2])),
                # Lower triangularity of the plasma boundary. [-]
                # "triangularity_lower":
                function_like(psi_norm, np.hstack([0, ((rmax[1:]+rmin[1:])*0.5 - rzmin[1:])/(rmax[1:] - rmin[1:])*2])),
                # Radial coordinate(major radius) on the inboard side of the magnetic axis[m]
                function_like(psi_norm, r_inboard),  # "r_inboard":
                # Radial coordinate(major radius) on the outboard side of the magnetic axis[m]
                function_like(psi_norm, r_outboard),  # "r_outboard":
            )
        else:
            return MagneticSurfaceAnalyze.ShapeProperty(
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

    ###############################
    # grid

    @cached_property
    def grid(self) -> Grid:

        if self._grid_type != 13:
            raise NotImplementedError(self._grid_type)

        return CurvilinearMesh([surf for _, surf in self.find_surface_by_psi_norm(self._psi_norm, o_point=True)],
                               [self._psi_norm, self._theta/TWOPI], cycle=[False, True])

    def r(self) -> np.ndarray:
        return self.grid.xy[:, :, 0]

    def z(self) -> np.ndarray:
        return self.grid.xy[:, :, 1]

    def rz(self, dim1=None, dim2=None, grid_type=None):
        """生成二维grid。用于将磁面坐标grid（默认）转换为目标grid
        """
        logger.debug((int(grid_type), str(grid_type.name)))

        if (grid_type == self._grid_type and all(dim1 == self._psi_norm) and all(dim2 == self._theta)) \
                or (dim1 is None and dim2 is None and grid_type is None):
            return self.grid.xy[:, :, 0], self.grid.xy[:, :, 1]
        elif int(grid_type) == 1 or getattr(grid_type, 'name', 'unnamed') == 'rectangle':
            rect_grid = RectilinearMesh(dim1, dim2)
            return rect_grid.xy[:, :, 0], rect_grid.xy[:, :, 1]
        else:
            raise NotImplementedError(f"grid_type.index={int(grid_type)} name='{getattr(grid_type,'name','unnamed')}'")

    def psi(self, r: _T, z: _T, *args, **kwargs) -> _T:
        return self._psirz(r, z, *args, **kwargs)

    def psi_norm(self, r: _T, z: _T, *args, **kwargs) -> _T:
        return (self.psi(r, z, *args, **kwargs)-self.psi_magnetic_axis)/(self.psi_boundary-self.psi_magnetic_axis)

    def Br(self, r: _T, z: _T) -> _T:
        return self.psi(r, z, dy=1) / r/TWOPI

    def Bz(self, r: _T, z: _T) -> _T:
        return -self.psi(r,  z, dx=1) / r/TWOPI

    def Btor(self, r: _T, z: _T) -> _T:
        return self._fpol(self.psi_norm(r, z)) / r

    def Bpol(self, r: _T, z: _T) -> _T:
        r"""
            $B_{pol}= \left|\nabla \psi \right|/2 \pi R $
        """
        return self.grad_psi(r, z) / r / (TWOPI)

    def B2(self, r: _T, z: _T) -> _T:
        return (self.Br(r, z)**2+self.Bz(r, z)**2 + self.Btor(r, z)**2)

    def grad_psi2(self,  r: _T, z: _T) -> _T:
        return self.psi(r, z, dx=1)**2+self.psi(r, z, dy=1)**2

    def grad_psi(self,  r: _T, z: _T) -> _T:
        return np.sqrt(self.grad_psi2(r, z))

    ###############################
    # surface integral
    @cached_property
    def o_point(self) -> OXPoint:
        opts, _ = self.critical_points
        return opts[0]

    @cached_property
    def ddpsi(self):
        r0 = self.o_point.r
        z0 = self.o_point.z
        return np.sqrt(self.psirz(r0, z0, dx=2) * self.psirz(r0, z0, dy=2) + self.psirz(r0, z0, dx=1, dy=1)**2)

    def _surface_integral(self, func: typing.Callable[[float, float], float] = None, surface_list=None) -> np.ndarray:
        r0 = self.o_point.r
        z0 = self.o_point.z

        ddpsi = np.sqrt(self.psirz(r0, z0, dx=2) * self.psirz(r0, z0, dy=2) + self.psirz(r0, z0, dx=1, dy=1)**2)

        c0 = TWOPI*r0**2/ddpsi

        if surface_list is None:
            surface_list = self.grid.axis_iter()
        else:
            surface_list = self.find_surface_by_psi_norm(surface_list, o_point=True)

        if func is not None:
            return np.asarray([(axis.integral(lambda r, z, _func=func, _bpol=self.Bpol:_func(r, z)/_bpol(r, z)) if not np.isclose(p, 0) else func(r0, z0) * c0) for p, axis in surface_list], dtype=float)
        else:
            return np.asarray([(axis.integral(lambda r, z, _func=func, _bpol=self.Bpol:1.0/_bpol(r, z)) if not np.isclose(p, 0) else c0) for p, axis in surface_list], dtype=float)

    @cached_property
    def dvolume_dpsi(self) -> np.ndarray:
        r"""
            $ V^{\prime} =  2 \pi  \int{ R / \left|\nabla \psi \right| * dl }$

            $ V^{\prime}(psi)= 2 \pi  \int{ dl * R / \left|\nabla \psi \right|}$
        """
        return self._surface_integral()

    def surface_average(self,  func,   /, value_axis: typing.Union[float, typing.Callable[[float, float], float]] = None, **kwargs) -> np.ndarray:
        r"""
            $\left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}$
        """
        return self._surface_integral(func, value_axis)/self.dvolume_dpsi

    @cached_property
    def phi_boundary(self) -> float:
        if not np.isclose(self._psi_norm[-1], 1.0):
            logger.warning(f"FIXME: psi_norm boudnary is {self._psi_norm[-1]} != 1.0 ")

        return self.phi[-1]

    @cached_property
    def rho_boundary(self) -> float:
        return np.sqrt(self.phi_boundary/(constants.pi * self._B0))

    def plot_contour(self, axis, levels=16):
        import matplotlib.pyplot as plt

        if isinstance(levels, int):
            levels = np.linspace(0, 1, levels)
        elif isinstance(levels, (collections.abc.Sequence)):
            l_min, l_max = levels
            levels = np.linspace(l_min, l_max, 16)

        levels = levels*(self.psi_boundary-self.psi_magnetic_axis)+self.psi_magnetic_axis

        field = self._psirz

        R, Z = field.grid.xy

        F = np.asarray(field(R, Z), dtype=float)

        for level, col in find_countours(F, R, Z, levels=levels):
            for segment in col:
                axis.add_patch(plt.Polygon(segment, fill=False, closed=np.all(
                    np.isclose(segment[0], segment[-1])), color="b", linewidth=0.2))
        return axis


TOLERANCE = 1.0e-6

EPS = np.finfo(float).eps

TWOPI = 2.0*constants.pi


class EquilibriumGlobalQuantities(_T_equilibrium_global_quantities):

    def _coord(self) -> MagneticSurfaceAnalyze:
        return self._parent._coord()

    @sp_property
    def magnetic_axis(self) -> _T_equilibrium_global_quantities_magnetic_axis:
        """Magnetic axis position and toroidal field	structure"""
        return (self._coord().magnetic_axis)

    @sp_property
    def psi_magnetic_axis(self) -> float: return self._coord().critical_points[0].psi

    @sp_property
    def psi_boundary(self) -> float: return self._coord().critical_points[1].psi

    beta_pol: float = sp_property(type="dynamic", units="-")

    beta_tor: float = sp_property(type="dynamic", units="-")

    beta_normal: float = sp_property(type="dynamic", units="-")

    # ip: float = sp_property(type="dynamic", units="A", cocos_label_transformation="ip_like",
    #                         cocos_transformation_expression=".sigma_ip_eff", cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.ip")

    # li_3: float = sp_property(type="dynamic", units="-")

    # volume: float = sp_property(type="dynamic", units="m^3")

    # area: float = sp_property(type="dynamic", units="m^2")

    # surface: float = sp_property(type="dynamic", units="m^2")

    # length_pol: float = sp_property(type="dynamic", units="m")

    # psi_axis: float = sp_property(type="dynamic", units="Wb", cocos_label_transformation="psi_like",
    #                               cocos_transformation_expression=".fact_psi", cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.psi_axis")

    # psi_boundary: float = sp_property(type="dynamic", units="Wb", cocos_label_transformation="psi_like",
    #                                   cocos_transformation_expression=".fact_psi", cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.psi_boundary")

    # magnetic_axis: _T_equilibrium_global_quantities_magnetic_axis = sp_property()

    # current_centre: _T_equilibrium_global_quantities_current_centre = sp_property()

    # q_axis: float = sp_property(type="dynamic", units="-", cocos_label_transformation="q_like",
    #                             cocos_transformation_expression=".fact_q", cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.q_axis")

    # q_95: float = sp_property(type="dynamic", units="-", cocos_label_transformation="q_like",
    #                           cocos_transformation_expression=".fact_q", cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.q_95")

    # q_min: _T_equilibrium_global_quantities_qmin = sp_property()

    # energy_mhd: float = sp_property(type="dynamic", units="J")

    # psi_external_average: float = sp_property(type="dynamic", units="Wb", cocos_label_transformation="psi_like", cocos_transformation_expression=".fact_psi",
    #                                           cocos_leaf_name_aos_indices="equilibrium.time_slice{i}.global_quantities.psi_external_average")

    # v_external: float = sp_property(type="dynamic", units="V", cocos_label_transformation="ip_like", cocos_transformation_expression=".sigma_ip_eff", cocos_leaf_name_aos_indices=[
    #                                 "core_profiles.global_quantities.v_loop", "equilibrium.time_slice{i}.global_quantities.v_external"], introduced_after_version="3.37.2")

    # plasma_inductance: float = sp_property(type="dynamic", units="H")

    # plasma_resistance: float = sp_property(type="dynamic", units="ohm", introduced_after_version="3.37.2")


class EquilibriumProfiles1D(_T_equilibrium_profiles_1d):

    def _coord(self) -> MagneticSurfaceAnalyze:
        return self._parent._coord()

    ###############################
    # 1-D

    @property
    def psi_norm(self) -> np.ndarray:
        return self._coord()._psi_norm

    @sp_property
    def psi(self) -> np.ndarray:
        return self.psi_norm * (self._coord().psi_boundary-self._coord().psi_magnetic_axis) + self._coord().psi_magnetic_axis

    @sp_property
    def fpol(self) -> Profile[float]:
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        return self._coord()._fpol(self.psi_norm)

    @sp_property
    def ffprime(self) -> Profile[float]:
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        return self._coord()._fpol(self.psi_norm)*self._coord()._fpol.derivative(self.psi_norm)

    # @sp_property
    # def pprime(self) -> Profile[float]:
    #     """Diamagnetic function (F=R B_Phi)  [T.m]."""
    #     return self._pprime(self.psi_norm)

    @sp_property
    def dphi_dpsi(self) -> Profile[float]:
        return self.fpol * self.gm1 * self.dvolume_dpsi / TWOPI

    @sp_property
    def q(self) -> Profile[float]:
        r"""
            Safety factor
            (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)[-].
            $ q(\psi) =\frac{d\Phi}{2\pi d\psi} =\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{2\pi}$
        """
        return self.dphi_dpsi * self._coord()._s_Bp * self._coord()._s_2PI

    @sp_property
    def magnetic_shear(self) -> Profile[float]:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
        return self.rho_tor/self.q * function_like(self.rho_tor, self.q).derivative(self.rho_tor)

    @sp_property
    def phi(self) -> Profile[float]:
        r"""
            Note:
            $\Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi$
        """
        return function_like(self.psi_norm, self.dphi_dpsi).antiderivative(self.psi_norm)*(self._coord().psi_boundary-self._coord().psi_magnetic_axis)

    @sp_property
    def rho_tor(self) -> Profile[float]:
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
        return np.sqrt(self.phi/(constants.pi * self._coord()._B0))

    @sp_property
    def rho_tor_norm(self) -> Profile[float]:
        return np.sqrt(self.phi/self.phi_boundary)

    @sp_property
    def volume(self) -> Profile[float]:
        """Volume enclosed in the flux surface[m ^ 3]"""
        return function_like(self.psi, self.dvolume_dpsi).antiderivative(self.psi)

    @sp_property
    def surface(self) -> Profile[float]:
        """Surface area of the toroidal flux surface {dynamic} [m^2]"""
        return self.dvolume_drho_tor*self.gm7

    @sp_property
    def dvolume_drho_tor(self) -> Profile[float]:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
        return extrapolate_left(self.psi_norm, (TWOPI**2) * self.rho_tor/(self.gm1)/(self._coord()._fvac/self.fpol)/self._coord()._R0)

    # @sp_property
    # def volume1(self) -> Profile[float]:
    #     """Volume enclosed in the flux surface[m ^ 3]"""
    #     if self.rho_tor[0] > 1.0e-4:
    #         x = np.hstack([[0.0], self.rho_tor])
    #         dvdx = np.hstack([[self.dvolume_drho_tor[0]], self.dvolume_drho_tor])
    #     else:
    #         x = self.rho_tor
    #         dvdx = self.dvolume_drho_tor

    #     return Function(x, dvdx).antiderivative(self.rho_tor)

    @sp_property
    def drho_tor_dpsi(self) -> Profile[float]:
        r"""
            $\frac{d\rho_{tor}}{d\psi} =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                        =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                        =\frac{q}{2\pi B_{0}\rho_{tor}}
            $
        """
        return 1.0/self.dpsi_drho_tor

    @sp_property
    def dpsi_drho_tor(self) -> Profile[float]:
        """
            Derivative of Psi with respect to Rho_Tor[Wb/m].
        """
        return extrapolate_left(self.psi_norm, (self._coord()._s_Bp)*self._coord()._B0*self.rho_tor/self.q)

    @sp_property
    def dphi_dvolume(self) -> Profile[float]:
        return self.fpol * self.gm1

    @sp_property
    def gm1(self) -> Profile[float]: return self._coord().surface_average(lambda r, z: 1.0/(r**2), self.psi_norm)

    @sp_property
    def gm2_(self) -> Profile[float]: return self._coord().surface_average(lambda r,
                                                                           z: self.grad_psi2(r, z)/(r**2), self.psi_norm)

    @sp_property
    def gm2(self) -> Profile[float]:
        return extrapolate_left(self.psi_norm,
                                self._coord().surface_average(lambda r, z: self._coord().grad_psi2(r, z)/(r**2)) / (self.dpsi_drho_tor ** 2), self.psi_norm)

    @sp_property
    def gm3(self) -> Profile[float]:
        return extrapolate_left(self.psi_norm,
                                self._coord().surface_average(self._coord().grad_psi2) / (self.dpsi_drho_tor ** 2), self.psi_norm)

    @sp_property
    def gm4(self) -> Profile[float]: return self._coord().surface_average(lambda r,
                                                                          z: 1.0/self._coord().B2(r, z), self.psi_norm)

    @sp_property
    def gm5(self) -> Profile[float]: return self._coord().surface_average(lambda r,
                                                                          z: self._coord().B2(r, z), self.psi_norm)

    @sp_property
    def gm6(self) -> Profile[float]:
        return extrapolate_left(self.psi_norm,
                                self._coord().surface_average(lambda r, z: self._coord().grad_psi2(r, z)/self._coord().B2(r, z)) / (self.dpsi_drho_tor ** 2), self.psi_norm)

    @sp_property
    def gm7(self) -> Profile[float]:
        return extrapolate_left(self.psi_norm,
                                self._coord().surface_average(lambda r, z: np.sqrt(self._coord().grad_psi2(r, z))) / self.dpsi_drho_tor, self.psi_norm)

    @sp_property
    def gm8(self) -> Profile[float]:
        return self._coord().surface_average(lambda r, z: r, self.psi_norm)

    @sp_property
    def gm9(self) -> Profile[float]:
        return self._coord().surface_average(lambda r, z: 1.0 / r, self.psi_norm)

    @sp_property
    def plasma_current(self) -> Profile[float]:
        return self.gm2 * self.dvolume_drho_tor / self.dpsi_drho_tor/constants.mu_0

    @sp_property
    def j_tor(self) -> Profile[float]:
        return self.plasma_current.derivative() / (self._coord().psi_boundary - self._coord().psi_magnetic_axis)/self.dvolume_dpsi * self._coord().r0

    @sp_property
    def j_parallel(self) -> Profile[float]:
        fvac = self._coord()._fvac
        d = np.asarray(function_like(np.asarray(self.volume), np.asarray(
            fvac*self.plasma_current/self.fpol)).derivative())
        return self._coord().r0*(self.fpol / fvac)**2 * d

    @sp_property
    def dpsi_drho_tor_norm(self) -> Profile[float]: return self.dpsi_drho_tor*self.rho_tor[-1]

    def _shape_property(self) -> MagneticSurfaceAnalyze.ShapeProperty: return self._coord().shape_property()

    @sp_property
    def geometric_axis(self) -> RZTuple:
        return RZTuple({"r": self._shape_property().geometric_axis.r,
                        "z": self._shape_property().geometric_axis.z}, parent=self)

    @sp_property
    def minor_radius(self) -> Profile[float]: return self._shape_property().minor_radius(self.psi_norm)

    @sp_property
    def r_inboard(self) -> Profile[float]: return self._shape_property().r_inboard(self.psi_norm)

    @sp_property
    def r_outboard(self) -> Profile[float]: return self._shape_property().r_outboard(self.psi_norm)

    @sp_property
    def elongation(self) -> Profile[float]: return self._shape_property().elongation(self.psi_norm)

    @sp_property
    def triangularity(self) -> Profile[float]: return self._shape_property().triangularity(self.psi_norm)

    @sp_property
    def triangularity_upper(self) -> Profile[float]: return self._shape_property().triangularity_upper(self.psi_norm)

    @sp_property
    def triangularity_lower(self) -> Profile[float]: return self._shape_property().triangularity_lower(self.psi_norm)

    @sp_property
    def trapped_fraction(self, value) -> Profile[float]:
        """Trapped particle fraction[-]
            Tokamak 3ed, 14.10
        """
        if value is _not_found_:
            epsilon = self.rho_tor/self._coord().r0
            value = np.asarray(1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon)))
        return value


class EquilibriumProfiles2D(_T_equilibrium_profiles_2d):

    def _coord(self) -> MagneticSurfaceAnalyze:
        return self._parent._coord()

    def _rz(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        if getattr(self, "_t_rz", None) is None:
            self._t_rz = self._coord().rz(self.grid.dim1, self.grid.dim2, grid_type=self.grid_type)
        return self._t_rz

    @sp_property
    def r(self) -> Profile[float]: return self._rz()[0]

    @sp_property
    def z(self) -> Profile[float]: return self._rz()[1]

    @sp_property
    def psi(self): return self._coord().psi(self.r, self.z)

    @sp_property
    def phi(self): return self._coord().apply_psifunc(self._parent.profiles_1d.phi, self.r, self.z)

    @sp_property
    def theta(self): return self._coord().apply_psifunc(self._parent.profiles_1d.phi,  self.r, self.z)

    @sp_property
    def j_tor(self): return self._coord().apply_psifunc(self._parent.profiles_1d.j_tor, self.r, self.z)

    @sp_property
    def j_parallel(self): return self._coord().apply_psifunc(self._parent.profiles_1d.j_parallel, self.r, self.z)

    @sp_property
    def b_field_r(self) -> Profile[float]: return self._coord().Br(self.r, self.z)

    @sp_property
    def b_field_z(self) -> Profile[float]: return self._coord().Bz(self.r, self.z)

    @sp_property
    def b_field_tor(self) -> Profile[float]: return self._coord().Btor(self.r, self.z)


class EquilibriumBoundary(_T_equilibrium_boundary):

    def _coord(self) -> MagneticSurfaceAnalyze:
        return self._parent._coord()

    psi_norm: float = sp_property(default_value=0.999)
    """Value of the normalized poloidal flux at which the boundary is taken (typically 99.x %),
            the flux being normalized to its value at the separatrix """

    @sp_property
    def outline(self) -> RZTuple1D:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord().find_surface(self.psi, o_point=True))
        return {"r": surf.xyz[0], "z": surf.xyz[1]}

    @sp_property
    def psi_magnetic_axis(self) -> float:
        return self._coord().psi_magnetic_axis

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord().psi_boundary

    @sp_property
    def psi(self) -> float:
        """Value of the poloidal flux at which the boundary is taken  [Wb]"""
        return self.psi_norm*(self._coord().psi_boundary-self._coord().psi_magnetic_axis)+self._coord().psi_magnetic_axis

    def _shape_property(self) -> MagneticSurfaceAnalyze.ShapeProperty:
        return self._coord().shape_property(self.psi_norm)

    @sp_property
    def geometric_axis(self) -> RZTuple: return self._shape_property().geometric_axis

    @sp_property
    def minor_radius(self) -> float: return self._shape_property().minor_radius

    @sp_property
    def elongation(self) -> float: return self._shape_property().elongation

    @sp_property
    def elongation_upper(self) -> float: return self._shape_property().elongation_upper

    @sp_property
    def elongation_lower(self) -> float: return self._shape_property().elongation_lower

    @sp_property
    def triangularity(self) -> float: return self._shape_property().triangularity

    @sp_property
    def triangularity_upper(self) -> float: return self._shape_property().triangularity_upper

    @sp_property
    def triangularity_lower(self) -> float: return self._shape_property().triangularity_lower

    @sp_property
    def x_point(self) -> List[RZTuple]:
        _, xpt = self._coord().critical_points
        return xpt

    @sp_property
    def strike_point(self) -> List[RZTuple]:
        return NotImplemented

    @sp_property
    def active_limiter_point(self) -> List[RZTuple]:
        return NotImplemented


class EquilibriumBoundarySeparatrix(_T_equilibrium_boundary_separatrix):

    def _coord(self) -> MagneticSurfaceAnalyze:
        return self._parent._coord()

    @sp_property
    def outline(self) -> RZTuple1D:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord().find_surface_by_psi_norm(1.0, o_point=None))
        return {"r": surf.xyz[0], "z": surf.xyz[1]}

    @sp_property
    def psi_magnetic_axis(self) -> float:
        return self._coord().psi_magnetic_axis

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord().psi_boundary

    @sp_property
    def psi(self) -> float:
        return self._coord().psi_norm*(self._coord().psi_boundary-self._coord().psi_magnetic_axis)+self._coord().psi_magnetic_axis

    @sp_property
    def x_point(self) -> List[RZTuple]:
        _, x = self._coord().critical_points
        return List[RZTuple]([{"r": v.r, "z": v.z} for v in x[:]])

    @sp_property
    def strike_point(self) -> List[RZTuple]:
        raise NotImplementedError("TODO:")


class EquilibriumTimeSlice(_T_equilibrium_time_slice):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._m_coord = None

    profiles_1d: EquilibriumProfiles1D = sp_property()

    profiles_2d: List[EquilibriumProfiles2D] = sp_property()

    global_quantities: EquilibriumGlobalQuantities = sp_property()

    boundary: EquilibriumBoundary = sp_property()

    boundary_separatrix: EquilibriumBoundarySeparatrix = sp_property()

    def _coord(self) -> MagneticSurfaceAnalyze:
        if self._m_coord is None:
            self.update()
        return self._m_coord

    def update(self, *args, B0=None, R0=None,  **kwargs):
        self._cache.clear()

        psirz = self.__entry__().get("profiles_2d/0/psi", None)

        if psirz is None:
            raise ValueError("No psi found in the equilibrium")
        elif isinstance(psirz, np.ndarray):
            psirz = Function(self.get("profiles_2d/0/grid/dim1"),
                             self.get("profiles_2d/0/grid/dim2"),
                             psirz,
                             grid_type="rectilinear")

        elif not isinstance(psirz, Function):
            raise ValueError("psi must be a Function or a numpy array")

        psi_1d = self.__entry__().get("profiles_1d/psi", _not_found_)
        fpol_1d = self.__entry__().get("profiles_1d/f", _not_found_)
        if not isinstance(psi_1d, np.ndarray) or len(psi_1d) != len(fpol_1d):
            psi_1d = np.linspace(0, 1.0, len(fpol_1d))

        if isinstance(psi_1d, np.ndarray):
            psi_1d = (psi_1d-psi_1d[0])/(psi_1d[-1]-psi_1d[0])

        # pprime_1d = self.profiles_1d._entry.get("dpressure_dpsi", None)
        self._m_coord = MagneticSurfaceAnalyze(
            psirz=psirz,
            B0=self._parent.vacuum_toroidal_field.b0[-1] if B0 is None else B0,
            R0=self._parent.vacuum_toroidal_field.r0 if R0 is None else R0,
            Ip=self.global_quantities.ip,
            fpol=function_like(psi_1d, fpol_1d),
            # pprime=self.profiles_1d._entry.get("dpressure_dpsi", None),
            # fpol=function_like(psi_norm, self.profiles_1d._entry.get("f", None)),
            # pprime=function_like(psi_norm, self.profiles_1d._entry.get("dpressure_dpsi", None)),
            grid_type=self.profiles_2d[0].grid_type,
        )


@Equilibrium.register(["eq_analyze"])
class FyEqAnalyze(Equilibrium):
    """ 
    FyEqAnalyze 标准磁面分析工具
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
