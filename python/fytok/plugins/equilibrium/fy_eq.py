import collections
import collections.abc
from dataclasses import dataclass
from functools import cached_property
from math import isclose
from typing import Callable, Iterator, Sequence, Tuple, Union

import numpy as np
from _imas.equilibrium import (_T_equilibrium_boundary,
                               _T_equilibrium_boundary_separatrix,
                               _T_equilibrium_global_quantities,
                               _T_equilibrium_profiles_1d,
                               _T_equilibrium_profiles_2d)
from scipy import constants
from spdm.data.Dict import Dict
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.geometry.CubicSplineCurve import CubicSplineCurve
from spdm.geometry.GeoObject import GeoObject, _TCoord
from spdm.geometry.Point import Point
from spdm.mesh.CurvilinearMesh import CurvilinearMesh
from spdm.mesh.Mesh import Mesh
from spdm.numlib.contours import find_countours
from spdm.numlib.optimize import find_critical_points
from spdm.utils.logger import logger
from spdm.utils.misc import convert_to_named_tuple
from spdm.utils.tags import _not_found_

from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.Utilities import RZTuple


TOLERANCE = 1.0e-6
EPS = np.finfo(float).eps

TWOPI = 2.0*constants.pi


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
    # return Function(x[1:], d[1:])(x)

# OXPoint = collections.namedtuple('OXPoint', "r z psi")


class RadialGrid(Dict):
    r"""
        Radial grid
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            super().__init__(kwargs)
        else:
            super().__init__(*args, **kwargs)

    def remesh(self, label: str = "psi_norm", new_axis: np.ndarray = None, ):

        axis = self._as_child(label)

        if isinstance(axis, np.ndarray) and isinstance(new_axis, np.ndarray) \
                and axis.shape == new_axis.shape and np.allclose(axis, new_axis):
            return self

        if new_axis is None:
            new_axis = np.linspace(axis[0], axis[-1], len(axis))
        elif isinstance(new_axis, int):
            new_axis = np.linspace(0, 1.0, new_axis)
        elif not isinstance(new_axis, np.ndarray):
            raise TypeError(new_axis)

        return RadialGrid(
            r0=self.r0,
            b0=self.b0,
            psi_axis=self.psi_axis,
            psi_boundary=self.psi_boundary,
            rho_tor_boundary=self.rho_tor_boundary,
            psi_norm=function_like(axis,  self.psi_norm)(new_axis) if label != "psi_norm" else new_axis,
            rho_tor_norm=function_like(axis,  self.rho_tor_norm)(new_axis) if label != "rho_tor_norm" else new_axis,
            # rho_pol_norm=Function(axis,  self.rho_pol_norm)(new_axis) if label != "rho_pol_norm" else new_axis,
            # area=Function(axis,  self.area)(new_axis) if label != "area" else new_axis,
            # surface=Function(axis,  self.surface)(new_axis) if label != "surface" else new_axis,
            # volume=Function(axis,  self.volume)(new_axis) if label != "volume" else new_axis,
            dvolume_drho_tor=function_like(axis,  self.dvolume_drho_tor)(new_axis),
        )

    r0: float = sp_property()
    b0: float = sp_property()

    psi_axis: float = sp_property()
    """Poloidal flux at the magnetic axis  [Wb]."""

    psi_magnetic_axis: float = sp_property()

    psi_boundary: float = sp_property()
    """Poloidal flux at the selected plasma boundary  [Wb]."""

    rho_tor_boundary: float = sp_property()

    psi_norm:  np.ndarray = sp_property()

    psi: np.ndarray = sp_property(lambda self: self.psi_norm * (self.psi_boundary-self.psi_axis)+self.psi_axis)
    """Poloidal magnetic flux {dynamic} [Wb]. This quantity is COCOS-dependent, with the following transformation"""

    rho_tor_norm: np.ndarray = sp_property()
    """Normalized toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate
            at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibrium calculation,
            see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]
    """

    rho_tor: np.ndarray = sp_property(lambda self: self.rho_tor_norm*self.rho_tor_boundary)
    """Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m].
            The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""

    rho_pol_norm:  np.ndarray = sp_property()
    """Normalized poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))) {dynamic} [-]"""

    area:  np.ndarray = sp_property()
    """Cross-sectional area of the flux surface {dynamic} [m^2]"""

    surface:  np.ndarray = sp_property()
    """Surface area of the toroidal flux surface {dynamic} [m^2]"""

    volume:  np.ndarray = sp_property()
    """Volume enclosed inside the magnetic surface {dynamic} [m^3]"""

    dvolume_drho_tor: np.ndarray = sp_property()


class MagneticCoordSystem(Dict[Node]):
    r"""
        Flux surface coordinate system on a square grid of flux and poloidal angle

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
                 psirz: Field,
                 B0: float,
                 R0: float,
                 Ip: float,
                 fpol:     np.ndarray,
                 psi_norm: Union[int, np.ndarray] = 128,
                 theta:    Union[int, np.ndarray] = 32,
                 grid_type_index=13,
                 **kwargs):
        """
            Initialize FluxSurface
        """
        super().__init__(*args, **kwargs)
        self._grid_type_index = grid_type_index

        self._psirz = psirz

        self._Ip = Ip
        self._B0 = B0
        self._R0 = R0
        self._fvac = self._B0*self._R0

        # @TODO: COCOS transformation
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
        elif isinstance(fpol, np.ndarray):
            if len(fpol) == len(self._psi_norm):
                self._fpol = function_like(self._psi_norm, fpol)
            else:
                self._fpol = function_like(np.linspace(0, 1.0, len(fpol)), fpol)
        else:
            raise RuntimeError(f"fpol is not defined!")

        logger.debug(f"Create MagneticCoordSystem: type index={self._grid_type_index} primary='psi'  ")

    @property
    def r0(self) -> float:
        return self._R0

    @property
    def b0(self) -> float:
        return self._B0

    @property
    def vacuum_toroidal_field(self):
        return {"r0": self.r0, "b0": self.b0}

    @cached_property
    def radial_grid(self) -> RadialGrid:
        return RadialGrid(
            r0=self.r0,
            b0=self.b0,
            psi_axis=self.psi_axis,
            psi_boundary=self.psi_boundary,
            rho_tor_boundary=self.rho_tor[-1],
            psi_norm=self.psi_norm,
            rho_tor_norm=self.rho_tor_norm,
            # rho_pol_norm=self.rho_pol_norm,
            # area=self.area,
            # surface=self.surface,
            dvolume_drho_tor=self.dvolume_drho_tor,
            # volume=self.volume,
        )

    @property
    def grid_type_index(self) -> int:
        return self._grid_type_index

    @property
    def cocos(self) -> int:
        logger.warning("NOT IMPLEMENTED!")
        return self._cocos

    @cached_property
    def cocos_flag(self) -> int:
        return 1 if self.psi_boundary > self.psi_axis else -1

    @cached_property
    def critical_points(self) -> Tuple[Sequence[OXPoint], Sequence[OXPoint]]:
        opoints = []
        xpoints = []

        for r, z, psi, D in find_critical_points(self._psirz, *self._psirz.mesh.bbox, tolerance=self._psirz.mesh.dx):
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

            bbox = self._psirz.mesh.bbox
            Rmid = (bbox[0] + bbox[2])/2.0
            Zmid = (bbox[1] + bbox[3])/2.0

            opoints.sort(key=lambda x: (x.r - Rmid)**2 + (x.z - Zmid)**2)

            o_r = opoints[0].r
            o_z = opoints[0].z
            # TOOD: NEED　IMPROVMENT!!
            xpoints.sort(key=lambda x: (x.r - o_r)**2 + (x.z - o_z)**2)
            # psi_axis = opoints[0].psi
            #xpoints.sort(key=lambda x: (x.psi - psi_axis)**2)

        return opoints, xpoints

    def find_surface(self, psi: Union[float, Sequence] = None, o_point: OXPoint = True) -> Iterator[Tuple[float, GeoObject]]:
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

        R, Z = self._psirz.mesh.xy

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
                        logger.warning(f"The magnetic surface average is not well defined on the separatrix!")
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

                        self._mesh = [theta]
                        self._points = points
                    surf = CubicSplineCurve(points, [theta])

                    yield level, surf
                    break

                if surf is None:
                    if np.isclose(level, o_point.psi):
                        yield level, Point(o_point.r, o_point.z)
                    else:
                        raise RuntimeError(f"{level},{o_point.psi},{(max(theta),min(theta))}")

    def find_surface_by_psi_norm(self, psi_norm: Union[float, Sequence], *args,   **kwargs) -> Iterator[Tuple[float, GeoObject]]:
        yield from self.find_surface(np.asarray(psi_norm, dtype=float)*(self.psi_boundary-self.psi_axis)+self.psi_axis, *args,  **kwargs)

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
    def psi_axis(self) -> float:
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
        geometric_axis_rz: np.ndarray
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

    def shape_property(self, psi_norm: Union[float, Sequence[float]] = None) -> ShapeProperty:
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
            return MagneticCoordSystem.ShapeProperty(
                # RZ position of the geometric axis of the magnetic surfaces (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the surface)
                RZTuple(
                    function_like(psi_norm, (rmin+rmax)*0.5),
                    function_like(psi_norm, (zmin+zmax)*0.5)),
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
            return MagneticCoordSystem.ShapeProperty(
                # RZ position of the geometric axis of the magnetic surfaces (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the surface)
                RZTuple((rmin+rmax)*0.5, (zmin+zmax)*0.5),
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
    # mesh

    @cached_property
    def mesh(self) -> Mesh:

        if self._grid_type_index != 13:
            raise NotImplementedError(self._grid_type_index)

        mesh = CurvilinearMesh([surf for _, surf in self.find_surface_by_psi_norm(self._psi_norm, o_point=True)],
                               [self._psi_norm, self._theta/TWOPI], cycle=[False, True])

        return mesh

    @property
    def r(self) -> np.ndarray:
        return self.mesh.xy[:, :, 0]

    @property
    def z(self) -> np.ndarray:
        return self.mesh.xy[:, :, 1]

    def psirz(self, r: _TCoord, z: _TCoord, *args, **kwargs) -> _TCoord:
        return self._psirz(r, z, *args, **kwargs)

    def psi_norm_rz(self, r: _TCoord, z: _TCoord, *args, **kwargs) -> _TCoord:
        return (self.psirz(r, z, *args, **kwargs)-self.psi_axis)/(self.psi_boundary-self.psi_axis)

    def Br(self, r: _TCoord, z: _TCoord) -> _TCoord:
        return self.psirz(r, z, dy=1) / r/TWOPI

    def Bz(self, r: _TCoord, z: _TCoord) -> _TCoord:
        return -self.psirz(r,  z, dx=1) / r/TWOPI

    def Btor(self, r: _TCoord, z: _TCoord) -> _TCoord:
        return self._fpol(self.psi_norm_rz(r, z)) / r

    def Bpol(self, r: _TCoord, z: _TCoord) -> _TCoord:
        r"""
            $B_{pol}= \left|\nabla \psi \right|/2 \pi R $
        """
        return self.grad_psi(r, z) / r / (TWOPI)

    def B2(self, r: _TCoord, z: _TCoord) -> _TCoord:
        return (self.Br(r, z)**2+self.Bz(r, z)**2 + self.Btor(r, z)**2)

    def grad_psi2(self,  r: _TCoord, z: _TCoord) -> _TCoord:
        return self.psirz(r, z, dx=1)**2+self.psirz(r, z, dy=1)**2

    def grad_psi(self,  r: _TCoord, z: _TCoord) -> _TCoord:
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

    def _surface_integral(self, func: Callable[[float, float], float] = None, surface_list=None) -> np.ndarray:
        r0 = self.o_point.r
        z0 = self.o_point.z

        ddpsi = np.sqrt(self.psirz(r0, z0, dx=2) * self.psirz(r0, z0, dy=2) + self.psirz(r0, z0, dx=1, dy=1)**2)

        c0 = TWOPI*r0**2/ddpsi

        if surface_list is None:
            surface_list = self.mesh.axis_iter()
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

    def surface_average(self,  func,   /, value_axis: Union[float, Callable[[float, float], float]] = None, **kwargs) -> np.ndarray:
        r"""
            $\left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}$
        """
        return self._surface_integral(func)/self.dvolume_dpsi

    ###############################
    # 1-D

    @property
    def psi_norm(self) -> np.ndarray:
        return self._psi_norm

    @property
    def psi(self) -> np.ndarray:
        return self.psi_norm * (self.psi_boundary-self.psi_axis) + self.psi_axis

    @cached_property
    def fpol(self) -> np.ndarray:
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        return self._fpol(self.psi_norm)

    @cached_property
    def ffprime(self) -> np.ndarray:
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        return self._fpol(self.psi_norm)*self._fpol.derivative(self.psi_norm)

    # @cached_property
    # def pprime(self) -> np.ndarray:
    #     """Diamagnetic function (F=R B_Phi)  [T.m]."""
    #     return self._pprime(self.psi_norm)

    @cached_property
    def dphi_dpsi(self) -> np.ndarray:
        return self.fpol * self.gm1 * self.dvolume_dpsi / TWOPI

    @cached_property
    def q(self) -> np.ndarray:
        r"""
            Safety factor
            (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)[-].
            $ q(\psi) =\frac{d\Phi}{2\pi d\psi} =\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{2\pi}$
        """
        return self.dphi_dpsi * self._s_Bp * self._s_2PI

    @cached_property
    def magnetic_shear(self) -> np.ndarray:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
        return self.rho_tor/self.q * function_like(self.rho_tor, self.q).derivative(self.rho_tor)

    @cached_property
    def phi(self) -> np.ndarray:
        r"""
            Note:
            $\Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi$
        """
        return function_like(self.psi_norm, self.dphi_dpsi).antiderivative(self.psi_norm)*(self.psi_boundary-self.psi_axis)

    @cached_property
    def phi_boundary(self) -> float:
        if not np.isclose(self.psi_norm[-1], 1.0):
            logger.warning(f"FIXME: psi_norm boudnary is {self.psi_norm[-1]} != 1.0 ")

        return self.phi[-1]

    @cached_property
    def rho_boundary(self) -> float:
        return np.sqrt(self.phi_boundary/(constants.pi * self._B0))

    @cached_property
    def rho_tor(self) -> np.ndarray:
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
        return np.sqrt(self.phi/(constants.pi * self._B0))

    @cached_property
    def rho_tor_norm(self) -> np.ndarray:
        return np.sqrt(self.phi/self.phi_boundary)

    @cached_property
    def volume(self) -> np.ndarray:
        """Volume enclosed in the flux surface[m ^ 3]"""
        return function_like(self.psi_norm, self.dvolume_dpsi).antiderivative(self.psi_norm)*(self.psi_boundary-self.psi_axis)

    @cached_property
    def surface(self) -> np.ndarray:
        """Surface area of the toroidal flux surface {dynamic} [m^2]"""
        return self.dvolume_drho_tor*self.gm7

    @cached_property
    def dvolume_drho_tor(self) -> np.ndarray:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
        return extrapolate_left(self._psi_norm, (TWOPI**2) * self.rho_tor/(self.gm1)/(self._fvac/self.fpol)/self._R0)

    # @cached_property
    # def volume1(self) -> np.ndarray:
    #     """Volume enclosed in the flux surface[m ^ 3]"""
    #     if self.rho_tor[0] > 1.0e-4:
    #         x = np.hstack([[0.0], self.rho_tor])
    #         dvdx = np.hstack([[self.dvolume_drho_tor[0]], self.dvolume_drho_tor])
    #     else:
    #         x = self.rho_tor
    #         dvdx = self.dvolume_drho_tor

    #     return Function(x, dvdx).antiderivative(self.rho_tor)

    @cached_property
    def drho_tor_dpsi(self) -> np.ndarray:
        r"""
            $\frac{d\rho_{tor}}{d\psi} =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                        =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                        =\frac{q}{2\pi B_{0}\rho_{tor}}
            $
        """
        return 1.0/self.dpsi_drho_tor

    @cached_property
    def dpsi_drho_tor(self) -> np.ndarray:
        """
            Derivative of Psi with respect to Rho_Tor[Wb/m].
        """
        return extrapolate_left(self._psi_norm, (self._s_Bp)*self._B0*self.rho_tor/self.q)

    @cached_property
    def dphi_dvolume(self) -> np.ndarray:
        return self.fpol * self.gm1

    @cached_property
    def gm1(self) -> np.ndarray:
        r"""
            Flux surface averaged $1/R ^ 2  \left[m ^ {-2}\right]$

            $\left\langle \frac{1}{R^{2}} \right\rangle$
        """
        return self.surface_average(lambda r, z: 1.0/(r**2))

    @cached_property
    def gm2_(self) -> np.ndarray:
        return self.surface_average(lambda r, z: self.grad_psi2(r, z)/(r**2))

    @cached_property
    def gm2(self) -> np.ndarray:
        r"""
            Flux surface averaged $\left| \nabla \rho_{tor}\right|^2/R^2  [m^-2]$

            $\left\langle\left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle$
        """

        return extrapolate_left(self._psi_norm,
                                self.surface_average(lambda r, z: self.grad_psi2(r, z)/(r**2)) / (self.dpsi_drho_tor ** 2))

    @cached_property
    def gm3(self) -> np.ndarray:
        r"""
            Flux surface averaged $\left| \nabla \rho_{tor}\right|^2  [-]$
            $\left\langle \left|\nabla\rho\right|^{2}\right\rangle$
        """
        return extrapolate_left(self._psi_norm,
                                self.surface_average(self.grad_psi2) / (self.dpsi_drho_tor ** 2))

    @cached_property
    def gm4(self) -> np.ndarray:
        r"""Flux surface averaged $1/B^2$  [T ^ -2]    $\left\langle \frac{1}{B^{2}}\right\rangle$    """
        return self.surface_average(lambda r, z: 1.0/self.B2(r, z))

    @cached_property
    def gm5(self) -> np.ndarray:
        r""" Flux surface averaged $B ^ 2  [T ^ 2]$     $ \left\langle B^{2}\right\rangle$ """
        return self.surface_average(lambda r, z: self.B2(r, z))

    @cached_property
    def gm6(self) -> np.ndarray:
        r"""
            Flux surface averaged  $\left| \nabla \rho_{tor}\right|^2/B^2  [T^-2]$
            $\left\langle \frac{\left |\nabla\rho\right|^{2}}{B^{2}}\right\rangle$

        """
        return extrapolate_left(self._psi_norm,
                                self.surface_average(lambda r, z: self.grad_psi2(r, z)/self.B2(r, z)) / (self.dpsi_drho_tor ** 2))

    @cached_property
    def gm7(self) -> np.ndarray:
        r"""
            Flux surface averaged $\left| \nabla \rho_{tor}\right|$ [-]
            $\left\langle \left |\nabla\rho\right |\right\rangle$
        """
        return extrapolate_left(self._psi_norm,
                                self.surface_average(lambda r, z: np.sqrt(self.grad_psi2(r, z))) / self.dpsi_drho_tor)

    @cached_property
    def gm8(self) -> np.ndarray:
        r"""
            Flux surface averaged R[m]
            $\left\langle R\right\rangle$
        """
        return self.surface_average(lambda r, z: r)

    @cached_property
    def gm9(self) -> np.ndarray:
        r"""
            Flux surface averaged $1/R[m ^{-1}]$
            $\left\langle \frac{1}{R}\right\rangle$
        """
        return self.surface_average(lambda r, z: 1.0 / r)

    def plot_contour(self, axis, levels=16):
        import matplotlib.pyplot as plt

        if isinstance(levels, int):
            levels = np.linspace(0, 1, levels)
        elif isinstance(levels, (collections.abc.Sequence)):
            l_min, l_max = levels
            levels = np.linspace(l_min, l_max, 16)

        levels = levels*(self.psi_boundary-self.psi_axis)+self.psi_axis

        field = self._psirz

        R, Z = field.mesh.xy

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

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @cached_property
    def magnetic_axis(self):
        """Magnetic axis position and toroidal field	structure"""
        return convert_to_named_tuple(self._coord.magnetic_axis)

    @sp_property
    def x_points(self):
        _, x = self._coord.critical_points
        return x

    @sp_property
    def psi_axis(self) -> float:
        """Poloidal flux at the magnetic axis[Wb]."""
        o, _ = self._coord.critical_points
        return o[0].psi

    @sp_property
    def psi_boundary(self) -> float:
        """Poloidal flux at the selected plasma boundary[Wb]."""
        _, x = self._coord.critical_points
        if len(x) > 0:
            return x[0].psi
        else:
            raise ValueError(f"No x-point")


class EquilibriumProfiles1D(_T_equilibrium_profiles_1d):

    @cached_property
    def _predefined_psi_norm(self):
        psi = self._entry.get("psi", None)
        return (psi-psi[0])/(psi[-1]-psi[0])

    @sp_property
    def pressure(self) -> Function:
        return function_like(self._predefined_psi_norm, self._entry.get("pressure", None))

    @sp_property
    def dpressure_dpsi(self) -> Function:
        return function_like(self._predefined_psi_norm,  self._entry.get("dpressure_dpsi", None))

    @property
    def pprime(self) -> Function:
        return self.dpressure_dpsi

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def ffprime(self) -> Function:
        """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
        return function_like(self._coord.psi_norm, self._coord.ffprime)

    @sp_property
    def f_df_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.ffprime)

    @sp_property
    def fpol(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.fpol)

    @sp_property
    def f(self) -> Function:
        return self.fpol

    @sp_property
    def plasma_current(self) -> Function:
        return self.gm2 * self.dvolume_drho_tor / self.dpsi_drho_tor/constants.mu_0

    @sp_property
    def j_tor(self) -> Function:
        return self.plasma_current.derivative() / (self._coord.psi_boundary - self._coord.psi_axis)/self.dvolume_dpsi * self._coord.r0

    @sp_property
    def j_parallel(self) -> Function:
        fvac = self._coord._fvac
        d = np.asarray(Function(np.asarray(self.volume), np.asarray(fvac*self.plasma_current/self.fpol)).derivative())
        return self._coord.r0*(self.fpol / fvac)**2 * d

    @sp_property
    def psi_norm(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.psi_norm)

    @sp_property
    def psi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.psi)

    @sp_property
    def dphi_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dphi_dpsi)

    @sp_property
    def q(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.q)

    @sp_property
    def phi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.phi)

    @sp_property
    def rho_tor(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.rho_tor)

    @sp_property
    def rho_tor_norm(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.rho_tor_norm)

    @sp_property
    def drho_tor_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.drho_tor_dpsi)

    @sp_property
    def rho_volume_norm(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.rho_volume_norm)

    @sp_property
    def area(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.area)

    @sp_property
    def darea_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.darea_dpsi)

    @sp_property
    def darea_drho_tor(self) -> Function	:
        return function_like(self._coord.psi_norm, self._coord.darea_drho_tor)

    @sp_property
    def surface(self):
        return function_like(self._coord.psi_norm, self._coord.surface)

    @sp_property
    def volume(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.volume)

    @sp_property
    def dvolume_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dvolume_dpsi)

    @sp_property
    def dpsi_drho_tor(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dpsi_drho_tor)

    @sp_property
    def dpsi_drho_tor_norm(self) -> Function:
        return self.dpsi_drho_tor*self.rho_tor[-1]

    @sp_property
    def dvolume_drho_tor(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dvolume_drho_tor)

    @cached_property
    def shape_property(self) -> MagneticCoordSystem.ShapeProperty:
        return self._coord.shape_property()

    @sp_property
    def geometric_axis(self) -> RZTuple[Function]:
        gaxis = self.shape_property.geometric_axis
        return RZTuple[Function](function_like(self._coord.psi_norm, gaxis.r),
                                 function_like(self._coord.psi_norm, gaxis.z))

    @sp_property
    def minor_radius(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.minor_radius)

    @sp_property
    def r_inboard(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.r_inboard)

    @sp_property
    def r_outboard(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.r_outboard)

    # @sp_property
    # def elongation(self) -> Function:
    #     """Elongation. {dynamic}[-]"""
    #     return self.shape_property.elongation
    @sp_property
    def elongation(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.elongation)

    @sp_property
    def triangularity(self) -> Function	:
        return function_like(self._coord.psi_norm, self.shape_property.triangularity)

    @sp_property
    def triangularity_upper(self) -> Function	:
        return function_like(self._coord.psi_norm, self.shape_property.triangularity_upper)

    @sp_property
    def triangularity_lower(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.triangularity_lower)

    @sp_property
    def gm1(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm1)

    @sp_property
    def gm2(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm2)

    @sp_property
    def gm3(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm3)

    @sp_property
    def gm4(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm4)

    @sp_property
    def gm5(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm5)

    @sp_property
    def gm6(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm6)

    @sp_property
    def gm7(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm7)

    @sp_property
    def gm8(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm8)

    @sp_property
    def gm9(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm9)

    @sp_property
    def magnetic_shear(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.magnetic_shear)

    @sp_property
    def trapped_fraction(self, value) -> Function:
        """Trapped particle fraction[-]
            Tokamak 3ed, 14.10
        """
        if value is _not_found_:
            epsilon = self.rho_tor/self._coord.r0
            value = np.asarray(1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon)))
        return function_like(self._coord.psi_norm, value)


class EquilibriumProfiles2D(_T_equilibrium_profiles_2d):
    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @cached_property
    def psi(self) -> Field:
        return self._coord._psirz  # (self._coord.r,self._coord.z)

    @sp_property
    def r(self) -> np.ndarray:
        """Values of the major radius on the grid  [m] """
        return self._coord.r

    @sp_property
    def z(self) -> np.ndarray:
        """Values of the Height on the grid  [m] """
        return self._coord.z

    # @sp_property
    # def psi(self):
    #     """Values of the poloidal flux at the grid in the poloidal plane  [Wb]. """
    #     return self.apply_psifunc(lambda p: p, unit="Wb")

    @sp_property
    def phi(self):
        return self.apply_psifunc("phi")

    @sp_property
    def j_tor(self):
        return self.apply_psifunc("j_tor")

    @sp_property
    def j_parallel(self):
        return self.apply_psifunc("j_parallel")

    @sp_property
    def b_field_r(self):
        return Field(self._coord.Br, self._coord.r, self._coord.z, mesh_type="curvilinear")

    @sp_property
    def b_field_z(self):
        return Field(self._coord.Bz, self._coord.r, self._coord.z, mesh_type="curvilinear")

    @sp_property
    def b_field_tor(self):
        return Field(self._coord.Btor, self._coord.r, self._coord.z, mesh_type="curvilinear")


class EquilibriumBoundary(_T_equilibrium_boundary):

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def outline(self) -> RZTuple:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface(self.psi, o_point=True))
        return RZTuple(surf.xyz[0], surf.xyz[1])

    @sp_property
    def x_point(self):
        _, xpt = self._parent.critical_points
        return xpt

    psi_norm: float = sp_property(default_value=0.999)
    """Value of the normalized poloidal flux at which the boundary is taken (typically 99.x %),
            the flux being normalized to its value at the separatrix """

    @sp_property
    def psi_axis(self) -> float:
        return self._coord.psi_axis

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord.psi_boundary

    @sp_property
    def psi(self) -> float:
        """Value of the poloidal flux at which the boundary is taken  [Wb]"""
        return self.psi_norm*(self._coord.psi_boundary-self._coord.psi_axis)+self._coord.psi_axis

    @property
    def shape_property(self) -> MagneticCoordSystem.ShapeProperty:
        return self._coord.shape_property(self.psi_norm)

    @sp_property
    def geometric_axis(self) -> RZTuple:
        """RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)"""
        return self.shape_property.geometric_axis

    @sp_property
    def minor_radius(self) -> float:
        """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
        return self.shape_property.minor_radius

    @sp_property
    def elongation(self) -> float:
        """Elongation of the plasma boundary. [-]	"""
        return self.shape_property.elongation

    @sp_property
    def elongation_upper(self) -> float:
        """Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]	"""
        return self.shape_property.elongation_upper

    @sp_property
    def elongation_lower(self) -> float:
        """Elongation(lower half w.r.t. geometric axis) of the plasma boundary. [-]	"""
        return self.shape_property.elongation_lower

    @sp_property
    def triangularity(self) -> float:
        """Triangularity of the plasma boundary. [-]	"""
        return self.shape_property.triangularity

    @sp_property
    def triangularity_upper(self) -> float:
        """Upper triangularity of the plasma boundary. [-]	"""
        return self.shape_property.triangularity_upper

    @sp_property
    def triangularity_lower(self) -> float:
        """Lower triangularity of the plasma boundary. [-]"""
        return self.shape_property.triangularity_lower

    @sp_property
    def strike_point(self)	:
        """Array of strike points, for each of them the RZ position is given	struct_array [max_size=unbounded]	1- 1...N"""
        return NotImplemented

    @sp_property
    def active_limiter_point(self):
        """	RZ position of the active limiter point (point of the plasma boundary in contact with the limiter)"""
        return NotImplemented


class EquilibriumBoundarySeparatrix(_T_equilibrium_boundary_separatrix):

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def outline(self) -> RZTuple:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface_by_psi_norm(1.0, o_point=None))
        return RZTuple(*surf.xyz)

    @sp_property
    def psi_axis(self) -> float:
        return self._coord.psi_axis

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord.psi_boundary

    @sp_property
    def psi(self) -> float:
        return self._coord.psi_norm*(self._coord.psi_boundary-self._coord.psi_axis)+self._coord.psi_axis


@Equilibrium.register(["fy_eq"])
class FyEquilibrium(Equilibrium):
    pass

    # def update(self,  *args,
    #            wall: Wall = _undefined_,
    #            pf_active: PFActive = _undefined_,
    #            core_profiles=_undefined_,
    #            **kwargs):
    #     super().update(*args, **kwargs)

    #     # self.profiles_1d.pressure = core_profiles.profiles_1d.pressure
    #     # self.profiles_1d.pressure = core_profiles.profiles_1d.fpol

    #     # # call Eq solver
    #     # psi_2d = self._eq_solver(
    #     #     {
    #     #         "vacuum_toroidal_field": self.vacuum_toroidal_field,
    #     #         "global_quantities": {"ip": self.global_quantities.ip,
    #     #                               "bet": self.global_quantities.betn
    #     #                               }
    #     #     }
    #     # )
    #     # self.profiles_2d.psi = psi_2d
    #     # return {
    #     #     "psi": psi_2d,
    #     #     "fpol": fpol,
    #     #     "pprime": pprime,
    #     # }
    #     return

    # profiles_1d: EquilibriumProfiles1D = sp_property()

    # profiles_2d: List[EquilibriumProfiles2D] = sp_property()

    # global_quantities: EquilibriumGlobalQuantities = sp_property()

    # boundary: EquilibriumBoundary = sp_property()

    # boundary_separatrix: EquilibriumBoundarySeparatrix = sp_property()

    # @sp_property
    # def coordinate_system(self, desc) -> MagneticCoordSystem:
    #     psirz = self.profiles_2d[0].psi

    #     if not isinstance(psirz, Field):
    #         psirz = Field(self.profiles_2d[0].psi,
    #                       self.profiles_2d[0].grid.dim1,
    #                       self.profiles_2d[0].grid.dim2,
    #                       mesh="rectilinear")

    #     psi_1d = self.profiles_1d._entry.get("psi")
    #     fpol_1d = self.profiles_1d._entry.get("f", _not_found_)
    #     if not isinstance(psi_1d, np.ndarray) or len(psi_1d) != len(fpol_1d):
    #         psi_1d = np.linspace(0, 1.0, len(fpol_1d))

    #     if isinstance(psi_1d, np.ndarray):
    #         psi_1d = (psi_1d-psi_1d[0])/(psi_1d[-1]-psi_1d[0])

    #     # pprime_1d = self.profiles_1d._entry.get("dpressure_dpsi", None)

    #     res = MagneticCoordSystem(
    #         psirz=psirz,
    #         B0=self.vacuum_toroidal_field.b0,
    #         R0=self.vacuum_toroidal_field.r0,
    #         Ip=self.global_quantities.ip,
    #         fpol=function_like(psi_1d, fpol_1d),
    #         # pprime=self.profiles_1d._entry.get("dpressure_dpsi", None),
    #         # fpol=function_like(psi_norm, self.profiles_1d._entry.get("f", None)),
    #         # pprime=function_like(psi_norm, self.profiles_1d._entry.get("dpressure_dpsi", None)),
    #         **desc
    #     )
    #     return res

    # @property
    # def radial_grid(self) -> RadialGrid:
    #     return self.coordinate_system.radial_grid


__SP_EXPORT__ = FyEquilibrium
