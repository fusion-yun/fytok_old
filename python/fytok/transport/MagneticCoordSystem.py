import collections
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Iterator, Sequence, Tuple, TypeVar, Union

from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.data.Node import Dict, List
from spdm.geometry.CubicSplineCurve import CubicSplineCurve
from spdm.geometry.GeoObject import GeoObject, _TCoord
from spdm.geometry.Point import Point
from spdm.mesh.CurvilinearMesh import CurvilinearMesh
from spdm.mesh.Mesh import Mesh
from spdm.numlib import constants, np
from spdm.numlib.contours import find_countours
from spdm.numlib.optimize import find_critical_points
from spdm.util.logger import deprecated, logger
from spdm.util.utilities import _not_found_, try_get


from ..common.Misc import RZTuple, VacuumToroidalField

TOLERANCE = 1.0e-6
EPS = np.finfo(float).eps

TWOPI = 2.0*constants.pi


@dataclass
class OXPoint:
    r: float
    z: float
    psi: float


# OXPoint = collections.namedtuple('OXPoint', "r z psi")


class MagneticCoordSystem(object):
    r"""
        Flux surface coordinate system on a square grid of flux and poloidal angle

        .. math::
            V^{\prime}\left(\rho\right)=\frac{\partial V}{\partial\rho}=2\pi\int_{0}^{2\pi}\sqrt{g}d\theta=2\pi\oint\frac{R}{\left|\nabla\rho\right|}dl

        .. math::
            \left\langle\alpha\right\rangle\equiv\frac{2\pi}{V^{\prime}}\int_{0}^{2\pi}\alpha\sqrt{g}d\theta=\frac{2\pi}{V^{\prime}}\varoint\alpha\frac{R}{\left|\nabla\rho\right|}dl

        Magnetic Flux Coordinates
        psi         :                     ,  flux function , $B \cdot \nabla \psi=0$ need not to be the poloidal flux funcion $\Psi$
        theta       : 0 <= theta   < 2*pi ,  poloidal angle
        phi         : 0 <= phi     < 2*pi ,  toroidal angle
    """
    COCOS_INDEX = 11
    COCOS_TABLE = [
        # e_Bp ,    \sigma_{Bp},    \sigma_{R\varphi\Z}
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

    def __init__(self,
                 psirz: Field,
                 fpol: Union[np.ndarray, Function],
                 R0: float,
                 B0: float,
                 psi_norm: np.ndarray = None,
                 ntheta: int = 128,
                 grid_type_index: int = 13,
                 cocos_index=11):
        """
            Initialize FluxSurface
        """

        super().__init__()
        self._grid_type_index = grid_type_index

        # @TODO: COCOS transformation
        self._s_B0 = np.sign(B0)
        self._s_fpol = np.sign(fpol[0])

        self._ntheta = ntheta if ntheta is not None else 128
        self._b0 = np.abs(B0)
        self._r0 = R0
        self._fvac = self._b0*self._r0

        self._psirz = psirz

        if psi_norm is None:
            psi_norm = 128
        if isinstance(psi_norm, int):
            self._psi_norm = np.linspace(0.001, 0.99, psi_norm)
        else:
            self._psi_norm = np.asarray(psi_norm)

        if isinstance(fpol, Function):
            self._fpol = np.abs(fpol)
        elif isinstance(fpol, np.ndarray) and fpol.shape == self._psi_norm.shape:
            self._fpol = Function(self._psi_norm, np.abs(fpol))
        else:
            raise TypeError(f"{type(fpol)}")

        # if isinstance(pprime, Function):
        #     self._ffprime = pprime
        # elif isinstance(pprime, np.ndarray) and pprime.shape == self._psi_norm.shape:
        #     self._pprime = Function(self._psi_norm, pprime)
        # else:
        #     raise TypeError(f"{type(pprime)}")

        # dim1=self["grid.dim1"]
        # dim2=self["grid.dim2"]

        # if isinstance(dim1, np.ndarray):
        #     u=dim1
        # elif dim1 == None:
        #     u=np.linspace(0.0001,  0.99,  len(self._ffprime))
        # elif isinstance(dim1, int):
        #     u=np.linspace(0.0001,  0.99,  dim1)
        # else:
        #     u=np.asarray([dim1])

        # if isinstance(dim2, np.ndarray):
        #     v=dim2
        # elif dim2 == None:
        #     v=np.linspace(0.0,  1.0,  128)
        # elif isinstance(dim2, int):
        #     v=np.linspace(0.0, 1.0,  dim2)
        # elif isinstance(dim2, np.ndarray):
        #     v=dim2
        # else:
        #     v=np.asarray([dim2])

        # self._uv=[u, v]

    @property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        return VacuumToroidalField(self._r0, self._b0)

    def radial_grid(self, primary_axis=None, axis=None):
        """
            Radial grid
        """
        psi_norm = self.psi_norm

        if primary_axis == "psi_norm" or primary_axis is None:
            if axis is not None:
                psi_norm = axis
        else:
            p_axis = try_get(self, primary_axis)
            if isinstance(p_axis, Function):
                p_axis = p_axis.__array__()
            if not isinstance(p_axis, np.ndarray):
                raise NotImplementedError(primary_axis)
            psi_norm = Function(p_axis, psi_norm)(np.linspace(p_axis[0], p_axis[-1], p_axis.shape[0]))
        return RadialGrid(self, psi_norm)

    @property
    def grid_type_index(self) -> int:
        return self._grid_type_index

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

            psi_axis = opoints[0].psi

            xpoints.sort(key=lambda x: (x.psi - psi_axis)**2)

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

        if not isinstance(psi, (collections.abc.Sequence, np.ndarray)):
            psi = [psi]

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
                        xpt = np.asarray([x_point.r, x_point.z])
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
                    logger.debug(level)
                    if np.isclose(level, o_point.psi):
                        yield level, Point(o_point.r, o_point.z)
                    else:
                        raise RuntimeError(f"{level},{o_point.psi},{(max(theta),min(theta))}")

                    # idx = [i for i, pt in points if np.allclose(pt, x_pt)]
                    # if len(idx) != 2:
                    #     logger.debug(f"irregular magnetic surface! ")
                    #     continue
                    # theta = theta[idx[0]:idx[1]]
                    # points = points[idx[0]:idx[1]]

    def find_surface_by_psi_norm(self, psi_norm: Union[float, Sequence], *args,   **kwargs) -> Iterator[Tuple[float, GeoObject]]:

        yield from self.find_surface(np.asarray(psi_norm)*(self.psi_boundary-self.psi_axis)+self.psi_axis, *args,  **kwargs)

    ###############################
    # 0-D

    @cached_property
    def magnetic_axis(self):
        o, _ = self.critical_points
        if not o:
            raise RuntimeError(f"Can not find magnetic axis")

        return Dict({
            "r": o[0].r,
            "z": o[0].z,
            "b_field_tor": NotImplemented
        })

    # @cached_property
    # def boundary(self) -> Curve:
    #     return next(self.find_surface(self.psi_boundary, only_closed=True))

    @cached_property
    def cocos_flag(self) -> int:
        return 1 if self.psi_boundary > self.psi_axis else -1

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
    class ShapePropety:
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

    def shape_property(self, psi_norm: Union[float, Sequence[float]] = None) -> ShapePropety:
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

        sbox = np.asarray([[*shape_box(s)] for _, s in self.find_surface_by_psi_norm(psi_norm)])

        if sbox.shape[0] == 1:
            rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox[0]
        else:
            rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox.T

        return MagneticCoordSystem.ShapePropety(
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
        # return self.create_mesh(self._psi_norm, None, type_index=13)

        if self._grid_type_index == 13:
            primary = "psi"

        if isinstance(self._ntheta, int):
            theta = np.linspace(0, 1, self._ntheta)
        else:
            theta = np.asarray(self._ntheta)

        mesh = CurvilinearMesh([surf for _, surf in self.find_surface_by_psi_norm(self._psi_norm, o_point=True)],
                               [self._psi_norm, theta], cycle=[False, True])

        logger.debug(f"Create mesh: type index={self._grid_type_index} primary={primary}  ")

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
        return self.fpol(self.psi_norm_rz(r, z)) / r

    def Bpol(self, r: _TCoord, z: _TCoord) -> _TCoord:
        r"""
            .. math:: B_{pol} =   |\nabla \psi|/2 \pi R  
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

    def _surface_integral(self, func: Callable[[float, float], float]) -> np.ndarray:
        return np.asarray([axis.integral(func) for _, axis in self.mesh.axis_iter()])

    @cached_property
    def dvolume_dpsi(self) -> np.ndarray:
        r"""
            .. math:: V^{\prime} =  2 \pi  \int{ R / |\nabla \psi| * dl }
            .. math:: V^{\prime}(psi)= 2 \pi  \int{ dl * R / |\nabla \psi|}
        """
        return self._surface_integral(lambda r, z, _bpol=self.Bpol: 1.0/_bpol(r, z))

    def surface_average(self,  func,   /, **kwargs) -> np.ndarray:
        r"""
            .. math:: \left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}
        """
        return self._surface_integral(lambda r, z, _func=func, _bpol=self.Bpol, _kwargs=kwargs: _func(r, z, **_kwargs)/_bpol(r, z))/self.dvolume_dpsi

    ###############################
    # 1-D

    @cached_property
    def fpol(self) -> np.ndarray:
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        return self._fpol(self.psi_norm)

    @cached_property
    def plasma_current(self) -> np.ndarray:
        """Toroidal current driven inside the flux surface.
          .. math:: I_{pl}\equiv\int_{S_{\zeta}}\mathbf{j}\cdot dS_{\zeta}=\frac{\text{gm2}}{4\pi^{2}\mu_{0}}\frac{\partial V}{\partial\psi}\left(\frac{\partial\psi}{\partial\rho}\right)^{2}
         {dynamic}[A]"""
        return self.gm2 * self.dvolume_drho_tor/(TWOPI**2) * self.dpsi_drho_tor/constants.mu_0

    @cached_property
    def j_parallel(self) -> np.ndarray:
        r"""Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0 {dynamic}[A/m ^ 2]. """
        d = np.asarray(Function(self.volume, self._fvac*self.plasma_current/self.fpol).derivative())
        return TWOPI*self._r0*(self.fpol/self._fvac)**2 * d

    @property
    def psi_norm(self) -> np.ndarray:
        return self._psi_norm

    @cached_property
    def psi(self) -> np.ndarray:
        return self.psi_norm * (self.psi_boundary-self.psi_axis) + self.psi_axis

    @cached_property
    def dphi_dpsi(self) -> np.ndarray:
        return self.fpol * self.gm1 * self.dvolume_dpsi / TWOPI

    @cached_property
    def q(self) -> np.ndarray:
        r"""
            Safety factor
            (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)[-].
            .. math:: q(\psi) =\frac{d\Phi}{2\pi d\psi} =\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{2\pi}
        """
        return self.dphi_dpsi  # self.fpol * self.gm1 * self.dvolume_dpsi

    @cached_property
    def magnetic_shear(self) -> np.ndarray:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
        return self.rho_tor/self.q * Function(self.psi, self.q).derivative(self.psi)

    @cached_property
    def phi(self) -> np.ndarray:
        r"""
            Note:
            .. math::
                \Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi
        """
        # res = Function(self.volume, self.dphi_dvolume).antiderivative(self.volume)
        # res = Function(self.psi_norm, self.dphi_dpsi).antiderivative(self.psi_norm) \
        #     + self.dphi_dpsi[0]*(self.psi_norm[0])*2

        # res *= self.psi_boundary-self.psi_axis
        if self.psi_norm[0] > 0.0:
            x = np.hstack([[0.0], self.psi_norm])
            dvdx = np.hstack([[self.dphi_dpsi[0]], self.dphi_dpsi])
        else:
            x = self.psi_norm
            dvdx = self.dphi_dpsi
        return Function(x, dvdx).antiderivative(self.psi_norm)*(self.psi_boundary-self.psi_axis)

    @cached_property
    def rho_tor(self) -> np.ndarray:
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
        return np.sqrt(self.phi/(constants.pi * self._b0))

    @cached_property
    def rho_tor_norm(self) -> np.ndarray:
        return np.sqrt(self.phi/self.phi[-1])

    @cached_property
    def volume(self) -> np.ndarray:
        """Volume enclosed in the flux surface[m ^ 3]"""
        if self.psi_norm[0] > 0.0:
            x = np.hstack([[0.0], self.psi_norm])
            dvdx = np.hstack([[self.dvolume_dpsi[0]], self.dvolume_dpsi])
        else:
            x = self.psi_norm
            dvdx = self.dvolume_dpsi

        return Function(x, dvdx).antiderivative(self.psi_norm)*(self.psi_boundary-self.psi_axis)

    @cached_property
    def dvolume_drho_tor(self) -> np.ndarray:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
        return (TWOPI**2) * self.rho_tor/(self.gm1)/(self._fvac/self.fpol)/self._r0

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
            .. math::
                \frac{d\rho_{tor}}{d\psi} =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                        =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                        =\frac{q}{2\pi B_{0}\rho_{tor}}
        """
        return self.dvolume_dpsi / self.dvolume_drho_tor
        # return self.q/((TWOPI)*self._b0*self.rho_tor)

    @cached_property
    def dpsi_drho_tor(self) -> np.ndarray:
        """
            Derivative of Psi with respect to Rho_Tor[Wb/m].
        """
        return (TWOPI)*self._b0*self.rho_tor/self.q

    @cached_property
    def dphi_dvolume(self) -> np.ndarray:
        return self.fpol * self.gm1

    @cached_property
    def gm1(self) -> np.ndarray:
        r"""
            Flux surface averaged 1/R ^ 2  [m ^ -2]
            .. math: : \left\langle\frac{1}{R^{2}}\right\rangle
        """
        return self.surface_average(lambda r, z: 1.0/(r**2))

    @cached_property
    def gm2(self) -> np.ndarray:
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2/R^2  [m^-2]
            .. math:: \left\langle\left |\frac{\nabla\rho}{R}\right|^{2}\right\rangle
        """
        return self.surface_average(lambda r, z: self.grad_psi2(r, z)/(r**2)) * (self.drho_tor_dpsi ** 2)

    @cached_property
    def gm3(self) -> np.ndarray:
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2  [-]
            .. math:: {\left\langle \left |\nabla\rho\right|^{2}\right\rangle}
        """
        return self.surface_average(self.grad_psi2) * (self.drho_tor_dpsi ** 2)

    @cached_property
    def gm4(self) -> np.ndarray:
        r"""
            Flux surface averaged 1/B ^ 2  [T ^ -2]
            .. math: : \left\langle \frac{1}{B^{2}}\right\rangle
        """
        return self.surface_average(lambda r, z: 1.0/self.B2(r, z))

    @cached_property
    def gm5(self) -> np.ndarray:
        r"""
            Flux surface averaged B ^ 2  [T ^ 2]
            .. math: : \left\langle B^{2}\right\rangle
        """
        return self.surface_average(lambda r, z: self.B2(r, z))

    @cached_property
    def gm6(self) -> np.ndarray:
        r"""
            Flux surface averaged  .. math: : \left | \nabla \rho_{tor}\right|^2/B^2  [T^-2]
            .. math:: \left\langle \frac{\left |\nabla\rho\right|^{2}}{B^{2}}\right\rangle
        """
        return self.surface_average(lambda r, z: self.grad_psi2(r, z)/self.B2(r, z)) * (self.drho_tor_dpsi ** 2)

        # return np.ndarray(self._grid.psi_norm, self.surface_average(self.norm_grad_rho_tor**2/self.B2))

    @cached_property
    def gm7(self) -> np.ndarray:
        r"""
            Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right |  [-]
            .. math: : \left\langle \left |\nabla\rho\right |\right\rangle
        """
        return self.surface_average(lambda r, z: np.sqrt(self.grad_psi2(r, z))) * self.drho_tor_dpsi
        # d = self.surface_average(self.norm_grad_rho_tor)
        # return np.ndarray(self._grid.psi_norm, np.ndarray(self._grid.psi_norm[1:], d[1:]))

    @cached_property
    def gm8(self) -> np.ndarray:
        r"""
            Flux surface averaged R[m]
            .. math: : \left\langle R\right\rangle
        """
        return self.surface_average(lambda r, z: r)

    @cached_property
    def gm9(self) -> np.ndarray:
        r"""
            Flux surface averaged 1/R[m ^ -1]
            .. math: : \left\langle \frac{1}{R}\right\rangle
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

        F = np.asarray(field)

        for level, col in find_countours(F, R, Z, levels=levels):
            for segment in col:
                axis.add_patch(plt.Polygon(segment, fill=False, closed=np.all(
                    np.isclose(segment[0], segment[-1])), color="b", linewidth=0.2))
        return axis


class RadialGrid:
    r"""
        Radial grid
    """

    def __init__(self,  coordinate_system: MagneticCoordSystem, psi_norm=None) -> None:

        self._coordinate_system = coordinate_system
        self._psi_axis = coordinate_system.psi_axis
        self._psi_boundary = coordinate_system.psi_boundary

        if isinstance(psi_norm, int):
            self._psi_norm = np.linspace(0, 1.0,  psi_norm)
        elif psi_norm is None:
            self._psi_norm = coordinate_system.psi_norm
        else:
            self._psi_norm = np.asarray(psi_norm)

    def __serialize__(self) -> Dict:
        return {
            "psi_norm": self.psi_norm,
            "rho_tor_norm": self.rho_tor_norm,
        }

    def pullback(self, psi_norm):
        return RadialGrid(self._coordinate_system, psi_norm)

    @property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        return self._coordinate_system.vacuum_toroidal_field

    @cached_property
    def psi_magnetic_axis(self) -> float:
        """Poloidal flux at the magnetic axis  [Wb]."""
        return self._psi_axis

    @cached_property
    def psi_boundary(self) -> float:
        """Poloidal flux at the selected plasma boundary  [Wb]."""
        return self._psi_boundary

    @property
    def psi_norm(self) -> np.ndarray:
        return self._psi_norm

    @cached_property
    def psi(self) -> np.ndarray:
        """Poloidal magnetic flux {dynamic} [Wb]. This quantity is COCOS-dependent, with the following transformation"""
        return self.psi_norm * (self.psi_boundary-self.psi_magnetic_axis)+self.psi_magnetic_axis

    @cached_property
    def rho_tor_norm(self) -> np.ndarray:
        r"""Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate
            at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation,
            see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]
        """
        return Function(self._coordinate_system.psi_norm, self._coordinate_system.rho_tor_norm)(self.psi_norm)

    @cached_property
    def rho_tor(self) -> np.ndarray:
        r"""Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m].
            The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""
        return Function(self._coordinate_system.psi_norm, self._coordinate_system.rho_tor)(self.psi_norm)

    @cached_property
    def rho_pol_norm(self) -> np.ndarray:
        r"""Normalised poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))) {dynamic} [-]"""
        return Function(self._coordinate_system.psi_norm, self._coordinate_system.rho_pol_norm)(self.psi_norm)

    @cached_property
    def area(self) -> np.ndarray:
        """Cross-sectional area of the flux surface {dynamic} [m^2]"""
        return Function(self._coordinate_system.psi_norm, self._coordinate_system.area)(self.psi_norm)

    @cached_property
    def surface(self) -> np.ndarray:
        """Surface area of the toroidal flux surface {dynamic} [m^2]"""
        return Function(self._coordinate_system.psi_norm, self._coordinate_system.surface)(self.psi_norm)

    @cached_property
    def volume(self) -> np.ndarray:
        """Volume enclosed inside the magnetic surface {dynamic} [m^3]"""
        return Function(self._coordinate_system.psi_norm, self._coordinate_system.volume)(self.psi_norm)

    @cached_property
    def dvolume_drho_tor(self) -> np.ndarray:
        return Function(self._coordinate_system.psi_norm, self._coordinate_system.dvolume_drho_tor)(self.psi_norm)
