import collections
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
from fytok.RadialGrid import RadialGrid
from numpy import arctan2, cos, sin, sqrt
from numpy.lib.arraysetops import isin
from scipy.optimize import fsolve, root_scalar
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Field import Field
from spdm.data.mesh.CurvilinearMesh import CurvilinearMesh
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.geometry.CubicSplineCurve import CubicSplineCurve
from spdm.geometry.Curve import Curve
from spdm.geometry.Point import Point
from spdm.numerical.Function import Function
from spdm.util.logger import logger
from spdm.util.utilities import try_get

from ...Profiles import Profiles

TOLERANCE = 1.0e-6
EPS = np.finfo(float).eps

TWOPI = 2.0*scipy.constants.pi


class MagneticSurfaceCoordinateSystem:
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

    def __init__(self, uv, psirz: Field, ffprime: Function, fvac: float, *args, **kwargs):
        """
            Initialize FluxSurface
        """
        super().__init__()

        self._mesh = None
        self._uv = uv
        self._psirz = psirz
        self._ffprime = ffprime
        self._fvac = fvac

    @cached_property
    def critical_points(self):
        opoints = []
        xpoints = []
        for r, z, psi, D in self._psirz.find_peak():
            p = PhysicalGraph({"r": r, "z": z, "psi": psi})

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

            bbox = self._psirz.coordinates.mesh.bbox
            Rmid = (bbox[0][0] + bbox[0][1])/2.0
            Zmid = (bbox[1][0] + bbox[1][1])/2.0

            opoints.sort(key=lambda x: (x.r - Rmid)**2 + (x.z - Zmid)**2)

            psi_axis = opoints[0].psi

            xpoints.sort(key=lambda x: (x.psi - psi_axis)**2)

        return opoints, xpoints

    def flux_surface_map(self, u, v=None):
        o_points, x_points = self.critical_points

        if len(o_points) == 0:
            raise RuntimeError(f"Can not find o-point!")
        else:
            R0 = o_points[0].r
            Z0 = o_points[0].z
            psi0 = o_points[0].psi

        if len(x_points) == 0:
            R1 = self.psirz.coordinates.bbox[1][0]
            Z1 = Z0
            psi1 = 0.0
            theta0 = 0
        else:
            R1 = x_points[0].r
            Z1 = x_points[0].z
            psi1 = x_points[0].psi
            theta0 = arctan2(Z1 - Z0, R1 - R0)
        Rm = sqrt((R1-R0)**2+(Z1-Z0)**2)

        if not isinstance(u, (np.ndarray, collections.abc.MutableSequence)):
            u = [u]
        u = np.asarray(u)
        if v is None:
            v = self._uv[1]
        elif not isinstance(v, (np.ndarray, collections.abc.MutableSequence)):
            v = np.asarray([v])

        theta = v*TWOPI  # +theta0
        psi = u*(psi1-psi0)+psi0

        for psi_val in psi:
            for theta_val in theta:
                r0 = R0
                z0 = Z0
                r1 = R0+Rm * cos(theta_val)
                z1 = Z0+Rm * sin(theta_val)

                if not np.isclose(self.psirz(r1, z1), psi_val):
                    try:
                        def func(r): return float(self.psirz((1.0-r)*r0+r*r1, (1.0-r)*z0+r*z1)) - psi_val
                        sol = root_scalar(func,  bracket=[0, 1], method='brentq')
                    except ValueError as error:
                        raise ValueError(f"Find root fialed! {error} {psi_val}")

                    if not sol.converged:
                        raise ValueError(f"Find root fialed!")

                    r1 = (1.0-sol.root)*r0+sol.root*r1
                    z1 = (1.0-sol.root)*z0+sol.root*z1

                yield r1, z1

    def create_mesh(self, u, v, *args, type_index=13):
        logger.debug(f"create mesh! type index={type_index}")
        if type_index == 13:
            rz = np.asarray([[r, z] for r, z in self.flux_surface_map(u, v[:-1])]).reshape(len(u), len(v)-1, 2)
            rz = np.hstack((rz, rz[:, :1, :]))
            mesh = CurvilinearMesh(rz, [u, v], cycle=[False, True])
        else:
            raise NotImplementedError(f"TODO: reconstruct mesh {type_index}")

        return mesh

    def create_surface(self, psi_norm, v=None):
        if isinstance(psi_norm, CubicSplineCurve):
            return psi_norm
        elif psi_norm is None:
            return [self.create_surface(p) for p in self.psi_norm]
        elif isinstance(psi_norm, collections.abc.MutableSequence):
            return [self.create_surface(p) for p in psi_norm]
        elif isinstance(psi_norm, int):
            psi_norm = self.psi_norm[psi_norm]

        if np.abs(psi_norm) <= EPS:
            o, _ = self.critical_points
            return Point(*o)
        if v is None:
            v = self._uv[1]
        else:
            v = np.asarray(v)

        xy = np.asarray([[r, z] for r, z in self.flux_surface_map(psi_norm,  v[:-1])])
        xy = np.vstack((xy, xy[:1, :]))
        return CubicSplineCurve(v, xy, is_closed=True)

    @cached_property
    def surface_mesh(self):
        return self.create_surface(*self._uv)

    def br(self, r, z):
        return -self.psirz(r,  z, dy=1)/r

    def bz(self, r, z):
        return self.psirz(r, z, dx=1)/r

    def bpol(self, r, z):
        return np.sqrt(self.br(r, z)**2+self.bz(r, z)**2)

    def surface_integrate2(self, fun, *args, **kwargs):
        return np.asarray([surf.integrate(lambda r, z: fun(r, z, *args, **kwargs)/self.bpol(r, z)) * (2*scipy.constants.pi) for surf in self.surface_mesh])

    @cached_property
    def mesh(self):
        return self.create_mesh(*self._uv, type_index=13)

    @property
    def r(self):
        return self.mesh.xy[:, :, 0]

    @property
    def z(self):
        return self.mesh.xy[:, :, 1]

    @cached_property
    def psi_axis(self):
        """Poloidal flux at the magnetic axis  [Wb]."""
        o, _ = self.critical_points
        return o[0].psi

    @cached_property
    def psi_boundary(self):
        """Poloidal flux at the selected plasma boundary  [Wb]."""
        _, x = self.critical_points
        if len(x) > 0:
            return x[0].psi
        else:
            raise ValueError(f"No x-point")

    @property
    def psi_norm(self):
        return self.mesh.uv[0]

    @cached_property
    def psi(self):
        return self.psi_norm * (self.psi_boundary-self.psi_axis) + self.psi_axis

    def psirz(self, r, z, *args, **kwargs):
        return self._psirz(r, z, *args, **kwargs)

    @cached_property
    def cocos_flag(self):
        return 1.0 if self.psi_boundary > self.psi_axis else -1.0

    @property
    def ffprime(self):
        return self._ffprime

    @cached_property
    def fpol(self):
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        fpol2 = self.ffprime.antiderivative * (2 * (self.psi_boundary - self.psi_axis))
        return Function(self.psi_norm,  np.sqrt(fpol2 - fpol2[-1] + self._fvac**2))

    @cached_property
    def dl(self):
        return np.asarray([self.mesh.axis(idx, axis=0).geo_object.dl(self.mesh.uv[1]) for idx in range(self.mesh.shape[0])])

    @cached_property
    def Br(self):
        return -self.psirz(self.r, self.z, dy=1).view(np.ndarray)/self.r

    @cached_property
    def Bz(self):
        return self.psirz(self.r, self.z, dx=1).view(np.ndarray)/self.r

    @cached_property
    def Btor(self):
        return 1.0 / self.r.view(np.ndarray) * self.fpol.reshape(self.mesh.shape[0], 1).view(np.ndarray)

    @cached_property
    def Bpol(self):
        r"""
            .. math:: B_{pol} =   R / |\nabla \psi|
        """
        return np.sqrt(self.Br**2+self.Bz**2)

    @cached_property
    def B2(self):
        return (self.Br**2+self.Bz**2 + self.Btor**2)

    @cached_property
    def norm_grad_psi(self):
        return np.sqrt(self.psirz(self.r, self.z, dx=1)**2+self.psirz(self.r, self.z, dy=1)**2)

    def surface_integrate(self, alpha=None, *args, **kwargs):
        r"""
            .. math:: \left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}
        """
        if alpha is None:
            alpha = 1.0/self.Bpol
        else:
            alpha = alpha/self.Bpol
        return np.sum((np.roll(alpha, 1, axis=1)+alpha) * self.dl, axis=1) * (0.5*2*scipy.constants.pi)

    def surface_average(self,  *args, **kwargs):
        return self.surface_integrate(*args, **kwargs) / self.dvolume_dpsi

    @cached_property
    def dvolume_dpsi(self):
        r"""
            .. math:: V^{\prime} =  2 \pi  \int{ R / |\nabla \psi| * dl }
            .. math:: V^{\prime}(psi)= 2 \pi  \int{ dl * R / |\nabla \psi|}
        """
        return self.surface_integrate()

    def bbox(self, s=None):
        rz = np.asarray([(s.bbox[0]+self.bbox[1])*0.5 for s in self.surface_mesh]).T

    def shape_property(self, psi=None):
        def shape_box(s: Curve):
            r, z = s.xy()
            rmin = np.min(r)
            rmax = np.max(r)
            zmin = np.min(z)
            zmax = np.max(z)
            rzmin = r[np.argmin(z)]
            rzmax = r[np.argmax(z)]
            r_inboard = s.point(0.5)[0]
            r_outboard = s.point(0)[0]
            return rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard

        if psi is None:
            pass
        elif not isinstance(psi, (np.ndarray, collections.abc.MutableSequence)):
            psi = [psi]

        sbox = np.asarray([[*shape_box(s)] for s in self.create_surface(psi)])

        if sbox.shape[0] == 1:
            rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox[0]
        else:
            rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox.T

        return AttributeTree({
            # RZ position of the geometric axis of the magnetic surfaces (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the surface)
            "geometric_axis": {"r": (rmin+rmax)*0.5,  "z": (zmin+zmax)*0.5},
            # Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary)[m]
            "minor_radius": (rmax - rmin)*0.5,
            # Elongation of the plasma boundary. [-]
            "elongation": (zmax-zmin)/(rmax-rmin),
            # Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]
            "elongation_upper": (zmax-(zmax+zmin)*0.5)/(rmax-rmin),
            # longation(lower half w.r.t. geometric axis) of the plasma boundary. [-]
            "elongation_lower": ((zmax+zmin)*0.5-zmin)/(rmax-rmin),
            # Triangularity of the plasma boundary. [-]
            "triangularity": (rzmax-rzmin)/(rmax - rmin)*2,
            # Upper triangularity of the plasma boundary. [-]
            "triangularity_upper": ((rmax+rmin)*0.5 - rzmax)/(rmax - rmin)*2,
            # Lower triangularity of the plasma boundary. [-]
            "triangularity_lower": ((rmax+rmin)*0.5 - rzmin)/(rmax - rmin)*2,
            # Radial coordinate(major radius) on the inboard side of the magnetic axis[m]
            "r_inboard": r_inboard,
            # Radial coordinate(major radius) on the outboard side of the magnetic axis[m]
            "r_outboard": r_outboard,
        })

    @cached_property
    def magnetic_axis(self):
        o, _ = self.critical_points
        if not o:
            raise RuntimeError(f"Can not find magnetic axis")

        return AttributeTree({
            "r": o[0].r,
            "z": o[0].z,
            "b_field_tor": NotImplemented
        })

    @cached_property
    def boundary(self):
        return self.create_surface(1.0)


class Equilibrium(PhysicalGraph):
    r"""Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

        Reference:
            - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.

        COCOS  11

        #    Top view
        #             ***************
        #            *               *
        #           *   ***********   *
        #          *   *           *   *
        #         *   *             *   *
        #         *   *             *   *
        #     Ip  v   *             *   ^  \phi
        #         *   *    Z o--->R *   *
        #         *   *             *   *
        #         *   *             *   *
        #         *   *     Bpol    *   *
        #          *   *     o     *   *
        #           *   ***********   *
        #            *               *
        #             ***************
        #               Bpol x
        #            Poloidal view
        #        ^Z
        #        |
        #        |       ************
        #        |      *            *
        #        |     *         ^    *
        #        |     *   \rho /     *
        #        |     *       /      *
        #        +-----*------X-------*---->R
        #        |     *  Ip, \phi   *
        #        |     *              *
        #        |      *            *
        #        |       *****<******
        #        |       Bpol,\theta
        #        |
        #            Cylindrical coordinate      : (R,\phi,Z)
        #    Poloidal plane coordinate   : (\rho,\theta,\phi)
    """

    IDS = "transport.equilibrium"

    def __init__(self,  *args, uv=None, psirz=None,   **kwargs):
        super().__init__(*args, **kwargs)

        self._vacuum_toroidal_field = self["vacuum_toroidal_field"]

        if not isinstance(psirz, Field):
            psirz = Field(self["profiles_2d.psi"],
                          self["profiles_2d.grid.dim1"],
                          self["profiles_2d.grid.dim2"],
                          mesh_type="rectilinear")

        self._psirz = psirz

        if uv is not None:
            self._uv = uv
        else:
            dim1 = self["coordinate_system.grid.dim1"]
            dim2 = self["coordinate_system.grid.dim2"]

            if isinstance(dim1, np.ndarray):
                u = dim1
            elif dim1 == None:
                u = np.linspace(0.0001,  0.99,  len(self._ffprime))
            elif isinstance(dim2, int):
                u = np.linspace(0.0001,  0.99,  dim1)
            elif isinstance(dim2, np.ndarray):
                u = dim1
            else:
                u = np.asarray([dim1])

            if isinstance(dim2, np.ndarray):
                v = dim2
            elif dim2 == None:
                v = np.linspace(0.0,  1.0,  128)
            elif isinstance(dim2, int):
                v = np.linspace(0.0, 1.0,  dim2)
            elif isinstance(dim2, np.ndarray):
                v = dim2
            else:
                v = np.asarray([dim2])

            self._uv = [u, v]

    @property
    def vacuum_toroidal_field(self):
        return self._vacuum_toroidal_field

    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, test_convergence=False,  ** kwargs):
        logger.debug(f"Update {self.__class__.__name__} ")
        logger.warning(f"TODO: not implemented!")
        # del self.global_quantities
        # del self.profiles_1d
        # del self.profiles_2d
        # del self.boundary
        # del self.boundary_separatrix
        # del self.coordinate_system
        return test_convergence

    @cached_property
    def coordinate_system(self):
        fvac = np.abs(self["vacuum_toroidal_field.r0"] * self["vacuum_toroidal_field.b0"])
        ffprime = Function(self["profiles_1d.psi_norm"], self["profiles_1d.f_df_dpsi"])
        return MagneticSurfaceCoordinateSystem(self._uv, self._psirz,  ffprime, fvac)

    @cached_property
    def constraints(self):
        return PhysicalGraph(self["constraints"], parent=self)

    def radial_grid(self, primary_axis=None, axis=None):
        """ """
        psi_norm = self.profiles_1d.psi_norm

        if primary_axis == "psi_norm" or primary_axis is None:
            if axis is not None:
                psi_norm = axis
        else:
            p_axis = try_get(self.profiles_1d, primary_axis)
            if not isinstance(p_axis, np.ndarray):
                raise NotImplementedError(primary_axis)
            psi_norm = Function(p_axis,psi_norm)(np.linspace(p_axis[0], p_axis[-1], p_axis.shape[0]))
        return RadialGrid(psi_norm, equilibrium=self)

    @cached_property
    def profiles_1d(self):
        return Equilibrium.Profiles1D(self["profiles_1d"], coord=self.coordinate_system, parent=self)

    @cached_property
    def profiles_2d(self):
        return Equilibrium.Profiles2D(self.coordinate_system,  parent=self)

    @cached_property
    def global_quantities(self):
        return Equilibrium.GlobalQuantities(self.coordinate_system,  parent=self)

    @cached_property
    def boundary(self):
        return Equilibrium.Boundary(self.coordinate_system,   parent=self)

    @cached_property
    def boundary_separatrix(self):
        return Equilibrium.BoundarySeparatrix(self.coordinate_system,   parent=self)

    class GlobalQuantities(PhysicalGraph):
        def __init__(self, coord: MagneticSurfaceCoordinateSystem,    *args,  **kwargs):
            super().__init__(*args, **kwargs)
            self._coord = coord

        @property
        def beta_pol(self):
            """Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
            return NotImplemented

        @property
        def beta_tor(self):
            """Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by(B0 ^ 2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0 ^ 2  [-]"""
            return NotImplemented

        @property
        def beta_normal(self):
            """Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0[T] / ip[MA][-]"""
            return NotImplemented

        @property
        def ip(self):
            """Plasma current(toroidal component). Positive sign means anti-clockwise when viewed from above.  [A]."""
            return self._parent.profiles_1d.plasma_current[-1]

        @property
        def li_3(self):
            """Internal inductance[-]"""
            return NotImplemented

        @property
        def volume(self):
            """Total plasma volume[m ^ 3]"""
            return NotImplemented

        @property
        def area(self):
            """Area of the LCFS poloidal cross section[m ^ 2]"""
            return NotImplemented

        @property
        def surface(self):
            """Surface area of the toroidal flux surface[m ^ 2]"""
            return NotImplemented

        @property
        def length_pol(self):
            """Poloidal length of the magnetic surface[m]"""
            return NotImplemented

        @property
        def magnetic_axis(self):
            """Magnetic axis position and toroidal field	structure"""
            return self._coord.magnetic_axis
            # return PhysicalGraph({"r":  self["magnetic_axis.r"],
            #                       "z":  self["magnetic_axis.z"],
            #                       # self.profiles_2d.b_field_tor(opt[0][0], opt[0][1])
            #                       "b_field_tor": self["magnetic_axis.b_field_tor"]
            #                       })

        @cached_property
        def x_points(self):
            _, x = self._coord.critical_points
            return x

        @cached_property
        def psi_axis(self):
            """Poloidal flux at the magnetic axis[Wb]."""
            o, _ = self._coord.critical_points
            return o[0].psi

        @cached_property
        def psi_boundary(self):
            """Poloidal flux at the selected plasma boundary[Wb]."""
            _, x = self._coord.critical_points
            if len(x) > 0:
                return x[0].psi
            else:
                raise ValueError(f"No x-point")

        @property
        def q_axis(self):
            """q at the magnetic axis[-]."""
            return NotImplemented

        @property
        def q_95(self):
            """q at the 95 % poloidal flux surface
            (IMAS uses COCOS=11: only positive when toroidal current
            and magnetic field are in same direction)[-]."""
            return NotImplemented

        @property
        def q_min(self):
            """Minimum q value and position structure"""
            return NotImplemented

        @property
        def energy_mhd(self):
            """Plasma energy content: 3/2 * int(p, dV) with p being the total pressure(thermal + fast particles)[J].  Time-dependent  Scalar[J]"""
            return NotImplemented

    class Profiles1D(Profiles):
        """Equilibrium profiles(1D radial grid) as a function of the poloidal flux	"""

        def __init__(self,  *args,  coord: MagneticSurfaceCoordinateSystem,  **kwargs):
            if coord is None:
                coord = self._parent.coordinate_system

            super().__init__(*args, axis=coord.psi_norm, **kwargs)

            self._coord = coord
            self._b0 = np.abs(self._parent.vacuum_toroidal_field.b0)
            self._r0 = self._parent.vacuum_toroidal_field.r0

        @property
        def psi_norm(self):
            """Normalized poloidal flux[Wb]. """
            return self._coord.psi_norm

        @cached_property
        def psi(self):
            """Poloidal flux[Wb]. """
            return Function(self.psi_norm, self.psi_norm * (self._coord.psi_boundary - self._coord.psi_axis) + self._coord.psi_axis)

        @cached_property
        def vprime(self):
            r"""
                .. math: : V^{\prime} =  2 \pi  \int{R / |\nabla \psi | * dl }
                .. math: : V^{\prime}(psi) = 2 \pi  \int{dl * R / |\nabla \psi|}
            """
            return Function(self.psi_norm, self._coord.dvolume_dpsi*(self._coord.psi_boundary-self._coord.psi_axis))

        @cached_property
        def dvolume_dpsi(self):
            r"""
                Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1].
            """
            return Function(self.psi_norm, self._coord.dvolume_dpsi)

        @cached_property
        def volume(self):
            """Volume enclosed in the flux surface[m ^ 3]"""
            return self.vprime.antiderivative

        @cached_property
        def ffprime(self):
            """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
            return Function(self.psi_norm, self._coord.ffprime(self.psi_norm))

        @property
        def f_df_dpsi(self):
            """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
            return self.ffprime

        @cached_property
        def fpol(self):
            """Diamagnetic function(F=R B_Phi)[T.m]."""
            return Function(self.psi_norm, self._coord.fpol(self.psi_norm))

        @property
        def f(self):
            """Diamagnetic function(F=R B_Phi)[T.m]."""
            return self.fpol

        @cached_property
        def dpressure_dpsi(self):
            return Function(self.psi_norm, self["dpressure_dpsi"])

        @cached_property
        def j_tor(self):
            r"""Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R) {dynamic}[A.m ^ -2]. """
            d = (self.gm2*self.dvolume_drho_tor*self.dpsi_drho_tor).derivative / \
                self.dvolume_dpsi*self._r0/scipy.constants.mu_0
            return Function(self.psi_norm, d.view(np.ndarray))

        @cached_property
        def j_parallel(self):
            r"""Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0 {dynamic}[A/m ^ 2]. """
            d = (-1/self._b0) * (self.fpol*self.dpressure_dpsi +
                                 self.gm5*self.f_df_dpsi/self.fpol/scipy.constants.mu_0)
            return Function(self.psi_norm, d.view(np.ndarray))

        @cached_property
        def dphi_dpsi(self):
            return self.gm1 * self.fpol * self.dvolume_dpsi / (4*scipy.constants.pi**2)

        @property
        def q(self):
            r"""
                Safety factor
                (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)[-].
                .. math: : q(\psi) =\frac{d\Phi}{d\psi} =\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{4\pi^{2}}
            """
            return Function(self.psi_norm, self.fpol * self.dvolume_dpsi*self._coord.surface_average(1.0/(self._coord.r**2))/(4*scipy.constants.pi**2))

        @cached_property
        def phi(self):
            r"""
                Note:
                .. math::
                    \Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi
            """
            return Function(self.psi_norm, self.fpol*self._coord.surface_integrate(1.0/(self._coord.r**2)) *
                            ((self._coord.psi_boundary-self._coord.psi_axis) / (TWOPI))).antiderivative

        @cached_property
        def rho_tor(self):
            """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
            return np.sqrt(self.phi/(scipy.constants.pi * self._b0))

        @cached_property
        def rho_tor_norm(self):
            return np.sqrt(self.phi/self.phi[-1])

        @cached_property
        def drho_tor_dpsi(self):
            r"""
                .. math::
                    \frac{d\rho_{tor}}{d\psi} =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                            =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                            =\frac{q}{2\pi B_{0}\rho_{tor}}
            """
            return self.rho_tor.derivative/(self._coord.psi_boundary-self._coord.psi_axis)

        @cached_property
        def dpsi_drho_tor(self)	:
            """
                Derivative of Psi with respect to Rho_Tor[Wb/m].
            """
            return (TWOPI*self._b0)*self.rho_tor/self.dphi_dpsi

        @cached_property
        def dpsi_drho_tor_norm(self)	:
            """
                Derivative of Psi with respect to Rho_Tor[Wb/m].
            """
            return self.dpsi_drho_tor*self.rho_tor[-1]

        @cached_property
        def dvolume_drho_tor(self)	:
            """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
            return (4*scipy.constants.pi**2*self._b0)*self.rho_tor/(self.fpol*self.gm1)

        @cached_property
        def shape_property(self):
            return self._coord.shape_property()

        @cached_property
        def geometric_axis(self):
            return PhysicalGraph({
                "r": Function(self.psi_norm, self.shape_property.geometric_axis.r),
                "z": Function(self.psi_norm, self.shape_property.geometric_axis.z),
            })

        @property
        def minor_radius(self):
            """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
            return self.shape_property.minor_radius

        @cached_property
        def r_inboard(self):
            """Radial coordinate(major radius) on the inboard side of the magnetic axis[m]"""
            return Function(self.psi_norm, self.shape_property.r_inboard)

        @cached_property
        def r_outboard(self):
            """Radial coordinate(major radius) on the outboard side of the magnetic axis[m]"""
            return Function(self.psi_norm, self.shape_property.r_outboard)

        @cached_property
        def elongation(self):
            """Elongation. {dynamic}[-]"""
            return Function(self.psi_norm, self.shape_property.elongation)

        @cached_property
        def triangularity(self)	:
            """Upper triangularity w.r.t. magnetic axis. {dynamic}[-]"""
            return Function(self.psi_norm, self.shape_property.triangularity)

        @cached_property
        def triangularity_upper(self)	:
            """Upper triangularity w.r.t. magnetic axis. {dynamic}[-]"""
            return Function(self.psi_norm, self.shape_property.triangularity_upper)

        @cached_property
        def triangularity_lower(self)	:
            """Lower triangularity w.r.t. magnetic axis. {dynamic}[-]"""
            return Function(self.psi_norm, self.shape_property.triangularity_lower)

        @cached_property
        def plasma_current(self):
            r"""
                Ip
            """
            return self.dpsi_drho_tor*self.gm2*self.dvolume_drho_tor/(TWOPI**2)/scipy.constants.mu_0

        @cached_property
        def norm_grad_rho_tor(self):
            return self._coord.norm_grad_psi * self.drho_tor_dpsi.view(np.ndarray).reshape(list(self.drho_tor_dpsi.shape)+[1])

        @cached_property
        def gm1(self):
            r"""
                Flux surface averaged 1/R ^ 2  [m ^ -2]
                .. math: : \left\langle\frac{1}{R^{2}}\right\rangle
            """
            return Function(self.psi_norm, self._coord.surface_average(1.0/(self._coord.r**2)))

        @cached_property
        def gm2(self):
            r"""
                Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2/R^2  [m^-2]
                .. math:: \left\langle\left |\frac{\nabla\rho}{R}\right|^{2}\right\rangle
            """
            d = self._coord.surface_average((self.norm_grad_rho_tor/self._coord.r)**2)
            return Function(self.psi_norm, Function(self.psi_norm[1:], d[1:])(self.psi_norm))

        @cached_property
        def gm3(self):
            r"""
                Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2  [-]
                .. math:: {\left\langle \left |\nabla\rho\right|^{2}\right\rangle}
            """
            d = self._coord.surface_average(self.norm_grad_rho_tor**2)
            return Function(self.psi_norm, Function(self.psi_norm[1:], d[1:])(self.psi_norm))

        @cached_property
        def gm4(self):
            r"""
                Flux surface averaged 1/B ^ 2  [T ^ -2]
                .. math: : \left\langle \frac{1}{B^{2}}\right\rangle
            """
            return Function(self.psi_norm, self._coord.surface_average(1.0/self._coord.B2))

        @cached_property
        def gm5(self):
            r"""
                Flux surface averaged B ^ 2  [T ^ 2]
                .. math: : \left\langle B^{2}\right\rangle
            """
            return Function(self.psi_norm, self._coord.surface_average(self._coord.B2))

        @cached_property
        def gm6(self):
            r"""
                Flux surface averaged  .. math: : \left | \nabla \rho_{tor}\right|^2/B^2  [T^-2]
                .. math:: \left\langle \frac{\left |\nabla\rho\right|^{2}}{B^{2}}\right\rangle
            """
            return Function(self.psi_norm, self._coord.surface_average(self.norm_grad_rho_tor**2/self._coord.B2))

        @cached_property
        def gm7(self):
            r"""
                Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right |  [-]
                .. math: : \left\langle \left |\nabla\rho\right |\right\rangle
            """
            d = self._coord.surface_average(self.norm_grad_rho_tor)
            return Function(self.psi_norm, Function(self.psi_norm[1:], d[1:])(self.psi_norm))

        @cached_property
        def gm8(self):
            r"""
                Flux surface averaged R[m]
                .. math: : \left\langle R\right\rangle
            """
            return Function(self.psi_norm, self._coord.surface_average(self._coord.r))

        @cached_property
        def gm9(self):
            r"""
                Flux surface averaged 1/R[m ^ -1]
                .. math: : \left\langle \frac{1}{R}\right\rangle
            """
            return Function(self.psi_norm, self._coord.surface_average(1.0/self._coord.r))

        @cached_property
        def magnetic_shear(self):
            """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
            return Function(self.psi_norm, self.rho_tor/self.q * self.q.derivative)

        @cached_property
        def rho_volume_norm(self)	:
            """Normalised square root of enclosed volume(radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary
                (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)[-]"""
            return NotImplemented

        @cached_property
        def area(self):
            """Cross-sectional area of the flux surface[m ^ 2]"""
            return NotImplemented

        @cached_property
        def darea_dpsi(self):
            """Radial derivative of the cross-sectional area of the flux surface with respect to psi[m ^ 2.Wb ^ -1]. """
            return NotImplemented

        @cached_property
        def darea_drho_tor(self)	:
            """Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor[m]"""
            return NotImplemented

        @cached_property
        def surface(self):
            """Surface area of the toroidal flux surface[m ^ 2]"""
            return NotImplemented

        @cached_property
        def trapped_fraction(self)	:
            """Trapped particle fraction[-]"""
            return Function(self.psi_norm, self["trapped_fraction"] or 0.0)

        @cached_property
        def b_field_max(self):
            """Maximum(modulus(B)) on the flux surface(always positive, irrespective of the sign convention for the B-field direction)[T]"""
            return NotImplemented

        @cached_property
        def beta_pol(self):
            """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
            return NotImplemented

    class Profiles2D(PhysicalGraph):
        """
            Equilibrium 2D profiles in the poloidal plane.
        """

        def __init__(self, coord: MagneticSurfaceCoordinateSystem, *args, ** kwargs):
            super().__init__(*args, **kwargs)
            self._coord = coord

        @property
        def grid_type(self):
            return self._coord.grid_type

        @cached_property
        def grid(self):
            return self._coord.grid

        @property
        def r(self):
            """Values of the major radius on the grid  [m] """
            return self._coord.r

        @property
        def z(self):
            """Values of the Height on the grid  [m] """
            return self._coord.z

        @cached_property
        def psi(self):
            """Values of the poloidal flux at the grid in the poloidal plane  [Wb]. """
            return self.apply_psifunc(lambda p: p, unit="Wb")

        @cached_property
        def theta(self):
            """	Values of the poloidal angle on the grid  [rad] """
            return NotImplementedError()

        @cached_property
        def phi(self):
            """	Toroidal flux  [Wb]"""
            return self.apply_psifunc("phi")

        @cached_property
        def j_tor(self):
            """	Toroidal plasma current density  [A.m^-2]"""
            return self.apply_psifunc("j_tor")

        @cached_property
        def j_parallel(self):
            """	Parallel (to magnetic field) plasma current density  [A.m^-2]"""
            return self.apply_psifunc("j_parallel")

        @cached_property
        def b_field_r(self):
            """R component of the poloidal magnetic field  [T]"""
            return Field(self._coord.Br, self._coord.r, self._coord.z, mesh_type="curvilinear")

        @cached_property
        def b_field_z(self):
            """Z component of the poloidal magnetic field  [T]"""
            return Field(self._coord.Bz, self._coord.r, self._coord.z, mesh_type="curvilinear")

        @cached_property
        def b_field_tor(self):
            """Toroidal component of the magnetic field  [T]"""
            return Field(self._coord.Btor, self._coord.r, self._coord.z, mesh_type="curvilinear")

    class Boundary(PhysicalGraph):
        def __init__(self, coord: MagneticSurfaceCoordinateSystem,   *args,  ** kwargs):
            super().__init__(*args, **kwargs)
            self._coord = coord or self._parent.coordinate_system

        @cached_property
        def type(self):
            """0 (limiter) or 1 (diverted)  """
            return 1

        @cached_property
        def outline(self):
            """RZ outline of the plasma boundary  """
            RZ = np.asarray([[r, z] for r, z in self._coord.flux_surface_map(1.0)])
            return PhysicalGraph({"r": RZ[:, 0], "z": RZ[:, 1]})

        @cached_property
        def x_point(self):
            _, xpt = self._parent.critical_points
            return xpt

        @cached_property
        def psi(self):
            """Value of the poloidal flux at which the boundary is taken  [Wb]"""
            return self._parent.psi_boundary

        @cached_property
        def psi_norm(self):
            """Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %),
                the flux being normalised to its value at the separatrix """
            return self.psi*0.99

        @cached_property
        def shape_property(self):
            return self._coord.shape_property(1.0)

        @property
        def geometric_axis(self):
            """RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)"""
            return self.shape_property.geometric_axis

        @property
        def minor_radius(self):
            """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
            return self.shape_property.minor_radius

        @property
        def elongation(self):
            """Elongation of the plasma boundary. [-]	"""
            return self.shape_property.elongation

        @property
        def elongation_upper(self):
            """Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]	"""
            return self.shape_property.elongation_upper

        @property
        def elongation_lower(self):
            """Elongation(lower half w.r.t. geometric axis) of the plasma boundary. [-]	"""
            return self.shape_property.elongation_lower

        @property
        def triangularity(self):
            """Triangularity of the plasma boundary. [-]	"""
            return self.shape_property.triangularity

        @property
        def triangularity_upper(self):
            """Upper triangularity of the plasma boundary. [-]	"""
            return self.shape_property.triangularity_upper

        @property
        def triangularity_lower(self):
            """Lower triangularity of the plasma boundary. [-]"""
            return self.shape_property.triangularity_lower

        @cached_property
        def strike_point(self)	:
            """Array of strike points, for each of them the RZ position is given	struct_array [max_size=unbounded]	1- 1...N"""
            return NotImplemented

        @cached_property
        def active_limiter_point(self):
            """	RZ position of the active limiter point (point of the plasma boundary in contact with the limiter)"""
            return NotImplemented

    class BoundarySeparatrix(PhysicalGraph):
        def __init__(self, coord: MagneticSurfaceCoordinateSystem,  *args,  ** kwargs):
            super().__init__(*args, **kwargs)

    ####################################################################################
    # Plot proflies
    def plot(self, axis=None, *args,
             scalar_field=[],
             vector_field=[],
             mesh=True,
             boundary=True,
             contour_=False,
             levels=32, oxpoints=True,
             **kwargs):
        """learn from freegs
        """
        if axis is None:
            axis = plt.gca()

        # R = self.profiles_2d.r
        # Z = self.profiles_2d.z
        # psi = self.profiles_2d.psi(R, Z)

        # axis.contour(R[1:-1, 1:-1], Z[1:-1, 1:-1], psi[1:-1, 1:-1], levels=levels, linewidths=0.2)

        if oxpoints is not False:
            o_point, x_point = self.coordinate_system.critical_points
            axis.plot(o_point[0].r,
                      o_point[0].z,
                      'g.',
                      linewidth=0.5,
                      markersize=2,
                      label="Magnetic axis")

            if len(x_point) > 0:
                for idx, p in enumerate(x_point):
                    axis.plot(p.r, p.z, 'rx')
                    axis.text(p.r, p.z, idx,
                              horizontalalignment='center',
                              verticalalignment='center')

                axis.plot([], [], 'rx', label="X-Point")

        if boundary is not False:
            boundary_points = np.vstack([self.boundary.outline.r,
                                         self.boundary.outline.z]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                       linewidth=0.5, fill=False, closed=True))
            axis.plot([], [], 'r--', label="Separatrix")

        if mesh is not False:
            for idx in range(0, self.coordinate_system.mesh.shape[0], 4):
                ax0 = self.coordinate_system.mesh.axis(idx, axis=0)
                axis.add_patch(plt.Polygon(ax0.xy, fill=False, closed=True, color="b", linewidth=0.2))

            for idx in range(0, self.coordinate_system.mesh.shape[1], 4):
                ax1 = self.coordinate_system.mesh.axis(idx, axis=1)
                axis.plot(ax1.xy[:, 0], ax1.xy[:, 1],  "r", linewidth=0.2)

        for s, opts in scalar_field:
            if s == "psirz":
                self._psirz.plot(axis, **opts)
            else:
                if "." not in s:
                    sf = f"profiles_2d.{s}"
                # self.coordinate_system.norm_grad_psi
                sf = try_get(self, s, None)
                if isinstance(sf, Field):
                    sf.plot(axis, **opts)
                elif isinstance(sf, np.ndarray):
                    axis.contour(self.profiles_2d.r, self.profiles_2d.z, sf, **opts)
                else:
                    logger.error(f"Can not find field {sf} {type(sf)}!")

        for u, v, opts in vector_field:
            uf = self.profiles_2d[u]
            vf = self.profiles_2d[v]
            axis.streamplot(self.profiles_2d.grid.dim1,
                            self.profiles_2d.grid.dim2,
                            vf, uf, **opts)

        return axis

    def fetch_profile(self, d):
        if isinstance(d, str):
            data = d
            opts = {"label": d}
        elif isinstance(d, collections.abc.Mapping):
            data = d.get("name", None)
            opts = d.get("opts", {})
        elif isinstance(d, tuple):
            data, opts = d
        elif isinstance(d, PhysicalGraph):
            data = d.data
            opts = d.opts
        else:
            raise TypeError(f"Illegal profile type! {d}")

        if isinstance(opts, str):
            opts = {"label": opts}

        if isinstance(data, str):
            nlist = data.split(".")
            if len(nlist) == 1:
                data = self.profiles_1d[nlist[0]]
            elif nlist[0] == 'cache':
                data = self.profiles_1d[nlist[1:]]
            else:
                data = self.profiles_1d[nlist]
        elif isinstance(data, list):
            data = np.array(data)
        elif isinstance(d, np.ndarray):
            pass
        else:
            raise TypeError(f"Illegal data type! {type(data)}")

        return data, opts

    def plot_profiles(self, fig_axis, axis, profiles):
        if not isinstance(profiles, list):
            profiles = [profiles]

        for idx, data in enumerate(profiles):
            ylabel = None
            opts = {}
            if isinstance(data, tuple):
                data, ylabel = data
            if isinstance(data, str):
                ylabel = data

            if not isinstance(data, list):
                data = [data]

            for d in data:
                value, opts = self.fetch_profile(d)

                if value is not NotImplemented and value is not None and len(value) > 0:
                    fig_axis[idx].plot(axis.data, value, **opts)
                else:
                    logger.error(f"Can not find profile '{d}'")

            fig_axis[idx].legend(fontsize=6)

            if ylabel:
                fig_axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
            fig_axis[idx].labelsize = "media"
            fig_axis[idx].tick_params(labelsize=6)
        return fig_axis[-1]

    def plot_full(self, *args,
                  axis=("psi_norm",   r'$(\psi-\psi_{axis})/(\psi_{boundary}-\psi_{axis}) [-]$'),
                  profiles=None,
                  profiles_2d=[],
                  vec_field=[],
                  surface_mesh=False,
                  **kwargs):

        axis, axis_opts = self.fetch_profile(axis)

        assert (axis.data is not NotImplemented)
        nprofiles = len(profiles) if profiles is not None else 0
        if profiles is None or nprofiles <= 1:
            fig, ax_right = plt.subplots(ncols=1, nrows=1, sharex=True)
        else:
            fig, axs = plt.subplots(ncols=2, nrows=nprofiles, sharex=True)
            # left
            ax_left = self.plot_profiles(axs[:, 0], axis, profiles)

            ax_left.set_xlabel(axis_opts.get("label", "[-]"), fontsize=6)

            # right
            gs = axs[0, 1].get_gridspec()
            for ax in axs[:, 1]:
                ax.remove()  # remove the underlying axes
            ax_right = fig.add_subplot(gs[:, 1])

        if surface_mesh:
            self.coordinate_system.plot(ax_right)

        self.plot(ax_right, profiles=profiles_2d, vec_field=vec_field, **kwargs.get("equilibrium", {}))

        self._tokamak.plot_machine(ax_right, **kwargs.get("machine", {}))

        ax_right.legend()
        fig.tight_layout()

        fig.subplots_adjust(hspace=0)
        fig.align_ylabels()

        return fig

    # # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2]  [-]
    # self.global_quantities.beta_pol = NotImplemented
    # # Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2  [-]
    # self.global_quantities.beta_tor = NotImplemented
    # # Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]  [-]
    # self.global_quantities.beta_normal = NotImplemented
    # # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above.  [A].
    # self.global_quantities.ip = NotImplemented
    # # Internal inductance  [-]
    # self.global_quantities.li_3 = NotImplemented
    # # Total plasma volume  [m^3]
    # self.global_quantities.volume = NotImplemented
    # # Area of the LCFS poloidal cross section  [m^2]
    # self.global_quantities.area = NotImplemented
    # # Surface area of the toroidal flux surface  [m^2]
    # self.global_quantities.surface = NotImplemented
    # # Poloidal length of the magnetic surface  [m]
    # self.global_quantities.length_pol = NotImplemented
    # # Poloidal flux at the magnetic axis  [Wb].
    # self.global_quantities.psi_axis = NotImplemented
    # # Poloidal flux at the selected plasma boundary  [Wb].
    # self.global_quantities.psi_boundary = NotImplemented
    # # Magnetic axis position and toroidal field	structure
    # self.global_quantities.magnetic_axis = NotImplemented
    # # q at the magnetic axis  [-].
    # self.global_quantities.q_axis = NotImplemented
    # # q at the 95% poloidal flux surface (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)  [-].
    # self.global_quantities.q_95 = NotImplemented
    # # Minimum q value and position	structure
    # self.global_quantities.q_min = NotImplemented
    # # Plasma energy content = 3/2 * int(p,dV) with p being the total pressure (thermal + fast particles) [J]. Time-dependent; Scalar  [J]
    # self.global_quantities.energy_mhd = NotImplemented
