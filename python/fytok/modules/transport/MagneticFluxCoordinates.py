import collections
from functools import cached_property

import numpy as np
import scipy.constants
import scipy.integrate
from numpy import arctan2, cos, sin, sqrt
from packaging import version

from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.data.mesh import Mesh
from spdm.data.mesh.CurvilinearMesh import CurvilinearMesh
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.util.logger import logger
from scipy.optimize import fsolve, root_scalar

# if version.parse(scipy.__version__) <= version.parse("1.4.1"):
#     from scipy.integrate import cumtrapz as cumtrapz
# else:
#     from scipy.integrate import cumulative_trapezoid as cumtrapz

logger.debug(f"Using SciPy Version: {scipy.__version__}")


def power2(a):
    return np.inner(a, a)


class MagneticFluxCoordinates:
    r"""
        Magnetic Flux Coordinates

        .. math::
            V^{\prime}\left(\rho\right)=\frac{\partial V}{\partial\rho}=2\pi\int_{0}^{2\pi}\sqrt{g}d\theta=2\pi\oint\frac{R}{\left|\nabla\rho\right|}dl

        .. math::
            \left\langle\alpha\right\rangle\equiv\frac{2\pi}{V^{\prime}}\int_{0}^{2\pi}\alpha\sqrt{g}d\theta=\frac{2\pi}{V^{\prime}}\varoint\alpha\frac{R}{\left|\nabla\rho\right|}dl

        Magnetic Flux Coordinates
        psi         :                     ,  flux function , $B \cdot \nabla \psi=0$ need not to be the poloidal flux funcion $\Psi$
        theta       : 0 <= theta   < 2*pi ,  poloidal angle
        phi         : 0 <= phi     < 2*pi ,  toroidal angle
    """

    def __init__(self, psirz: Field, *args,
                 b0=None,
                 fvac=None,
                 ffprime=None,
                 psi_norm=None,
                 grid_type=None,
                 grid_shape=[64, 128],
                 parent=None,
                 ** kwargs):
        """
            Initialize FluxSurface
        """

        if not isinstance(psirz, (Field, Function)):
            raise TypeError(psirz)
        self._parent = parent
        self._psirz = psirz
        self._mesh = None
        self._grid_type = grid_type
        self._grid_shape = grid_shape

        self._b0 = b0
        self._fvac = fvac
        self._ffprime = ffprime
        self._psi_norm = psi_norm if psi_norm is not None else [0.01, 0.99]

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

    @cached_property
    def cocos_flag(self):
        return 1.0 if self.psi_boundary > self.psi_axis else -1.0

    @property
    def b0(self):
        return self._b0

    @cached_property
    def psi_axis(self):
        return self.critical_points[0][0].psi

    @cached_property
    def psi_boundary(self):
        _, x = self.critical_points
        if len(x) == 0:
            raise ValueError(f"No x-point")
        return x[0].psi

    def find_by_psinorm(self, u, v=None):
        o_points, x_points = self.critical_points

        if len(o_points) == 0:
            raise RuntimeError(f"Can not find O-point!")
        else:
            R0 = o_points[0].r
            Z0 = o_points[0].z
            psi0 = o_points[0].psi

        if len(x_points) == 0:
            R1 = self._psirz.coordinates.bbox[1][0]
            Z1 = Z0
            psi1 = 0.0
        else:
            R1 = x_points[0].r
            Z1 = x_points[0].z
            psi1 = x_points[0].psi

        theta0 = arctan2(R1 - R0, Z1 - Z0)
        Rm = sqrt((R1-R0)**2+(Z1-Z0)**2)

        if not isinstance(u, (np.ndarray, collections.abc.Sequence)):
            u = [u]

        if v is None:
            v = np.linspace(0, 2.0*scipy.constants.pi, self._grid_shape[1], endpoint=False)
        elif not isinstance(v, (np.ndarray, collections.abc.Sequence)):
            v = [v]

        for p in u:
            for t in v:
                psival = p*(psi1-psi0)+psi0

                r0 = R0
                z0 = Z0
                r1 = R0+Rm*sin(t+theta0)
                z1 = Z0+Rm*cos(t+theta0)
                
                if not np.isclose(self._psirz(r1, z1), psival):
                    try:
                        def func(r): return float(self._psirz((1.0-r)*r0+r*r1, (1.0-r)*z0+r*z1)) - psival
                        sol = root_scalar(func,  bracket=[0, 1], method='brentq')
                    except ValueError as error:
                        raise ValueError(f"Find root fialed! {error} {psival}")

                    if not sol.converged:
                        raise ValueError(f"Find root fialed!")

                    r1 = (1.0-sol.root)*r0+sol.root*r1
                    z1 = (1.0-sol.root)*z0+sol.root*z1

                yield r1, z1

    def construct_flux_mesh(self,  psi_norm=None,    grid_shape=None):
        grid_shape = grid_shape or self._grid_shape
        psi_norm = psi_norm or self._psi_norm

        if len(psi_norm) == 2:
            psi_norm = np.linspace(*psi_norm, grid_shape[0])
        elif not isinstance(psi_norm, np.ndarray):
            raise TypeError(type(psi_norm))

        u = psi_norm

        v = np.linspace(0.0, 1.0,  grid_shape[1])
        rz = []
        for p in psi_norm:
            d = [[r, z] for r, z in self.find_by_psinorm(p, v[:-1]*scipy.constants.pi*2.0)]
            rz.append(d+d[:1])

        rz = np.asarray(rz)
        return CurvilinearMesh([rz[:, :, 0], rz[:, :, 1]], [u, v], cycle=[False, True])

    def reconstruct_radial_label(self, radial_label, psi_norm=None):
        if psi_norm is None:
            psi_norm = np.linspace(0.0, 1.0, self.grid_shape[0])
        elif isinstance(psi_norm, int):
            psi_norm = np.linspace(0.0, 1.0, psi_norm)

        if radial_label == 0:
            raise NotImplemented()
        elif radial_label == 1:  # psi
            u = psi_norm*(self.psi_boundary-self.psi_axis)+self.psi_axis
        elif radial_label == 2:  # sqrt[(psi-psi_axis)/(psi_edge-psi_axis)]
            u = np.sqrt(psi_norm)
        elif radial_label == 3:  # sqrt[Phi/Phi_edge]
            u = np.linspace(0, 1.0, grid_shape[0])*self.phi(1.0)
            psi_norm = self.phi.invert(u)
        elif radial_label == 4:  # sqrt[psi-psi_axis]
            u = psi_norm*(self.psi_boundary-self.psi_axis)
        elif radial_label == 5:  # sqrt[Phi/pi/B0]
            raise NotImplementedError(f"")
        else:
            raise NotImplementedError(f"unknown")

        return self.construct_flux_mesh(psi_norm)

    def reconstruct_poloidal_angle_label(self, poloidal_angle_label, ntheta=None, mesh=None):
        r"""
            radial label (dim1)
                1x      :  psi
                2x      :  sqrt[(psi-psi_axis)/(psi_edge-psi_axis)]
                3x      :  sqrt[Phi/Phi_edge]
                4x      :  sqrt[psi-psi_axis]
                5x      :  sqrt[Phi/pi/B0]

            poloidal angle label (dim2)
                x1      :  the straight-field line
                x2      :  the equal arc poloidal angle
                x3      :  the polar poloidal angle
                x4      :  Fourier modes in the straight-field line poloidal angle
                x5      :  Fourier modes in the equal arc poloidal angle
                x6      :  Fourier modes in the polar poloidal angle

            @startuml
            [*] --> poloidal_angle

            poloidal_angle       --> equal_arc
            poloidal_angle       --> straight_field_line



            straight_field_line --> [*]

            @enduml
        """

        mesh = mesh or self.mesh

        assert(isinstance(mesh, Mesh))

        u, _ = mesh.uv

        v = np.linspace(0.0, 1.0, ntheta or self._grid_shape[1], endpoint=False)  # *scipy.constants.pi*2.0

        npsi = len(u)

        if poloidal_angle_label == 0:
            raise NotImplementedError()
        elif poloidal_angle_label == 1:  # the straight-field line
            def _map(idx):
                axis = mesh.axis(idx, axis=0)
                arc_length = axis.dl.antiderivative
                return axis.point(arc_length.invert(v*arc_length(1.0)))

        elif poloidal_angle_label == 2:  # the equal arc poloidal angle
            def _map(idx):
                axis = mesh.axis(idx, axis=0)
                arc_length = axis.dl.antiderivative
                return axis.point(arc_length.invert(v*arc_length(1.0)))

            rz = np.asarray([_map(idx) for idx in range(npsi)])

        elif poloidal_angle_label == 3:  # the polar poloidal angle
            def func(r, z):
                return r
        elif poloidal_angle_label == 4:
            raise NotImplementedError(f"Fourier modes in the straight-field line poloidal angle ")
        elif poloidal_angle_label == 5:
            raise NotImplementedError(f"Fourier modes in the equal arc poloidal angle  ")
        else:
            raise NotImplementedError(f"unknown")

        return CurvilinearMesh([rz[:, :, 0], rz[:, :, 1]], [u, v], cycle=[False, True])

    @ property
    def mesh(self):
        if self._mesh is None:
            self._mesh = self.construct_flux_mesh()
        return self._mesh

    @ property
    def R(self):
        return self.mesh.xy[0]

    @ property
    def Z(self):
        return self.mesh.xy[1]

    #################################
    @ cached_property
    def grad_psi(self):
        r = self.mesh.xy[0]
        z = self.mesh.xy[1]
        return self._psirz(r, z, dx=1), self._psirz(r, z, dy=1)

    @ cached_property
    def B2(self):
        r = self.mesh.xy[0]
        return (self.norm_grad_psi**2) + self.fpol.reshape(list(self.drho_tor_dpsi.shape)+[1]) ** 2/(r**2)

    @ cached_property
    def norm_grad_psi(self):
        r"""
            .. math:: V^{\prime} =   R / |\nabla \psi|
        """
        grad_psi_r, grad_psi_z = self.grad_psi
        return np.sqrt(grad_psi_r**2+grad_psi_z**2)

    @ cached_property
    def dl(self):
        return np.asarray([self.mesh.axis(idx, axis=0).geo_object.dl(self.mesh.uv[1]) for idx in range(self.mesh.shape[0])])

    def _surface_integral(self, J, *args, **kwargs):
        r"""
            .. math:: \left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}
        """
        return np.sum(0.5*(np.roll(J, 1, axis=1)+J) * self.dl, axis=1)

    def surface_average(self, J, *args, **kwargs):
        return Function(self.psi_norm, self._surface_integral(J, *args, **kwargs) / self.vprime * (2*scipy.constants.pi))
    #################################

    @ property
    def psi_norm(self):
        return self.mesh.uv[0]

    @ cached_property
    def psi(self):
        return self.psi_norm * (self.psi_boundary-self.psi_axis) + self.psi_axis

    @ cached_property
    def vprime(self):
        r"""
            .. math:: V^{\prime} =  2 \pi  \int{ R / |\nabla \psi| * dl }
            .. math:: V^{\prime}(psi)= 2 \pi  \int{ dl * R / |\nabla \psi|}
        """
        return Function(self.psi_norm, self._surface_integral(self.R/self.norm_grad_psi) * (2*scipy.constants.pi))

    @ cached_property
    def dvolume_dpsi(self):
        r"""
            Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1].
        """
        return self.vprime*self.cocos_flag

    @ cached_property
    def volume(self):
        """Volume enclosed in the flux surface[m ^ 3]"""
        return self.vprime.antiderivative

    @ cached_property
    def ffprime(self):
        """	Derivative of F w.r.t. Psi, multiplied with F  [T^2.m^2/Wb]. """
        d = self._ffprime
        if isinstance(d, Function):
            res = Function(self.psi_norm, d(self.psi_norm))
        elif isinstance(d, np.ndarray) and len(d) == len(self.psi_norm):
            res = Function(self.psi_norm, d)
        else:
            raise TypeError(type(d))

        return res

    @ property
    def f_df_dpsi(self):
        """	Derivative of F w.r.t. Psi, multiplied with F  [T^2.m^2/Wb]. """
        return self.ffprime

    @ cached_property
    def fpol(self):
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        psi_axis = self.psi_axis
        psi_boundary = self.psi_boundary
        f2 = self.ffprime.antiderivative * (2.0*(psi_boundary-psi_axis)) + self._fvac**2
        return Function(self.psi_norm, np.sqrt(f2), unit="T.m")

    @ property
    def f(self):
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        return self.fpol

    @ cached_property
    def q(self):
        r"""
            Safety factor
            (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)  [-].
            .. math:: q(\psi)=\frac{d\Phi}{d\psi}=\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{4\pi^{2}}
        """
        return Function(self.psi_norm, self.surface_average(1.0/(self.R**2)) * self.fpol * self.vprime / (scipy.constants.pi*2)**2)

    @ cached_property
    def dvolume_drho_tor(self)	:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
        return Function(self.psi_norm, self.dvolume_dpsi * self.dpsi_drho_tor)

    @ cached_property
    def dvolume_dpsi_norm(self):
        """Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1]. """
        return NotImplemented

    @ cached_property
    def phi(self):
        r"""
            Note:
                !!! COORDINATE　DEPENDENT!!!

            .. math ::
                \Phi_{tor}\left(\psi\right)=\int_{0}^{\psi}qd\psi
        """
        return self.q.antiderivative

    @ cached_property
    def rho_tor(self):
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0  [m]"""
        return Function(self.psi_norm,  np.sqrt(self.phi/(scipy.constants.pi * self.b0)), unit="m")

    @ cached_property
    def rho_tor_norm(self):
        return Function(self.psi_norm, self.rho_tor/self.rho_tor[-1])

    @ cached_property
    def norm_grad_rho_tor(self):
        return self.norm_grad_psi * self.drho_tor_dpsi.reshape(list(self.drho_tor_dpsi.shape)+[1])

    @ cached_property
    def drho_tor_dpsi(self)	:
        r"""
            .. math ::

                \frac{d\rho_{tor}}{d\psi}=\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                        =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                        =\frac{q}{2\pi B_{0}\rho_{tor}}

        """
        return Function(self.psi_norm, self.q/(2.0*scipy.constants.pi*self.b0))

    @ cached_property
    def dpsi_drho_tor(self)	:
        """
            Derivative of Psi with respect to Rho_Tor[Wb/m].

            Todo:
                FIXME: dpsi_drho_tor(0) = ???
        """
        return Function(self.psi_norm, (2.0*scipy.constants.pi*self.b0)*self.rho_tor/self.q, unit="Wb/m")

    @ cached_property
    def gm1(self):
        r"""
            Flux surface averaged 1/R ^ 2  [m ^ -2]
            .. math:: \left\langle\frac{1}{R^{2}}\right\rangle
        """
        return self.surface_average(1.0/(self.R**2))

    @ cached_property
    def gm2(self):
        r"""
            Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right|^2/R^2  [m^-2]
            .. math:: \left\langle\left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle
        """
        return self.surface_average((self.norm_grad_rho_tor/self.R)**2)

    @ cached_property
    def gm3(self):
        r"""
            Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right|^2  [-]
            .. math:: {\left\langle \left|\nabla\rho\right|^{2}\right\rangle}
        """
        return self.surface_average(self.norm_grad_rho_tor**2)

    @ cached_property
    def gm4(self):
        r"""
            Flux surface averaged 1/B ^ 2  [T ^ -2]
            .. math:: \left\langle \frac{1}{B^{2}}\right\rangle
        """
        return self.surface_average(1.0/self.B2)

    @ cached_property
    def gm5(self):
        r"""
            Flux surface averaged B ^ 2  [T ^ 2]
            .. math:: \left\langle B^{2}\right\rangle
        """
        return self.surface_average(self.B2)

    @ cached_property
    def gm6(self):
        r"""
            Flux surface averaged  .. math:: \left | \nabla \rho_{tor}\right|^2/B^2  [T^-2]
            .. math:: \left\langle \frac{\left|\nabla\rho\right|^{2}}{B^{2}}\right\rangle
        """
        return self.surface_average(self.norm_grad_rho_tor**2/self.B2)

    @ cached_property
    def gm7(self):
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right |  [-]
            .. math:: \left\langle \left|\nabla\rho\right|\right\rangle
        """
        return self.surface_average(self.norm_grad_rho_tor)

    @ cached_property
    def gm8(self):
        r"""
            Flux surface averaged R[m]
            .. math:: \left\langle R\right\rangle
        """
        return self.surface_average(self.R)

    @ cached_property
    def gm9(self):
        r"""
            Flux surface averaged 1/R[m ^ -1]
            .. math:: \left\langle \frac{1}{R}\right\rangle
        """
        return self.surface_average(1.0/self.R)

    @ cached_property
    def magnetic_shear(self):
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
        return Function(self.psi_norm, self.rho_tor/self.q * self.q.derivative)

    @ cached_property
    def r_inboard(self):
        """Radial coordinate(major radius) on the inboard side of the magnetic axis[m]"""
        return NotImplemented

    @ cached_property
    def r_outboard(self):
        """Radial coordinate(major radius) on the outboard side of the magnetic axis[m]"""
        return NotImplemented

    @ cached_property
    def rho_volume_norm(self)	:
        """Normalised square root of enclosed volume(radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary
            (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)[-]"""
        return NotImplemented

    @ cached_property
    def area(self):
        """Cross-sectional area of the flux surface[m ^ 2]"""
        return NotImplemented

    @ cached_property
    def darea_dpsi(self):
        """Radial derivative of the cross-sectional area of the flux surface with respect to psi[m ^ 2.Wb ^ -1]. """
        return NotImplemented

    @ cached_property
    def darea_drho_tor(self)	:
        """Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor[m]"""
        return NotImplemented

    @ cached_property
    def surface(self):
        """Surface area of the toroidal flux surface[m ^ 2]"""
        return NotImplemented

    @ cached_property
    def trapped_fraction(self)	:
        """Trapped particle fraction[-]"""
        return NotImplemented

    @ cached_property
    def b_field_max(self):
        """Maximum(modulus(B)) on the flux surface(always positive, irrespective of the sign convention for the B-field direction)[T]"""
        return NotImplemented

    @ cached_property
    def beta_pol(self):
        """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
        return NotImplemented
