
import collections
import inspect
from functools import cached_property
from itertools import cycle

import numpy as np
import scipy.constants
import scipy.integrate
from numpy import arctan2, cos, sin, sqrt
from packaging import version
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.optimize import fsolve, root_scalar
from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.data.mesh.CurvilinearMesh import CurvilinearMesh
from spdm.data.mesh.RectilinearMesh import RectilinearMesh
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.util.logger import logger

if version.parse(scipy.__version__) <= version.parse("1.4.1"):
    from scipy.integrate import cumtrapz as cumtrapz
else:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
logger.debug(f"SciPy Version: {scipy.__version__}")


def power2(a):
    return np.inner(a, a)


def find_peaks_2d_image(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    idxs = np.where(detected_peaks)

    for n in range(len(idxs[0])):
        yield idxs[0][n], idxs[1][n]


def find_critical(fun2d: Field):

    fxy2 = fun2d.derivative(dx=1)**2+fun2d.derivative(dy=1)**2
    X, Y = fun2d.coordinates.mesh.points
    span = 1
    for ix, iy in find_peaks_2d_image(-fxy2[span:-span, span:-span]):
        ix += span
        iy += span
        x = X[ix, iy]
        y = Y[ix, iy]
        if abs(fxy2[ix, iy]) > 1.0e-5:  # FIXME: replace magnetic number
            xmin = X[ix-1, iy]
            xmax = X[ix+1, iy]
            ymin = Y[ix, iy-1]
            ymax = Y[ix, iy+1]

            def f(r):
                if r[0] < xmin or r[0] > xmax or r[1] < ymin or r[1] > ymax:
                    raise LookupError(r)
                fx = fun2d.derivative(r[0], r[1], dx=1)
                fy = fun2d.derivative(r[0], r[1], dy=1)

                return fx, fy

            def fprime(r):
                fxx = fun2d.derivative(r[0], r[1], dx=2)
                fyy = fun2d.derivative(r[0], r[1], dy=2)
                fxy = fun2d.derivative(r[0], r[1], dy=1, dx=1)

                return [[fxx, fxy], [fxy, fyy]]  # FIXME: not sure, need double check

            try:
                x1, y1 = fsolve(f, [x, y],   fprime=fprime)
            except LookupError:
                continue
            else:
                x = x1
                y = y1

        # D = fxx * fyy - (fxy)^2
        D = fun2d.derivative(x, y, dx=2) * fun2d.derivative(x, y, dy=2) - (fun2d.derivative(x, y,  dx=1, dy=1))**2

        yield x, y, D

    # RZ = np.full([npsi.shape[0], ntheta.shape[0], 2], np.nan)
    # def func(psival, ntheta=ntheta):
    #     return [[r, z] for r, z in self.find_by_psinorm(psival, ntheta)]
    # # We can use a with statement to ensure threads are cleaned up promptly
    # with futures.ThreadPoolExecutor( ) as executor:
    #     # Start the load operations and mark each future with its URL
    #     future_to_i = {executor.submit(func, psi): i for i, psi in enumerate(npsi)}
    #     for future in futures.as_completed(future_to_i):
    #         i = future_to_i[future]
    #         try:
    #             RZ[i] = future.result()
    #         except Exception as exc:
    #             print('%r generated an exception: %s' % (i, exc))


class FluxSurface:
    r"""
        Flux surface average

        .. math::
            V^{\prime}\left(\rho\right)=\frac{\partial V}{\partial\rho}=2\pi\int_{0}^{2\pi}\sqrt{g}d\theta=2\pi\oint\frac{R}{\left|\nabla\rho\right|}dl

        .. math::
            \left\langle\alpha\right\rangle\equiv\frac{2\pi}{V^{\prime}}\int_{0}^{2\pi}\alpha\sqrt{g}d\theta=\frac{2\pi}{V^{\prime}}\varoint\alpha\frac{R}{\left|\nabla\rho\right|}dl

        grid_type :
                   =1  rectangular	,
                       Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2). In this case the position
                       arrays should not be filled since they are redundant with grid/dim1 and dim2.
    """

    def __init__(self, psirz: Field, *args,
                 wall=None,
                 fvac=None,
                 ffprime=None,
                 grid_shape=[64, 128],
                 tolerance=1.0e-9,
                 **kwargs):
        """
            Initialize FluxSurface
        """

        if not isinstance(psirz, (Field, Function)):
            raise TypeError(psirz)
        self._psirz = psirz
        self._wall = wall
        self._fvac = fvac
        self._ffprime = ffprime
        self._grid_shape = grid_shape
        self._mesh = None

    @property
    def grid_index(self):
        return self._grid_index

    @cached_property
    def critical_points(self):
        opoints = []
        xpoints = []

        for r, z, tag in find_critical(self._psirz):
            # Remove points outside the vacuum wall
            if not self._wall.in_limiter(r, z):
                continue
            p = PhysicalGraph({"r": r, "z": z, "psi": self._psirz(r, z)})

            if tag < 0.0:  # saddle/X-point
                xpoints.append(p)
            else:  # extremum/ O-point
                opoints.append(p)

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

    @cached_property
    def psi_axis(self):
        return self.critical_points[0][0].psi

    @cached_property
    def psi_boundary(self):
        _, x = self.critical_points
        if len(x) == 0:
            raise ValueError(f"No x-point")
        return x[0].psi

    @property
    def grid_shape(self):
        return self._grid_shape

    def reconstruct(self,  radial_label=1, poloidal_angle_label=3, shape=None):
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
                x6      :  Fourier modes in the polar poloidal angl
        """

        grid_shape = shape or self._grid_shape

        u = None
        v = None

        if radial_label == 1:
            u = np.linspace(0, 1.0, grid_shape[0])
            # raise NotImplementedError(f"psi")
        elif radial_label == 2:
            u = np.linspace(0, 1.0, grid_shape[0])**2
            # raise NotImplementedError(f"sqrt[(psi-psi_axis)/(psi_edge-psi_axis)]")
        elif radial_label == 3:
            raise NotImplementedError(f"sqrt[Phi/Phi_edge]")
        elif radial_label == 4:
            u = np.linspace(0, 1.0, grid_shape[0])**2
            # raise NotImplementedError(f"sqrt[psi-psi_axis]")
        elif radial_label == 5:
            raise NotImplementedError(f"sqrt[Phi/pi/B0]")
        else:
            raise NotImplementedError(f"unknown")

        if poloidal_angle_label == 0:
            raise NotImplementedError(f"the straight-field line")
        elif poloidal_angle_label == 1:
            raise NotImplementedError(f" the straight-field line")
        elif poloidal_angle_label == 2:
            raise NotImplementedError(f" the equal arc poloidal angle")
        elif poloidal_angle_label == 3:  # the polar poloidal angle
            v = np.linspace(0, 1.0, grid_shape[1])*scipy.constants.pi*2.0
        elif poloidal_angle_label == 4:
            raise NotImplementedError(
                f"Fourier modes in the straight-field line poloidal angle ")
        elif poloidal_angle_label == 5:
            raise NotImplementedError(f"Fourier modes in the equal arc poloidal angle  ")
        else:
            raise NotImplementedError(f"unknown")

        if self._mesh is None and poloidal_angle_label == 3:
            mesh = self._mesh

    def find_by_psinorm(self, u, v=128):
        o_points, x_points = self.critical_points

        if len(o_points) == 0:
            raise RuntimeError(f"Can not find O-point!")
        else:
            R0 = o_points[0].r
            Z0 = o_points[0].z
            psi0 = o_points[0].psi

        if len(x_points) == 0:
            R1 = R0
            Z1 = self._psirz.coordinates.bbox[1][1]
            psi1 = 0.0
        else:
            R1 = x_points[0].r
            Z1 = x_points[0].z
            psi1 = x_points[0].psi

        theta0 = arctan2(R1 - R0, Z1 - Z0)
        Rm = sqrt((R1-R0)**2+(Z1-Z0)**2)

        if isinstance(u, int):
            u = np.linspace(0, 1.0, u)
        elif not isinstance(u, (np.ndarray, collections.abc.Sequence)):
            u = [u]

        if isinstance(v, int):
            v = np.linspace(0, 1.0, v)
        elif not isinstance(v, (np.ndarray, collections.abc.Sequence)):
            v = [v]

        for p in u:
            for t in v:

                psival = p*(psi1-psi0)+psi0
                r0 = R0
                z0 = Z0
                r1 = R0+Rm*sin(t*scipy.constants.pi*2.0+theta0)
                z1 = Z0+Rm*cos(t*scipy.constants.pi*2.0+theta0)
                if not np.isclose(self._psirz(r1, z1), psival):
                    try:
                        sol = root_scalar(lambda r: self._psirz((1.0-r)*r0+r*r1, (1.0-r)*z0+r*z1) - psival,
                                          bracket=[0, 1], method='brentq')
                    except ValueError as error:
                        raise ValueError(f"Find root fialed! {error}")

                    if not sol.converged:
                        raise ValueError(f"Find root fialed!")

                    r1 = (1.0-sol.root)*r0+sol.root*r1
                    z1 = (1.0-sol.root)*z0+sol.root*z1

                yield r1, z1

    @property
    def mesh(self):
        if self._mesh is None:
            u = np.linspace(0, 1.0, self._grid_shape[0])
            v = np.linspace(0, 1.0, self._grid_shape[1])
            rz = np.asarray([[r, z] for r, z in self.find_by_psinorm(u, v)]).reshape(self._grid_shape+[2])
            self._psi_norm = u
            self._mesh = CurvilinearMesh([rz[:, :, 0], rz[:, :, 1]], [u, v], cycle=[False, True])

        return self._mesh

    def R(self):
        return self._mesh.points[0]

    @cached_property
    def Z(self):
        return self._mesh.points[1]

    #################################

    def grad_psi(self, r, z):
        return self._psirz(r, z, dx=1), self._psirz(r, z, dy=1)

    def psi_norm_rz(self, r, z):
        psi = self._psirz(r, z)
        return (psi-self.psi_boundary)/(self.psi_axis-self.psi_boundary)

    def B2(self, r, z):
        br, bz = self.grad_psi(r, z)
        return ((br**2+bz**2)+self.fpol(self.psi_norm_rz(r, z))**2)/(r**2)

    @cached_property
    def _grad_psi(self):
        r = self.mesh.points[0]
        z = self.mesh.points[1]
        return self._psirz(r, z, dx=1), self._psirz(r, z, dy=1)

    def average(self, func, *args, **kwargs):
        r"""
            .. math:: \left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}
        """

        d = np.asarray([scipy.integrate.romberg(lambda u, _axis=self.mesh.axis(idx, axis=0), _J=self.Jdl[idx]:  _axis.pullback(func, u) * _J(u),
                                                0.0, 1.0) for idx in range(self.mesh.shape[0])])

        return (2*scipy.constants.pi) * d / self.vprime

        # if inspect.isfunction(func):
        #     res = (2*scipy.constants.pi) * np.sum(func(self.R, self.Z, *args, **kwargs)*self.Jdl, axis=1) / self.vprime
        # else:
        #     res = (2*scipy.constants.pi) * np.sum(func * self.Jdl, axis=1) / self.vprime
        # # res[0] = res[1]
        # return res

    #################################

    @property
    def psi_norm(self):
        return self._psi_norm

    @cached_property
    def psi(self):
        return self.psi_norm * (self.psi_boundary-self.psi_axis) + self.psi_axis

    @cached_property
    def Jdl(self):
        def J(r, z, _grad_psi=self.grad_psi): return r / np.linalg.norm(_grad_psi(r, z))
        # return [self.mesh.axis(idx, axis=0).make_one_form(J) for idx in range(self.mesh.shape[0])]
        res = [self.mesh.axis(idx, axis=0).pullback(J, form=1) for idx in range(self.mesh.shape[0])]
        return res

    @cached_property
    def vprime(self):
        r"""
            .. math:: V^{\prime} =  2 \pi  \int{ R / |\nabla \psi| * dl }
            .. math:: V^{\prime}(psi)= 2 \pi  \int{ dl * R / |\nabla \psi|}
        """
        return np.asarray([self.Jdl[idx].integrate() for idx in range(self.mesh.shape[0])]) * (2*scipy.constants.pi)

    @cached_property
    def dvolume_dpsi(self):
        r"""
            Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1].
        """
        return self.vprime*self.cocos_flag

    @cached_property
    def volume(self):
        """Volume enclosed in the flux surface[m ^ 3]"""
        return self.dvolume_dpsi.antiderivative

    @cached_property
    def ffprime(self):
        """	Derivative of F w.r.t. Psi, multiplied with F  [T^2.m^2/Wb]. """
        d = self["_ffprime"]
        if isinstance(d, Function):
            return Function(self.psi_norm, d(self.psi_norm))
        elif isinstance(d, np.ndarray) and len(d) == len(self.psi_norm):
            return Function(self.psi_norm, d)
        else:
            raise TypeError(type(d))

    @property
    def f_df_dpsi(self):
        """	Derivative of F w.r.t. Psi, multiplied with F  [T^2.m^2/Wb]. """
        return self.ffprime

    @cached_property
    def fpol(self):
        """Diamagnetic function (F=R B_Phi)  [T.m]."""

        psi_axis = self.vacuum_toroidal_field.psi_axis
        psi_boundary = self.vacuum_toroidal_field.psi_boundary
        f2 = self.ffprime.antiderivative * (2.0*(psi_boundary-psi_axis)) + self._fvac**2
        return Function(self.psi_norm, np.sqrt(f2), unit="T.m")

    @property
    def f(self):
        """Diamagnetic function (F=R B_Phi)  [T.m]."""
        return self.fpol

    @cached_property
    def q(self):
        r"""
            Safety factor
            (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)  [-].
            .. math:: q(\psi)=\frac{d\Phi}{d\psi}=\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{4\pi^{2}}
        """
        return self.average(lambda r, z: 1.0/r**2) * (1.0 / (2*scipy.constants.pi)) * self.fpol

    @cached_property
    def dvolume_drho_tor(self)	:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
        return self.dvolume_dpsi * self.dpsi_drho_tor

    @cached_property
    def dvolume_dpsi_norm(self):
        """Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1]. """
        return NotImplemented

    @cached_property
    def phi(self):
        r"""
            Note:
                !!! COORDINATE　DEPENDENT!!!

            .. math ::
                \Phi_{tor}\left(\psi\right)=\int_{0}^{\psi}qd\psi
        """
        return self.q.antiderivative

    @cached_property
    def rho_tor(self):
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0  [m]"""
        data = np.sqrt(self.phi)/np.sqrt(scipy.constants.pi * self.vacuum_toroidal_field.b0)
        return Function(self.psi_norm, data, unit="m")

    @cached_property
    def rho_tor_norm(self):
        return self.rho_tor/self.rho_tor[-1]

    @cached_property
    def drho_tor_dpsi(self)	:
        r"""
            .. math ::

                \frac{d\rho_{tor}}{d\psi}=\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                        =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                        =\frac{q}{2\pi B_{0}\rho_{tor}}

        """
        res = self.q/(2.0*scipy.constants.pi*self.vacuum_toroidal_field.b0)
        res[1:] /= self.rho_tor[1:]
        # res[0] = res[1:5](0)  # self.fpol[0]*self.gm1[0]/(2.0*scipy.constants.pi*self.vacuum_toroidal_field.b0)
        # return self.q/(2.0*scipy.constants.pi*self.vacuum_toroidal_field.b0)
        res[0] = 2*res[1]-res[2]
        return res

    @cached_property
    def dpsi_drho_tor(self)	:
        """
            Derivative of Psi with respect to Rho_Tor[Wb/m].

            Todo:
                FIXME: dpsi_drho_tor(0) = ???
        """
        res = (2.0*scipy.constants.pi*self.vacuum_toroidal_field.b0)*self.rho_tor[:]/self.q[:]
        res[0] = 2*res[1]-res[2]
        return Function(self.psi, res, unit="Wb/m")

    @cached_property
    def gm1(self):
        r"""
            Flux surface averaged 1/R ^ 2  [m ^ -2]
            .. math:: \left\langle\frac{1}{R^{2}}\right\rangle
        """
        return self.average(lambda r, z: 1.0/r**2)

    @cached_property
    def gm2(self):
        r"""
            Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right|^2/R^2  [m^-2]
            .. math:: \left\langle\left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle
        """
        return self.average(lambda r, z: power2(self.grad_psi(r, z)/(r**2)))*(self.drho_tor_dpsi**2)

    @cached_property
    def gm3(self):
        r"""
            Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right|^2  [-]
            .. math:: {\left\langle \left|\nabla\rho\right|^{2}\right\rangle}
        """
        return self.average(lambda r, z: power2(self.grad_psi(r, z)))*(self.drho_tor_dpsi**2)

    @cached_property
    def gm4(self):
        r"""
            Flux surface averaged 1/B ^ 2  [T ^ -2]
            .. math:: \left\langle \frac{1}{B^{2}}\right\rangle
        """
        return self.average(lambda r, z: 1.0/self.B2(r, z))

    @cached_property
    def gm5(self):
        r"""
            Flux surface averaged B ^ 2  [T ^ 2]
            .. math:: \left\langle B^{2}\right\rangle
        """
        return self.average(lambda r, z: self.B2(r, z))

    @cached_property
    def gm6(self):
        r"""
            Flux surface averaged  .. math:: \left | \nabla \rho_{tor}\right|^2/B^2  [T^-2]
            .. math:: \left\langle \frac{\left|\nabla\rho\right|^{2}}{B^{2}}\right\rangle
        """
        return self.average(lambda r, z: power2(self.grad_psi(r, z))/self.B2(r, z)) * (self.drho_tor_dpsi**2)

    @cached_property
    def gm7(self):
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right |  [-]
            .. math:: \left\langle \left|\nabla\rho\right|\right\rangle
        """
        return self.average(lambda r, z: np.linalg.norm(self.grad_psi(r, z))) * self.drho_tor_dpsi

    @cached_property
    def gm8(self):
        r"""
            Flux surface averaged R[m]
            .. math:: \left\langle R\right\rangle
        """
        return self.average(lambda r, z: r)

    @cached_property
    def gm9(self):
        r"""
            Flux surface averaged 1/R[m ^ -1]
            .. math:: \left\langle \frac{1}{R}\right\rangle
        """
        return self.average(lambda r, z: 1.0/r)

    @cached_property
    def magnetic_shear(self):
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
        return self.rho_tor/self.q * self.q.derivative

    @cached_property
    def r_inboard(self):
        """Radial coordinate(major radius) on the inboard side of the magnetic axis[m]"""
        return NotImplemented

    @cached_property
    def r_outboard(self):
        """Radial coordinate(major radius) on the outboard side of the magnetic axis[m]"""
        return NotImplemented

    @cached_property
    def rho_tor(self):
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
        return NotImplemented

    @cached_property
    def rho_tor_norm(self):
        """Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary
            (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)[-]"""
        return NotImplemented

    @cached_property
    def drho_tor_dpsi(self)	:
        return NotImplemented

    @cached_property
    def dpsi_drho_tor(self)	:
        """Derivative of Psi with respect to Rho_Tor[Wb/m]. """
        return NotImplemented

    @cached_property
    def dvolume_drho_tor(self)	:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
        return NotImplemented

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
        return NotImplemented

    @cached_property
    def b_field_max(self):
        """Maximum(modulus(B)) on the flux surface(always positive, irrespective of the sign convention for the B-field direction)[T]"""
        return NotImplemented

    @cached_property
    def beta_pol(self):
        """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
        return NotImplemented
