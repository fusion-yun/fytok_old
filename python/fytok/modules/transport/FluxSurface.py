
from spdm.data.Quantity import Quantity
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.data.Field import Field
from spdm.data.Coordinates import Coordinates
from scipy.optimize import fsolve, root_scalar
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.ndimage.filters import maximum_filter, minimum_filter
from numpy import arctan2, cos, sin, sqrt
import collections
import inspect
from functools import cached_property

import numpy as np
import scipy.constants
from packaging import version
from spdm.util.logger import logger


if version.parse(scipy.__version__) <= version.parse("1.4.1"):
    from scipy.integrate import cumtrapz as cumtrapz
else:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
logger.debug(f"SciPy Version: {scipy.__version__}")


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
    X, Y = fun2d.coordinates.mesh.mesh
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


class FluxSurface(PhysicalGraph):
    r"""Flux surface average

        .. math::
            V^{\prime}\left(\rho\right)=\frac{\partial V}{\partial\rho}=2\pi\int_{0}^{2\pi}\sqrt{g}d\theta=2\pi\oint\frac{R}{\left|\nabla\rho\right|}dl

        .. math::
            \left\langle\alpha\right\rangle\equiv\frac{2\pi}{V^{\prime}}\int_{0}^{2\pi}\alpha\sqrt{g}d\theta=\frac{2\pi}{V^{\prime}}\varoint\alpha\frac{R}{\left|\nabla\rho\right|}dl

        grid_type :
                   =1  rectangular	,
                       Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2). In this case the position
                       arrays should not be filled since they are redundant with grid/dim1 and dim2.
    """

    def __init__(self,
                 psirz: Field,
                 *args,
                 vacuum_toroidal_field=None,
                 wall=None,
                 ffprime=None,
                 psi_norm=None,
                 tolerance=1.0e-9,
                 **kwargs):
        """
            Initialize FluxSurface
        """
        super().__init__(None, *args, **kwargs)

        self.__dict__["_tolerance"] = tolerance
        self.__dict__["_wall"] = wall
        self.__dict__["_vacuum_toroidal_field"] = vacuum_toroidal_field

        if not isinstance(psirz, Field):
            raise TypeError(psirz)

        self.__dict__["_psirz"] = psirz

        psi_norm = psi_norm or len(self._psirz.coordinates.mesh.axis[0])
        if isinstance(psi_norm, int):
            psi_norm = np.linspace(0, 1.0, psi_norm)
        elif isinstance(psi_norm, np.ndarray):
            pass
        elif not psi_norm:
            raise TypeError(type(psi_norm))

        if ffprime is None:
            pass
        else:
            if not isinstance(ffprime, np.ndarray):
                raise TypeError(type(ffprime))

            if psi_norm is None:
                if isinstance(ffprime, Field):
                    psi_norm = ffprime.coordinates.mesh.axis[0]
                elif isinstance(ffprime, np.ndarray):
                    psi_norm = np.ndarray(0, 1.0, len(ffprime))
                else:
                    raise TypeError(type(ffprime))
            elif len(psi_norm) == len(ffprime):
                pass
            elif isinstance(ffprime, Field):
                ffprime = ffprime(psi_norm)
            else:
                raise TypeError(type(ffprime))
            self.__dict__["_ffprime"] = ffprime

        self.__dict__["_psi_norm"] = psi_norm

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
        return 1 if self.psi_boundary > self.psi_axis else -1

    @cached_property
    def psi_axis(self):
        o, _ = self.critical_points
        return o[0].psi

    @cached_property
    def psi_boundary(self):
        _, x = self.critical_points
        if len(x) > 0:
            return x[0].psi
        else:
            raise ValueError(f"No x-point")

    def find_by_psinorm(self, psival, *args, **kwargs):
        yield from self.find_by_psi(psival*(self.psi_boundary-self.psi_axis)+self.psi_axis, *args, **kwargs)

    def find_by_psi(self, psival, ntheta=64):

        if type(ntheta) is int:
            dim_theta = np.linspace(0, scipy.constants.pi*2.0,  ntheta, endpoint=False)
        elif isinstance(ntheta, collections.abc.Sequence) or isinstance(ntheta, np.ndarray):
            dim_theta = ntheta
        else:
            dim_theta = [ntheta]

        o_points, x_points = self.critical_points

        R0 = o_points[0].r
        Z0 = o_points[0].z

        if len(x_points) > 0:
            R1 = x_points[0].r
            Z1 = x_points[0].z

            theta0 = arctan2(R1 - R0, Z1 - Z0)  # + scipy.constants.pi/npoints  # avoid x-point
            Rm = sqrt((R1-R0)**2+(Z1-Z0)**2)
        else:
            theta0 = 0
            Rm = R0

        for t in dim_theta:
            theta = t+theta0
            r0 = R0
            z0 = Z0
            r1 = R0 + Rm * sin(theta)
            z1 = Z0 + Rm * cos(theta)

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

    @cached_property
    def surface_mesh(self):
        if not self._parent.coordinate_system.grid_type.index:
            pass
        elif self._parent.coordinate_system.grid_type.index > 1:
            raise ValueError(f"Unknown grid type {self._coordinate_system.grid_type}")

        npsi = self._parent.coordinate_system.grid.dim1
        ntheta = self._parent.coordinate_system.grid.dim2

        # TODO: Using futures.ThreadPoolExecutor() cannot improve performance. Why ?
        return np.array([[[r, z] for r, z in self.find_by_psinorm(psival, ntheta)] for psival in npsi])

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

    @cached_property
    def R(self):
        return self._psirz.coordinates.mesh.mesh[0]

    @cached_property
    def Z(self):
        return self._psirz.coordinates.mesh.mesh[1]

    @cached_property
    def grad_psi(self):
        return self._psirz(self.R, self.Z, dx=1), self._psirz(self.R, self.Z, dy=1)

    @cached_property
    def dl(self):
        dR = (np.roll(self.R, 1, axis=1) - np.roll(self.R, -1, axis=1))/2.0
        dZ = (np.roll(self.Z, 1, axis=1) - np.roll(self.Z, -1, axis=1))/2.0
        return sqrt(dR ** 2 + dZ ** 2)

    @cached_property
    def grad_psi2(self):
        dpsi_dr, dpsi_dz = self.grad_psi
        return dpsi_dr**2 + dpsi_dz**2

    @cached_property
    def Jdl(self):
        return (self.R / np.sqrt(self.grad_psi2)) * self.dl

    @cached_property
    def B2(self):
        return (self.grad_psi2+self.fpol.reshape((-1, 1))**2)/(self.R**2)

    #################################

    @property
    def vacuum_toroidal_field(self):
        return self._vacuum_toroidal_field

    @property
    def ffprime(self):
        return self._ffprime

    @cached_property
    def fpol(self):
        f2 = cumtrapz(self.ffprime, self.psi_norm, initial=0) \
            * (self.psi_axis - self.psi_boundary) * 2.0
        return np.sqrt(f2 + (self.vacuum_toroidal_field.r0*self.vacuum_toroidal_field.b0)**2)

    @cached_property
    def psi_norm(self):
        return self._psi_norm

    @cached_property
    def psi(self):
        return Field(self.psi_norm * (self.psi_boundary-self.psi_axis) + self.psi_axis, coordinates=self.psi_norm, unit="Wb")

    @cached_property
    def dvolume_dpsi(self):
        return self.vprime*self.cocos_flag

    @cached_property
    def vprime(self):
        r""".. math:: V^{\prime} =  2 \pi  \int{ R / |\nabla \psi| * dl }
            .. math:: V^{\prime}(psi)= 2 \pi  \int{ dl * R / |\nabla \psi|}
        """
        return Field((2.0*scipy.constants.pi)*np.sum(self.Jdl, axis=1), coordinates=self.psi_norm)

    @cached_property
    def volume(self):
        """Volume enclosed in the flux surface[m ^ 3]"""
        return Field(cumtrapz(self.dvolume_dpsi, self.psi_norm, initial=0) * (self.psi_boundary-self.psi_axis), coordinates=self.psi_norm, unit="m**3")

    @cached_property
    def dvolume_drho_tor(self)	:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
        return self.dvolume_dpsi * self.dpsi_drho_tor

    @cached_property
    def q(self):
        r"""Safety factor
            (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)  [-].
            .. math:: q(\psi)=\frac{d\Phi}{d\psi}=\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{4\pi^{2}}
        """
        # logger.debug(r"Calculate q as  F V^{\prime} \left\langle R^{-2}\right \rangle /(4 \pi^2) ")
        return self.fpol * np.sum(self.Jdl/(self.R**2), axis=1)*(self.cocos_flag / (2*scipy.constants.pi))

    @cached_property
    def phi(self):
        r"""
            Note:
                !!! COORDINATE　DEPENDENT!!!

            .. math ::
                \Phi_{tor}\left(\psi\right)=\int_{0}^{\psi}qd\psi
        """
        return Field(cumtrapz(self.q, self.psi_norm) * (self.psi_boundary-self.psi_axis), coordinates=self.psi_norm, unit="Wb")

    @cached_property
    def rho_tor(self):
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0  [m]"""
        data = np.sqrt(self.phi)/np.sqrt(scipy.constants.pi * self.vacuum_toroidal_field.b0)
        return Field(data, coordinates=self.psi_norm, unit="m")

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
        return Field(res, coordinates=self.psi_norm, unit="Wb/m")

    @cached_property
    def gm1(self):
        r""".. math:: \left\langle\frac{1}{R^{2}}\right\rangle """
        return self.average(1.0/self.R**2)

    @cached_property
    def gm2(self):
        r""".. math:: \left\langle\left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle """
        return self.average(self.grad_psi2/(self.R**2))*(self.drho_tor_dpsi**2)

    @cached_property
    def gm3(self):
        r""".. math:: {\left\langle \left|\nabla\rho\right|^{2}\right\rangle}"""
        return self.average(self.grad_psi2)*(self.drho_tor_dpsi**2)

    @cached_property
    def gm4(self):
        r""".. math:: \left\langle \frac{1}{B^{2}}\right\rangle """
        return self.average(1/self.B2)

    @cached_property
    def gm5(self):
        r""".. math:: \left\langle B^{2}\right\rangle """
        return self.average(self.B2)

    @cached_property
    def gm6(self):
        r""".. math:: \left\langle \frac{\left|\nabla\rho\right|^{2}}{B^{2}}\right\rangle """
        return self.average(self.grad_psi2/self.B2) * (self.drho_tor_dpsi**2)

    @cached_property
    def gm7(self):
        r""".. math:: \left\langle \left|\nabla\rho\right|\right\rangle """
        return self.average(np.sqrt(self.grad_psi2)) * self.drho_tor_dpsi

    @cached_property
    def gm8(self):
        r""".. math:: \left\langle R\right\rangle """
        return self.average(self.R)

    @cached_property
    def gm9(self):
        r""".. math:: \left\langle \frac{1}{R}\right\rangle """
        return self.average(1.0/self.R)

    def average(self, func, *args, **kwargs):
        if inspect.isfunction(func):
            res = (2*scipy.constants.pi) * np.sum(func(self.R, self.Z, *args, **kwargs)*self.Jdl, axis=1) / self.vprime
        else:
            res = (2*scipy.constants.pi) * np.sum(func * self.Jdl, axis=1) / self.vprime
        # res[0] = res[1]
        return res
