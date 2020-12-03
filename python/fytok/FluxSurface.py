
import collections
import functools
import inspect
from concurrent import futures
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import constants
from numpy import arctan2, cos, sin, sqrt
from scipy.interpolate import (RectBivariateSpline, SmoothBivariateSpline,
                               UnivariateSpline)
from scipy.optimize import root_scalar
from spdm.util.AttributeTree import AttributeTree, _next_
from spdm.util.Interpolate import (Interpolate1D, Interpolate2D, derivate,
                                   find_critical, find_root, integral,
                                   interpolate)
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles
from sympy import Point, Polygon


class FluxSurface(Profiles):
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

    def __init__(self,  psirz,
                 coordinate_system, *args,
                 limiter=None,
                 r0=None,
                 b0=None,
                 ffprime=None,
                 psi_norm=129,
                 tolerance=1.0e-6,   ** kwargs):
        """Initialize FluxSurface

        """
        super().__init__(None, *args, x_axis=psi_norm, **kwargs)
        self._limiter = limiter
        self._psirz = psirz
        self._r0 = r0
        self._b0 = b0
        self._ffprime = ffprime
        self._psi_norm = self._x_axis
        if callable(ffprime):
            self._ffprime = ffprime
        elif isinstance(ffprime, np.ndarray):
            self._ffprime = UnivariateSpline(self._psi_norm, ffprime)
        else:
            raise TypeError(f"ffprime:{ffprime}")

        self.tolerance = tolerance

        if not isinstance(coordinate_system, AttributeTree):
            coordinate_system = AttributeTree(coordinate_system)
        if not coordinate_system.grid.grid_type.index or coordinate_system.grid.grid_type.index == 1:
            self._coordinate_system = coordinate_system
        else:
            raise NotImplementedError(f"coordinate_system type error! {coordinate_system}")

    @cached_property
    def critical_points(self):
        psirz = self._psirz
        opoints = []
        xpoints = []

        if not self._limiter:
            raise RuntimeError(f"Missing 'limiter'!")

        bounds = [v for v in map(float, self._limiter.bounds)]

        NX = 128
        NY = int(NX*(bounds[3] - bounds[1])/(bounds[2] - bounds[0])+0.5)

        X, Y = np.meshgrid(np.linspace(bounds[0], bounds[2], NX),
                           np.linspace(bounds[1], bounds[3], NY), indexing="ij")

        Rmid = float(bounds[2] + bounds[0])/2.0
        Zmid = float(bounds[3] + bounds[1])/2.0

        for r, z, tag in find_critical(psirz, X, Y):
            # Remove points outside the vacuum wall
            if not self._limiter and self._limiter.encloses(Point(r, z)):
                continue

            if tag < 0.0:  # saddle/X-point
                xpoints.append(AttributeTree(r=r, z=z, psi=float(psirz(r, z))))
            else:  # extremum/ O-point
                opoints.append(AttributeTree(r=r, z=z, psi=float(psirz(r, z))))

        if not opoints:
            raise RuntimeError(f"Can not find o-point!")
        else:
            opoints.sort(key=lambda x: (x.r - Rmid)**2 + (x.z - Zmid)**2)
            psi_axis = opoints[0].psi

            xpoints.sort(key=lambda x: (x.psi - psi_axis)**2)

        return opoints, xpoints

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
            dim_theta = np.linspace(0, constants.pi*2.0,  ntheta, endpoint=False)
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

            theta0 = arctan2(R1 - R0, Z1 - Z0)  # + constants.pi/npoints  # avoid x-point
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
        if not self._coordinate_system.grid_type.index:
            pass
        elif self._coordinate_system.grid_type.index > 1:
            raise ValueError(f"Unknown grid type {self._coordinate_system.grid_type}")

        npsi = self._coordinate_system.grid.dim1
        ntheta = self._coordinate_system.grid.dim2

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
        return self.surface_mesh[:, :, 0]

    @cached_property
    def Z(self):
        return self.surface_mesh[:, :, 1]

    @cached_property
    def grad_psi(self):
        return self._psirz(self.R, self.Z, dx=1), self._psirz(self.R, self.Z, dy=1)

    @cached_property
    def dl(self):
        dR = (np.roll(self.R, 1, axis=1) - np.roll(self.R, -1, axis=1))/2.0
        dZ = (np.roll(self.Z, 1, axis=1) - np.roll(self.Z, -1, axis=1))/2.0
        return sqrt(dR ** 2 + dZ ** 2)

    @cached_property
    def Jdl(self):
        return (self.R / np.sqrt(self.grad_psi2)) * self.dl

    @cached_property
    def grad_psi2(self):
        dpsi_dr, dpsi_dz = self.grad_psi
        return dpsi_dr**2 + dpsi_dz**2

    @cached_property
    def B2(self):
        return (self.grad_psi2+self.fpol.reshape((-1, 1))**2)/(self.R**2)

    @cached_property
    def fpol(self):
        """Diamagnetic function (F=R B_tor)  [T.m]."""
        f2 = np.array([self._ffprime.integral(p, 1.0) for p in self._psi_norm]) * (self.psi_axis-self.psi_boundary)
        return np.sqrt(f2 * 2.0 + (self._r0*self._b0)**2)

    @cached_property
    def dvolume_dpsi(self):
        r""".. math:: V^{\prime} =  2 \pi  \int{ R / |\nabla \psi| * dl }
            .. math:: V^{\prime}(psi)= 2 \pi  \int{ dl * R / |\nabla \psi|}
        """
        return (2*constants.pi) * np.sum(self.Jdl, axis=1)

    @cached_property
    def volume(self):
        """Volume enclosed in the flux surface[m ^ 3]"""
        return self.integral(self.dvolume_dpsi, 0.0, self._psi_norm) * (self.psi_axis-self.psi_boundary)

    @property
    def vprime(self):
        return self.dvolume_dpsi

    @cached_property
    def q(self):
        r"""Safety factor 
            (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)  [-].

            .. math:: q(\psi)=\frac{d\Phi}{d\psi}=\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{4\pi^{2}}
        """
        logger.debug(r"Calculate q as  F V^{\prime} \left\langle R^{-2}\right \rangle /(4 \pi^2) ")
        return self.fpol*self.dvolume_dpsi * self.gm1 / (4*scipy.constants.pi**2)

    @cached_property
    def phi(self):
        r""" 
            .. math ::
                \Phi_{tor}\left(\psi\right)=\int_{0}^{\psi}qd\psi
        """
        return self.integral(self.q, 0, self._psi_norm)

    @cached_property
    def rho_tor(self):
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0  [m]"""
        return np.sqrt(self.phi)/np.sqrt(scipy.constants.pi * self._b0)

    @cached_property
    def drho_tor_dpsi(self)	:
        r"""
            .. math ::

                \frac{d\rho_{tor}}{d\psi}=\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                        =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                        =\frac{q}{2\pi B_{0}\rho_{tor}}

        """
        res = self.q/self.rho_tor/(2.0*constants.pi*self._b0)
        res[0] = res[1]*2-res[2]
        return res

    @cached_property
    def dpsi_drho_tor(self)	:
        """Derivative of Psi with respect to Rho_Tor[Wb/m]. """
        return (2.0*constants.pi*self._b0)*self.rho_tor/self.q

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
            res = (2*constants.pi) * np.sum(func(self.R, self.Z, *args, **kwargs)*self.Jdl, axis=1) / self.vprime
        else:
            res = (2*constants.pi) * np.sum(func * self.Jdl, axis=1) / self.vprime
        # res[0] = res[1]
        return res

    def apply(self, func, R, Z, *args, **kwargs):
        pval = self._psirz(R, Z, *args, **kwargs)
        if isinstance(pval, np.ndarray):
            pass
        return func(pval)

    def plot(self, axis):
        axis.plot(self.R, self.Z, "b--", linewidth=0.1)
        axis.plot(self.R.transpose(1, 0), self.Z.transpose(1, 0), "b--", linewidth=0.1)
