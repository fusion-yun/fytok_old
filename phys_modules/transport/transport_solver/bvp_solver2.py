"""

"""

import collections
import enum
from itertools import chain
from math import log
from typing import Callable, Mapping, Optional, Type, Union, Sequence

from matplotlib.pyplot import loglog

from fytok.common.IDS import IDS
from fytok.common.Misc import Identifier
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport
from fytok.transport.Equilibrium import Equilibrium
from fytok.transport.MagneticCoordSystem import RadialGrid
from fytok.transport.TransportSolver import TransportSolver
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, _not_found_, sp_property
from spdm.numlib import constants, np
from spdm.numlib.bvp import solve_bvp, BVPResult
from spdm.numlib.misc import array_like
from spdm.util.logger import logger
from spdm.util.utilities import convert_to_named_tuple

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class TransportSolverBVP2(TransportSolver):
    r"""
        Solve transport equations :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
        See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

        self._tau = self._core_profiles.time - self._core_profiles.previous_state.time

        self._core_profiles_next = self._core_profiles.profiles_1d
        self._core_profiles_prev = self._core_profiles.previous_state.profiles_1d
        self._c_transp = self._core_transport.model.combine.profiles_1d
        self._c_source = self._core_sources.source.combine.profiles_1d
        self._eq = self._equilibrium.time_slice.profiles_1d

        self._inv_tau = 0 if abs(self._tau) < EPSILON else 1.0/self._tau

        # $R_0$ characteristic major radius of the device   [m]
        self._R0 = self._equilibrium.time_slice.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        self._B0 = self._equilibrium.time_slice.vacuum_toroidal_field.b0

        self._B0m = self._equilibrium.previous_state.time_slice.vacuum_toroidal_field.b0
        # $rho_tor_{norm}$ normalized minor radius                [-]
        self._rho_tor_norm = self._core_profiles_next.grid.rho_tor_norm

        self._psi_norm = self._core_profiles_next.grid.psi_norm

        # Grid
        # $rho_tor$ not  normalized minor radius                [m]
        self._rho_tor = Function(self._rho_tor_norm, self._core_profiles_next.grid.rho_tor)

        self._rho_tor_boundary = self._core_profiles_next.grid.rho_tor[-1]

        self._rho_tor_boundary_m = self._core_profiles_prev.grid.rho_tor[-1]

        self._k_B = (self._B0 - self._B0m) / (self._B0 + self._B0m) * self._inv_tau * 2.0

        self._k_rho_bdry = (self._rho_tor_boundary - self._rho_tor_boundary_m) / \
            (self._rho_tor_boundary + self._rho_tor_boundary_m)*self._inv_tau*2.0

        self._k_phi = self._k_B + self._k_rho_bdry

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        self._fpol = Function(self._rho_tor_norm, self._eq.fpol(self._psi_norm))

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        self._vpr = Function(self._rho_tor_norm, self._eq.dvolume_drho_tor(self._psi_norm))

        self._vprm = Function(self._rho_tor_norm,
                              self._equilibrium.previous_state.time_slice.profiles_1d.dvolume_drho_tor(self._psi_norm))

        self._vpr35 = self._vpr**(5/3)
        self._vpr35m = self._vprm**(5/3)

        if np.isclose(self._eq.dvolume_drho_tor(self._psi_norm[0]), 0.0):
            self._inv_vpr23 = Function(self._rho_tor_norm[1:], self._eq.dvolume_drho_tor(self._psi_norm[1:])**(-2/3))
        else:
            self._inv_vpr23 = Function(self._rho_tor_norm, self._eq.dvolume_drho_tor(self._psi_norm)**(-2/3))

        # $q$ safety factor                                 [-]
        self._qsf = Function(self._rho_tor_norm, self._eq.q(self._psi_norm))
        self._gm1 = Function(self._rho_tor_norm, self._eq.gm1(self._psi_norm))
        self._gm2 = Function(self._rho_tor_norm, self._eq.gm2(self._psi_norm))
        self._gm3 = Function(self._rho_tor_norm, self._eq.gm3(self._psi_norm))

        self._Qimp_k_ns = (3*self._k_rho_bdry - self._k_phi * self._vpr.derivative())

    def create_transport_eq(self,
                            coeff: Sequence = [],  # (i,x,y,gamma)
                            bc: Sequence = [0, 1, 0, 1, 0, 0],
                            hyper_diff=0.0001,
                            **kwargs
                            ) -> BVPResult:

        def func(i: int, x: np.ndarray, Y: np.ndarray, G: np.ndarray,  Ym: Function,  h: float,
                 _coeff: Sequence[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = coeff,
                 _hyper_diff=hyper_diff):

            a, b, c, d, e, S = _coeff

            y = Y[i]
            g = G[i]

            yp = Function(x, y).derivative(x)

            dy = (-g + e(i, x,  Y, G)*y + _hyper_diff * yp)/(d(i, x,  Y, G) + _hyper_diff)

            dg = S(i, x, Y, G)

            if h is not None and Ym is not None and Ym[i] is not None:
                dg = dg - (a(i, x,  Y, G)*y - b(i, x, Y, G)*Ym[i](x))/h

            if callable(c):
                dg = dg*c(i, x,  Y, G)
            else:
                dg = dg*c

            return dy, dg

        def bc_func(ya: float, ga: float, yb: float, gb: float, /, _bc: Sequence = bc):
            r"""
                u*y(b)+v*\Gamma(b)=w
                Ya=[y(a),y'(a)]
                Yb=[y(b),y'(b)]
                P=[[u(a),u(b)],[v(a),v(b)],[w(a),w(b)]]
            """
            u0, v0, w0,  u1, v1, w1 = _bc
            return np.hstack([(u0 * ya + v0 * ga - w0), (u1 * yb + v1 * gb - w1)])

        return func, bc_func

    def current_transport(self,
                          transp: CoreTransport.Model.Profiles1D,
                          source: CoreSources.Source.Profiles1D,
                          bc: TransportSolver.BoundaryConditions1D.BoundaryConditions,
                          **kwargs):

        # -----------------------------------------------------------------
        # Transport
        # plasma parallel conductivity,                                 [(Ohm*m)^-1]
        conductivity_parallel = transp.conductivity_parallel

        # -----------------------------------------------------------
        # Sources

        j_exp = source.j_parallel  # + source.get("j_decomposed.explicit_part", Function(0))
        j_imp = 0  # source.get("j_decomposed.implicit_part", Function(0))

        def a(i, x, y, gamma): return (conductivity_parallel*self._rho_tor)(x)
        def b(i, x, y, gamma): return (conductivity_parallel*self._rho_tor)(x)
        def c(i, x, y, gamma): return ((constants.mu_0 * self._B0 * self._rho_tor_boundary)/(self._fpol**2))(x)
        def d(i, x, y, gamma): return(self._vpr * self._gm2 / self._fpol / (self._rho_tor_boundary)/(TWOPI))(x)
        def e(i, x, y, gamma): return ((- constants.mu_0 * self._B0 * self._k_phi)
                                       * (conductivity_parallel * self._rho_tor**2/self._fpol**2))(x)

        def S(i, x, y, gamma): return (-self._vpr * (j_exp + j_imp*y[i])/TWOPI)(x)
        # -----------------------------------------------------------
        # boundary condition, value
        # axis
        u0, v0, w0 = 0, 1, 0

        # Identifier of the boundary condition type.
        #   ID =    1: poloidal flux;
        #           2: ip;
        #           3: loop voltage;
        #           4: undefined;
        #           5: generic boundary condition y expressed as a1y'+a2y=a3.
        #           6: equation not solved; [eV]

        if bc.identifier.index == 1:  # poloidal flux;
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        elif bc.identifier.index == 2:  # ip, total current inside x=1
            Ip = bc.value[0]
            u1 = 0
            v1 = 1
            w1 = constants.mu_0 * Ip / self._fpol[-1]
        elif bc.identifier.index == 3:  # loop voltage;
            Uloop_bdry = bc.value[0]
            u1 = 0
            v1 = 1
            w1 = (self._tau*Uloop_bdry + self._core_profiles_prev.psi[-1])*d(1.0)
        elif bc.identifier.index == 5:  # generic boundary condition y expressed as a1y' + a2y=a3;
            u1 = bc.value[1]
            v1 = bc.value[0]
            w1 = bc.value[2]
        elif bc.identifier.index == 6:  # equation not solved;
            raise NotImplementedError(bc.identifier.index)
        else:
            raise NotImplementedError(bc)

        return "psi", *self.create_transport_eq((a, b, c, d, e, S), (u0, v0, w0, u1, v1, w1), **kwargs)

    def particle_transport(self,
                           path,
                           transp: CoreTransport.Model.Profiles1D,
                           source: CoreSources.Source.Profiles1D,
                           bc: TransportSolver.BoundaryConditions1D,
                           ** kwargs
                           ):
        # Particle Transport
        transp = transp.fetch(path)

        diff = transp.particles.d

        conv = transp.particles.v

        source = source.fetch(path)

        se_exp = source.particles  # + source.fetch("particles_decomposed.explicit_part", 0)

        se_imp = 0  # source.fetch("particles_decomposed.implicit_part", 0)

        def a(i, x, y, gamma): return self._vpr(x)
        def b(i, x, y, gamma): return self._vprm(x)
        def c(i, x, y, gamma): return self._rho_tor_boundary
        def d(i, x, y, gamma): return (self._vpr * self._gm3 * diff / self._rho_tor_boundary)(x)
        def e(i, x, y, gamma): return (self._vpr * self._gm3 * conv - self._vpr * self._rho_tor * self._k_phi)(x)
        def S(i, x, y, gamma): return (self._vpr * (se_exp + se_imp*y[i] + self._k_rho_bdry))(x)

        # -----------------------------------------------------------
        # boundary condition, value
        bc = bc.fetch(path)

        # axis
        u0, v0, w0 = 0, 1, 0.0

        if bc.particles.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.particles.value[0]
        else:
            raise NotImplementedError(bc.particles.identifier)

        return path+["density"], *self.create_transport_eq((a, b, c, d, e, S), (u0, v0, w0, u1, v1, w1), **kwargs)

    def neutral_condition(self, path):
        if path[0] == "electrons":
            pass
        else:
            pass

        def a(x, y, yp, i): return 0
        def b(x, y, yp, i): return 0
        def c(x, y, yp, i): return 0
        def d(x, y, yp, i): return 0
        def e(x, y, yp, i): return 0
        def S(x, y, yp, i): return 0

        return path+["density"], (a, b, c, d, e, S), (0,  0,  0,  0, 0, 0)

    def energy_transport(self,
                         path,
                         transp: CoreTransport.Model.Profiles1D,
                         source: CoreSources.Source.Profiles1D,
                         bc: TransportSolver.BoundaryConditions1D,
                         **kwargs
                         ):

        # energy transport
        transp = transp.fetch(path)

        chi = transp.energy.d
        v_pinch = transp.energy.v

        Qs_exp = source.energy  # + source.energy_decomposed.explicit_part
        Qs_imp = 0  # source.energy_decomposed.implicit_part

        # gamma_s = 3/2 * core_profiles_next.get("density_flux", 0)

        def a(x, y, yp, i): (3/2) * self._vpr35 * y[i-1]
        def b(x, y, yp, i): (3/2) * self._vpr35m * ym[i-1]
        def c(x, y, yp, i): self._rho_tor_boundary * self._inv_vpr23
        def d(x, y, yp, i): self._vpr * self._gm3 * y[i-1] * chi / self._rho_tor_boundary
        # - self._vpr * (3/4)*self._k_phi * self._rho_tor * density_next
        def e(x, y, yp, i): self._vpr * self._gm3 * y[i-1] * v_pinch  # + gamma_s
        def S(x, y, yp, i): self._vpr35 * (Qs_exp + (Qs_imp + self._Qimp_k_ns)*y[i-1])
        # + np.sum([self.nu_ab(label, other.label)*other.temperature for other in species])
        # +np.sum([self.nu_ab(label, other.label) for other in species])

        # ----------------------------------------------
        # Boundary Condition
        # axis
        u0, v0, w0 = 0,  1, 0

        if bc.energy.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.energy.value[0]
        else:
            raise NotImplementedError(bc.energy)

        return path+["temperature"], (a, b, c, d, e, S), (u0, v0, w0,  u1, v1, w1)

    def rotation_transport(self,
                           core_profiles_next: CoreProfiles.Profiles1D,
                           core_profiles_prev: CoreProfiles.Profiles1D,
                           transp:  CoreTransport.Model.Profiles1D,
                           source: CoreSources.Source.Profiles1D,
                           bc:  TransportSolver.BoundaryConditions1D,
                           **kwargs):
        r"""
            Rotation Transport
            .. math::  \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime\frac{5}{3}}\left\langle R\right\rangle \
                        m_{i}n_{i}u_{i,\varphi}\right)+\frac{\partial}{\partial\rho}\Phi_{i}=V^{\prime}\left[U_{i,\varphi,exp}-U_{i,\varphi}\cdot u_{i,\varphi}+U_{zi,\varphi}\right]
                :label: transport_rotation
        """
        logger.warning(f"TODO: Rotation Transport is not implemented!")
        return 0.0

    def solve_core(self,  /,

                   enable_ion_particle_solver: bool = False,
                   ion_species: Sequence = None,
                   impurities: Sequence = [],
                   tolerance=1.0e-3,
                   max_nodes=250,
                   **kwargs) -> float:
        r"""
            Solve transport equations

            Current Equation

            Args:
                core_profiles       : profiles at :math:`t-1`
                equilibrium         : Equilibrium
                transports          : CoreTransport
                sources             : CoreSources
                boundary_condition  :

            Note:
                .. math ::  \sigma_{\parallel}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho} \right) \psi= \
                            \frac{F^{2}}{\mu_{0}B_{0}\rho}\frac{\partial}{\partial\rho}\left[\frac{V^{\prime}}{4\pi^{2}}\left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle \
                            \frac{1}{F}\frac{\partial\psi}{\partial\rho}\right]-\frac{V^{\prime}}{2\pi\rho}\left(j_{ni,exp}+j_{ni,imp}\psi\right)
                    :label: transport_current


                if :math:`\psi` is not solved, then

                ..  math ::  \psi =\int_{0}^{\rho}\frac{2\pi B_{0}}{q}\rho d\rho

            Particle Transport
            Note:

                .. math::
                    \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                    \left(V^{\prime}n_{s}\right)+\frac{\partial}{\partial\rho}\Gamma_{s}=\
                    V^{\prime}\left(S_{s,exp}-S_{s,imp}\cdot n_{s}\right)
                    :label: particle_density_transport

                .. math::
                    \Gamma_{s}\equiv-D_{s}\cdot\frac{\partial n_{s}}{\partial\rho}+v_{s}^{pinch}\cdot n_{s}
                    :label: particle_density_gamma

            Heat transport equations

            Note:

                ion

                .. math:: \frac{3}{2}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                            \left(n_{i}T_{i}V^{\prime\frac{5}{3}}\right)+V^{\prime\frac{2}{3}}\frac{\partial}{\partial\rho}\left(q_{i}+T_{i}\gamma_{i}\right)=\
                            V^{\prime\frac{5}{3}}\left[Q_{i,exp}-Q_{i,imp}\cdot T_{i}+Q_{ei}+Q_{zi}+Q_{\gamma i}\right]
                    :label: transport_ion_temperature

                electron

                .. math:: \frac{3}{2}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                            \left(n_{e}T_{e}V^{\prime\frac{5}{3}}\right)+V^{\prime\frac{2}{3}}\frac{\partial}{\partial\rho}\left(q_{e}+T_{e}\gamma_{e}\right)=
                            V^{\prime\frac{5}{3}}\left[Q_{e,exp}-Q_{e,imp}\cdot T_{e}+Q_{ei}-Q_{\gamma i}\right]
                    :label: transport_electron_temperature
        """

        if ion_species is None:
            ion_species_list = [["ion", {"label": ion.label}]
                                for ion in self._core_profiles_prev.ion if ion.label not in impurities]
        elif isinstance(ion_species, collections.abc.Sequence):
            ion_species_list = [["ion", {"label": label}] for label in ion_species if label not in impurities]

        impurities_list = [["ion", {"label": label}] for label in impurities]

        if self._core_profiles_next.get("psi", None) is None:
            self._core_profiles_next["psi"] = self._core_profiles_next.grid.psi

        if enable_ion_particle_solver is False:
            equations = [
                self.current_transport(self._c_transp,  self._c_source, self.boundary_conditions_1d.current),
                self.particle_transport(["electrons"], self._c_transp,  self._c_source, self.boundary_conditions_1d),
                # self.energy_transport(["electrons"], self._c_transp,  self._c_source, self.boundary_conditions_1d),
                # *[self.energy_transport(["ion", {"label": label}], self._c_transp,  self._c_source,
                #                         self.boundary_conditions_1d) for label in ion_species_list],
            ]

        if False:
            for rpath in ["electrons"]+ion_species_list:
                residual += self.energy_transport(
                    self._core_profiles_next.fetch(rpath),
                    self._core_profiles_prev.fetch(rpath),
                    self._c_transp.fetch(rpath),
                    self._c_source.fetch(rpath),
                    self.boundary_conditions_1d.fetch(rpath),
                    tolerance=tolerance,
                    max_nodes=max_nodes,
                    verbose=verbose,
                    **kwargs
                )
                count += 1

        if False:
            residual += self.rotation_transport(
                self._core_profiles_next,
                self._core_profiles_prev,
                self._c_transp.momentum,
                self._c_source.momentum_tor,
                self.boundary_conditions_1d.momentum_tor,
                **kwargs)
            count += 1

        x = self._rho_tor_norm

        Y = np.vstack([array_like(self._rho_tor_norm, self._core_profiles_next.fetch(path, None))
                       for path, *_ in equations])
        logger.debug(Y.shape)

        Gamma = np.zeros_like(Y)

        # Gamma = np.vstack([gamma(idx, x, Y, Gamma) for idx, (path, func, bc, gamma) in enumerate(equations)])

        if self._tau > 0:
            Ym = np.vstack([array_like(self._rho_tor_norm, self._core_profiles_prev.fetch(path, None))
                            for path, *_ in equations])
        else:
            Ym = [None]*len(equations)

        def func(x, Y, p=None, /, Ym: np.ndarray = Ym, h: float = self._tau, equations=equations):
            num = int(len(Y)/2)
            if num*2 != len(Y):
                raise RuntimeError(f"The number of 'y'={len(Y)} should be an even number.")

            y = Y[: num]
            g = Y[num:]

            d = [func(idx, x, y, g, Ym, h) for idx, (path, func, *_) in enumerate(equations)]

            return np.vstack([array_like(x, d[i][0]) for i in range(num)]+[array_like(x, d[i][1]) for i in range(num)])

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, /, equations=equations):
            r"""
                u*y(b)+v*\Gamma(b)=w
                Ya=[y(a),y'(a)]
                Yb=[y(b),y'(b)]
                P=[[u(a),u(b)],[v(a),v(b)],[w(a),w(b)]]
            """

            num = int(min(len(Ya)/2, len(Yb)/2))

            d = np.asarray([bc(Ya[idx], Ya[num+idx], Yb[idx], Yb[num+idx])
                            for idx, (path, func, bc, *_) in enumerate(equations)])
            return np.hstack([d[:, 0], d[:, 1]])

        sol = solve_bvp(func, bc_func, x, np.vstack([Y, Gamma]), tolerance=tolerance, max_nodes=max_nodes, **kwargs)

        rms_residuals = np.max(sol.rms_residuals)

        eq_list = []

        for idx, (path, * _) in enumerate(equations):
            self._core_profiles_next[path] = Function(sol.x, sol.y[idx])
            eq_list.append(path)
        self._core_profiles_next["rms_residuals"] = Function((sol.x[: -1]+sol.x[1:])*0.5, sol.rms_residuals)

        logger.info(
            f"Solve transport equations : {eq_list}\t [{'Success' if sol.success else 'Failed'}] max reduisal={rms_residuals}")

        return rms_residuals


__SP_EXPORT__ = TransportSolverBVP2
