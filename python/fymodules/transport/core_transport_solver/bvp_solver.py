"""

"""

import collections
import enum
from math import log
from typing import Mapping, Optional, Sequence, Type, Union

from fytok.common.Misc import Identifier
from fytok.IDS import IDS
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreSources import CoreSources
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.CoreTransportSolver import CoreTransportSolver
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from fytok.numlib.bvp import solve_bvp
from scipy import constants
from spdm.data import Dict, Function, List, sp_property
from spdm.logger import logger
from spdm.tags import _not_found_
from spdm.util.utilities import convert_to_named_tuple

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class TransportSolverBVP(CoreTransportSolver):
    r"""
        Solve transport equations :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
        See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)

        self._tau = self._core_profiles.time - self._core_profiles.previous_state.time

        self._core_profiles_next = self._core_profiles.profiles_1d
        self._core_profiles_prev = self._core_profiles.previous_state.profiles_1d
        self._c_transp = self._core_transport.model.combine.profiles_1d
        self._c_source = self._core_sources.source.combine.profiles_1d
        self._eq = self._equilibrium.profiles_1d

        self._inv_tau = 0 if abs(self._tau) < EPSILON else 1.0/self._tau

        # $R_0$ characteristic major radius of the device   [m]
        self._R0 = self._equilibrium.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        self._B0 = self._equilibrium.vacuum_toroidal_field.b0

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
                              self._equilibrium.previous_state.profiles_1d.dvolume_drho_tor(self._psi_norm))

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

    def solve_general_form(self, x0, y0, flux0,   coeff,  bc,  hyper_diff=[0.0, 0.0],  **kwargs):
        r"""
            solve standard form

            Args:
                x      : :math:`\rho_{tor,norm}`
                y0     : :math:`y(t-1)`
                yp0    : :math:`y^{\prime}(t-1)=\left. {\partial Y}/{\partial x}\right|_{t-1}`
                inv_tau    : :math:`1/\tau`  inverse of time step
                coeff  : coefficients for `bvp` solver, :math:`a,b,c,d,e,f,g`,
                bc     : boundary condition ,  :math:`u,v,w`

            Returns:
                result of scipy.integrate.solve_bvp

            Note:
                Generalized form of transport equations:

            .. math::
                \frac{a\left(x\right)\cdot Y\left(x,t\right)-b\left(x\right)\cdot Y\left(x,t-1\right)}{\tau}+\
                \frac{1}{c\left(x\right)}\frac{\partial}{\partial x}\Gamma\left(x,t\right)=f\left(x\right)-g\left(x\right)\cdot Y\left(x,t\right)
                :label: generalized_trans_eq

            .. math::
                \Gamma\left(x,t\right)\equiv-d\left(x\right)\cdot\frac{\partial Y\left(x,t\right)}{\partial\rho}+e\left(x\right)\cdot Y\left(x,t\right)
                :label: generalized_trans_eq_gamma

            where   :math:`Y` is the function, :math:`t` is time , :math:`x` is the radial coordinate.

            The boundary conditions are given by

            .. math::
                    u\left(x_{bnd}\right)\cdot Y\left(x_{bnd},t\right)+v\left(x_{bnd}\right)\cdot\Gamma\left(x_{bnd},t\right)=w\left(x_{bnd}\right)
                    :label: generalized_trans_eq_gamma

            These equations were rewriten as a first-order ODE group

            .. math::
                \begin{cases}
                    \frac{\partial Y\left(x,t\right)}{\partial\rho} & =  -\frac{\Gamma\left(x,t\right)-e\left(x\right)\cdot Y\left(x,t\right)}{d\left(x\right)}\\
                    \frac{\partial\Gamma\left(x,t\right)}{\partial x} & =    c\left(x\right)\left[f\left(x\right)-g\left(x\right)\cdot Y\left(x,t\right)-\
                                \frac{a\left(x\right)\cdot Y\left(x,t\right)-b\left(x\right)\cdot Y\left(x,t-1\right)}{\tau}\right]
                \end{cases}
                :label: generalized_trans_eq_first_order
        """
        a, b, c, d, e, f, g = coeff

        if not isinstance(x0, np.ndarray):
            raise TypeError(type(x0))

        if isinstance(y0, Function):
            y0_func = y0
            y0 = y0_func(x0)
        elif isinstance(y0, np.ndarray) and y0.shape == x0.shape:
            y0_func = Function(x0, y0)
        else:
            raise ValueError(f"{y0.shape} != {x0.shape} ")

        if flux0 is None:
            flux0 = -d(x0) * y0_func.derivative(x0) + e(x0) * y0_func(x0)
        elif isinstance(flux0, Function):
            pass
        elif not isinstance(flux0, np.ndarray) or flux0.shape != x0.shape:
            raise ValueError(f"{flux0.shape} != {x0.shape} ")

        if hyper_diff is not None:
            hyper_diff_exp, hyper_diff_imp = hyper_diff
            hyper_diff = hyper_diff_exp+hyper_diff_imp*max(d)

        def S(x, y):
            return (f(x) - g(x) * y - (a(x) * y - b(x) * y0_func(x)))*c(x)

        def fun(x, Y):
            y, gamma = Y
            yp = Function(x, y).derivative(x)
            if np.isclose(x[0], 0.0):
                yp[0] = 0.0
            dy = (-gamma + e(x)*y + hyper_diff * yp)/(d(x)+hyper_diff)
            dgamma = (f(x) - g(x) * y - (a(x) * y - b(x) * y0_func(x)))*c(x)
            dy[0] = dy[1]
            return np.vstack((dy, dgamma))

        u0, v0, w0 = bc[0]
        u1, v1, w1 = bc[1]

        def bc_func(Ya, Yb):
            r"""
                u*y(b)+v*\Gamma(b)=w
                Ya=[y(a),y'(a)]
                Yb=[y(b),y'(b)]
                P=[[u(a),u(b)],[v(a),v(b)],[w(a),w(b)]]
            """

            ya, gammaa = Ya
            yb, gammab = Yb

            return (u0 * ya + v0 * gammaa - w0,
                    u1 * yb + v1 * gammab - w1)

        sol = solve_bvp(fun, bc_func, x0, np.vstack([y0, flux0]),  **kwargs)

        # s_exp_flux = Function(sol.x,  Function(sol.x,  S(sol.x, sol.y[0])).antiderivative(sol.x))
        diff_flux = Function(sol.x, np.asarray(-d(sol.x) * sol.yp[0]))
        conv_flux = Function(sol.x, np.asarray(e(sol.x) * sol.y[0]))

        y = Function(sol.x, sol.y[0])
        yp = Function(sol.x, sol.yp[0])
        profiles = convert_to_named_tuple({
            "diff_flux": diff_flux,
            "conv_flux": conv_flux,
            # "s_exp_flux": s_exp_flux,
            "y": y,
            "yp": yp,
            "flux":  Function(sol.x, sol.y[1]),
        })

        if not sol.success:
            logger.warning(sol.message)

        return sol, profiles

    def current_transport(self,
                          core_profiles_next: CoreProfiles.Profiles1D,
                          core_profiles_prev: CoreProfiles.Profiles1D,
                          transp: CoreTransport.Model.Profiles1D,
                          source: CoreSources.Source.Profiles1D,
                          bc: CoreTransportSolver.BoundaryConditions1D.BoundaryConditions,
                          **kwargs):

        # -----------------------------------------------------------------
        # Transport
        # plasma parallel conductivity,                                 [(Ohm*m)^-1]
        conductivity_parallel = transp.conductivity_parallel

        # -----------------------------------------------------------
        # Sources
        j_exp = source.j_parallel

        # j_decomp = source.get("j_decomposed", None)
        # if j_decomp is not None:
        #     j_exp = j_exp + j_decomp.get("explicit_part", 0)
        #     j_imp = j_decomp.get("implicit_part", 0)
        # else:
        #     j_imp = 0.0
        j_imp = 0.0

        a = conductivity_parallel * self._rho_tor * self._inv_tau

        b = conductivity_parallel * self._rho_tor * self._inv_tau

        c = (constants.mu_0 * self._B0 * self._rho_tor_boundary)/(self._fpol**2)

        d = self._vpr*self._gm2 / self._fpol / (self._rho_tor_boundary)/(TWOPI)  # FIXME: should be TWOPI **2

        e = (- constants.mu_0 * self._B0 * self._k_phi) * (conductivity_parallel * self._rho_tor**2/self._fpol**2)

        f = - self._vpr * j_exp / TWOPI

        g = - self._vpr * j_imp    # + ....

        # -----------------------------------------------------------
        # boundary condition, value
        # axis
        u0, v0, w0 = 0, 1, 0.0

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

        sol, profiles = self.solve_general_form(
            self._rho_tor_norm,
            core_profiles_next.get("psi", self._core_profiles_next.grid.psi),
            core_profiles_next.get("dpsi_drho_tor_norm", None),
            (a, b, c, d, e, f, g),
            ((u0, v0, w0), (u1, v1, w1)),
            hyper_diff=[0, 0.0001],
            **kwargs)

        rms_residuals = np.max(sol.rms_residuals)

        logger.info(
            f"Solve transport equations: Current  \t [{'Success' if sol.success else 'Failed'}] max reduisal={rms_residuals}")

        core_profiles_next["psi"] = profiles.y
        core_profiles_next["psi_error"] = Function((sol.x[1:]+sol.x[:-1])*0.5, sol.rms_residuals)
        core_profiles_next["dpsi_drho_tor_norm"] = profiles.yp

        return rms_residuals

    def particle_transport(self,
                           core_profiles_next: Union[CoreProfiles.Profiles1D.Electrons, CoreProfiles.Profiles1D.Ion],
                           core_profiles_prev: Union[CoreProfiles.Profiles1D.Electrons, CoreProfiles.Profiles1D.Ion],
                           transp: Union[CoreTransport.Model.Profiles1D.Electrons, CoreTransport.Model.Profiles1D.Ion],
                           source: Union[CoreSources.Source.Profiles1D.Electrons, CoreSources.Source.Profiles1D.Ion],
                           bc: Union[CoreTransportSolver.BoundaryConditions1D.Electrons,
                                     CoreTransportSolver.BoundaryConditions1D.Ion],
                           ** kwargs
                           ):
        # Particle Transport

        diff = transp.particles.d
        conv = transp.particles.v

        se_exp = source.particles  # + source.particles_decomposed.explicit_part
        se_imp = 0  # source.particles_decomposed.implicit_part

        a = self._vpr * self._inv_tau
        b = self._vprm * self._inv_tau
        c = Function(None, self._rho_tor_boundary)
        d = self._vpr * self._gm3 * diff / self._rho_tor_boundary
        e = self._vpr * self._gm3 * conv - self._vpr * self._rho_tor * self._k_phi
        f = self._vpr * se_exp
        g = self._vpr * (se_imp + self._k_rho_bdry)

        # -----------------------------------------------------------
        # boundary condition, value
        # axis
        u0, v0, w0 = 0, 1, 0.0

        if bc.particles.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.particles.value[0]
        else:
            raise NotImplementedError(bc.particles.identifier)

        sol, profiles = self.solve_general_form(
            self._rho_tor_norm,
            core_profiles_prev.density,
            core_profiles_prev.get("density_flux", None),
            (a, b, c, d, e, f, g),
            ((u0, v0, w0), (u1, v1, w1)),
            hyper_diff=[0, 0.0001],
            **kwargs
        )

        core_profiles_next["density"] = profiles.y
        core_profiles_next["density_error"] = Function((sol.x[1:]+sol.x[:-1])*0.5, sol.rms_residuals)
        core_profiles_next["density_flux"] = profiles.flux

        core_profiles_next["diff_flux"] = profiles.diff_flux
        core_profiles_next["conv_flux"] = profiles.conv_flux

        rms_residuals = np.max(sol.rms_residuals)

        logger.info(
            f"Solve transport equations: Particle '{core_profiles_next.label}' \t [{'Success' if sol.success else 'Failed'}] max reduisal={rms_residuals}")

        return rms_residuals

    def energy_transport(self,
                         core_profiles_next: Union[CoreProfiles.Profiles1D.Electrons, CoreProfiles.Profiles1D.Ion],
                         core_profiles_prev: Union[CoreProfiles.Profiles1D.Electrons, CoreProfiles.Profiles1D.Ion],
                         transp: Union[CoreTransport.Model.Profiles1D.Electrons, CoreTransport.Model.Profiles1D.Ion],
                         source: Union[CoreSources.Source.Profiles1D.Electrons, CoreSources.Source.Profiles1D.Ion],
                         bc: Union[CoreTransportSolver.BoundaryConditions1D.Electrons,
                                   CoreTransportSolver.BoundaryConditions1D.Ion],
                         **kwargs
                         ):

        # energy transport
        chi = transp.energy.d
        v_pinch = transp.energy.v

        Qs_exp = source.energy  # + source.energy_decomposed.explicit_part
        Qs_imp = 0  # source.energy_decomposed.implicit_part

        density_prev = core_profiles_prev.density

        density_next = core_profiles_next.density

        gamma_s = 3/2 * core_profiles_next.get("density_flux", 0)

        a = self._inv_tau * (3/2) * self._vpr35 * density_next

        b = self._inv_tau * (3/2) * self._vpr35m * density_prev

        c = self._rho_tor_boundary * self._inv_vpr23

        d = self._vpr * self._gm3 * density_next * chi / self._rho_tor_boundary

        e = self._vpr * self._gm3 * density_next * v_pinch + gamma_s  # \
        # - self._vpr * (3/4)*self._k_phi * self._rho_tor * density_next

        f = self._vpr35 * (Qs_exp)
        # + np.sum([self.nu_ab(label, other.label)*other.temperature for other in species])

        g = self._vpr35 * (Qs_imp + self._Qimp_k_ns*density_next)
        # +np.sum([self.nu_ab(label, other.label) for other in species])

        # ----------------------------------------------
        # Boundary Condition
        # axis
        u0, v0, w0 = e[0], -1, 0

        if bc.energy.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.energy.value[0]
        else:
            raise NotImplementedError(bc.energy)

        sol, profiles = self.solve_general_form(
            self._rho_tor_norm,
            core_profiles_prev.temperature,
            core_profiles_prev.get("heat_flux", None),
            (a, b, c, d, e, f, g),
            ((u0, v0, w0), (u1, v1, w1)),
            hyper_diff=[1e-4, 0.0001],
            **kwargs
        )

        core_profiles_next["temperature"] = profiles.y
        core_profiles_next["heat_flux"] = profiles.flux
        core_profiles_next["temperature_error"] = Function((sol.x[1:]+sol.x[:-1])*0.5, sol.rms_residuals)

        rms_residuals = np.max(sol.rms_residuals)

        logger.info(
            f"Solve transport equations: Energy '{core_profiles_next.label}' \t [{'Success' if sol.success else 'Failed'}] max reduisal={rms_residuals}")

        return rms_residuals

    def rotation_transport(self,
                           core_profiles_next: CoreProfiles.Profiles1D,
                           core_profiles_prev: CoreProfiles.Profiles1D,
                           transp:  CoreTransport.Model.Profiles1D,
                           source: CoreSources.Source.Profiles1D,
                           bc:  CoreTransportSolver.BoundaryConditions1D,
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
                   tolerance=1.0e-3,
                   max_nodes=250,
                   verbose=2,
                   enable_ion_particle_solver: bool = False,
                   ion_species: Sequence = None,
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

        residual = 0
        count = 0
        residual += self.current_transport(
            self._core_profiles_next,
            self._core_profiles_prev,
            self._c_transp,
            self._c_source,
            self.boundary_conditions_1d.current,
            tolerance=tolerance,
            max_nodes=max_nodes,
            verbose=verbose,
            **kwargs
        )
        count += 1

        if enable_ion_particle_solver is False:
            residual += self.particle_transport(
                self._core_profiles_next.electrons,
                self._core_profiles_prev.electrons,
                self._c_transp.electrons,
                self._c_source.electrons,
                self.boundary_conditions_1d.electrons,
                tolerance=tolerance,
                max_nodes=max_nodes,
                verbose=verbose,
                **kwargs
            )

            count += 1

            nDT = 0
            Gamma_DT = 0

            # for ion in self._core_profiles_next.ion:
            #     if ion.label not in impurities:
            #         continue
            #     nDT -= ion.density(self._rho_tor_norm)
            #     flux = ion.get("density_flux", _not_found_)
            #     if isinstance(flux, Function):
            #         flux = flux(self._rho_tor_norm)

            #     if isinstance(flux, np.ndarray):
            #         Gamma_DT -= flux

            num = len(ion_species_list)

            for ion in self._core_profiles_next.ion:
                if ion.label not in impurities:
                    continue
                nDT -= ion.density(self._rho_tor_norm)*ion.z
                flux = ion.get("density_flux", 0)*ion.z
                if isinstance(flux, Function):
                    flux = flux(self._rho_tor_norm)

                Gamma_DT -= flux

            nDT += self._core_profiles_next.electrons.density(self._rho_tor_norm)
            Gamma_DT += self._core_profiles_next.electrons.fetch("density_flux", 0)(self._rho_tor_norm)

            nDT /= num
            Gamma_DT /= num

            for ion in self._core_profiles_next.ion:
                if ion.label in impurities:
                    continue
                ion["density"] = nDT
                ion["density_flux"] = Gamma_DT

        elif enable_ion_particle_solver is True:
            n_ele = 0.0
            n_ele_flux = 0.0
            for rpath in ion_species_list:
                count += 1
                residual += self.particle_transport(
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
                ion = self._core_profiles_next.fetch(rpath)
                n_ele = n_ele - ion.density*ion.z
                n_ele_flux = n_ele_flux - ion.get("density_flux", 0)*ion.z

            # Quasi-neutral condition
            self._core_profiles_next.electrons["density"] = n_ele
            self._core_profiles_next.electrons["density_flux"] = n_ele

        if True:
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

        return residual/count

        # while time < end_time
        #     repeat until converged
        #         calculate equilibrium
        #         begin parallel
        #             calculate source_1
        #             calculate source_2
        #             calculate source_3
        #             ···
        #             calculate source_n
        #             calculate transport_coefficients_1
        #             calculate transport_coefficients_2
        #             calculate transport_coefficients_3
        #             ···
        #             calculate transport_coefficients_n
        #         end parallel
        #
        #         begin parallel
        #             combine sources
        #             combine transport coefficients
        #         end parallel
        #
        #         calculate new profile
        #
        #
        #     update time and time_step
        # terminate


__SP_EXPORT__ = TransportSolverBVP
