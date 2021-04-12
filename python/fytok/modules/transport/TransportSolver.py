"""

"""

import collections
import copy
from functools import cached_property

import numpy as np
import scipy.constants
from fytok.util.Misc import Identifier
from spdm.numerical.Function import Function
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.data.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.utilities import try_get
from spdm.numerical.bvp import solve_bvp
from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .EdgeProfiles import EdgeProfiles
from .EdgeSources import EdgeSources
from .EdgeTransport import EdgeTransport
from .Equilibrium import Equilibrium
from ..utilities.RadialGrid import RadialGrid
EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * scipy.constants.pi


class TransportSolver(PhysicalGraph):
    r"""
        Solve transport equations :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`

        See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`

    """
    IDS = "transport_solver_numerics"

    def __init__(self,  *args, grid: RadialGrid = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid

    @property
    def radial_grid(self):
        return self._grid

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
        a, b, d, e, f, g = coeff

        if not isinstance(y0, Function):
            y0 = Function(x0, y0)

        if hyper_diff is not None:
            hyper_diff_exp, hyper_diff_imp = hyper_diff
            hyper_diff = hyper_diff_exp+hyper_diff_imp*max(d)

        def S(x, y):
            return f(x) - g(x) * y - (a(x) * y - b(x) * y0(x))

        def fun(x, Y):
            y, gamma = Y

            try:
                yp = Function(x, y).derivative

                dy = (-gamma + e(x)*y + hyper_diff * yp)/(d(x)+hyper_diff)

                dgamma = S(x, y)

            except RuntimeWarning as error:
                raise RuntimeError(error)
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

        if flux0 is None:
            flux0 = -d(x0) * y0.derivative + e(x0)*y0

        sol = solve_bvp(fun, bc_func, x0, np.vstack([y0.view(np.ndarray), flux0.view(np.ndarray)]),  **kwargs)

        s_exp_flux = Function(sol.x,  S(sol.x, sol.y[0])).antiderivative(sol.x)
        diff_flux = Function(sol.x, -d(sol.x) * sol.yp[0])
        conv_flux = Function(sol.x, e(sol.x) * sol.y[0])

        profiles = AttributeTree({
            "diff_flux": diff_flux,
            "conv_flux": conv_flux,
            "s_exp_flux": s_exp_flux,
            "residual": (diff_flux + conv_flux - s_exp_flux),

            "y": Function(sol.x, sol.y[0]),
            "yp":  Function(sol.x, sol.yp[0]),
            "flux": Function(sol.x, sol.y[1]),
            "flux_prime": Function(sol.x, sol.yp[1]),
        })

        if not sol.success:
            logger.warning(sol.message)

        return sol, profiles

    def solve(self,
              core_profiles_prev: CoreProfiles,
              *args,
              time=None,
              equilibrium: Equilibrium,
              core_transport: CoreTransport,
              core_sources: CoreSources,
              boundary_conditions: PhysicalGraph,
              enable_ion_solver: bool = False,
              tolerance=1.0e-3,
              max_nodes=1000,
              verbose=2,
              **kwargs) -> CoreProfiles:
        r"""
            Solve transport equations


            Current Equation

            Args:
                core_profiles_prev  : profiles at :math:`t-1`
                core_profiles_next　: profiles at :math:`t`
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
        if time is None:
            time = equilibrium.time

        core_profiles_next = CoreProfiles(grid=self.radial_grid, time=time)

        tau = core_profiles_next.time - core_profiles_prev.time

        inv_tau = 0 if abs(tau) < EPSILON else 1.0/tau

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_next.grid.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_next.grid.vacuum_toroidal_field.b0

        B0m = core_profiles_prev.grid.vacuum_toroidal_field.b0

        # Grid
        # $rho_tor$ not  normalized minor radius                [m]
        rho_tor = core_profiles_next.grid.rho_tor

        rho_tor_boundary = core_profiles_next.grid.rho_tor_boundary

        rho_tor_boundary_m = core_profiles_prev.grid.rho_tor_boundary

        # $rho_tor_{norm}$ normalized minor radius                [-]
        rho_tor_norm = core_profiles_next.grid.rho_tor_norm

        psi_norm = core_profiles_next.grid.psi_norm

        k_B = (B0 - B0m) / (B0 + B0m) * inv_tau * 2.0

        k_rho_bdry = (rho_tor_boundary - rho_tor_boundary_m) / (rho_tor_boundary + rho_tor_boundary_m)*inv_tau*2.0

        k_phi = k_B + k_rho_bdry

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = equilibrium.profiles_1d.fpol.pullback(psi_norm, rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = equilibrium.profiles_1d.dvolume_drho_tor.pullback(psi_norm, rho_tor_norm)

        vprm = core_profiles_prev.vprime

        if not isinstance(vprm, np.ndarray) or vprm == None:
            vprm = vpr

        # $q$ safety factor                                 [-]
        qsf = equilibrium.profiles_1d.q.pullback(psi_norm, rho_tor_norm)

        gm1 = equilibrium.profiles_1d.gm1.pullback(psi_norm, rho_tor_norm)

        gm2 = equilibrium.profiles_1d.gm2.pullback(psi_norm, rho_tor_norm)

        gm3 = equilibrium.profiles_1d.gm3.pullback(psi_norm, rho_tor_norm)

        core_profiles_next.vprime = vpr

        core_profiles_next.gm3 = gm3

        if not enable_ion_solver:
            spec = ["electrons"]
        else:
            raise NotImplementedError()

        #　Current Equation
        if True:

            # -----------------------------------------------------------------
            # Transport
            # plasma parallel conductivity,                                 [(Ohm*m)^-1]
            conductivity_parallel = core_transport.conductivity_parallel

            # -----------------------------------------------------------
            # Sources
            # total non inductive current, PSI independent component,          [A/m^2]
            j_ni_exp = core_sources.j_parallel

            # total non inductive current, component proportional to PSI,      [A/m^2/V/s]
            j_ni_imp = 0.0  # sources.j_ni_imp if sources is not None else 0.0   # can not find data in imas dd

            c = (scipy.constants.mu_0*B0 * rho_tor_boundary)/(fpol**2)

            a = conductivity_parallel * rho_tor * c * inv_tau

            b = conductivity_parallel * rho_tor * c * inv_tau

            d = vpr*gm2 / fpol / ((TWOPI**2)*rho_tor_boundary)

            e = (- scipy.constants.mu_0 * B0 * k_phi) * (conductivity_parallel * rho_tor**2/fpol**2)

            f = - vpr * j_ni_exp/TWOPI * c

            g = vpr * j_ni_imp/TWOPI * c

            # -----------------------------------------------------------
            # boundary condition, value
            # Identifier of the boundary condition type.
            #   ID =    1: poloidal flux;
            #           2: ip;
            #           3: loop voltage;
            #           4: undefined;
            #           5: generic boundary condition y expressed as a1y'+a2y=a3.
            #           6: equation not solved; [eV]

            sp_bc = boundary_conditions["current"]

            # Boundary conditions for electron diffusion equation in form:
            #      U*Y + V*Gamma =W
            # On axis:
            #     dNi/drho_tor(rho_tor=0)=0:  - this is Ne, not N

            if sp_bc.identifier.index == 1:
                u = 1
                v = 0
                w = sp_bc.value
            else:
                raise NotImplementedError(sp_bc)

            # $\Psi$ flux function from current                 [Wb]
            psi0 = core_profiles_prev.psi(psi_norm)

            # $\frac{\partial\Psi}{\partial\rho_{tor,norm}}$               [Wb/m]
            gamma0 = core_profiles_prev.dpsi_drho_tor_norm(psi_norm)

            if not isinstance(gamma0, np.ndarray):
                gamma0 = -d * Function(rho_tor_norm, psi0).derivative + e * psi0

            sol, profiles = self.solve_general_form(rho_tor_norm,
                                                    psi0,
                                                    gamma0,
                                                    (a, b,  d, e, f, g),
                                                    ((e[0], -1, 0.0), (u, v, w)),
                                                    hyper_diff=[0, 0.0001],
                                                    tolerance=tolerance,
                                                    verbose=verbose,
                                                    max_nodes=max_nodes)

            logger.info(f"Solve transport equations: Current [{'Success' if sol.success else 'Failed'}]")

            if sol.success:
                core_profiles_next.psi = profiles.y
                core_profiles_next.dpsi_drho_tor_norm = profiles.yp
                # core_profiles_next.dgamma_current = profiles.gamma
                # core_profiles_next.j_tor = (profiles.y*fpol).derivative / vpr * \
                #     (-TWOPI*R0 / scipy.constants.mu_0/rho_tor_boundary)

                # core_profiles_next.j_parallel = profiles.gamma * \
                #     (- TWOPI / (scipy.constants.mu_0 * B0 * rho_tor_boundary)) * fpol**2/vpr

                # core_profiles_next.e_field.parallel = (core_profiles_next.j_parallel-j_ni_exp -
                #                                        j_ni_exp*core_profiles_next.psi)/conductivity_parallel
            else:
                psi_prime = (scipy.constants.pi*2.0)*B0 * rho_tor / qsf * rho_tor_boundary
                core_profiles_next.dpsi_drho_tor_norm = psi_prime
                core_profiles_next.psi = psi0[0] + psi_prime.antiderivative * 2

            # core_profiles_next.f_current = f*c
            # core_profiles_next.j_total = j_ni_exp

        # Particle Transport
        for sp in spec:

            diff = core_transport[sp].particles.d
            conv = core_transport[sp].particles.v

            se_exp = core_sources[sp].particles
            se_imp = 0.0

            if core_sources[sp].particles_decomposed != None:
                se_exp += core_sources[sp].particles_decomposed.explicit_part
                se_imp += core_sources[sp].particles_decomposed.implicit_part

            a = rho_tor_boundary * vpr * inv_tau
            b = rho_tor_boundary * vprm * inv_tau
            d = vpr * gm3 * diff / rho_tor_boundary
            e = vpr * gm3 * conv - vpr * rho_tor * k_phi
            f = rho_tor_boundary * vpr * se_exp
            g = rho_tor_boundary * vpr * (se_imp + k_rho_bdry)

            sp_bc = boundary_conditions[sp]["particles"]

            # Boundary conditions for electron diffusion equation in form:
            #      U*Y + V*Gamma =W
            # On axis:
            #     dNi/drho_tor(rho_tor=0)=0:  - this is Ne, not N
            if sp_bc.identifier.index == 1:
                u = 1
                v = 0
                w = sp_bc.value
            else:
                raise NotImplementedError(sp_bc)

            ne0 = core_profiles_prev[sp].density

            if rho_tor_norm is not ne0.x:
                ne0 = Function(rho_tor_norm, ne0(rho_tor_norm))

            gamma0 = core_profiles_prev[sp].gamma

            if not isinstance(gamma0, np.ndarray):
                gamma0 = -d * ne0.derivative + e * ne0

            sol, profiles = self.solve_general_form(
                rho_tor_norm,
                ne0,
                gamma0,
                (a, b,  d, e, f, g),
                ((e[0], -1, 0), (u, v, w)),
                hyper_diff=[0, 0.0001],
                tolerance=tolerance,
                verbose=verbose,
                max_nodes=max_nodes,
                ignore_x=[d.x[np.argmax(np.abs(d.derivative))]]
            )

            logger.info(
                f"Solve transport equations: {sp.capitalize()} particle [{'Success' if sol.success else 'Failed'}] ")

            core_profiles_next[sp] = {
                "diff": diff,
                "conv": conv,

                "diff_flux": profiles.diff_flux,
                "conv_flux": profiles.conv_flux,
                "s_exp_flux": profiles.s_exp_flux,
                "residual": profiles.residual,

                "density":  profiles.y,
                "density_prime": profiles.yp,
                "density_flux": profiles.gamma,
                "density_flux_prime": profiles.gamma_prime,
                "n_rms_residuals":  profiles.rms_residuals,
            }

        # Energy Transport
        for sp in []:  # spec:

            diff = core_transport[sp].energy.d
            v_pinch = core_transport[sp].energy.v

            qs_exp = core_sources[sp].energy
            qs_imp = 0.0

            if core_sources[sp].energy_decomposed != None:
                qs_imp = core_sources[sp].energy_decomposed.implicit_part
                qs_exp = core_sources[sp].energy_decomposed.explicit_part
            # FIXME: Collisions is not implemented
            # qs_exp += qei + qzi + qgi
            # qi_imp += vei + vzi
            logger.warning(
                f"Energy Transport: Collisions is not implemented! [qs_exp += qei + qzi + qgi, qi_imp += vei + vzi] ")

            sp_prev = core_profiles_prev[sp]
            sp_next = core_profiles_next[sp]

            density_prev = sp_prev["density"]

            density_next = sp_next["density"]

            density_flux_next = sp_next["density_flux"]

            a = (3/2) * density_next * vpr * rho_tor_boundary * inv_tau

            b = (3/2) * density_prev * (vprm**(5/3)/vpr**(2/3)) * rho_tor_boundary * inv_tau

            d = vpr * gm3 * density_next * diff / rho_tor_boundary

            e = vpr * gm3 * density_next * v_pinch + density_flux_next - vpr * (3/2)*k_phi * rho_tor * density_next

            f = vpr * (qs_exp) * rho_tor_boundary

            g = vpr * (qs_imp + (3*k_rho_bdry - k_phi * vpr.derivative)*density_next) * rho_tor_boundary

            sp_bc = boundary_conditions[sp]["energy"]

            if sp_bc.identifier.index == 1:
                u = 1
                v = 0
                w = sp_bc.value
            else:
                raise NotImplementedError(sp_bc)

            Ts_prev = sp_prev.temperature
            Hs_prev = sp_prev.head_flux or None

            sol, profiles = self.solve_general_form(
                rho_tor_norm,
                Ts_prev,
                Hs_prev,
                (a, b, d, e, f, g),
                ((e[0], -1,  0), (u, v, w)),
                hyper_diff=[0, 0.0001],
                tolerance=tolerance,
                verbose=verbose,
                max_nodes=max_nodes,
                ignore_x=[d.x[np.argmax(np.abs(d.derivative))]]
                ** kwargs
            )

            logger.debug(
                f"Solve transport equations: Temperature {sp_next.label}: [{'Done' if  sol.success else 'Failed' }]")

            core_profiles_next[sp]["temperature"] = profiles.y,
            core_profiles_next[sp]["heat_flux"] = profiles.flux

        # Rotation Transport
        if False:
            r"""
                Note:

                    .. math::  \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime\frac{5}{3}}\left\langle R\right\rangle \
                                m_{i}n_{i}u_{i,\varphi}\right)+\frac{\partial}{\partial\rho}\Phi_{i}=V^{\prime}\left[U_{i,\varphi,exp}-U_{i,\varphi}\cdot u_{i,\varphi}+U_{zi,\varphi}\right]
                        :label: transport_rotation
            """
            raise NotImplementedError("Rotation")

        return core_profiles_next
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
