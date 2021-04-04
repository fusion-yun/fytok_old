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

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6


class TransportSolver(PhysicalGraph):
    r"""
        Solve transport equations :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`

        See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`

    """
    IDS = "transport_solver_numerics"

    def __init__(self,  *args,
                 equilibrium: Equilibrium = None,
                 core_transport: CoreTransport = None,
                 core_sources: CoreSources = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._equilibrium = equilibrium
        self._core_transport = core_transport
        self._core_sources = core_sources

    @property
    def equilibrium(self) -> Equilibrium:
        return self._equilibrium

    @property
    def core_transport(self):
        return self._core_transport

    @property
    def core_sources(self) -> CoreSources:
        return self._core_sources

    @cached_property
    def solver(self) -> Identifier:
        """Solver identifier """
        c = self._cache.solver
        return Identifier(
            name="FyTok",
            index=0,
            description="Default core transport solver"
        ) if c is NotImplemented or c is None or len(c) == 0 else Identifier(c)

    @cached_property
    def primary_coordinate(self) -> Identifier:
        r"""
            Primary coordinate system with which the transport equations are solved.

            For a 1D transport solver:

                name        :   Short string identifier

                index       :   Integer identifier (enumeration index within a list).
                                index = 1 means :math:`\rho_{tor,norm}=\rho_{tor}/\rho_{tor,boundary}`;  2 = :math:`\rho_{tor}=\sqrt{ \Phi/\pi B_{0}}`.
                                Private identifier values must be indicated by a negative index.

                description	:   Verbose description
        """
        c = self["primary_coordinate"]
        return Identifier(
            name="rho_tor_norm",
            index=1,
            description=""
        ) if c is NotImplemented or c is None or len(c) == 0 else Identifier(c)

    class BoundaryCondition(PhysicalGraph):
        def __init__(self,   *args, **kwargs):
            super().__init__(*args, **kwargs)

        @cached_property
        def identifier(self):
            """Identifier of the boundary condition type.

                ID=
                    - 1: value of the field y;
                    - 2: radial derivative of the field (-dy/drho_tor);
                    - 3: scale length of the field y/(-dy/drho_tor);
                    - 4: flux;
                    - 5: generic boundary condition y expressed as a1y'+a2y=a3.
                    - 6: equation not solved; [eV]

            """
            return NotImplemented

        @cached_property
        def value(self):
            """Value of the boundary condition.
                For ID = 1 to 4, only the first position in the vector is used.
                For ID = 5, all three positions are used, meaning respectively a1, a2, a3. {dynamic} [mixed]	"""
            return NotImplemented

        @cached_property
        def rho_tor_norm(self):
            """Position, in normalised toroidal flux, at which the boundary
                condition is imposed. Outside this position, the value of the data are
                considered to be prescribed. {dynamic} [-]	"""
            return NotImplemented

    @cached_property
    def boundary_conditions(self):
        """ Boundary conditions of the radial transport equations for various time slices.
            To be removed when the solver_1d structure is finalized. {dynamic}"""
        return TransportSolver.BoundaryCondition(self["boundary_conditions"])

    def solve_general_form(self, x, y0, gamma0, inv_tau, coeff,  bc,  hyper_diff=[0.0, 0.0],  **kwargs):
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

        if isinstance(hyper_diff, collections.abc.MutableSequence):
            hyper_diff_exp, hyper_diff_imp = hyper_diff
            hyper_diff = hyper_diff_exp+hyper_diff_imp*max(d)
        elif hyper_diff is None:
            hyper_diff = 0.0

        D = d
        E = e
        F = (f + b*inv_tau*y0)*c
        G = (g + a*inv_tau)*c

        def fun(x, Y):
            y, gamma = Y
            try:
                yp = Function(x, y).derivative
                dgamma = F(x) - G(x) * y
                dy = (-gamma + E(x)*y+hyper_diff * yp)/(D(x)+hyper_diff)
            except RuntimeWarning as error:
                raise RuntimeError(y)
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

        if gamma0 is None:
            gamma0 = - Function(x, y0).derivative*D(x) + y0*E(x)

        sol = solve_bvp(fun, bc_func, x, np.vstack([y0.view(np.ndarray), gamma0.view(np.ndarray)]),  **kwargs)

        y1 = Function(sol.x, sol.y[0])
        yp1 = Function(sol.x, sol.yp[0])
        s_exp_flux = Function(sol.x, f(sol.x)-g(sol.x)*y1).antiderivative(sol.x) * c
        diff_flux = Function(sol.x, -d(sol.x) * yp1)
        conv_flux = Function(sol.x, e(sol.x) * y1)

        profiles = AttributeTree({
            "diff_flux": diff_flux,
            "conv_flux": conv_flux,
            "s_exp_flux": s_exp_flux,
            "residual": (diff_flux + conv_flux - s_exp_flux),

            "density": y1,
            "density_prime": yp1,
            "gamma": Function(sol.x, sol.y[1]),
            "gamma_prime": Function(sol.x, sol.yp[1]),
        })
        return sol, profiles

    def solve(self, core_profiles_prev: CoreProfiles, bc=None,   enable_ion_solver=False, **kwargs):
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

        core_profiles_next = CoreProfiles(self._parent.radial_grid, time=self.equilibrium.time,   parent=self._parent)

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
        rho_tor_boundary = rho_tor[-1]
        rho_tor_boundary_m = core_profiles_prev.grid.rho_tor[-1]
        # $rho_tor_{norm}$ normalized minor radius                [-]
        rho_tor_norm = core_profiles_next.grid.rho_tor_norm
        psi_norm = core_profiles_next.grid.psi_norm
        k_phi = ((B0 - B0m) / (B0 + B0m) + (rho_tor_boundary - rho_tor_boundary_m) /
                 (rho_tor_boundary + rho_tor_boundary_m))*inv_tau

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = self.equilibrium.profiles_1d.fpol.pullback(psi_norm, rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = (self.equilibrium.profiles_1d.dvolume_dpsi).pullback(psi_norm, rho_tor_norm)

        vprm = core_profiles_prev.vprime

        if vprm == None:
            vprm = vpr

        gm1 = self.equilibrium.profiles_1d.gm1.pullback(psi_norm, rho_tor_norm)

        gm2 = self.equilibrium.profiles_1d.gm2.pullback(psi_norm, rho_tor_norm)

        gm3 = self.equilibrium.profiles_1d.gm3.pullback(psi_norm, rho_tor_norm)

        H = vpr * gm3

        logger.debug(vpr[:10])

        diff_hyper = 0

        if not enable_ion_solver:
            spec = ["electrons"]
        else:
            raise NotImplementedError()

        #　Current Equation
        if False:

            # $\Psi$ flux function from current                 [Wb]
            psi0 = self.equilibrium.profiles_1d.psi.pullback(psi_norm, rho_tor_norm)
            # $\frac{\partial\Psi}{\partial\rho_{tor,norm}}$               [Wb/m]
            psi0_prime = psi0.derivative
            # $q$ safety factor                                 [-]
            # equilibrium.profiles_1d.mapping("rho_tor_norm", "q")(rho_tor_norm)
            qsf = self.equilibrium.profiles_1d.q.pullback(psi_norm, rho_tor_norm)

            # plasma parallel conductivity,                     [(Ohm*m)^-1]
            conductivity_parallel = Function(rho_tor_norm, 0.0, description={"name": "conductivity_parallel"})

            for trans in self.core_transport:
                sigma = trans.profiles_1d["conductivity_parallel"]
                if isinstance(sigma, np.ndarray):
                    conductivity_parallel += sigma

            # -----------------------------------------------------------
            # Sources
            # total non inductive current, PSI independent component,          [A/m^2]
            j_ni_exp = Function(rho_tor_norm, 0.0, description={"name": "j_ni_exp"})
            for src in self.core_sources:
                j = src.j_parallel
                if isinstance(j, np.ndarray):
                    j_ni_exp += j

            # total non inductive current, component proportional to PSI,      [A/m^2/V/s]
            j_ni_imp = 0.0  # sources.j_ni_imp if sources is not None else 0.0   # can not find data in imas dd

            # for src in sources(equilibrium):
            #     j_ni_exp += src.profiles_1d.j_parallel

            a = conductivity_parallel * rho_tor

            b = conductivity_parallel * rho_tor

            c = (scipy.constants.mu_0*B0 * rho_tor_boundary)/(fpol**2)

            d = vpr*gm2 / fpol / (4.0*(scipy.constants.pi**2)*rho_tor_boundary)

            e = (- scipy.constants.mu_0 * B0 * k_phi) * (conductivity_parallel * rho_tor**2/fpol**2)

            f = - vpr * j_ni_exp/(2.0 * scipy.constants.pi)

            g = vpr * j_ni_imp/(2.0 * scipy.constants.pi)

            # + rho_tor* conductivity_parallel*k_phi*(2.0-2.0*rho_tor *  fprime/fpol + rho_tor * conductivity_parallel_prime/conductivity_parallel)

            # Boundary conditions for current diffusion equation in form:
            #     u*Y + v*Y' = w

            # -----------------------------------------------------------
            # boundary condition, value
            # Identifier of the boundary condition type.
            #   ID =    1: poloidal flux;
            #           2: ip;
            #           3: loop voltage;
            #           4: undefined;
            #           5: generic boundary condition y expressed as a1y'+a2y=a3.
            #           6: equation not solved; [eV]

            # At the edge:
            # if boundary_condition.identifier.index == 1:  # poloidal flux
            #     u = 1.0
            #     v = 0.0
            #     w = boundary_condition.value[0]
            # elif boundary_condition.identifier.index == 2:  # ip
            #     u = 0.0
            #     v = 1.0
            #     w =   scipy.constants.mu_0 * boundary_condition.value[0]/fpol[-1]
            # elif boundary_condition.identifier.index == 3:  # loop voltage
            #     u = 1.0
            #     v = 0.0
            #     w = tau*boundary_condition.value[0]+psi0[-1]
            # elif boundary_condition.identifier.index == 5:  # generic boundary condition  y expressed as a1y'+a2y=a3.
            #     v, u, w = boundary_condition.value
            # else:
            #     # Current equation is not solved:
            #     #  Interpretative value of safety factor should be given
            #     # if any(qsf != 0.0):  # FIXME
            #     # dy = 2.0* scipy.constants.pi*B0*rho_tor/qsf

            #     # a[-1] = 1.0*inv_tau
            #     # b[-1] = 1.0*inv_tau
            #     # # c[-1] = 1.0
            #     # d[-1] = 0.0
            #     # e[-1] = 0.0
            #     # f[-1] = 0.0

            #     u = 1.0
            #     v = 0.0
            #     w =  - scipy.constants.mu_0 * boundary_condition.value[0]/fpol[-1]

            # Ip = sum(j_ni_exp)
            # psi0_prime[-1]*d[-1]/ scipy.constants.mu_0 * fpol[-1]

            sol = self.solve_general_form(rho_tor_norm,
                                          psi0, psi0_prime,
                                          inv_tau,
                                          (a, b, c, d, e, f, g),
                                          ((0, 1, 0.0),
                                           (1, 0, psi0[-1])),  # 　Ip  - scipy.constants.mu_0 * Ip/fpol[-1]
                                          verbose=2,  max_nodes=2500
                                          )
            logger.info(
                f"Solve transport equations: Current : {'Done' if  sol.success else 'Failed' }  \n Message: {sol.message} ")

            # core_profiles_next.profiles_1d.psi0 = psi0
            # core_profiles_next.profiles_1d.psi0_prime = psi0_prime
            # core_profiles_next.profiles_1d.psi0_eq = Function(
            #     equilibrium.profiles_1d.psi, axis=equilibrium.profiles_1d.rho_tor_norm)
            # core_profiles_next.profiles_1d.psi0_prime_eq = Function(
            #     equilibrium.profiles_1d.dpsi_drho_tor*rho_tor_boundary, axis=equilibrium.profiles_1d.rho_tor_norm)

            # core_profiles_next.profiles_1d.q0 = Function(
            #     equilibrium.profiles_1d.q, axis=equilibrium.profiles_1d.rho_tor_norm)

            # j_total0 = (d * psi0_prime).derivative / c / vpr*(2.0 * scipy.constants.pi)

            # # j_total0[0] = 2*j_total0[1]-j_total0[2]
            # core_profiles_next.profiles_1d.j_total0 = j_total0
            # core_profiles_next.profiles_1d.j_ni_exp = j_ni_exp

            # core_profiles_next.profiles_1d.rho_star = vpr/(4.0*(scipy.constants.pi**2)*R0)

            # core_profiles_next.profiles_1d.rho_tor = rho_tor_norm*equilibrium.profiles_1d.rho_tor[-1]

            if sol.success:
                core_profiles_next.psi = Function(sol.x, sol.y[0])
                core_profiles_next.dpsi_drho_tor = Function(sol.x, sol.yp[0])
                core_profiles_next.dgamma_current = Function(sol.x, sol.yp[1])

                q = (scipy.constants.pi*2.0)*B0 * rho_tor_boundary**2 * sol.x
                # q[1:] /= sol.yp[0][1:]
                # q[0] = 2*q[1]-q[2]

                core_profiles_next.q = Function(sol.x, q)

                core_profiles_next.j_tor = Function(sol.x, sol.y[1]*fpol(sol.x)).derivative / vpr(sol.x)\
                    * (-2.0 * scipy.constants.pi*R0 / scipy.constants.mu_0/rho_tor_boundary)

                # d = sol.yp[1] * (- scipy.constants.pi*2.0 / (scipy.constants.mu_0 *
                #                                              B0 * rho_tor_boundary)) * fpol(sol.x)**2/vpr(sol.x)

                # core_profiles_next.j_parallel = Function(sol.x, d)

                # core_profiles_next.e_field.parallel = (j_parallel-j_ni_exp -
                #                                                    j_ni_exp*core_profiles_next.psi)/conductivity_parallel

            else:
                psi_prime = Function(rho_tor_norm,  (scipy.constants.pi*2.0)*B0 * rho_tor / qsf * rho_tor_boundary)
                core_profiles_next.psi_prime = psi_prime
                core_profiles_next.psi = psi0[0] + psi_prime.antiderivative * 2

            core_profiles_next.f_current = f*c

            core_profiles_next.j_total = j_ni_exp

            # core_profiles_next.vpr = vpr
            # core_profiles_next.gm2 = gm2

            # core_profiles_next.a = a(sol.x)
            # core_profiles_next.b = b(sol.x)
            # # core_profiles_next.c =(sol.x)
            # core_profiles_next.d = d(sol.x)
            # core_profiles_next.e = e(sol.x)
            # core_profiles_next.f = f(sol.x)
            # core_profiles_next.g = g(sol.x)
            # core_profiles_next.dvolume_dpsi = Function(equilibrium.dvolume_dpsi,
            #                                                       axis=equilibrium.rho_tor_norm)
            # core_profiles_next.dpsi_drho_tor = Function(equilibrium.dpsi_drho_tor,
            #                                                        axis=equilibrium.rho_tor_norm)

            # core_profiles_next.q = 2.0* scipy.constants.pi*B0*rho_tor/psi_prime1

            # current density, toroidal,                        [A/m^2]
            # j_tor = - 2.0* scipy.constants.pi*R0/ scipy.constants.mu_0/vpr * dfun4
            # core_profiles_next.j_tor = j_tor
            # j_par = - 2.0* scipy.constants.pi/R0/ scipy.constants.mu_0/vpr * (fpol/B0)**2*dfun5
            # core_profiles_next.current_parallel_inside = j_par

            # # $E_\parallel$  parallel electric field,,          [V/m]
            # core_profiles_next.e_field.parallel = (j_par - j_ni_exp - j_ni_imp*psi1) / conductivity_parallel

            # # Total Ohmic currents                              [A]
            # # fun7 = vpr * j_par / 2.0 /  scipy.constants.pi * B0 / fpol**2
            # core_profiles_next.j_ohmic = fpol[-1]*core_profiles_next.integral(
            #     vpr * j_par / 2.0 /  scipy.constants.pi * B0 / fpol**2, 0, 1)

            # core_profiles_next.j_tor = j_tor

            # # Total non-inductive currents                       [A]
            # core_profiles_next.j_non_inductive = fpol[-1] * core_profiles_next.integral(
            #     vpr * (j_ni_exp + j_ni_imp * psi1) / (2.0 *  scipy.constants.pi) * B0 / fpol**2)

        diff_hyper = 1000
        # Particle Transport
        for sp in spec:
            diff = np.zeros(rho_tor_norm.shape)
            conv = np.zeros(rho_tor_norm.shape)

            for trans in self.core_transport:
                D = trans[sp].particles.d

                if isinstance(D, np.ndarray) or isinstance(D, (int, float)):
                    diff += D

                v = trans[sp].particles.v
                if isinstance(v, np.ndarray) or isinstance(v, (int, float)):
                    conv += v

            se_exp = np.zeros(rho_tor_norm.shape)
            se_imp = np.zeros(rho_tor_norm.shape)

            for src in self.core_sources:
                si = src[sp].particles_decomposed.implicit_part
                if isinstance(si, np.ndarray) or type(si) in (int, float):
                    se_imp += si

                se = src[sp].particles_decomposed.explicit_part
                if isinstance(se, np.ndarray) or type(se) in (int, float):
                    se_exp += se

                se = src[sp].particles
                if isinstance(se, np.ndarray) or type(se) in (int, float):
                    se_exp += se

            ne0 = try_get(core_profiles_prev, sp).density

            if not isinstance(ne0, np.ndarray) and ne0 == None:
                ne0 = np.zeros(rho_tor_norm.shape)

            a = vpr
            b = vprm
            c = rho_tor_boundary
            d = H * diff / c
            e = H * conv - k_phi*rho_tor*vpr
            f = vpr * se_exp
            g = vpr * se_imp

            """
                # Boundary conditions for electron diffusion equation in form:
                #     V*Y' + U*Y =W
                # On axis:
                #     dNi/drho_tor(rho_tor=0)=0:  - this is Ne, not N
                if(solver_type != 4):
                    v[0] = 1.
                    u[0] = 0.
                else:
                    IF (DIFF[0]>1.0E-6) :
                if((diff[0]+diff_hyper) > 1.0e-6):
                    #         V[0] = -DIFF[0]
                    v = -diff[0]-diff_hyper
                else:
                    v = -1.0e-
                #       U[0] = VCONV[0]
                    u = conv[0]+diff_hyper*ne0_prime[0]/ne0[0]
                w = 0
                At the edge:
                    FIXED Ne
                if(boundary_condition.type == 1):
                    v = 0.
                    u = 1.
                    w = boundary_condition[1, 0]
                #       FIXED grad_Ne
                elif(boundary_condition.type == 2):
                    v = 1.
                    u = 0.
                    w = -boundary_condition[1, 0
                #       FIXED L_Ne
                elif(boundary_condition.type == 3):
                    v = boundary_condition[1, 0]
                    u = 1.
                    w = 0
                #       FIXED Flux_Ne
                elif(boundary_condition.type == 4):
                    v = -vpr[-1]*gm3[-1]*diff[-1]
                    u = vpr[-1]*gm3[-1]*conv[-1]
                    w = boundary_condition[1, 0
                #       Generic boundary condition
                elif(boundary_condition.type == 5):
                    v = boundary_condition[1, 0]
                    u = boundary_condition(2, 2)
                    w = boundary_condition(2, 3
                # Density equation is not solved:
                else:  # if(boundary_condition.type == 0):


                v = 0.0
                u = 1.0
                w = ne0[-1]
            """

            sol, profiles = self.solve_general_form(rho_tor_norm,
                                                    ne0,
                                                    None,
                                                    inv_tau,
                                                    (a, b, c, d, e, f, g),
                                                    ((-e[0], 1, 0), (1, 0, 4.9e19)),
                                                    hyper_diff=[0.0, 0.0],
                                                    tol=1e-6,
                                                    verbose=2,
                                                    max_nodes=1000,
                                                    ignore_x=[np.sqrt(0.88)]
                                                    )

            logger.info(f"""Solve transport equations: Electron density: { 'Success' if sol.success else 'Failed' } """)

            if not sol.success:
                logger.debug(sol.message)

            core_profiles_next[sp] = {
                "n_diff_flux": profiles.diff_flux,
                "n_conv_flux": profiles.conv_flux,
                "n_s_exp_flux": profiles.s_exp_flux,
                "n_residual": profiles.residual,
                "n_diff": Function(rho_tor_norm, diff),
                "n_conv": Function(rho_tor_norm, conv),
                "density":  profiles.density,
                "density_prime": profiles.density_prime,
                "n_gamma": profiles.gamma,
                "n_gamma_prime": profiles.gamma_prime,
                "n_rms_residuals":  profiles.rms_residuals,
                "vpr": vpr
            }

            # else:
            # diff_flux = -d * ne0_prime  # * H*diff/rho_tor_boundary
            # ne0 = Function(sol.x, sol.y[0])
            # ne0_prime = Function(sol.x, sol.yp[0])
            # s_exp_flux = Function(sol.x, f(sol.x)-g(sol.x)*ne0).antiderivative(sol.x) * c
            # diff_flux = Function(sol.x, -d(sol.x) * ne0_prime)  # * H * conv
            # conv_flux = Function(sol.x, e(sol.x) * ne0)  # * H * conv

            # {
            #     # "density_prime": -d * ne0_prime,  # * H*diff/rho_tor_boundary
            #     "diff_flux": diff_flux,
            #     "conv_flux": conv_flux,
            #     "s_exp_flux": s_exp_flux,
            #     "residual": (diff_flux + conv_flux - s_exp_flux),
            #     "diff": Function(rho_tor_norm, diff),
            #     "conv": Function(rho_tor_norm, conv),
            #     "density": ne0,
            #     "density_prime": ne0_prime,
            #     "gamma": Function(sol.x, sol.y[1]),
            #     "gamma_prime": Function(sol.x, sol.yp[1]),
            #     "rms_residuals": sol.rms_residuals
            # }

            # Temperature equation
        if False:
            for sp in spec:

                r"""
                Boundary conditions for ion heat transport equation in form:
                    V*Y' + U*Y =W
                On axis:
                    dTi/drho_tor(rho_tor=0)=0
                if(solver_type != 4):
                    v[iion][0] = 1.
                    u[iion][0] = 0.
                else:
                    if((diff_ti[0]+diff_hyper) > 1.0e-6):
                        v[iion][0] = -(diff_ti[0]+diff_hyper)*ni[0]
                    else:
                        v[iion][0] = -1.0e-6*ni[0]
                    u[iion][0] = (conv_ti[0]+diff_hyper*ti0p[0]/ti0[0])*ni[0]+local_flux_ni_conv_s4[iion]
                w[iion][0] = 0.0
                # At the edge:
                #       FIXED Ti
                if(ti_bnd_type(2, iion) == 1):
                    v[iion][1] = 0.
                    u[iion][1] = 1.
                    w[iion][1] = ti_bnd[1, 0]
                #       FIXED grad_Ti
                elif(ti_bnd_type(2, iion) == 2):
                    v[iion][1] = 1.
                    u[iion][1] = 0.
                    w[iion][1] = -ti_bnd[1, 0]
                #       FIXED L_Ti
                if(ti_bnd_type(2, iion) == 3):
                    v[iion][1] = ti_bnd[1, 0]
                    u[iion][1] = 1.
                    w[iion][1] = 0.
                #       FIXED Flux_Ti
                elif(ti_bnd_type(2, iion) == 4):
                    v[iion][1] = -vpr[-1]*gm3[-1]*diff_ti[-1]*ni[-1]
                    u[iion][1] = vpr[-1]*gm3[-1]*conv_ti[-1]*ni[-1]+flux_ni[-1]
                    w[iion][1] = ti_bnd[1, 0]
                #       Generic boundary condition
                elif(ti_bnd_type(2, iion) == 5):
                    v[iion][1] = ti_bnd[1, 0]
                    u[iion][1] = ti_bnd(2, 2)
                    w[iion][1] = ti_bnd(2, 3)
                # Temperature equation is not solved:
                elif(ti_bnd_type(2, iion) == 0):
                    dy = derivative(y, rho_tor)  # temperature gradient
                    flag = 0
                    a[iion][-1] = 1.0
                    b[iion][-1] = 1.0
                    c[iion][-1] = 1.0
                    d[iion][-1] = 0.0
                    e[iion][-1] = 0.0
                    f[iion][-1] = 0.0
                    g[iion][-1] = 0.0
                    v[iion][1] = 0.0
                    u[iion][1] = 1.0
                    w[iion][1] = y[-1]
                Defining coefficients for numerical solver:
                solver["CM1"][ndim, iion] = vie
                solver["CM1"][iion, ndim] = vie
                for zion in range(nion):
                    solver["CM1"][iion, zion] = vii[zion]
                """

                diff = core_profiles_next._create(0.0, name="diff")
                conv = core_profiles_next._create(0.0, name="conv")

                for trans in self.core_transport:
                    d = trans[sp].energy.d
                    if isinstance(d, np.ndarray) or type(d) in (int, float):
                        diff += d

                    v = trans[sp].energy.v
                    if isinstance(v, np.ndarray) or type(v) in (int, float):
                        conv += v

                qs_exp = core_profiles_next._create(0.0, name="se_exp")
                qs_imp = core_profiles_next._create(0.0, name="se_imp")

                # FIXME: collisions
                # qs_exp += qei + qzi + qgi
                # qi_imp += vei + vzi

                for src in self.core_sources:
                    # si = src.electrons.particles_decomposed.implicit_part
                    # if isinstance(si, np.ndarray) or type(si) in (int, float):
                    #     se_imp += si

                    # se = src.electrons.particles_decomposed.explicit_part
                    # if isinstance(se, np.ndarray) or type(se) in (int, float):
                    #     se_exp += se
                    se = src[sp].energy
                    if isinstance(se, np.ndarray) or type(se) in (int, float):
                        se_exp += se

                sp_prev = core_profiles_prev[sp]
                sp_next = core_profiles_next[sp]

                ts_prev = sp_prev.temperature

                if not ts_prev:
                    logger.error("Temperature profile is not initialized!")
                    continue

                tsp_prev = sp_prev.temperature_prime

                if tsp_prev is None:
                    tsp_prev = Function(ts_prev, axis=rho_tor_norm).derivative

                ns_next = sp_next.density

                ns_prev = sp_prev.density

                ns_flux_next = sp_next.density_flux

                a = (3/2) * vpr * ns_next

                b = (3/2) * (vprm**(5/3)/vpr**(2/3)) * ns_prev

                c = rho_tor_boundary

                d = vpr*gm3 / rho_tor_boundary * ns_next * (diff + diff_hyper)

                e = vpr*gm3 * ns_next * (conv + diff_hyper*tsp_prev/ts_prev) + \
                    ns_flux_next - vpr * (3/2)*k_phi * rho_tor * ns_next

                f = vpr * (qs_exp)

                g = vpr * (qs_imp + (3*k_rho - k_phi * dvpr)*ns_next)

                solution = self.solve_general_form(rho_tor_norm,
                                                   ts_prev, tsp_prev,
                                                   inv_tau,
                                                   (a, b, c, d, e, f, g),
                                                   ((u0, v0, w0), (u1, v1, w1)),
                                                   **kwargs
                                                   )

                logger.debug(
                    f"""Solve transport equations: Temperature {sp_next.label}: {'Done' if  solution.success else 'Failed' }
                    Message: {sol.message} """)

                if solution.success:
                    sp_next.temperature = Function(solution.x, solution.y[0])
                    sp_next.temperature_prime = Function(solution.x, solution.y0[0])
                    sp_next.temperature_flux = Function(solution.x, solution.y[1])

        core_profiles_next.vpr = vpr
        core_profiles_next.gm3 = gm3

        return core_profiles_next

    def rotation(self,
                 core_profiles_prev,
                 core_profiles_next,
                 *args,
                 collisions=None,
                 core_transport=None,
                 core_sources=None,
                 boundary_condition=None,
                 hyper_diff=[0, 0],
                 **kwargs):
        r"""
            Note:

                .. math::  \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime\frac{5}{3}}\left\langle R\right\rangle \
                            m_{i}n_{i}u_{i,\varphi}\right)+\frac{\partial}{\partial\rho}\Phi_{i}=V^{\prime}\left[U_{i,\varphi,exp}-U_{i,\varphi}\cdot u_{i,\varphi}+U_{zi,\varphi}\right]
                    :label: transport_rotation
        """

        if core_transport is None:
            core_transport = self._tokamak.core_transport

        if core_sources is None:
            core_sources = self._tokamak.core_sources

        if boundary_conditions is None:
            boundary_conditions = self.boundary_conditions

        # Allocate types for interface with PLASMA_COLLISIONS:
        self.allocate_collisionality(nrho_tor, nion, collisions, ifail)

        # Allocate types for interface with numerical solver:
        self. allocate_numerics(ndim, nrho_tor, solver, ifail)

        # Set up local variables:
        amix = control["AMIX"]
        tau = control["TAU"]
        solver_type = control["SOLVER_TYPE"]

        B0 = kwargs["BGEO"]
        B0m = kwargs["BTM"]
        B0dot = (B0-B0m)*inv_tau

        # Flux surface averaged R {dynamic} [m]
        gm8 = equilibrium.profiles_1d.interpolate("gm8")(rho_tor_norm)
        gm3 = equilibrium.profiles_1d.interpolate("gm3")(rho_tor_norm)

        rho_tor = kwargs["RHO"]
        vpr = kwargs["VPR"]
        vprm = kwargs["VPRM"]
        gm3 = kwargs["G1"]
        gm8 = kwargs["G2"]
        gm8m = kwargs["G2M"]

        mtor_tot = 0.
        flux_mtor_tot = 0.

        # Exchange terms (defined on previous iteration):
        #      self.PLASMA_COLLISIONS (GEOMETRY,PROFILES,COLLISIONS,ifail)

        # Numeric types common for all ions:
        solver["TYPE"] = solver_type
        solver["NDIM"] = ndim
        solver["NRHO"] = nrho_tor
        solver["AMIX"] = amix
        solver["DERIVATIVE_FLAG"][0] = 0

        #     solution of rotation transport equation for
        #     individual ion species

        for iion in range(nion):

            idim = iion

            # Set equation to 'predictive' and all coefficients to zero:
            flag = 1
            y   .fill(0.0)
            dy  .fill(0.0)
            ym  .fill(0.0)
            dym .fill(0.0)
            a   .fill(0.0)
            b   .fill(0.0)
            c   .fill(0.0)
            d   .fill(0.0)
            e   .fill(0.0)
            f   .fill(0.0)
            g   .fill(0.0)
            h = 0.0
            v = 0.0
            u = 0.0
            w = 0.0

            # Set up ion mass:
            mion = profiles["MION"][iion]  # *MP

            # Set up boundary conditions for particular ion type:
            vtor_bnd_type = profiles["VTOR_BND_TYPE"][iion]
            vtor_bnd = profiles["VTOR_BND"][iion]

            # Set up local variables for particular ion type:
            vtor = profiles["VTOR"][iion]
            dvtor = profiles["DVTOR"][iion]
            vtorm = kwargs["VTORM"][iion]
            dvtorm = kwargs["DVTORM"][iion]
            wtor = profiles["WTOR"][iion]
            ni = profiles["NI"][iion]
            dni = profiles["DNI"][iion]
            nim = kwargs["NIM"][iion]
            dnim = kwargs["DNIM"][iion]
            mtor = profiles["MTOR"][iion]

            flux_mtor = profiles["FLUX_MTOR"][iion]
            flux_ni = profiles["FLUX_NI"][iion]

            diff = transport["DIFF_VTOR"][iion]
            conv = transport["VCONV_VTOR"][iion]

            ui_exp = kwargs["UI_EXP"][iion]
            ui_imp = kwargs["UI_IMP"][iion]

            wzi = collisions["WZI"][iion]
            uzi = collisions["UZI"][iion]

            for zion in range(nion):
                # DPC was           WII(IRHO,IION)   = COLLISIONS["WII(IRHO,IION,ZION)   ### check if the following is what was intended
                wii[zion] = collisions["WII"][iion, zion]

            # Coefficients for rotation transport equation in form:
            #
            #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

            y = vtor
            dy = dvtor
            ym = vtorm
            dym = dvtorm

            a = vpr*gm8*ni*mion
            b = vprm*gm8m*nim*mion
            c = 1.
            # AF, 14.May.2011 - multipication by G2, which in analytics is 1
            d = vpr*gm3*ni*mion*diff*gm8
            e = (vpr*gm3*ni*conv + flux_ni - B0dot/2./B0*rho_tor*ni*vpr) * gm8*mion
            f = vpr*(ui_exp + uzi)
            g = vpr*(ui_imp + wzi)

            h = tau

            # Boundary conditions for numerical solver in form:
            #
            #     V*Y' + U*Y =W

            # On axis
            #     dVtor,i/drho_tor(rho_tor=0)=0: # AF - 24.Jun.2010, replaces "#     dTi/drho_tor(rho_tor=0)=0:"
            if(solver_type != 4):  # AF 11.Oct.2011
                v[0] = 1.
                u[0] = 0.
            else:
                if(diff[0] > 1.0e-6):
                    v[0] = -diff[0]*ni[0]
                else:
                    v[0] = -1.0e-6*ni[0]

                u[0] = conv[0]*ni[0] + local_flux_ni_s4[iion]
            w[0] = 0.

            # At the edge:
            #     FIXED Vtor,i
            if(vtor_bnd_type[1] == 1):
                v = 0.
                u = 1.
                w = vtor_bnd[1, 0]

            #     FIXED grad_Vtor,i
            if(vtor_bnd_type[1] == 2):
                v = 1.
                u = 0.
                w = -vtor_bnd[1, 0]

            #     FIXED L_Vtor,i
            if(vtor_bnd_type[1] == 3):
                v = 1.
                u = 1./vtor_bnd[1, 0]
                w = 0.

            #     FIXED Flux_Mtor,i
            if(vtor_bnd_type[1] == 4):
                v = -vpr[-1]*gm3[-1]*gm8[-1]*diff[-1]*ni[-1]*mion
                u = vpr[-1]*gm3[-1]*gm8[-1]*conv[-1] * \
                    ni[-1]*mion + gm8[-1]*flux_ni[-1]*mion
                w = vtor_bnd[1, 0]

            #     Generic boundary condition
            if(vtor_bnd_type[1] == 5):
                v = vtor_bnd[1, 0]
                u = vtor_bnd(2, 2)
                w = vtor_bnd(2, 3)

            # Rotation equation is not solved:
            if(vtor_bnd_type[1] == 0):
                dy = derivative(y, rho_tor)
                flag = 0
                for irho_tor in range(rho_tor.shape[0]):
                    a = 1.0
                    b = 1.0
                    c = 1.0
                    d = 0.0
                    e = 0.0
                    f = 0.0
                    g = 0.0

                v = 0.0
                u = 1.0
                w = y[-1]

            # Defining coefficients for numerical solver:
            solver["EQ_FLAG"][idim] = flag

            solver["RHO"] = rho_tor

            solver["Y"][idim] = y
            solver["DY"][idim] = dy
            solver["YM"][idim] = ym
            solver["A"][idim] = a
            solver["B"][idim] = b
            solver["C"][idim] = c
            solver["D"][idim] = d
            solver["E"][idim] = e
            solver["F"][idim] = f
            solver["G"][idim] = g

            solver["CM1"][idim, idim] = 0.

            for zion in range(nion):
                solver["CM1"][iion, zion] = wii[irho_tor, zion]

            solver["H"] = h

            solver["V"][idim] = v
            solver["U"][idim] = u
            solver["W"][idim] = w

        # Solution of rotation transport equation:
        self.solve_eq(a, b, c, d, e, f, g, h)

        # dy check for nans in solution, return if true)
        if(any(np.isnan(solver["y"]))):
            raise RuntimeError('Error in the vtor equation: nans in the solution, stop')

        for iion in range(nion):
            idim = iion

            # dpc 2011-08-11: I think we need most of the following
            mion = profiles["MION"][iion]  # *MP
            vtor_bnd_type = profiles["VTOR_BND_TYPE"][iion]
            vtor_bnd = profiles["VTOR_BND"][iion]

            # dpc 2011-08-11: I think we need most of the following
            vtor = profiles["VTOR"][iion]
            vtorm = kwargs["VTORM"][iion]
            wtor = profiles["WTOR"][iion]
            ni = profiles["NI"][iion]
            nim = kwargs["NIM"][iion]
            mtor = profiles["MTOR"][iion]
            flux_mtor = profiles["FLUX_MTOR"][iion]
            flux_ni = profiles["FLUX_NI"][iion]
            diff = transport["DIFF_VTOR"][iion]
            conv = transport["VCONV_VTOR"][iion]
            ui_exp = kwargs["UI_EXP"][iion]
            ui_imp = kwargs["UI_IMP"][iion]
            wzi = collisions["WZI"][iion]
            uzi = collisions["UZI"][iion]
            # dpc end

            # New solution and its derivative:
            y = solver["Y"](idim, irho_tor)
            dy = solver["DY"](idim, irho_tor)

            if(vtor_bnd_type[1] == 0):
                y = profiles["VTOR"][iion]
                dy = derivative(y, rho_tor)

            # New rotation velocity and momentum flux:
            vtor = y
            dvtor = dy
            # dy 2017-10-06        WTOR(IRHO)           = Y(IRHO)/G2(NRHO)
            wtor = y/gm8

            mtor = gm8*ni*mion*y
            mtor_tot = mtor_tot + mtor

            if(rho_tor != 0.):  # FIXME
                fun1 = vpr/rho_tor * (ui_exp + uzi + (wzi + gm8*mion*ni*inv_tau - ui_imp) * y)
            else:
                fun1 = (ui_exp + uzi + (wzi + gm8*mion*ni*inv_tau - ui_imp) * y)

            intfun1 = integral(fun1, rho_tor)  # Integral source

            flux_mtor_conv = gm8*mion*flux_ni*y

            flux_mtor_cond = vpr*gm3*gm8*mion*ni                          \
                * (y*conv - dy*diff)

            flux_mtor = flux_mtor_conv + flux_mtor_cond

            int_source = intfun1 + vpr*gm8*B0dot/2./B0*rho_tor*mion*ni*y

            # if equation is not solved, conductive component of electron heat flux is determined from the integral of kwargs
            if(vtor_bnd_type[1] == 0):

                diff = 1.e-6
                flux_mtor = int_source

                flux_mtor_cond = int_source - gm8*mion*flux_ni*y

                if((vpr*gm3*gm8*mion*ni != 0.0) and (dy != 0.0)):
                    diff = - flux_mtor_cond / dy / (vpr*gm3*gm8*mion*ni)
                    conv = 0.0
                if(diff <= 1.e-6):
                    diff = 1.e-6
                    conv = (flux_mtor_cond / (max(abs(vpr), 1.e-6)*gm3*gm8
                                              * mion*ni) + dy*diff) / max(abs(y), 1.e-6)

                flux_mtor_tot = flux_mtor_tot + flux_mtor

                # Return new profiles to the work flow:
                profiles["VTOR"][iion] = vtor
                profiles["DVTOR"][iion] = dvtor
                # dy 2019-08-31 make sure profiles from db are written to the coreprof
                #       way around, address afterwords for consistent fix
                #        PROFILES["WTOR(IRHO,IION)            = WTOR(IRHO)
                profiles["MTOR"][iion] = mtor
                profiles["DIFF_VTOR"][iion] = diff
                profiles["VCONV_VTOR"][iion] = conv
                profiles["FLUX_MTOR"][iion] = flux_mtor
                profiles["FLUX_MTOR_CONV"][iion] = flux_mtor_conv
                profiles["FLUX_MTOR_COND"][iion] = flux_mtor_cond
                profiles["SOURCE_MTOR"][iion] = ui_exp + uzi - (ui_imp + wzi) * vtor
                profiles["INT_SOURCE_MTOR"][iion] = int_source

            fun1 = profiles["SOURCE_MTOR"][iion]*vpr
            intfun1 = integral(fun1, rho_tor)
            profiles["INT_SOURCE_MTOR"][iion] = intfun1

        profiles["MTOR_TOT"] = mtor_tot
        profiles["FLUX_MTOR_TOT"] = flux_mtor_tot


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
