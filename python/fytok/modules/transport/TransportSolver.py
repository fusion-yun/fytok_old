"""

"""

import collections
import copy
from functools import cached_property

import numpy as np
import scipy.constants
from fytok.util.Misc import Identifier
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.data.Profile import Profile
from spdm.util.sp_export import sp_find_module

from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .EdgeProfiles import EdgeProfiles
from .EdgeSources import EdgeSources
from .EdgeTransport import EdgeTransport
from .Equilibrium import Equilibrium

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6


class TransportSolver(AttributeTree):
    r"""
        Solve transport equations :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`

        See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`

    """
    IDS = "transport_solver_numerics"

    @staticmethod
    def __new__(cls,  cache, *args, **kwargs):
        if cls is not TransportSolver:
            return super(TransportSolver, cls).__new__(cls)

        backend = cache.solver.name
        if isinstance(backend, LazyProxy):
            backend = backend()

        if backend is NotImplemented or not backend:
            n_cls = cls
        else:
            plugin_name = f"{__package__}.plugins.transport_solver.Plugin{backend}"

            n_cls = sp_find_module(plugin_name, fragment=f"TransportSolver{backend}")

            if n_cls is None:
                raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#TransportSolver{backend}")

        return object.__new__(n_cls)

    def __init__(self,  cache, *args, tokamak={},
                 solver="electron",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__['_cache'] = cache
        self._tokamak = tokamak
        self._solver = solver

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
        c = self._cache.primary_coordinate
        return Identifier(
            name="rho_tor_norm",
            index=1,
            description=""
        ) if c is NotImplemented or c is None or len(c) == 0 else Identifier(c)

    class BoundaryCondition(AttributeTree):
        def __init__(self, cache, *args, **kwargs):
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
        res = AttributeTree(
            current=AttributeTree(),
            electrons=AttributeTree,
            ions=AttributeTree(
                default_factory_array=lambda _holder=self: TransportSolver.BoundaryCondition(None, parent=_holder)),
            energy_ion_total=AttributeTree(),
            momentum_tor=AttributeTree(),
            default_factory=lambda k, _holder=self: TransportSolver.BoundaryCondition(None, holder=_holder, key=k))
        return res

    def update(self,
               core_profiles_prev,
               core_profiles_next,
               *args,
               equilibrium=None,
               core_transport=None,
               core_sources=None,
               boundary_condition=None,
               particle_transport="electron",
               **kwargs):
        """Solve transport equations
        """

        if equilibrium is None:
            equilibrium = self._tokamak.equilibrium

        if core_transport is None:
            core_transport = self._tokamak.core_transport

        if core_sources is None:
            core_sources = self._tokamak.core_sources

        if boundary_condition is None:
            boundary_condition = self._tokamak.boundary_conditions

        self.current(core_profiles_prev,
                     core_profiles_next,
                     equilibrium=equilibrium,
                     core_sources=core_sources,
                     core_transport=core_transport,
                     boundary_condition=boundary_condition.current,
                     ** kwargs)

        if particle_transport == "electron":
            self.electron_density(core_profiles_prev,
                                  core_profiles_next,
                                  equilibrium=equilibrium,
                                  core_transport=core_transport,
                                  core_sources=core_sources,
                                  boundary_condition=boundary_condition,
                                  **kwargs)
        else:
            self.ion_density(core_profiles_prev,
                             core_profiles_next,
                             equilibrium=equilibrium,
                             core_transport=core_transport,
                             core_sources=core_sources,
                             boundary_condition=boundary_condition,
                             **kwargs)

        # self.temperatures(core_profiles_prev,
        #                   core_profiles_next,
        #                   equilibrium=equilibrium,
        #                   core_transports=core_transport,
        #                   core_sources=core_sources,
        #                   boundary_condition=boundary_condition,
        #                   **kwargs)

        # if enable_rotation:
        #     self.rotation(core_profiles_prev,
        #                   core_profiles_next,
        #                   equilibrium=equilibrium,
        #                   core_transports=core_transport,
        #                   core_sources=core_sources,
        #                   boundary_condition=boundary_condition,
        #                   **kwargs)

        self.update_global_quantities(core_profiles_prev,  core_profiles_next)

        return True

    def update_global_quantities(self, core_profiles_prev,  core_profiles_next):
        # self.core_profiles_prev.global_quantities = NotImplemented
        pass

    def solve_general_form(self, x, y0, yp0, inv_tau, coeff,  bc, parameters=None, **kwargs):
        r"""solve standard form

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
                        \frac{\partial Y\left(x,t\right)}{\partial\rho} & = \
                              -\frac{\Gamma\left(x,t\right)-e\left(x\right)\cdot Y\left(x,t\right)}{d\left(x\right)}\\
                        \frac{\partial\Gamma\left(x,t\right)}{\partial x} & = \
                             c\left(x\right)\left[f\left(x\right)-g\left(x\right)\cdot Y\left(x,t\right)-\
                                 \frac{a\left(x\right)\cdot Y\left(x,t\right)-b\left(x\right)\cdot Y\left(x,t-1\right)}{\tau}\right]
                    \end{cases}
                    :label: generalized_trans_eq_first_order

        """
        if parameters is not None:
            return self.solve_general_form_with_parameter(x, y0, yp0, inv_tau, coeff,  bc, parameters, **kwargs)

        a, b, c, d, e, f, g = coeff

        D = d
        E = e
        F = (f + b*inv_tau*y0)*c
        G = (g + a*inv_tau)*c

        fix_boundary = False
        dD = None
        dE = None

        if abs(D(x[0])) < TOLERANCE:
            fix_boundary = True
            dD = D.derivative
            if isinstance(E, Profile):
                dE = E.derivative
            else:
                dE = 0.0

        def fun(x, Y):
            y, gamma = Y
            vD = D(x)
            vE = E(x)
            dgamma = F(x) - G(x) * y

            if fix_boundary:
                vD[0] = 1.0
                dy = -(gamma-vE*y)/vD
                dy[0] = (dgamma[0] - dE(0)*y[0])/(E(0)-dD(0))
            else:
                dy = -(gamma-vE*y)/vD

            return np.vstack((dy.value, dgamma.value))

        u0, v0, w0 = bc[0]
        u1, v1, w1 = bc[1]

        def bc_func(Ya, Yb):
            r"""
                u*y(b)+v*\Gamma(b)=w
                Ya=[y(a),y'(a)]
                Yb=[y(b),y'(b)]
                P=[[u(a),u(b)],[v(a),v(b)],[w(a),w(b)]]
            """

            y0, gamma0 = Ya
            y1, gamma1 = Yb
            return (u0 * y0 + v0 * gamma0 - w0,
                    u1 * y1 + v1 * gamma1 - w1)

        if yp0 is None:
            yp0 = Profile(y0, axis=x).derivative

        gamma0 = -yp0*D(x) + y0*E(x)

        return scipy.integrate.solve_bvp(fun, bc_func, x[:], np.vstack((y0[:], gamma0[:])), **kwargs)

    def solve_general_form_with_parameter(self, x, y0, yp0, inv_tau, coeff,  bc, parameters=None, **kwargs):
        a, b, c, d, e, f, g = coeff

        def D(x, p): return d(x) if isinstance(d, Profile) else d(x, p)
        def E(x, p): return e(x) if isinstance(e, Profile) else e(x, p)
        F = (f + b*inv_tau*y0)*c
        G = (g + a*inv_tau)*c

        def fun(x, Y, p):
            y, gamma = Y
            dy = -(gamma-E(x, p)*y)/D(x, p)
            dgamma = F(x) - G(x) * y
            return np.vstack((dy, dgamma))

        u0, v0, w0 = bc[0]
        u1, v1, w1 = bc[1]

        def bc_func(Ya, Yb, p):
            r"""
                u*y(b)+v*\Gamma(b)=w
                Ya=[y(a),y'(a)]
                Yb=[y(b),y'(b)]
                P=[[u(a),u(b)],[v(a),v(b)],[w(a),w(b)]]
            """

            y0, gamma0 = Ya
            y1, gamma1 = Yb
            return (u0 * y0 + v0 * gamma0 - w0,
                    u1 * y1 + v1 * gamma1 - w1,
                    gamma1-E(1.0, p)(1.0)*y1)

        if yp0 is None:
            yp0 = Profile(y0, axis=x).derivative

        if not isinstance(parameters, collections.abc.Sequence):
            parameters = [parameters]

        gamma0 = -yp0*D(x, parameters) + y0*E(x, parameters)

        return scipy.integrate.solve_bvp(fun, bc_func, x, np.vstack((y0, gamma0)), p=parameters, **kwargs)

    def current(self,
                core_profiles_prev,
                core_profiles_next,
                *,
                equilibrium=None,
                core_transport=None,
                core_sources=None,
                boundary_condition=None,
                **kwargs):
        r"""

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

        """

        # -----------------------------------------------------------
        # time step                                         [s]
        tau = core_profiles_next.time - core_profiles_prev.time

        if abs(tau) < EPSILON:
            inv_tau = 0
        else:
            inv_tau = 1.0/tau

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_next.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_next.vacuum_toroidal_field.b0

        # -----------------------------------------------------------
        # Grid
        # $rho_tor$ not  normalised minor radius                [m]
        rho_tor = core_profiles_next.profiles_1d.grid.rho_tor

        rho_tor_norm = core_profiles_next.profiles_1d.grid.rho_tor_norm

        rho_tor_boundary = core_profiles_next.profiles_1d.grid.rho_tor[-1]

        k_phi = ((core_profiles_next.vacuum_toroidal_field.b0 -
                  core_profiles_prev.vacuum_toroidal_field.b0) /
                 (core_profiles_next.vacuum_toroidal_field.b0 +
                  core_profiles_prev.vacuum_toroidal_field.b0) +
                 (core_profiles_next.profiles_1d.grid.rho_tor[-1] -
                  core_profiles_prev.profiles_1d.grid.rho_tor[-1]) /
                 (core_profiles_next.profiles_1d.grid.rho_tor[-1] +
                  core_profiles_prev.profiles_1d.grid.rho_tor[-1]))*inv_tau

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        # equilibrium.profiles_1d.mapping("rho_tor_norm", "fpol")(rho_tor_norm)
        fpol = Profile(equilibrium.profiles_1d.fpol, axis=equilibrium.profiles_1d.rho_tor_norm)
        # fprime = core_profiles_next.profiles_1d.derivative("fpol")

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        # equilibrium.profiles_1d.mapping("rho_tor_norm",   "dvolume_drho_tor")(rho_tor_norm)
        vpr = Profile(equilibrium.profiles_1d.dvolume_drho_tor*rho_tor_boundary,
                      axis=equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)

        vppr = vpr.derivative/rho_tor_boundary
        # $gm2 \euqiv \left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $  [m^-2]
        gm2 = Profile(equilibrium.profiles_1d.gm2,
                      axis=equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)  # equilibrium.profiles_1d.mapping("rho_tor_norm", "gm2")(rho_tor_norm)
        gm1 = Profile(equilibrium.profiles_1d.gm1,
                      axis=equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)  # equilibrium.profiles_1d.mapping("rho_tor_norm", "gm1")(rho_tor_norm)
        # $\Psi$ flux function from current                 [Wb]
        psi0 = Profile(equilibrium.profiles_1d.psi,
                       axis=equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)  # equilibrium.profiles_1d.mapping("rho_tor_norm", "psi")(rho_tor_norm)
        # $\frac{\partial\Psi}{\partial\rho_{tor,norm}}$               [Wb/m]
        psi0_prime = psi0.derivative
        # $q$ safety factor                                 [-]
        qsf = Profile(equilibrium.profiles_1d.q,
                      axis=equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)  # equilibrium.profiles_1d.mapping("rho_tor_norm", "q")(rho_tor_norm)

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        conductivity_parallel = Profile(0.0, axis=core_profiles_next.profiles_1d.grid.rho_tor_norm, description={
                                        "name": "conductivity_parallel"})

        for trans in core_transport:
            sigma = trans.profiles_1d["conductivity_parallel"]
            if isinstance(sigma, np.ndarray):
                conductivity_parallel += sigma

        # -----------------------------------------------------------
        # Sources
        # total non inductive current, PSI independent component,          [A/m^2]
        j_ni_exp = core_profiles_next.profiles_1d._create(0.0, name="j_ni_exp")
        for src in core_sources:
            j = src.profiles_1d.j_parallel
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

        core_profiles_next.profiles_1d.psi0 = psi0
        core_profiles_next.profiles_1d.psi0_prime = psi0_prime
        core_profiles_next.profiles_1d.psi0_eq = Profile(
            equilibrium.profiles_1d.psi, axis=equilibrium.profiles_1d.rho_tor_norm)
        core_profiles_next.profiles_1d.psi0_prime_eq = Profile(
            equilibrium.profiles_1d.dpsi_drho_tor*rho_tor_boundary, axis=equilibrium.profiles_1d.rho_tor_norm)

        core_profiles_next.profiles_1d.q0 = Profile(
            equilibrium.profiles_1d.q, axis=equilibrium.profiles_1d.rho_tor_norm)

        j_total0 = (d * psi0_prime).derivative / c / vpr*(2.0 * scipy.constants.pi)

        # j_total0[0] = 2*j_total0[1]-j_total0[2]
        core_profiles_next.profiles_1d.j_total0 = j_total0
        core_profiles_next.profiles_1d.j_ni_exp = j_ni_exp

        core_profiles_next.profiles_1d.rho_star = vpr/(4.0*(scipy.constants.pi**2)*R0)

        core_profiles_next.profiles_1d.rho_tor = rho_tor_norm*equilibrium.profiles_1d.rho_tor[-1]

        if sol.success:
            core_profiles_next.profiles_1d.psi = Profile(sol.y[0], axis=sol.x)
            core_profiles_next.profiles_1d.psi1_prime = core_profiles_next.profiles_1d.psi.derivative
            core_profiles_next.profiles_1d.psi1_prime1 = Profile(sol.yp[0], axis=sol.x)

            core_profiles_next.profiles_1d.dgamma_current = Profile(sol.yp[1], axis=sol.x)
            core_profiles_next.profiles_1d.dpsi_drho_tor = Profile(sol.yp[0]/rho_tor_boundary, axis=sol.x)

            q = (scipy.constants.pi*2.0)*B0 * rho_tor_boundary**2 * sol.x
            # /sol.yp[0]
            q[1:] /= sol.yp[0][1:]
            q[0] = 2*q[1]-q[2]

            core_profiles_next.profiles_1d.q = Profile(q, axis=sol.x, description={"name": "q from transport"})

            core_profiles_next.profiles_1d.j_tor = (Profile(sol.y[1], axis=sol.x)*fpol).derivative / vpr\
                * (-2.0 * scipy.constants.pi*R0 / scipy.constants.mu_0/rho_tor_boundary)

            core_profiles_next.profiles_1d.j_parallel = Profile(sol.yp[1], axis=sol.x) * (- scipy.constants.pi*2.0 / (scipy.constants.mu_0*B0 * rho_tor_boundary)) * \
                (fpol**2/vpr)

            # core_profiles_next.profiles_1d.e_field.parallel = (j_parallel-j_ni_exp -
            #                                                    j_ni_exp*core_profiles_next.profiles_1d.psi)/conductivity_parallel

        else:
            psi_prime = (scipy.constants.pi*2.0)*B0 * rho_tor / qsf * rho_tor_boundary
            core_profiles_next.profiles_1d.psi_prime = psi_prime
            core_profiles_next.profiles_1d.psi = psi0[0] + \
                Profile(psi_prime, axis=rho_tor_norm).integral * 2

        core_profiles_next.profiles_1d.f_current = f*c

        core_profiles_next.profiles_1d.j_total = j_ni_exp

        # core_profiles_next.profiles_1d.vpr = vpr
        # core_profiles_next.profiles_1d.gm2 = gm2

        # core_profiles_next.profiles_1d.a = a(sol.x)
        # core_profiles_next.profiles_1d.b = b(sol.x)
        # # core_profiles_next.profiles_1d.c =(sol.x)
        # core_profiles_next.profiles_1d.d = d(sol.x)
        # core_profiles_next.profiles_1d.e = e(sol.x)
        # core_profiles_next.profiles_1d.f = f(sol.x)
        # core_profiles_next.profiles_1d.g = g(sol.x)
        # core_profiles_next.profiles_1d.dvolume_dpsi = Profile(equilibrium.profiles_1d.dvolume_dpsi,
        #                                                       axis=equilibrium.profiles_1d.rho_tor_norm)
        # core_profiles_next.profiles_1d.dpsi_drho_tor = Profile(equilibrium.profiles_1d.dpsi_drho_tor,
        #                                                        axis=equilibrium.profiles_1d.rho_tor_norm)

        # core_profiles_next.profiles_1d.q = 2.0* scipy.constants.pi*B0*rho_tor/psi_prime1

        # current density, toroidal,                        [A/m^2]
        # j_tor = - 2.0* scipy.constants.pi*R0/ scipy.constants.mu_0/vpr * dfun4
        # core_profiles_next.profiles_1d.j_tor = j_tor
        # j_par = - 2.0* scipy.constants.pi/R0/ scipy.constants.mu_0/vpr * (fpol/B0)**2*dfun5
        # core_profiles_next.profiles_1d.current_parallel_inside = j_par

        # # $E_\parallel$  parallel electric field,,          [V/m]
        # core_profiles_next.profiles_1d.e_field.parallel = (j_par - j_ni_exp - j_ni_imp*psi1) / conductivity_parallel

        # # Total Ohmic currents                              [A]
        # # fun7 = vpr * j_par / 2.0 /  scipy.constants.pi * B0 / fpol**2
        # core_profiles_next.profiles_1d.j_ohmic = fpol[-1]*core_profiles_next.profiles_1d.integral(
        #     vpr * j_par / 2.0 /  scipy.constants.pi * B0 / fpol**2, 0, 1)

        # core_profiles_next.profiles_1d.j_tor = j_tor

        # # Total non-inductive currents                       [A]
        # core_profiles_next.profiles_1d.j_non_inductive = fpol[-1] * core_profiles_next.profiles_1d.integral(
        #     vpr * (j_ni_exp + j_ni_imp * psi1) / (2.0 *  scipy.constants.pi) * B0 / fpol**2)

        return True

    def electron_density(self,
                         core_profiles_prev,
                         core_profiles_next,
                         *args,
                         equilibrium=None,
                         core_transport=None,
                         core_sources=None,
                         boundary_condition=None,
                         hyper_diff=[0.0, 0.0],
                         **kwargs):
        r"""

            Note:

                .. math::
                    \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                    \left(V^{\prime}n_{s}\right)+\frac{\partial}{\partial\rho}\Gamma_{s}=\
                    V^{\prime}\left(S_{s,exp}-S_{s,imp}\cdot n_{s}\right)
                    :label: particle_density_transport

                .. math::
                    \Gamma_{s}\equiv-D_{s}\cdot\frac{\partial n_{s}}{\partial\rho}+v_{s}^{pinch}\cdot n_{s}
                    :label: particle_density_gamma


        """

        # time step                                         [s]
        tau = core_profiles_next.time - core_profiles_prev.time

        if abs(tau) < EPSILON:
            inv_tau = 0.0
        else:
            inv_tau = 1.0/tau

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_next.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_next.vacuum_toroidal_field.b0

        # Grid
        # $rho_tor$ not  normalised minor radius                [m]
        rho_tor = core_profiles_next.profiles_1d.grid.rho_tor
        rho_tor_boundary = rho_tor[-1]

        # $rho_tor_{norm}$ normalised minor radius                [-]
        rho_tor_norm = core_profiles_next.profiles_1d.grid.rho_tor_norm

        k_phi = ((core_profiles_next.vacuum_toroidal_field.b0 -
                  core_profiles_prev.vacuum_toroidal_field.b0) /
                 (core_profiles_next.vacuum_toroidal_field.b0 +
                  core_profiles_prev.vacuum_toroidal_field.b0) +
                 (core_profiles_next.profiles_1d.grid.rho_tor[-1] -
                  core_profiles_prev.profiles_1d.grid.rho_tor[-1]) /
                 (core_profiles_next.profiles_1d.grid.rho_tor[-1] +
                  core_profiles_prev.profiles_1d.grid.rho_tor[-1]))*inv_tau

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = Profile(equilibrium.profiles_1d.fpol, axis=equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = Profile(equilibrium.profiles_1d.dvolume_drho_tor[:],
                      axis=equilibrium.profiles_1d.rho_tor_norm[:])(rho_tor_norm) * rho_tor_boundary

        core_profiles_next.profiles_1d.vprime = vpr

        vprm = core_profiles_prev.profiles_1d.vprime
        if vprm is None:
            vprm = vpr

        gm3 = Profile(equilibrium.profiles_1d.gm3, axis=equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)

        diff = Profile(0.0, axis=core_profiles_next.profiles_1d.grid.rho_tor_norm, description={"name": "diff"})

        vconv = Profile(0.0, axis=core_profiles_next.profiles_1d.grid.rho_tor_norm, description={"name": "vconv"})

        for trans in core_transport:

            D = trans.profiles_1d.electrons.particles.d

            if isinstance(D, np.ndarray) or isinstance(D, (int, float)):
                diff += D

            v = trans.profiles_1d.electrons.particles.v
            if isinstance(v, np.ndarray) or isinstance(v, (int, float)):
                vconv += v

        se_exp = Profile(0.0, axis=core_profiles_next.profiles_1d.grid.rho_tor_norm, description={"name": "se_exp"})
        se_imp = Profile(0.0, axis=core_profiles_next.profiles_1d.grid.rho_tor_norm, description={"name": "se_imp"})

        for src in core_sources:
            # si = src.profiles_1d.electrons.particles_decomposed.implicit_part
            # if isinstance(si, np.ndarray) or type(si) in (int, float):
            #     se_imp += si

            # se = src.profiles_1d.electrons.particles_decomposed.explicit_part
            # if isinstance(se, np.ndarray) or type(se) in (int, float):
            #     se_exp += se
            se = src.profiles_1d.electrons.particles
            if isinstance(se, np.ndarray) or type(se) in (int, float):
                se_exp += se

        ne0 = core_profiles_prev.profiles_1d.electrons.density

        ne0_prime = core_profiles_prev.profiles_1d.electrons.density_prime

        if ne0_prime is None:
            ne0_prime = ne0.derivative

        # diff_hyper = hyper_diff[0] + hyper_diff[1] * max(diff)

        H = vpr * gm3

        # dlnNe0 = ne0_prime/ne0

        a = vpr
        b = vprm
        c = rho_tor_boundary
        # def d(x, hyper_diff): return H(x) * (diff(x) + hyper_diff) / rho_tor_boundary
        # def e(x, hyper_diff): return H(x) * (vconv(x) + hyper_diff * dlnNe0(x)) - k_phi*rho_tor*vpr
        d = H * (diff) / rho_tor_boundary
        e = H * (vconv) - k_phi*rho_tor*vpr
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
                u = vconv[0]+diff_hyper*ne0_prime[0]/ne0[0]
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
                u = vpr[-1]*gm3[-1]*vconv[-1]
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

        sol = self.solve_general_form(rho_tor_norm,
                                      ne0, ne0_prime,
                                      inv_tau,
                                      (a, b, c, d, e, f, g),
                                      ((0, 1, e[0]), (1, 0, ne0[-1])),
                                      #   parameters=[diff_hyper],
                                      tol=0.5,
                                      verbose=2, max_nodes=400
                                      )

        logger.info(
            f"""Solve transport equations: Electron density: {'Done' if  sol.success else 'Failed' }
             Message: {sol.message} """)

        core_profiles_next.profiles_1d.electrons.density0 = ne0
        core_profiles_next.profiles_1d.electrons.density0_prime = ne0_prime

        core_profiles_next.profiles_1d.electrons.se_exp0 = f
        core_profiles_next.profiles_1d.electrons.se_exp0b = core_profiles_next.profiles_1d.electrons.density_flux0_prime

        core_profiles_next.profiles_1d.electrons.diff_flux = -d * ne0_prime  # * H*diff/rho_tor_boundary
        core_profiles_next.profiles_1d.electrons.vconv_flux = e * ne0  # * H * vconv
        core_profiles_next.profiles_1d.electrons.s_exp_flux = f.integral * c

        core_profiles_next.profiles_1d.electrons.density_residual = \
            core_profiles_next.profiles_1d.electrons.diff_flux\
            + core_profiles_next.profiles_1d.electrons.vconv_flux\
            - core_profiles_next.profiles_1d.electrons.s_exp_flux

        # core_profiles_next.profiles_1d.electrons.density0_residual_left1 = ().derivative
        core_profiles_next.profiles_1d.electrons.density0_residual_right = f*c
        core_profiles_next.profiles_1d.electrons.diff = diff[:]
        core_profiles_next.profiles_1d.electrons.vconv = trans.profiles_1d.electrons.particles.v[:]
        # if sol.success:
        core_profiles_next.profiles_1d.electrons.density = Profile(sol.y[0], axis=sol.x)
        core_profiles_next.profiles_1d.electrons.density_prime = Profile(sol.yp[0], axis=sol.x)
        core_profiles_next.profiles_1d.electrons.density_flux = Profile(sol.y[1], axis=sol.x)
        # core_profiles_next.profiles_1d.electrons.density_flux_error = - core_profiles_next.profiles_1d.electrons.density_prime * \
        #     H * diff_hyper / rho_tor_boundary + core_profiles_next.profiles_1d.electrons.density * \
        #     (H * (diff_hyper * dlnNe0))

        core_profiles_next.profiles_1d.a = a(sol.x)
        core_profiles_next.profiles_1d.b = b(sol.x)
        # core_profiles_next.profiles_1d.c =(sol.x)
        core_profiles_next.profiles_1d.d = d(sol.x)
        core_profiles_next.profiles_1d.e = e(sol.x)
        core_profiles_next.profiles_1d.f = f(sol.x)
        core_profiles_next.profiles_1d.g = g(sol.x)

        core_profiles_next.profiles_1d.vpr = vpr
        core_profiles_next.profiles_1d.gm3 = gm3

        if False:
            core_profiles_next.profiles_1d.n_i_total = core_profiles_next.profiles_1d.electrons.density
            # ni_tot_flux = core_transport.profiles_1d.electron.particles.flux
            # ni_tot_energy_flux = core_transport.profiles_1d.electron.energy.flux

            # for ion in core_profiles_prev.profiles_1d.ion:
            #     ni_tot  -= ion.zion*ion.density_fast
            #     if(profiles["NI_BND_TYPE"][iion] > 0.5):
            #         ni_tot = ni_tot - profiles["ZION"][iion]*profiles["NI"][iion]
            #         flux_ni_tot = flux_ni_tot - profiles["ZION"][iion]*profiles["FLUX_NI"][iion]
            #         contrib_2_energy_flux_ni_tot = contrib_2_energy_flux_ni_tot - \
            #             profiles["ZION"][iion] * profiles["CONTRIB_2_ENERGY_FLUX_NI"][iion]

    def ion_density_one(self,
                        iion,
                        core_profiles_prev,
                        core_profiles_next,
                        *args,
                        equilibrium=None,
                        core_transport=None,
                        core_sources=None,
                        boundary_condition=None,
                        hyper_diff=[0, 0],
                        **kwargs):

        # -----------------------------------------------------------
        # time step                                         [s]
        tau = core_profiles_next.time - core_profiles_prev.time

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_next.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_next.vacuum_toroidal_field.b0

        # -----------------------------------------------------------
        # Grid
        # $rho_tor$ not  normalised minor radius                [m]
        rho_tor = core_profiles_next.rho_tor

        # $\Psi$ flux function from current                 [Wb]
        psi0 = core_profiles_prev.profiles_1d.psi
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        psi_prime = core_profiles_prev.profiles_1d.psi_prime

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = core_profiles_next.fpol

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = core_profiles_next.dvolume_dpsi

        # $gm2 \euqiv \left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $  [m^-2]
        gm2 = core_profiles_next.gm2

        # $\left\langle \left|\nabla\rho\right|^{2}\right\rangle $  [-]
        gm3 = core_profiles_next.gm3

        # -----------------------------------------------------------
        # Profile

        # $q$ safety factor                                 [-]
        qsf = core_profiles_next.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        conductivity_parallel = core_profiles_next.conductivity_parallel

        hyper_diff_exp = hyper_diff[0]
        hyper_diff_imp = hyper_diff[1]

        ni0 = core_profiles_prev.profiles_1d.ion[iion].density

        ni0p = core_profiles_prev.profiles_1d.ion[iion].density_drho_tor or derivative(ni0, rho_tor)

        diff = np.zeros(shape=rho_tor.shape)
        vconv = np.zeros(shape=rho_tor.shape)

        for model in transports.model:
            # c1[imodel] = transport["C1"][imodel]
            diff += model.ion[iion].particles.d
            vconv += model.ion[iion].particles.v

        si_exp = np.zeros(shape=rho_tor.shape)
        si_imp = np.zeros(shape=rho_tor.shape)

        for source in sources.source:
            si_imp += source.ion[iion].particles_decomposed.implicit_part
            si_exp += source.ion[iion].particles_decomposed.explicit_part

        # Coefficients for ion diffusion equation  in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff)

        a = vpr
        b = vprm
        c = 1.
        d = vpr*gm3*(diff+diff_hyper)
        e = vpr*gm3*(vconv+diff_hyper*ni0p/ni0) - B0dot/2./B0*rho_tor*vpr
        f = vpr*si_exp
        g = vpr*si_imp
        h = tau

        u = [0, 0]
        v = [0, 0]
        w = [0, 0]

        # Boundary conditions for ion diffusion equation in form:
        #
        #     V*Y' + U*Y =W
        #
        # On axis:
        #       dNi/drho_tor(rho_tor=0)=0:
        if(boundary_condition.type != 4):  # AF 11.Oct.2011
            v[0] = 1.
            u[0] = 0.
        else:
            if((diff[0]+diff_hyper) > 1.0e-6):
                #         V[0] = -DIFF[0]
                v[0] = -diff[0]-diff_hyper
            else:
                v[0] = -1.0e-6

                #       U[0] = VCONV[0]
                u[0] = vconv[0]+diff_hyper*dnim[0]/nim[0]
            # AF 11.Oct.2011
            w[0] = 0.

        # At the edge:
        #       FIXED Ni
        if(boundary_condition.type == 1):
            v = 0.
            u = 1.
            w = boundary_condition.value[1, 0]

        #       FIXED grad_Ni
        elif(boundary_condition.type == 2):
            v = 1.
            u = 0.
            w = -boundary_condition.value[1, 0]

        #       FIXED L_Ni
        elif(boundary_condition.type == 3):
            v = boundary_condition.value[1, 0]
            u = 1.
            w = 0.

        #       FIXED Flux_Ni
        elif(boundary_condition.type == 4):
            #        V[1] = -G(NRHO)*DIFF(NRHO)
            #        U[1] = G(NRHO)*VCONV(NRHO)
            v = -vpr[-1]*gm3[-1]*diff[-1]
            u = vpr[-1]*gm3[-1]*vconv[-1]
            w = boundary_condition.value[1, 0]

        #       Generic boundary condition
        elif(boundary_condition.type == 5):
            v, u, w = boundary_condition.value[1]

        # Density equation is not solved:
        if(boundary_condition.type == 0):
            flag = 0

            a[-1] = 1.0
            b[-1] = 1.0
            c[-1] = 1.0
            d[-1] = 0.0
            e[-1] = 0.0
            f[-1] = 0.0
            g[-1] = 0.0

            v = 0.0
            u = 1.0
            w = ni0[-1]

        # Solution of ion density diffusion equation:
        try:
            sol = self.solve_bvp(rho_tor, ni0, ni0p,
                                 tau,
                                 TransportSolver.COEFF(a, b, c, d, e, f, g),
                                 TransportSolver.BCCOEFF(u, v, w)
                                 )
        except RuntimeError as error:
            raise RuntimeError(f"Fail to solve ion density transport equation! \n {error} ")
        else:
            ni1 = sol.y[0]
            ni1p = sol.yp[0]

        # for irho_tor in range(rho_tor.shape[0]):
            #        NI(IRHO)    = Y(IRHO)
            #        DNI(IRHO)   = DY(IRHO)
            #        IF (RHO(IRHO) #= 0.E0_R8) :
            #           FUN1(IRHO)  = 1.e0_R8/RHO(IRHO)*(VPR(IRHO)*SI_EXP(IRHO)
            #                +VPRM(IRHO)*NIM(IRHO)/TAU
            #                -NI(IRHO)*VPR(IRHO)*(1.e0_R8/TAU+SI_IMP(IRHO)))
            #        else:
            #           FUN1(IRHO) = 1.e0_R8*(SI_EXP(IRHO)+NIM(IRHO)/TAU-NI(IRHO)
            #                *(1.e0_R8/TAU+SI_IMP(IRHO)))
            #

        fun1 = vpr*si_exp + vprm*ni0*inv_tau - ni0*vpr*(1.*inv_tau+si_imp)

        intfun1 = integral(fun1, rho_tor)

        local_fun1_s4 = (si_exp[0] + ni0[0]*inv_tau - ni0[0]*(1.*inv_tau+si_imp[0]))/gm3[0]  # stripping G1
        # stripped integral for the axis - from the INTEGR2 routine, first point only
        local_intfun1_s4 = local_fun1_s4*rho_tor[0]/2.0

        int_source = intfun1 + B0dot/2./B0*rho_tor*vpr*ni0
        flux = vpr*gm3 * (ni1*vconv - ni1p*diff)

        # Contribution to ion energy transport:
        # flux_ni_conv = 0.

        # for imodel in range(nmodel):
        #     flux_ni_conv = flux_ni_conv + c1[imodel]*vpr*gm3 * (y*vconv_mod[imodel] - yp*diff_mod[imodel])

        # # If equation is not solved, flux is determined
        # #     by the integral of kwargs and transport coefficients
        # #     are updated with effective values:
        # if(boundary_condition.type == 0):
        #     diff = 1.e-6
        #     flux = int_source
        #     flux_ni_conv = 1.5*int_source
        #     if((vpr*gm3 != 0.0) and (yp != 0.0)):  # FIXME
        #         diff = - flux / yp / (vpr*gm3)
        #     if (abs(diff) >= nine_diff_limit):
        #         diff = sign(nine_diff_limit, diff)
        #         vconv = 0.0
        #     if(diff <= 1.e-6 .AND. vpr*gm3 != 0):
        #         diff = 1.e-6
        #         vconv = (flux / (max(abs(vpr), 1.e-6)*gm3) + dy*diff) / max(abs(y), 1.e-6)

        # Return new ion density and flux profiles to the work flow:
        core_profiles_next.ion[iion].density = ni0
        core_profiles_next.ion[iion].ddensity_rho_tor = ni0p
        core_profiles_next.ion[iion].transport.d = diff
        core_profiles_next.ion[iion].transport.v = vconv
        core_profiles_next.ion[iion].transport.flux = flux
        # core_profiles_next.ion[iion].transport.flux_conv = flux_ni_conv
        core_profiles_next.ion[iion].source = si_exp - si_imp * ni0
        core_profiles_next.ion[iion].int_source = integral(ni0*vpr, rho_tor)

    def ion_density(self,
                    core_profiles_prev,
                    core_profiles_next,
                    *args,
                    equilibrium=None,
                    core_transport=None,
                    core_sources=None,
                    boundary_condition=None,
                    hyper_diff=[0, 0],
                    enable_quasi_neutrality=True,
                    **kwargs):
        r"""
            Note:
                .. math:: \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime}n_{i}\right)+\
                            \frac{\partial}{\partial\rho}\Gamma_{i}=V^{\prime}\left(S_{i,exp}-S_{i,imp}\cdot n_{i}\right)
                    :label: transport_ion_density

                .. math::
                    \Gamma_{s}\equiv-D_{i}\cdot\frac{\partial n_{i}}{\partial\rho}+v_{i}^{pinch}\cdot n_{i}
                    :label: transport_ion_density_gamma
        """

        for iion, ion in enumerate(core_profiles_prev.profiles_1d.ion):
            self.ion_density_one(self, iion,
                                 core_profiles_next,
                                 core_profiles_prev,
                                 equilibrium=equilibrium,
                                 core_transport=core_transport,
                                 core_sources=core_sources,
                                 boundary_condition=boundary_condition,
                                 **kwargs
                                 )

        core_profiles_next.profiles_1d.electrons.density = core_profiles_next.profiles_1d.n_i_total

        # core_profiles_next.profiles_1d.electrons.density_thermal =
        # core_profiles_next.profiles_1d.electrons.density -
        # core_profiles_next.profiles_1d.electrons.density_fast

        if enable_quasi_neutrality:
            ele_transport = core_transport[-1].profiles_1d.electrons
            ele_transport.particles.flux[:] = 0.0
            ele_transport.energy.flux[:] = 0.0

            for trans in core_transport:
                for ion in enumerate(trans.profiles_1d.ion):
                    ele_transport.particles.flux += ion.z_ion * ion.particles.flux
                    ele_transport.energy.flux += ion.z_ion * ion.energy.flux

    def temperatures(self,
                     core_profiles_prev,
                     core_profiles_next,
                     *args,
                     equilibrium=None,
                     collisions=None,
                     core_transport=None,
                     core_sources=None,
                     boundary_conditions=None,
                     hyper_diff=[0, 0],
                     **kwargs):
        r"""heat transport equations

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

        hyper_diff_exp = hyper_diff[0]
        hyper_diff_imp = hyper_diff[1]

        # time step                                         [s]
        tau = core_profiles_next.time - core_profiles_prev.time

        if abs(tau) < EPSILON:
            inv_tau = 0.0
        else:
            inv_tau = 1.0/tau

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_next.vacuum_toroidal_field.r0

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_next.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_next.vacuum_toroidal_field.b0

        # Grid
        # $rho_tor$ not  normalised minor radius                [m]
        rho_tor = core_profiles_next.profiles_1d.grid.rho_tor
        rho_tor_boundary = rho_tor[-1]

        # $rho_tor_{norm}$ normalised minor radius                [-]
        rho_tor_norm = core_profiles_next.profiles_1d.grid.rho_tor_norm

        k_rho = ((core_profiles_next.profiles_1d.grid.rho_tor[-1] -
                  core_profiles_prev.profiles_1d.grid.rho_tor[-1]) /
                 (core_profiles_next.profiles_1d.grid.rho_tor[-1] +
                  core_profiles_prev.profiles_1d.grid.rho_tor[-1]))*inv_tau
        k_b0 = ((core_profiles_next.vacuum_toroidal_field.b0 -
                 core_profiles_prev.vacuum_toroidal_field.b0) /
                (core_profiles_next.vacuum_toroidal_field.b0 +
                 core_profiles_prev.vacuum_toroidal_field.b0))*inv_tau
        k_phi = k_rho + k_b0

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = equilibrium.profiles_1d.mapping("rho_tor_norm", "fpol")(rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = equilibrium.profiles_1d.mapping("rho_tor_norm", "dvolume_drho_tor")(rho_tor_norm)
        vprm = vpr  # profiles_prev.profiles_1d.dvolume_dpsi
        dvpr = rho_tor/vpr * equilibrium.profiles_1d.dvolume_drho_tor.derivative*equilibrium.profiles_1d.dpsi_drho_tor

        gm3 = equilibrium.profiles_1d.mapping("rho_tor_norm", "gm3")(rho_tor_norm)

        u0, v0, w0 = (0, 1, 0)
        u1, v1, w1 = (0, 1, 0)

        spec_list = ["electrons"] + [["ion"][iion] for iion, ion_prev in enumerate(core_profiles_prev.profiles_1d.ion)]

        logger.debug(spec_list)

        for spec in spec_list:

            """
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
                 u[iion][0] = (vconv_ti[0]+diff_hyper*ti0p[0]/ti0[0])*ni[0]+local_flux_ni_conv_s4[iion]
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
                 u[iion][1] = vpr[-1]*gm3[-1]*vconv_ti[-1]*ni[-1]+flux_ni[-1]
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

            diff = core_profiles_next.profiles_1d._create(0.0, name="diff")
            vconv = core_profiles_next.profiles_1d._create(0.0, name="vconv")

            for trans in core_transport:
                d = trans.profiles_1d[spec].energy.d
                if isinstance(d, np.ndarray) or type(d) in (int, float):
                    diff += d

                v = trans.profiles_1d[spec].energy.v
                if isinstance(v, np.ndarray) or type(v) in (int, float):
                    vconv += v

            diff_hyper = hyper_diff_exp + hyper_diff_imp*max(diff)

            qs_exp = core_profiles_next.profiles_1d._create(0.0, name="se_exp")
            qs_imp = core_profiles_next.profiles_1d._create(0.0, name="se_imp")

            # FIXME: collisions
            # qs_exp += qei + qzi + qgi
            # qi_imp += vei + vzi

            for src in core_sources:
                # si = src.profiles_1d.electrons.particles_decomposed.implicit_part
                # if isinstance(si, np.ndarray) or type(si) in (int, float):
                #     se_imp += si

                # se = src.profiles_1d.electrons.particles_decomposed.explicit_part
                # if isinstance(se, np.ndarray) or type(se) in (int, float):
                #     se_exp += se
                se = src.profiles_1d[spec].energy
                if isinstance(se, np.ndarray) or type(se) in (int, float):
                    se_exp += se

            sp_prev = core_profiles_prev.profiles_1d[spec]
            sp_next = core_profiles_next.profiles_1d[spec]

            ts_prev = sp_prev.temperature

            tsp_prev = sp_prev.temperature_prime

            if tsp_prev is None:
                tsp_prev = Profile(rho_tor_norm, ts_prev).derivative

            ns_next = sp_next.density

            ns_prev = sp_prev.density

            ns_flux_next = sp_next.density_flux

            a = (3/2) * vpr * ns_next

            b = (3/2) * (vprm**(5/3)/vpr**(2/3)) * ns_prev

            c = rho_tor_boundary

            d = vpr*gm3 / rho_tor_boundary * ns_next * (diff + diff_hyper)

            e = vpr*gm3 * ns_next * (vconv + diff_hyper*tsp_prev/ts_prev) + \
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
                sp_next.temperature = Profile(solution.x, solution.y[0])
                sp_next.temperature_prime = Profile(solution.x, solution.y0[0])
                sp_next.temperature_flux = Profile(solution.x, solution.y[1])

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
            vconv = transport["VCONV_VTOR"][iion]

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
            e = (vpr*gm3*ni*vconv + flux_ni - B0dot/2./B0*rho_tor*ni*vpr) * gm8*mion
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

                u[0] = vconv[0]*ni[0] + local_flux_ni_s4[iion]
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
                u = vpr[-1]*gm3[-1]*gm8[-1]*vconv[-1] * \
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
            vconv = transport["VCONV_VTOR"][iion]
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
                * (y*vconv - dy*diff)

            flux_mtor = flux_mtor_conv + flux_mtor_cond

            int_source = intfun1 + vpr*gm8*B0dot/2./B0*rho_tor*mion*ni*y

            # if equation is not solved, conductive component of electron heat flux is determined from the integral of kwargs
            if(vtor_bnd_type[1] == 0):

                diff = 1.e-6
                flux_mtor = int_source

                flux_mtor_cond = int_source - gm8*mion*flux_ni*y

                if((vpr*gm3*gm8*mion*ni != 0.0) and (dy != 0.0)):
                    diff = - flux_mtor_cond / dy / (vpr*gm3*gm8*mion*ni)
                    vconv = 0.0
                if(diff <= 1.e-6):
                    diff = 1.e-6
                    vconv = (flux_mtor_cond / (max(abs(vpr), 1.e-6)*gm3*gm8
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
                profiles["VCONV_VTOR"][iion] = vconv
                profiles["FLUX_MTOR"][iion] = flux_mtor
                profiles["FLUX_MTOR_CONV"][iion] = flux_mtor_conv
                profiles["FLUX_MTOR_COND"][iion] = flux_mtor_cond
                profiles["SOURCE_MTOR"][iion] = ui_exp + uzi - (ui_imp + wzi) * vtor
                profiles["INT_SOURCE_MTOR"][iion] = int_source

            fun1 = profiles["SOURCE_MTOR"][iion]*vpr
            intfun1 = integral(fun1, rho_tor)
            profiles["INT_SOURCE_MTOR"][iion] = intfun1

        for irho_tor in range(rho_tor.shape[0]):
            profiles["MTOR_TOT"] = mtor_tot
            profiles["FLUX_MTOR_TOT"] = flux_mtor_tot


# while time < end_time
#     repeat
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

#         begin parallel
#             combine sources
#             combine transport coefficients
#         end parallel

#         calculate new profile

#     until converged

#     update time and time_step
# terminate
