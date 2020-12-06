"""

"""

import collections
import copy
from functools import cached_property

import numpy as np
import scipy
from scipy import constants
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from spdm.util.Profiles import Profile

from fytok.Equilibrium import Equilibrium

from fytok.CoreProfiles import CoreProfiles
from fytok.CoreSources import CoreSources
from fytok.CoreTransport import CoreTransport

from fytok.EdgeProfiles import EdgeProfiles
from fytok.EdgeSources import EdgeSources
from fytok.EdgeTransport import EdgeTransport

from fytok.Misc import Identifier

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
                 enable_quasi_neutrality=True,
                 solver="electron",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__['_cache'] = cache
        self._tokamak = tokamak
        self._enable_quasi_neutrality = enable_quasi_neutrality
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

        assert(core_profiles_prev.profiles_1d.grid.rho_tor_norm.shape ==
               core_profiles_next.profiles_1d.grid.rho_tor_norm.shape)

        self.current(core_profiles_prev,
                     core_profiles_next,
                     equilibrium=equilibrium,
                     core_sources=core_sources,
                     core_transport=core_transport,
                     boundary_condition=boundary_condition.current,
                     ** kwargs)

        if self._solver == "electron":
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

    COEFF = collections.namedtuple("coeff", "a b c d e f ")
    BCCOEFF = collections.namedtuple("bc", "u v w")

    def solve_general_form(self, x, y0, yp0, inv_tau, coeff,  bc, **kwargs):
        r"""solve standard form
            Args:
                x      : :math:`\rho_{tor,norm}`
                y0     : :math:`y(t-1)`
                yp0    : :math:`y^{\prime}(t-1)=\left.\frac{\partial Y}{\partial x}\right|_{t-1}`
                tau    : :math:`\tau`  time step
                coeff  : coefficients for `bvp` solver,

                        .. math::
                            \frac{a\cdot y-b\cdot y^{-1}}{\tau}=\frac{d}{dx}\left(c\cdot\frac{dy}{dx}+d\cdot y\right)+e\cdot y+f \\
                            :label: bvp_eq

                        where   :math:`Y=[y,y^{\prime}]` is the function, :math:`a,b,c,d,e,f,g` are function of :math:`x` , and

                        .. math::
                             \lim_{\tau\rightarrow0}\frac{a\cdot y-b\cdot y^{-1}}{\tau}=0,

                bc     : boundary condition ,  :math:`u,v,w`

                         .. math::
                            u\cdot y+ v\cdot\left.y^{\prime}\right|_{x=bnd} =w
                            :label: bvp_eq_bc


            Returns:
                (np.ndarra,np.ndarray)  : :math:`y(t)` , :math:`y^{\prime}(t)`

            Note:

                BVP Problem of :math:`Y=[y,y^{\prime}]`

                .. math::
                   \begin{cases}
                    \frac{d y}{d x} & =y^{\prime}\\
                    \frac{d y^{\prime}}{d x} & =A\cdot y^{\prime}+B\cdot y+C
                    \end{cases}

                where

                .. math::
                    A	&=\left(-\frac{d}{dx}c-d\right) \\
                    B	&=-\frac{d}{dx}d+\left(\frac{a}{\tau}-e\right) \\
                    C	&=-\frac{b}{\tau} \cdot y^{-1} -f

                solved by  scipy.integrate.solve_bvp
        """

        a, b, c, d, e, f, g = coeff

        gamma0 = -yp0*d+y0*e

        D = UnivariateSpline(x, d)
        E = UnivariateSpline(x, e)
        F = UnivariateSpline(x, (f + b*inv_tau*y0)*c)
        G = UnivariateSpline(x, (g + a*inv_tau)*c)

        def fun(x, Y):
            y, gamma = Y
            dy = -(gamma-E(x)*y)/D(x)
            dgamma = F(x) - G(x)*y
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

            y0, gamma0 = Ya
            y1, gamma1 = Yb

            return (u0 * y0 - v0 * gamma0 - w0,
                    u1 * y1 - v1 * gamma1 - w1)

        return scipy.integrate.solve_bvp(fun, bc_func, x, np.vstack((y0, gamma0)), **kwargs)

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
                core_profiles_next　:  profiles at :math:`t`

                equilibrium         : Equilibrium
                transports          : CoreTransport
                sources             : CoreSources
                boundary_condition  :

            .. math ::  \sigma_{\parallel}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho} \right) \psi= \
                        \frac{F^{2}}{\mu_{0}B_{0}\rho}\frac{\partial}{\partial\rho}\left[\frac{V^{\prime}}{4\pi^{2}}\left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle \
                        \frac{1}{F}\frac{\partial\psi}{\partial\rho}\right]-\frac{V^{\prime}}{2\pi\rho}\left(j_{ni,exp}+j_{ni,imp}\psi\right)
                :label: transport_current

            Note:
                if :math:`\psi` is not solved, then

                ..  math ::  \psi =\int_{0}^{\rho}\frac{2\pi B_{0}}{q}\rho d\rho

        """

        # -----------------------------------------------------------
        # time step                                         [s]
        tau = core_profiles_next.time - core_profiles_prev.time

        if abs(tau) < EPSILON:
            inv_tau = 0.0
        else:
            inv_tau = 1.0/tau

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_next.vacuum_toroidal_field.b0

        # $B_0^-$ previous time steps$B_0$,                 [T]
        B0m = core_profiles_prev.vacuum_toroidal_field.b0

        # $\dot{B}_{0}$ time derivative or $B_0$,           [T/s]
        B0dot = (B0 - B0m)*inv_tau

        # -----------------------------------------------------------
        # Grid
        # $rho_tor$ not  normalised minor radius                [m]
        rho_tor = core_profiles_next.profiles_1d.grid.rho_tor

        rho_tor_norm = core_profiles_next.profiles_1d.grid.rho_tor_norm
        rho_tor_boundary = core_profiles_next.profiles_1d.grid.rho_tor_boundary
        # $rho_tor_{norm}$ normalised minor radius                [-]
        # rho_tor_norm = rho_tor/rho_tor_boundary
        # rho_tor_norm = core_profiles_next.profiles_1d.grid.rho_tor_norm

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = equilibrium.profiles_1d.mapping("rho_tor_norm", "fpol")(rho_tor_norm)
        # fprime = core_profiles_next.profiles_1d.derivative("fpol")

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = equilibrium.profiles_1d.mapping("rho_tor_norm", "dvolume_drho_tor")(rho_tor_norm)

        # $gm2 \euqiv \left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $  [m^-2]
        gm2 = equilibrium.profiles_1d.mapping("rho_tor_norm", "gm2")(rho_tor_norm)

        # $\Psi$ flux function from current                 [Wb]
        psi0 = equilibrium.profiles_1d.mapping("rho_tor_norm", "psi")(rho_tor_norm)
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        psi0_prime = equilibrium.profiles_1d.derivative_func("psi", "rho_tor_norm")(rho_tor_norm)

        # $q$ safety factor                                 [-]
        # qsf = core_profiles_next.profiles_1d.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        conductivity_parallel = core_profiles_next.profiles_1d._create(0.0, name="conductivity_parallel")

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

        # Coefficients for current diffusion equation in form:
        #   (a*y-b*y(t-1))/tau = d(-d*y' + e*y) + g*y+ f
        rho_tor[0] = 0.01

        a = conductivity_parallel

        b = conductivity_parallel

        c = (constants.mu_0*B0 * rho_tor*rho_tor_boundary)/fpol**2

        c[0] = 2*c[1]-c[2]

        d = vpr*gm2 / (4.0*(constants.pi**2)*fpol*rho_tor_boundary)

        e = - (constants.mu_0 * B0dot/2.0)*conductivity_parallel * (rho_tor/fpol)**2

        f = - vpr * j_ni_exp/(2.0*constants.pi)

        g = vpr * j_ni_imp/(2.0*constants.pi)

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
        if boundary_condition.identifier.index == 1:  # poloidal flux
            u = 1.0
            v = 0.0
            w = boundary_condition.value[0]
        elif boundary_condition.identifier.index == 2:  # ip
            u = 0.0
            v = 1.0
            w = - constants.mu_0/(vpr[-1]*gm3[-1])*(4.0*constants.pi**2)*boundary_condition.value[0]
        elif boundary_condition.identifier.index == 3:  # loop voltage
            u = 1.0
            v = 0.0
            w = tau*boundary_condition.value[0]+psi0[-1]
        elif boundary_condition.identifier.index == 5:  # generic boundary condition  y expressed as a1y'+a2y=a3.
            v, u, w = boundary_condition.value
        else:
            # Current equation is not solved:
            #  Interpretative value of safety factor should be given
            # if any(qsf != 0.0):  # FIXME
            # dy = 2.0*constants.pi*B0*rho_tor/qsf

            # a[-1] = 1.0*inv_tau
            # b[-1] = 1.0*inv_tau
            # # c[-1] = 1.0
            # d[-1] = 0.0
            # e[-1] = 0.0
            # f[-1] = 0.0

            u = 1.0
            v = 0.0
            w = psi0[-1]  # core_profiles_next.profiles_1d.integral(2.0*constants.pi*B0/qsf, 0, 1.0)

        try:

            sol = self.solve_general_form(rho_tor_norm,
                                          psi0, psi0_prime,
                                          inv_tau,
                                          (a, b, c, d, e, f, g),
                                          ((0, 1, 0), (u, v, w))
                                          )
        except Exception as error:
            logger.error(f"Unable to solve the 'current' transport equation! \n {error} ")
        else:
            if sol.success:
                logger.debug(f"Solve transport equations: Current : {sol.message}")
        finally:
            core_profiles_next.profiles_1d.psi = Profile(sol.x, sol.y[0])
            core_profiles_next.profiles_1d.psi_prime = Profile(sol.x, sol.y[1])

        core_profiles_next.profiles_1d.psi0 = psi0
        core_profiles_next.profiles_1d.psi0_prime = psi0_prime

        core_profiles_next.profiles_1d.j_total = j_ni_exp

        # core_profiles_next.profiles_1d.A = A
        # core_profiles_next.profiles_1d.B = B
        # core_profiles_next.profiles_1d.C = C

        # core_profiles_next.profiles_1d.I = j_ni_exp*core_profiles_next.profiles_1d.volume

        # New magnetic flux function and current density:
        # intfun7 = core_profiles_next.profiles_1d.integral(c*((a*psi1 - b*psi0)*inv_tau  - f))
        # # intfun7 = [func7.integral(0, r) for r in rho_tor_norm]

        # dy = (intfun7 + e*psi1) / d

        # # fun4 = fun2*dy
        # # fun5 = fun2*dy*R0*B0/fpol

        # dfun4 = UnivariateSpline(rho_tor, fun2*dy).derivative()(rho_tor)  # Derivation of function FUN4
        # dfun5 = UnivariateSpline(rho_tor, fun2*dy*R0*B0/fpol).derivative()(rho_tor)   # Derivation of function FUN5

        # # New profiles of plasma parameters obtained
        # #     from current diffusion equation:

        # core_profiles_next.profiles_1d.q = 2.0*constants.pi*B0*rho_tor/psi_prime1

        # current density, toroidal,                        [A/m^2]
        # j_tor = - 2.0*constants.pi*R0/constants.mu_0/vpr * dfun4
        # core_profiles_next.profiles_1d.j_tor = j_tor
        # j_par = - 2.0*constants.pi/R0/constants.mu_0/vpr * (fpol/B0)**2*dfun5
        # core_profiles_next.profiles_1d.current_parallel_inside = j_par

        # # $E_\parallel$  parallel electric field,,          [V/m]
        # core_profiles_next.profiles_1d.e_field.parallel = (j_par - j_ni_exp - j_ni_imp*psi1) / conductivity_parallel

        # # Total Ohmic currents                              [A]
        # # fun7 = vpr * j_par / 2.0 / constants.pi * B0 / fpol**2
        # core_profiles_next.profiles_1d.j_ohmic = fpol[-1]*core_profiles_next.profiles_1d.integral(
        #     vpr * j_par / 2.0 / constants.pi * B0 / fpol**2, 0, 1)

        # core_profiles_next.profiles_1d.j_tor = j_tor

        # # Total non-inductive currents                       [A]
        # core_profiles_next.profiles_1d.j_non_inductive = fpol[-1] * core_profiles_next.profiles_1d.integral(
        #     vpr * (j_ni_exp + j_ni_imp * psi1) / (2.0 * constants.pi) * B0 / fpol**2)

        return True

    def electron_density(self,
                         core_profiles_prev,
                         core_profiles_next,
                         *args,
                         equilibrium=None,
                         core_transport=None,
                         core_sources=None,
                         boundary_condition=None,
                         hyper_diff=[0.00, 1.0],
                         **kwargs):

        hyper_diff_exp = hyper_diff[0]
        hyper_diff_imp = hyper_diff[1]

        # -----------------------------------------------------------
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

        # $B_0^-$ previous time steps$B_0$,                 [T]
        B0m = core_profiles_prev.vacuum_toroidal_field.b0

        # $\dot{B}_{0}$ time derivative or $B_0$,           [T/s]
        B0dot = (B0 - B0m)*inv_tau

        # -----------------------------------------------------------
        # Grid
        # $rho_tor$ not  normalised minor radius                [m]
        rho_tor = core_profiles_next.profiles_1d.grid.rho_tor
        rho_tor_boundary = rho_tor[-1]

        # $rho_tor_{norm}$ normalised minor radius                [-]
        rho_tor_norm = core_profiles_next.profiles_1d.grid.rho_tor_norm

        # # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        # psi_prime = core_profiles_prev.profiles_1d.grid.psi_prime
        # # normalized psi  [0,1]                            [-]
        # psi_norm = core_profiles_prev.profiles_1d.grid.psi_norm

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = equilibrium.profiles_1d.mapping("rho_tor_norm", "fpol")(rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = equilibrium.profiles_1d.mapping("rho_tor_norm", "dvolume_dpsi")(rho_tor_norm)
        vprm = vpr  # core_profiles_prev.profiles_1d.dvolume_dpsi

        gm3 = equilibrium.profiles_1d.mapping("rho_tor_norm", "gm3")(rho_tor_norm)

        # Set up local variables for particular ion type:
        ne0 = core_profiles_prev.profiles_1d.electrons.density

        ne0_prime = core_profiles_prev.profiles_1d.electrons.density.derivative

        diff = core_profiles_next.profiles_1d._create(0.0, name="diff")
        vconv = core_profiles_next.profiles_1d._create(0.0, name="vconv")

        for trans in core_transport:
            d = trans.profiles_1d.electrons.particles.d
            if isinstance(d, np.ndarray) or type(d) in (int, float):
                diff += d

            v = trans.profiles_1d.electrons.particles.v
            if isinstance(v, np.ndarray) or type(v) in (int, float):
                vconv += v

        se_exp = core_profiles_next.profiles_1d._create(0.0, name="se_exp")
        se_imp = core_profiles_next.profiles_1d._create(0.0, name="se_imp")

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
        # Coefficients for electron diffusion equation in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y
        diff_hyper = hyper_diff_exp + hyper_diff_imp * max(diff)

        a = vpr
        b = vprm
        c = rho_tor_boundary
        d = vpr*gm3*(diff + diff_hyper)/rho_tor_boundary
        e = vpr*gm3*(vconv + diff_hyper*ne0_prime/ne0/rho_tor_boundary) - B0dot/(2.0*B0)*rho_tor*vpr
        f = vpr*se_exp
        g = vpr*se_imp

        # Boundary conditions for electron diffusion equation in form:
        #
        #     V*Y' + U*Y =W
        #
        # On axis:
        #       dNi/drho_tor(rho_tor=0)=0:  - this is Ne, not Ni

        # if(solver_type != 4):
        #     v[0] = 1.
        #     u[0] = 0.
        # else:
        #       IF (DIFF[0]>1.0E-6) :
        # if((diff[0]+diff_hyper) > 1.0e-6):
        #     #         V[0] = -DIFF[0]
        #     v = -diff[0]-diff_hyper
        # else:
        #     v = -1.0e-6

        # #       U[0] = VCONV[0]
        #     u = vconv[0]+diff_hyper*ne0_prime[0]/ne0[0]
        # w = 0.

        # At the edge:
        #       FIXED Ne
        # if(boundary_condition.type == 1):
        #     v = 0.
        #     u = 1.
        #     w = boundary_condition[1, 0]
        # #       FIXED grad_Ne
        # elif(boundary_condition.type == 2):
        #     v = 1.
        #     u = 0.
        #     w = -boundary_condition[1, 0]

        # #       FIXED L_Ne
        # elif(boundary_condition.type == 3):
        #     v = boundary_condition[1, 0]
        #     u = 1.
        #     w = 0.

        # #       FIXED Flux_Ne
        # elif(boundary_condition.type == 4):
        #     v = -vpr[-1]*gm3[-1]*diff[-1]
        #     u = vpr[-1]*gm3[-1]*vconv[-1]
        #     w = boundary_condition[1, 0]

        # #       Generic boundary condition
        # elif(boundary_condition.type == 5):
        #     v = boundary_condition[1, 0]
        #     u = boundary_condition(2, 2)
        #     w = boundary_condition(2, 3)

        # # Density equation is not solved:
        # else:  # if(boundary_condition.type == 0):

        v = 0.0
        u = 1.0
        w = ne0[-1]
        y0 = ne0
        y0p = ne0_prime

        sol = self.solve_general_form(rho_tor_norm,
                                      y0, y0p,
                                      inv_tau,
                                      (a, b, c, d, e, f, g),
                                      ((0, 1, 0), (1, 0, ne0[-1])),
                                      tol=0.001, verbose=2, max_nodes=2500)

        logger.debug(
            f"Solve transport equations: Electron density: {'Done' if  sol.success else 'Failed' }  \n Error Message: {sol.message} ")

        if sol.success:
            core_profiles_next.profiles_1d.electrons.density = Profile(sol.x, sol.y[0])
            core_profiles_next.profiles_1d.electrons.density_prime = Profile(sol.x, sol.yp[0])
            core_profiles_next.profiles_1d.electrons.gamma = Profile(sol.x, sol.y[1])
            core_profiles_next.profiles_1d.electrons.dgamma = Profile(sol.x, sol.yp[1])

        core_profiles_next.profiles_1d.vpr = vpr
        core_profiles_next.profiles_1d.gm3 = gm3

        core_profiles_next.profiles_1d.electrons.density0 = ne0
        core_profiles_next.profiles_1d.electrons.density0_prime = ne0_prime
        core_profiles_next.profiles_1d.electrons.gamma0 = Profile(rho_tor_norm, -ne0_prime*d+c*ne0)

        core_profiles_next.profiles_1d.electrons.dgamma0 = Profile(rho_tor_norm, -ne0_prime*d+c*ne0).derivative

        # Profile(rho_tor_norm, -d*ne0_prime + e*ne0).derivative
        core_profiles_next.profiles_1d.electrons.se_exp0 = vpr * se_exp*c

        core_profiles_next.profiles_1d.a = a
        core_profiles_next.profiles_1d.b = b
        core_profiles_next.profiles_1d.c = c
        core_profiles_next.profiles_1d.d = d
        core_profiles_next.profiles_1d.e = e
        core_profiles_next.profiles_1d.f = f
        core_profiles_next.profiles_1d.g = g

        if self._enable_quasi_neutrality:
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

        # $B_0^-$ previous time steps$B_0$,                 [T]
        B0m = core_profiles_prev.vacuum_toroidal_field.b0

        # $\dot{B}_{0}$ time derivative or $B_0$,           [T/s]
        B0dot = (B0 - B0m)*inv_tau

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
            .. math:: \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime}n_{i}\right)+\
                        \frac{\partial}{\partial\rho}\Gamma_{i}=V^{\prime}\left(S_{i,exp}-S_{i,imp}\cdot n_{i}\right)
                :label: transport_ion_density
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
                     collisions=None,
                     core_transport=None,
                     core_sources=None,
                     boundary_conditions=None,
                     hyper_diff=[0, 0],
                     **kwargs):
        r"""heat transport equations
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

        ########################################
        # -----------------------------------------------------------
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

        # $B_0^-$ previous time steps$B_0$,                 [T]
        B0m = core_profiles_prev.vacuum_toroidal_field.b0

        # $\dot{B}_{0}$ time derivative or $B_0$,           [T/s]
        B0dot = (B0 - B0m)*inv_tau

        # -----------------------------------------------------------
        # Grid
        # $rho_tor$ not  normalised minor radius                [m]
        rho_tor = core_profiles_next.grid.rho_tor
        rho_tor_norm = core_profiles_next.grid.rho_tor_norm

        # $\Psi$ flux function from current                 [Wb]
        psi0 = core_profiles_next.grid.psi
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        psi_prime = core_profiles_next.grid.psi_prime

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = core_profiles_next.profiles_1d.interpolate("f")(rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = core_profiles_next.profiles_1d.dvolume_dpsi
        dvpr = core_profiles_next.profiles_1d.derivative("dvolume_dpsi")  # Derivation of V'

        # $\left\langle \left|\nabla\rho\right|^{2}\right\rangle $  [-]
        gm3 = core_profiles_next.profiles_1d.gm3

        # -----------------------------------------------------------
        # Profile

        # $q$ safety factor                                 [-]
        qsf = core_profiles_prev.profiles_1d.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        # conductivity_parallel = core_transport.profiles_1d.conductivity_parallel

        # Energy exchange terms due to collisions
        #     (defined from previous iteration):
        # self.plasma_collisions(core_profiles, collisions)

        vie = collisions.vie

        # -------------------------------------------------------
        #  ION HEAT TRANSPORT:
        for iion, ion in enumerate(core_profiles_prev.profiles_1d.ion):

            # Set equation to 'predictive' and all coefficients to zero:

            # Set up boundary conditions for particular ion type:
            # ti_bnd_type[iion] = [[0, 0, 0], profiles["TI_BND_TYPE"][iion]]
            # ti_bnd = [[0, 0, 0], profiles["TI_BND"][iion]]

            # Set up local variables for particular ion type:
            ti0 = ion.temperature
            ti0p = ion.dtemperature_drho_tor or derivative(ti0, rho_tor)
            ni0 = ion.density
            ni0p = ion.ddensity_drho_tor or derivative(ni0, rho_tor)

            ni1 = core_profiles_next.ion[iion].density
            ni1p = core_profiles_next.ion[iion].ddensity_drho_tor or derivative(ni1, rho_tor)

            flux_ti = transports.ion[iion].particles.FLUX_TI
            flux_ti_cond = transports.ion[iion].particles.FLUX_TI_COND
            flux_ti_conv = transports.ion[iion].particles.FLUX_TI_CONV
            flux_ni = transports.ion[iion].particles.FLUX_NI_CONV

            diff_ti = transports.ion[iion].particles.d
            vconv_ti = transports.ion[iion].particles.v
            qgi = transports.ion[iion].QGI

            qi_exp = transports.ion[iion].energy_decomposed.explicit_part
            qi_imp = transports.ion[iion].energy_decomposed.implicit_part

            vei = collisions.VEI[iion]
            qei = collisions.QEI[iion]
            vzi = collisions.VZI[iion]
            qzi = collisions.QZI[iion]

            vii = []

            for jion in enumerate(core_profiles_prev.profiles_1d.ion):
                vii[jion] = collisions.VII[iion, zion]

            # Coefficients for ion heat transport equation in form:
            #
            #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

            diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff_ti)

            a[iion] = 1.5*vpr*ni
            b[iion] = 1.5*vprm**fivethird/vpr**twothird*nim
            c[iion] = 1.0
            d[iion] = vpr*gm3*ni*(diff_ti+diff_hyper)
            e[iion] = vpr*gm3*ni*(vconv_ti+diff_hyper*ti0p/ti0) + flux_ni - 1.5*B0dot/2./B0*rho_tor*ni*vpr
            f[iion] = vpr * (qi_exp + qei + qzi + qgi)
            g[iion] = vpr*(qi_imp + vei + vzi) - B0dot/2./B0*rho_tor*ni*dvpr

            # Boundary conditions for ion heat transport equation in form:
            #     V*Y' + U*Y =W

            # On axis:
            #       dTi/drho_tor(rho_tor=0)=0
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

            # Defining coefficients for numerical solver:

            solver["CM1"][ndim, iion] = vie
            solver["CM1"][iion, ndim] = vie

            for zion in range(nion):
                solver["CM1"][iion, zion] = vii[zion]

        # -------------------------------------------------------
        #  ELECTRON HEAT TRANSPORT:

        # Set up local variables for electron heat transport equation:

        ne0 = core_profiles_prev.profiles_1d.electrons.density
        ne0_prime = core_profiles_prev.profiles_1d.electrons.density_prime

        te0 = core_profiles_prev.profiles_1d.electrons.temperature
        te0p = core_profiles_prev.profiles_1d.electrons.temperature_prime

        diff_te = core_transport.profiles_1d.electrons.particles.d
        vconv_te = core_transport.profiles_1d.electrons.particles.v
        flux_ne = core_transport.profiles_1d.electrons.particles.flux

        qgi = 0.
        for iion in range(nion):
            qgi = qgi + transport["QGI"][iion]

        qe_exp = kwargs["QOH"] / itm_ev
        qe_imp = 0.

        qe_exp = qe_exp + core_sources.profiles_1d.electrons.energy_decomposed.explicit_part
        qe_imp = qe_imp + core_sources.profiles_1d.electrons.energy_decomposed.implicit_part

        qie = collisions["QIE"]
        vie = collisions["VIE"]

        # Set up boundary conditions for electron heat transport equation:
        te_bnd_type = profiles["TE_BND_TYPE"]
        te_bnd = profiles["TE_BND"]

        # Coefficients for electron heat transport equation in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        diff_hyper = hyper_diff_exp + hyper_diff_imp*max(diff_te)

        a = 1.5*vpr*ne
        # DPC temporary "fix" for NaNs

        b = 1.5*vprm**fivethird/vpr**twothird*ne0
        if vpr <= 0.0:
            b[0] = 0.0

        # DPC end of temporary fix
        c = 1.
        d = vpr * gm3 * ne * (diff_te+diff_hyper)
        e = vpr * gm3 * ne * (vconv_te+diff_hyper*dtem / tem) + flux_ne - 1.5*B0dot/2./B0*rho_tor*ne*vpr
        f = vpr * (qe_exp + qie - qgi)
        g = vpr * (qe_imp + vie) - B0dot/2./B0*rho_tor*ne*dvpr

        h = tau

        # Boundary conditions for electron heat
        #     transport equation in form:
        #
        #     V*Y' + U*Y =W

        # On axis
        #     dTe/drho_tor(rho_tor=0)=0:
        if(solver_type != 4):
            v[0] = 1.
            u[0] = 0.
        else:  # - Zero flux instead of zero gradient at the axis for solver 4
            #    IF (DIFF_TE[0]>1.0E-6) : #AF 19.Mar.2012 - To avoid problems with the axis boundary condition
            # AF 19.Mar.2012 - To avoid problems with the axis boundary condition
            if((diff_te[0]+diff_hyper) > 1.0e-6):
                #      V[0] = -DIFF_TE[0]*NE[0]
                v[0] = -(diff_te[0]+diff_hyper)*ne[0]
            else:
                v[0] = -1.0e-6*ne[0]

                #    U[0] = VCONV_TE[0]*NE[0]+LOCAL_FLUX_NE_CONV_S4
                u[0] = (vconv_te[0]+diff_hyper*dtem[0]/tem[0])*ne[0]+local_flux_ne_conv_s4
                w[0] = 0.

        # At the edge:

        #     FIXED Te
        if(te_bnd_type[1] == 1):
            v = 0.
            u = 1.
            w = te_bnd[1, 0]

        #     FIXED grad_Te
        if(te_bnd_type[1] == 2):
            v = 1.
            u = 0.
            w = -te_bnd[1, 0]

        #     FIXED L_Te
        if(te_bnd_type[1] == 3):
            v = te_bnd[1, 0]
            u = 1.
            w = 0.

        #    FIXED Flux_Te
        if(te_bnd_type[1] == 4):
            #     V[1] = -G(NRHO)*DIFF(NRHO)*NE(NRHO)
            #     U[1] = G(NRHO)*VCONV(NRHO)*NE(NRHO)+FLUX_NE(NRHO)
            v = -vpr[-1]*gm3[-1]*diff_te[-1]*ne[-1]
            u = vpr[-1]*gm3[-1]*vconv_te[-1]*ne[-1]+flux_ne[-1]
            w = te_bnd[1, 0]

        #    Generic boundary condition
        if(te_bnd_type[1] == 5):
            v = te_bnd[1, 0]
            u = te_bnd(2, 2)
            w = te_bnd(2, 3)

        # Temperature equation is not solved:
        if(te_bnd_type[1] == 0):

            dy = derivative(y, rho_tor)  # temperature gradient

            flag = 0

            a.fill(1.0)
            b.fill(1.0)
            c.fill(1.0)
            d.fill(0.0)
            e.fill(0.0)
            f.fill(0.0)
            g.fill(0.0)

            v = 0.0
            u = 1.0
            w = y[-1]

        # Defining coefficients for numerical solver:
        solver["TYPE"] = solver_type
        solver["EQ_FLAG"][ndim] = flag
        solver["NDIM"] = ndim
        solver["NRHO"] = rho_tor.shape[0]
        solver["AMIX"] = amix
        solver["DERIVATIVE_FLAG"][0] = 0

        solver["RHO"] = rho_tor

        solver["Y"][ndim] = y
        solver["DY"][ndim] = dy
        solver["YM"][ndim] = ym

        solver["A"][ndim] = a
        solver["B"][ndim] = b
        solver["C"][ndim] = c
        solver["D"][ndim] = d
        solver["E"][ndim] = e
        solver["F"][ndim] = f
        solver["G"][ndim] = g

        solver["CM1"][ndim, ndim].fill(0.0)

        solver["H"] = h

        solver["V"][ndim, 1] = v
        solver["U"][ndim, 1] = u
        solver["W"][ndim, 1] = w
        solver["V"][ndim, 2] = v[2]
        solver["U"][ndim, 2] = u[2]
        solver["W"][ndim, 2] = w[2]

        # Solution of heat transport equations:
        self.solve_eq(a, b, c, d, e, f, g, h)

        # dy check for nans in solution, return if true)
        if(any(np.isnan(solver["y"]))):
            raise RuntimeError('Error in the temperature equation: nans in the solution, stop')

        # New temperatures and heat fluxes:
        #-------------------------------------------------------#
        #     IONS:                                             #
        #-------------------------------------------------------#
        for iion in range(nion):
            ti_bnd_type[iion] = profiles["TI_BND_TYPE"][iion]
            ti_bnd = profiles["TI_BND"][iion]

            # dpc end

            ti = profiles["TI"][iion]
            ti0 = kwargs["TIM"][iion]
            ni = profiles["NI"][iion]
            nim = kwargs["NIM"][iion]
            flux_ti = profiles["FLUX_TI"][iion]
            flux_ti_cond = profiles["FLUX_TI_COND"][iion]
            flux_ti_conv = profiles["FLUX_TI_CONV"][iion]
            flux_ni = profiles["FLUX_NI_CONV"][iion]
            diff_ti = transport["DIFF_TI"][iion]
            vconv_ti = transport["VCONV_TI"][iion]
            qgi = transport["QGI"][iion]
            qi_exp = kwargs["QI_EXP"][iion]
            qi_imp = kwargs["QI_IMP"][iion]
            vei = collisions["VEI"][iion]
            qei = collisions["QEI"][iion]
            vzi = collisions["VZI"][iion]
            qzi = collisions["QZI"][iion]
            # dpc end

            y = solver["Y"](iion, irho_tor)
            dy = solver["DY"](iion, irho_tor)

            if(ti_bnd_type[iion] == 0):
                y = profiles["TI"][iion]
                dy = derivative(y, rho_tor)

            ti = y
            dti = dy
            if any(ti < 0.0):
                raise RuntimeError('Error in the temperature equation: on-axis ion temperature is negative, stop')

            else:
                # write(*, *) 'warning, temperature for ion ', iion, ' and irho_tor ', irho_tor, 'is negative, set it to the value at the previous irho_tor'
                ti[1:] = ti[0:-1]
                dti[1:] = dti(irho_tor-1)
                # end if
            # end if
            if(rho_tor != 0.):  # FIXME
                fun1 = vpr/rho_tor *                                           \
                    ((1.5*nim*ti0*inv_tau*(vprm/vpr)**fivethird
                      + qi_exp + qei + qzi + qgi)
                     - (1.5*ni*inv_tau + qi_imp + vei + vzi
                        - B0dot/2./B0*rho_tor*ni*dvpr) * y)
            else:
                fun1 = ((1.5*nim*ti0*inv_tau
                         + qi_exp + qei + qzi + qgi)
                        - (1.5*ni*inv_tau + qi_imp + vei + vzi
                           - B0dot/2./B0*rho_tor*ni*dvpr) * y)

            intfun1 = integral(fun1, rho_tor)  # Integral source

            for irho_tor in range(rho_tor.shape[0]):
                flux_ti_conv = y*flux_ni

                flux_ti_cond = vpr*gm3*ni                                     \
                    * (y*vconv_ti - dy*diff_ti)

                flux_ti = flux_ti_conv + flux_ti_cond

                int_source = intfun1 + y * 1.5*B0dot/2./B0*rho_tor*ni*vpr

                # If equation is not solved, total and conductive ion heat flux
                #     are determined from the integral of kwargs:
                if(ti_bnd_type(2, iion) == 0):

                    diff_ti = 1.e-6
                    flux_ti = int_source
                    flux_ti_cond = int_source - flux_ni*y

                    if((vpr*gm3 != 0.0)):
                        # dy limit also DY if less than sqrt(epsilon(1.0))
                        diff_ti = - flux_ti_cond / sign(max(abs(dy), sqrt(epsilon(1.0))), dy) / (vpr*gm3*ni)
                        # dy further limit diff_ti
                    if (abs(diff_ti) >= tite_diff_limit):
                        diff_ti = sign(tite_diff_limit, diff_ti)
                        vconv_ti = 0.0
                    if(diff_ti <= 1.e-6):
                        diff_ti = 1.e-6
                        vconv_ti = (flux_ti_cond / (max(abs(vpr), 1.e-6)*gm3*ni) + dy*diff_ti) / max(abs(y), 1.e-6)

                # Return ion temperature and flux:
                profiles["TI"][iion] = ti
                profiles["DTI"][iion] = dti
                profiles["DIFF_TI"][iion] = diff_ti
                profiles["VCONV_TI"][iion] = vconv_ti
                profiles["FLUX_TI_CONV"][iion] = flux_ti_conv
                profiles["FLUX_TI_COND"][iion] = flux_ti_cond
                profiles["FLUX_TI"][iion] = flux_ti
                profiles["SOURCE_TI"][iion] = qi_exp + qei + qzi + qgi - (qi_imp + vei + vzi) * ti
                profiles["INT_SOURCE_TI"][iion] = int_source
                profiles["QEI_OUT"] = profiles["QEI_OUT"]+qei

            fun1 = profiles["SOURCE_TI"][iion]*vpr
            intfun1 = integral(fun1, rho_tor)
            profiles["INT_SOURCE_TI"][iion] = intfun1

        #-------------------------------------------------------#
        #     ELECTRONS:                                        #
        #-------------------------------------------------------#

        qgi = 0.
        for iion in range(nion):
            qgi = qgi + transport["QGI"][iion]
        # dpc end

        y = solver["Y"][ndim]
        dy = solver["DY"][ndim]

        if(te_bnd_type[1] == 0):
            y = profiles["TE"]
            dy = derivative(y, rho_tor)

        te = y
        dte = dy

        if any(te < 0.0):
            raise RuntimeError('Error in the temperature equation: on-axis electron temperature is negative, stop')
        else:
            te[1:] = te[0:-1]
            dte[1:] = dte[0:-1]

        if(rho_tor != 0.):  # FIXME
            fun2 = vpr/rho_tor *                                         \
                (1.5*ne0*tem*inv_tau*(vprm/vpr)**fivethird
                 + qe_exp + qie - qgi
                 - y * (1.5*ne*inv_tau + qe_imp + vie
                        - B0dot/2./B0*rho_tor*ne*dvpr/vpr))
        else:
            fun2 = (1.5*ne0*tem*inv_tau
                    + qe_exp + qie - qgi
                    - y * (1.5*ne*inv_tau + qe_imp + vie
                           - B0dot/2./B0*ne*dvpr))

        intfun2 = integral(fun2, rho_tor)  # Integral source

        flux_te_conv = y*flux_ne

        flux_te_cond = vpr*gm3*ne * (y*vconv_te - dy*diff_te)

        flux_te = flux_te_conv + flux_te_cond

        int_source = intfun2 + y * 1.5*B0dot/2./B0*rho_tor*ne*vpr

        # If equation is not solved, conductive component of electron heat flux
        #     is determined from the integral of kwargs:
        if(te_bnd_type[1] == 0):

            diff_te = 1.e-6
            flux_te = int_source

            flux_te_cond = int_source - flux_ne*y

            if(vpr*gm3 != 0.0):  # FIXME !!
                # dy limit also DY if less than sqrt(epsilon(1.0))
                diff_te = - flux_te_cond / sign(max(abs(dy), sqrt(epsilon(1.0))), dy) / (vpr*gm3*ne)
                # dy further limit diff_ti
            if (abs(diff_te) >= tite_diff_limit):
                diff_te = sign(tite_diff_limit, diff_te)
                vconv_te = 0.0
            if(diff_te <= 1.e-6):
                diff_te = 1.e-6
                vconv_te = (flux_te_cond / (max(abs(vpr), 1.e-6)*gm3*ne) + dy*diff_te) / y

        # Return electron temperature and flux:
        profiles["TE"] = te
        profiles["DTE"] = dte
        profiles["DIFF_TE"] = diff_te
        profiles["VCONV_TE"] = vconv_te
        profiles["FLUX_TE"] = flux_te
        profiles["FLUX_TE_CONV"] = flux_te_conv
        profiles["FLUX_TE_COND"] = flux_te_cond
        profiles["SOURCE_TE"] = qe_exp + qie - qgi - (qe_imp + vie) * te
        profiles["INT_SOURCE_TE"] = int_source

        fun1 = profiles["SOURCE_TE"]*vpr
        intfun1 = integral(fun1, rho_tor)
        profiles["INT_SOURCE_TE"] = intfun1

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
