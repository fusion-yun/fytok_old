import collections
import copy

import numpy as np
import scipy
from scipy import constants
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.Interpolate import Interpolate1D, derivate, integral
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module

from .CoreProfiles import CoreProfiles

EPSILON = 1.0e-5


class Transport(AttributeTree):
    """
        Solve transport equations

        Schema:
            imas_dd:://transport_solver_numerics
        Refs:
            - Hinton/Hazeltine, Rev. Mod. Phys. vol. 48 (1976), pp.239-308
            - David P. Coster, Vincent Basiuk, Grigori Pereverzev, Denis Kalupin, Roman Zagórksi,
                Roman Stankiewicz, Philippe Huynh, and Fréd…,
              "The European Transport Solver", IEEE Transactions on Plasma Science 38, 9 PART 1 (2010), pp. 2085--2092.
            - G V Pereverzev, P N Yushmanov, and Eta, "ASTRA–Automated System for Transport Analysis in a Tokamak",
              Max-Planck-Institut für Plasmaphysik (1991), 147.

            $  \\rho\\equiv\\sqrt{\\frac{\\Phi}{\\pi B_{0}}} $
            $  gm2 \\euqiv \\left\\langle \\left|\\frac{\\nabla\\rho}{R}\\right|^{2}\\right\\rangle $
    """
    @staticmethod
    def __new__(cls,  config, *args, **kwargs):
        if cls is not Transport:
            return super(Transport, cls).__new__(cls)

        backend = config.engine

        if not backend:
            n_cls = cls
        else:
            plugin_name = f"{__package__}.plugins.transport.Plugin{backend}"

            n_cls = sp_find_module(plugin_name, fragment=f"Transport{backend}")

            if n_cls is None:
                raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#Transport{backend}")

        return object.__new__(n_cls)

    def __init__(self,  config, *args, tokamak={},  **kwargs):
        super().__init__(*args, **kwargs)

        self._tokamak = tokamak
        self.description = AttributeTree()
        self.description.name = "FyTok"
        self.description.primary_coordinate.name = "rho_tor_norm"    # or "rho_tor"
        self.description.primary_coordinate.index = "1"              # or 2

    def update(self, core_profiles,
               *args,
               equilibrium=None,
               core_transports=None,
               core_sources=None,
               rho_tor_norm=None,
               enable_quasi_neutrality=True,
               **kwargs):
        """Solve transport equations
            solve   core_profiles with 'equilibrium'
            scheme:
                core_profiles   :=> imas_dd://core_profiles
                equilibrium     :=> imas_dd://equilibrium.time_slice
                transports      :=> imas_dd://core_transports
                sources,        :=> imas_dd://core_sources

                .transport_solver :> imas_dd://self.transport_solver_numerics

        """
        core_profiles_prev = core_profiles

        if rho_tor_norm is None:
            rho_tor_norm = core_profiles_prev.profiles_1d.grid.rho_tor_norm

        core_profiles_iter = CoreProfiles(equilibrium=equilibrium, rho_tor_norm=rho_tor_norm)

        # -----------------------------------------------------------
        # Equilibrium

        # current density profile:
        self._current(core_profiles_prev,  core_profiles_iter,  **kwargs)

        # # ion density profiles:
        # self._ion_density(core_profiles_prev,  core_profiles_iter,  **kwargs)

        # # electron density profile:
        # if enable_quasi_neutrality:
        #     self._electron_density(core_profiles_prev,  core_profiles_iter,  **kwargs)

        # # electron/ion density profile from quasi-neutrality:
        # self._quasi_neutrality(core_profiles_prev,  core_profiles_iter,  **kwargs)

        # # ion temperature profiles:
        # self._temperatures(core_profiles_prev,  core_profiles_iter,  **kwargs)

        # # toroidal rotation profiles:
        # self._rotation(core_profiles_prev,  core_profiles_iter,  **kwargs)

        self.update_global_quantities(core_profiles_prev,  core_profiles_iter)

        tol = self.check_converge(core_profiles_prev, core_profiles_iter)

        return core_profiles_iter, tol

    def check_converge(self, core_profiles_prev, core_profiles_iter):
        return 0

    def update_global_quantities(self, core_profiles_prev,  core_profiles_iter):
        # self.core_profiles.global_quantities = NotImplemented
        pass

    def solve_general_form(self, x,
                           y0, yp0,                 #
                           a, b, c, d, e, f, g,     # coefficients, which are  fuction of x
                           h,                       # time step
                           u, v, w,                 # boundary condition
                           ** kwargs
                           ):
        """ solve standard form
            $\\f[
            \\frac{a\\cdot Y-a\\cdot Y^{-1}}{h} +\\frac{1}{c}\\frac{\\partial}{\\partial\\rho}\\left(-d\\cdot\\frac{\\partial Y}{\\partial\\rho}+e\\cdot Y\\right) =f-g\\cdot Y
            ]$ where $Y$ is the function, $a,b,c,d,e,f,g$ are function of $x$, the boundary condition is
            $\\f[
                v\\cdot\\left.\\frac{\\partial Y}{\\partial x}\\right|_{x=bnd}+u\\cdot Y=w
            ]$

            return y, dy/dx

        """
        if yp0 is None:
            yp0 = UnivariateSpline(x, y0, x).derivative()(x)

        dd = UnivariateSpline(x, d).derivative()(x)
        de = UnivariateSpline(x, e).derivative()(x)

        A = UnivariateSpline(x, (-dd+e)/d)
        B = UnivariateSpline(x, (c*a/h+de+c*g)/d)
        C = UnivariateSpline(x, -c*b*y0/(h*d)-c*f/d)

        def fun(x, Y):
            """
            Y=[y,y']

            $ \\frac{\\partial y}{\\partial\\rho}&=Y^{\\prime}  \\\\
              \\frac{\\partial y^{\\prime}}{\\partial\\rho}&=A\\cdot y^{\\prime}+B\\cdot y+C
            $
            dy/dx =y'
            dy'/dx=A*y'+B*Y+C
            """
            y = Y[0]
            yp = Y[1]
            dy = yp
            ddy = A(x) * yp + B(x) * y+C(x)
            return np.array([dy, ddy])

        def bc(Ya, Yb,   u=u, v=v, w=w):
            """
            u *y'(b)+v*y(b)=w

            Ya=[y(a),y'(a)]
            Yb=[y(b),y'(b)]
            P=[[u(a),u(b)],[v(a),v(b)],[w(a),w(b)]]

            """
            Ya = (w[0] - Ya*u[0])/v[0]
            Yb = (w[1] - Yb*u[1])/v[1]

            # for k in range(len(Ya)):
            #     if v[k] is None or v[k] == 1.0e-8:
            #         Ybc[k][0] = -w[k]/u[k]
            #     else:
            #         Ybc[k][1] = (w[k] - Ybc[k][0]*u[k])/v[k]
            return Ya[0], Yb[0]

        solution = scipy.integrate.solve_bvp(fun, bc, x, np.array([y0, yp0]))

        if not solution.success:
            raise RuntimeError(solution.message)
        else:
            logger.debug("Solve bvp DONE")
        return solution.sol

    ############################################################################################
    #  TRANSPORT EQUATIONS:
    def _current(self,
                 core_profiles_prev,
                 core_profiles_iter,
                 *args,
                 transports=None,
                 sources=None,
                 **kwargs):

        ########################################
        # -----------------------------------------------------------
        # time step                                         [s]
        logger.debug((core_profiles_iter, core_profiles_prev))
        tau = core_profiles_iter.time - core_profiles_prev.time

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_iter.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_iter.vacuum_toroidal_field.b0

        # $B_0^-$ previous time steps$B_0$,                 [T]
        B0m = core_profiles_prev.vacuum_toroidal_field.b0

        # $\dot{B}_{0}$ time derivative or $B_0$,           [T/s]

        if tau > EPSILON:
            B0dot = (B0 - B0m)/tau
        else:
            B0dot = 0.0
            tau = 0.0
            B0m = B0
        # -----------------------------------------------------------
        # Grid
        # $rho$ not  normalised minor radius                [m]
        rho = core_profiles_iter.profiles_1d.grid.rho_tor
        rho_tor_norm = core_profiles_iter.profiles_1d.grid.rho_tor_norm
        # $\Psi$ flux function from current                 [Wb]
        psi0 = core_profiles_prev.profiles_1d.psi
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        dpsi_drho0 = core_profiles_prev.profiles_1d.dpsi_drho_tor

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = core_profiles_iter.profiles_1d.fpol
        fprime = core_profiles_iter.profiles_1d.interpolate("fpol").derivative()(rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = core_profiles_iter.profiles_1d.dvolume_dpsi

        # $gm2 \euqiv \left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $  [m^-2]
        gm2 = core_profiles_iter.profiles_1d.gm2

        # -----------------------------------------------------------
        # Profile

        # $q$ safety factor                                 [-]
        qsf = core_profiles_iter.profiles_1d.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        sigma = core_profiles_iter.profiles_1d.conductivity_parallel
        dsigma_drho = core_profiles_iter.profiles_1d.interpolate("conductivity_parallel").derivative()(rho_tor_norm)

        # -----------------------------------------------------------
        # Sources
        # total non inductive current, PSI independent component,          [A/m^2]
        # sources.profiles_1d.j_parallel if sources.profiles_1d.j_parallel not in [{},None,NotImplemented]  else 0.0
        j_ni_exp = 0.0

        # total non inductive current, component proportional to PSI,      [A/m^2/V/s]
        j_ni_imp = 0.0  # sources.j_ni_imp if sources is not None else 0.0   # can not find data in imas dd

        # for src in sources(equilibrium):
        #     j_ni_exp += src.profiles_1d.j_parallel

        # coefficients for numerical solver
        h = tau

        v = [0, 0]                                     # boundary conditions for numerical solver
        u = [0, 0]                                     # boundary conditions for numerical solver
        w = [0, 0]                                     # boundary conditions for numerical solver

        # Coefficients for for current diffusion
        #   equation in form:
        #         (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        a = sigma                                      # $\sigma_{\parallel}$
        b = sigma                                      # $\sigma_{\parallel}$

        c = constants.mu_0*B0*rho / fpol**2            # $\frac{\mu_{0}B_{0}\rho}{F^{2}}$
        d = vpr/(4.0*(constants.pi**2)*fpol)*gm2       # $\frac{V^{\prime}g_{3}}{4\pi^{2}F}$

        # $fun1=\frac{\sigma_{\parallel}\mu_{0}\rho^{2}}{F^{2}}$
        fun1 = sigma*constants.mu_0*(rho/fpol)**2

        # $e=-\text{FUN1}\frac{\dot{\Phi}_{b}}{2\Phi_{b}}$
        e = -fun1 * B0dot/2.0

        # $f=&-\frac{V^{\prime}}{2\pi\rho}j_{ni,exp}$
        f = -vpr/(2.0*constants.pi*rho) * j_ni_exp

        # $g=-\frac{V^{\prime}}{2\pi\rho}j_{ni,imp}+\sigma_{\parallel}\frac{\partial}{\partial\rho}\left(\frac{\sigma_{\parallel}\mu_{0}\rho^{2}}{F^{2}}\right)\cdot\frac{\dot{\Phi}_{b}}{2\Phi_{b}}$
        # g = vpr/(2.0*constants.pi*rho) * j_ni_imp + B0dot/2.0*sigma*dfun1
        g = vpr/(2.0*constants.pi*rho) * j_ni_imp + B0dot/2.0 * (sigma*(2-2*rho*fprime/fpol)+rho*dsigma_drho)

        # $\frac{V^{\prime}}{4\pi^{2}}\left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $
        fun2 = vpr*gm2/(4.0*constants.pi**2)

        # for irho in range(nrho):  # fix magnetic axis
        #     if abs(rho[irho]) < EPSILON:
        #         f[irho] = 1.0/(2.0*constants.pi)*j_ni_exp[irho]
        #         g[irho] = 1.0/(2.0*constants.pi)*j_ni_imp[irho] + B0dot/2.0*sigma[irho]*dfun1[irho]

        #  Boundary conditions for current diffusion
        #     equation in form:
        #     V*Y' + U*Y =W
        # On axis:
        #     dpsi/drho(rho=0)=0
        v[0] = 1.0
        u[0] = 0.0
        w[0] = 0.0

        # -----------------------------------------------------------
        # boundary condition, value
        # boundary_condition={
        #   "type":  in [0,1,2,3,4],
        #   "value": [[u0,v0,w0],[u1,v1,w1]]
        # }

        # At the edge:
        bc = core_profiles_iter.boundary_conditions.current
        if not bc or bc.identifier.index not in [1, 2, 3, 5]:          # Current equation is not solved:
            #  Interpretative value of safety factor should be given
            # if any(qsf != 0.0):  # FIXME
            dy = 2.0*constants.pi*B0*rho/qsf

            a[-1] = 1.0
            b[-1] = 1.0
            c[-1] = 1.0
            d[-1] = 0.0
            e[-1] = 0.0
            f[-1] = 0.0
            g[-1] = 0.0

            v[1] = 0.0
            u[1] = 1.0
            w[1] = UnivariateSpline(rho_tor_norm, 2.0*constants.pi*B0/qsf).integral(0, 1.0)

        elif bc.identifier.index == 1:  # poloidal flux
            v[1] = 0.0
            u[1] = 1.0
            w[1] = bc.value[0]
        elif bc.identifier.index == 2:  # ip

            v[1] = 1.0
            u[1] = 0.0
            w[1] = - constants.mu_0/fun2[-1]*bc.value[1][0]
        elif bc.identifier.index == 3:  # loop voltage
            v[1] = 0.0
            u[1] = 1.0
            w[1] = tau*bc.value[0]+psi0[-1]
        elif bc.identifier.index == 5:  # generic boundary condition  y expressed as a1y'+a2y=a3.
            v[1], u[1], w[1] = bc.value

        # Solution of current diffusion equation:
        try:
            sol = self.solve_general_form(rho, psi0, dpsi_drho0, a, b, c, d, e, f, g, h, u, v, w)
        except RuntimeError as error:
            logger.error(f"Fail to solve current transport equation! \n {error} ")
            psi1 = psi0
            dpsi_drho1 = dpsi_drho0
        else:
            psi1 = sol.y[0]
            dpsi_drho1 = sol.yp[0]

        # New magnetic flux function and current density:
        # dpc2 = (c*((a*y-b*ym)/h + g*y - f) + e*y) / d
        func7 = UnivariateSpline(rho_tor_norm, c*((a*psi1-b*psi0)/h + g*psi1 - f))
        intfun7 = [func7.integral(0, r) for r in rho_tor_norm]

        dy = intfun7/d + e/d*psi1

        # fun4 = fun2*dy
        # fun5 = fun2*dy*R0*B0/fpol

        dfun4 = UnivariateSpline(rho, fun2*dy).derivative()(rho)  # Derivation of function FUN4
        dfun5 = UnivariateSpline(rho, fun2*dy*R0*B0/fpol).derivative()(rho)   # Derivation of function FUN5

        # New profiles of plasma parameters obtained
        #     from current diffusion equation:

        core_profiles_iter.profiles_1d.psi = psi1
        core_profiles_iter.profiles_1d.dpsi_drho_tor = dpsi_drho1
        core_profiles_iter.profiles_1d.q = 2.0*constants.pi*B0*rho/dpsi_drho1

        j_tor = - 2.0*constants.pi*R0/constants.mu_0/vpr * dfun4
        j_par = - 2.0*constants.pi/R0/constants.mu_0/vpr * (fpol/B0)**2*dfun5

        # $E_\parallel$  parallel electric field,,          [V/m]
        e_par = (j_par - j_ni_exp - j_ni_imp*psi1) / sigma
        core_profiles_iter.profiles_1d.e_field.parallel = e_par

        # Total Ohmic currents                              [A]
        # fun7 = vpr * j_par / 2.0e0 / constants.pi * B0 / fpol**2
        func7 = UnivariateSpline(rho_tor_norm, vpr * j_par / 2.0e0 / constants.pi * B0 / fpol**2)
        intfun7 = [func7.integral(0, r) for r in rho_tor_norm]
        core_profiles_iter.profiles_1d.j_ohmic = intfun7[-1] * fpol[-1]

        # current density, toroidal,                        [A/m^2]
        core_profiles_iter.profiles_1d.j_tor = j_tor

        # Total non-inductive currents                       [A]
        # fun7 = vpr * (j_ni_exp + j_ni_imp * psi) / (2.0e0 * constants.pi) * B0 / fpol**2
        func7 = UnivariateSpline(rho_tor_norm, vpr * (j_ni_exp + j_ni_imp * psi0) /
                                 (2.0e0 * constants.pi) * B0 / fpol**2)
        intfun7 = [func7.integral(0, r) for r in rho_tor_norm]
        core_profiles_iter.profiles_1d.j_non_inductive = intfun7[-1] * fpol[-1]

        return True

    def _ion_density_one(self,
                         iion,
                         core_profiles_iter,
                         core_profiles_prev,
                         *args,
                         transports=None,
                         sources=None,
                         hyper_diff=[0, 0],
                         **kwargs):

        # -----------------------------------------------------------
        # time step                                         [s]
        tau = core_profiles_iter.time - core_profiles_prev.time

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_iter.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_iter.vacuum_toroidal_field.b0

        # $B_0^-$ previous time steps$B_0$,                 [T]
        B0m = core_profiles_prev.vacuum_toroidal_field.b0

        # $\dot{B}_{0}$ time derivative or $B_0$,           [T/s]
        B0dot = (B0 - B0m)/tau

        # -----------------------------------------------------------
        # Grid
        # $rho$ not  normalised minor radius                [m]
        rho = core_profiles_iter.rho_tor

        # $\Psi$ flux function from current                 [Wb]
        psi0 = core_profiles_prev.profiles_1d.psi
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        psi0p = core_profiles_prev.profiles_1d.dpsi_drho

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = core_profiles_iter.fpol

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = core_profiles_iter.dvolume_dpsi

        # $gm2 \euqiv \left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $  [m^-2]
        gm2 = core_profiles_iter.gm2

        # $\left\langle \left|\nabla\rho\right|^{2}\right\rangle $  [-]
        gm3 = core_profiles_iter.gm3

        # -----------------------------------------------------------
        # Profile

        # $q$ safety factor                                 [-]
        qsf = core_profiles_iter.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        sigma = core_profiles_iter.conductivity_parallel

        hyper_diff_exp = hyper_diff[0]
        hyper_diff_imp = hyper_diff[1]

        ni0 = core_profiles_prev.profiles_1d.ion[iion].density

        ni0p = core_profiles_prev.profiles_1d.ion[iion].density_drho or derivate(ni0, rho)

        diff = np.zeros(shape=rho.shape)
        vconv = np.zeros(shape=rho.shape)

        for model in transports.model:
            # c1[imodel] = transport["C1"][imodel]
            diff += model.ion[iion].particles.d
            vconv += model.ion[iion].particles.v

        si_exp = np.zeros(shape=rho.shape)
        si_imp = np.zeros(shape=rho.shape)

        for source in sources.source:
            si_imp += source.ion[iion].particles_decomposed.implicit_part
            si_exp += source.ion[iion].particles_decomposed.explicit_part

        # Coefficients for ion diffusion equation  in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff)  # AF 25.Apr.2016, 22.Aug.2016

        a = vpr
        b = vprm
        c = 1.
        d = vpr*gm3*(diff+diff_hyper)  # AF 25.Apr.2016
        e = vpr*gm3*(vconv+diff_hyper*ni0p/ni0) - B0dot/2./B0*rho*vpr
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
        #       dNi/drho(rho=0)=0:
        if(boundary_condition.type != 4):  # AF 11.Oct.2011
            v[0] = 1.
            u[0] = 0.
        else:  # AF 11.Oct.2011 - Zero flux instead of zero gradient at the axis for solver 4
            #       IF (DIFF[0]>1.0E-6) : #AF 19.Mar.2012 - To avoid problems with the axis boundary condition
            # AF 19.Mar.2012 - To avoid problems with the axis boundary condition #AF 25.Apr.2016
            if((diff[0]+diff_hyper) > 1.0e-6):
                #         V[0] = -DIFF[0]
                v[0] = -diff[0]-diff_hyper  # AF 25.Apr.2016
            else:
                v[0] = -1.0e-6

                #       U[0] = VCONV[0] #AF 25.Apr.2016
                u[0] = vconv[0]+diff_hyper*dnim[0]/nim[0]  # AF 25.Apr.2016
            # AF 11.Oct.2011
            w[0] = 0.

        # At the edge:
        #       FIXED Ni
        if(boundary_condition.type == 1):
            v[1] = 0.
            u[1] = 1.
            w[1] = boundary_condition.value[1, 0]

        #       FIXED grad_Ni
        elif(boundary_condition.type == 2):
            v[1] = 1.
            u[1] = 0.
            w[1] = -boundary_condition.value[1, 0]

        #       FIXED L_Ni
        elif(boundary_condition.type == 3):
            v[1] = boundary_condition.value[1, 0]
            u[1] = 1.
            w[1] = 0.

        #       FIXED Flux_Ni
        elif(boundary_condition.type == 4):
            #        V[1] = -G(NRHO)*DIFF(NRHO)
            #        U[1] = G(NRHO)*VCONV(NRHO)
            v[1] = -vpr[-1]*gm3[-1]*diff[-1]
            u[1] = vpr[-1]*gm3[-1]*vconv[-1]
            w[1] = boundary_condition.value[1, 0]

        #       Generic boundary condition
        elif(boundary_condition.type == 5):
            v[1], u[1], w[1] = boundary_condition.value[1]

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

            v[1] = 0.0
            u[1] = 1.0
            w[1] = ni0[-1]

        # Solution of ion density diffusion equation:
        try:
            sol = self.solve_general_form(rho, ni0, ni0p, a, b, c, d, e, f, g, h, u, v, w)
        except RuntimeError as error:
            raise RuntimeError(f"Fail to solve ion density transport equation! \n {error} ")
        else:
            ni1 = sol.y[0]
            ni1p = sol.yp[0]

        # for irho in range(rho.shape[0]):
            #        NI(IRHO)    = Y(IRHO)
            #        DNI(IRHO)   = DY(IRHO)
            #        IF (RHO(IRHO) #= 0.E0_R8) :
            #           FUN1(IRHO)  = 1.e0_R8/RHO(IRHO)*(VPR(IRHO)*SI_EXP(IRHO)          \ #AF 11.Oct.2011 - the division by rho was needed because the routine INTEGR below actually integrates the function multiplied by rho
            #                +VPRM(IRHO)*NIM(IRHO)/TAU                                   \
            #                -NI(IRHO)*VPR(IRHO)*(1.e0_R8/TAU+SI_IMP(IRHO)))
            #        else:
            #           FUN1(IRHO) = 1.e0_R8*(SI_EXP(IRHO)+NIM(IRHO)/TAU-NI(IRHO)        \ #AF 11.Oct.2011 - this is only OK with solver_test since Vprime == rho in that case - get it fixed in the trunk#
            #                *(1.e0_R8/TAU+SI_IMP(IRHO)))
            #

        fun1 = vpr*si_exp + vprm*ni0/tau - ni0*vpr*(1./tau+si_imp)

        intfun1 = integral(fun1, rho)

        local_fun1_s4 = (si_exp[0] + ni0[0]/tau - ni0[0]*(1./tau+si_imp[0]))/gm3[0]  # stripping G1
        # stripped integral for the axis - from the INTEGR2 routine, first point only
        local_intfun1_s4 = local_fun1_s4*rho[0]/2.0

        int_source = intfun1 + B0dot/2./B0*rho*vpr*ni0
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
        core_profiles_iter.ion[iion].density = ni0
        core_profiles_iter.ion[iion].ddensity_rho = ni0p
        core_profiles_iter.ion[iion].transport.d = diff
        core_profiles_iter.ion[iion].transport.v = vconv
        core_profiles_iter.ion[iion].transport.flux = flux
        # core_profiles_iter.ion[iion].transport.flux_conv = flux_ni_conv
        core_profiles_iter.ion[iion].source = si_exp - si_imp * ni0
        core_profiles_iter.ion[iion].int_source = integral(ni0*vpr, rho)

    def _ion_density(self,
                     core_profiles_iter,
                     core_profiles_prev,
                     *args,
                     transports=None,
                     sources=None,
                     hyper_diff=[0, 0],
                     **kwargs):
        for iion, ion in enumerate(core_profiles_prev.profiles_1d.ion):
            self._ion_density_one(self, iion,
                                  core_profiles_iter,
                                  core_profiles_prev,
                                  equilibrium,
                                  transports=transports,
                                  sources=sources,
                                  **kwargs
                                  )

    def _electron_density(self,
                          core_profiles_iter,
                          core_profiles_prev,
                          *args,
                          transports=None,
                          sources=None,
                          hyper_diff=[0, 0],
                          **kwargs):

        hyper_diff_exp = hyper_diff[0]  # AF 22.Aug.2016
        hyper_diff_imp = hyper_diff[1]  # AF 22.Aug.2016

        # -----------------------------------------------------------
        # time step                                         [s]
        tau = core_profiles_new.time - core_profiles.time

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_new.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_new.vacuum_toroidal_field.b0

        # $B_0^-$ previous time steps$B_0$,                 [T]
        B0m = core_profiles.vacuum_toroidal_field.b0

        # $\dot{B}_{0}$ time derivative or $B_0$,           [T/s]
        B0dot = (B0 - B0m)/tau

        # -----------------------------------------------------------
        # Grid
        # $rho$ not  normalised minor radius                [m]
        rho = core_profiles_iter.grid.rho
        # $\Psi$ flux function from current                 [Wb]
        psi0 = core_profiles_prev.profiles_1d.grid.psi
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        psi0p = core_profiles_prev.profiles_1d.grid.dpsi or derivate(psi0, rho)
        # normalized psi  [0,1]                            [-]
        psi_norm = core_profiles_prev.profiles_1d.grid.psi_norm

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = equilibrium.profiles_1d.interpolate("f")(rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = equilibrium.profiles_1d.interpolate("dvolume_dpsi")(rho_tor_norm)
        vprm = kwargs["VPRM"]

        gm3 = equilibrium.profiles_1d.interpolate("gm3")(rho_tor_norm)

        # -----------------------------------------------------------
        # Profile

        # $q$ safety factor                                 [-]
        qsf = core_profiles_prev.profiles_1d.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        sigma = core_profiles_prev.profiles_1d.conductivity_parallel

        #    solution of particle transport equation

        # Set equation to 'predictive' and all coefficients to zero:
        flag = 1

        # Set up boundary conditions for particular ion type:
        # boundary_condition.type = profiles["NE_BND_TYPE"]
        # ne_bnd = [[0, 0, 0], profiles["NE_BND"]]

        # Set up local variables for particular ion type:
        ne0 = core_profiles_prev.profiles_1d.electron.density

        ne0p = core_profiles_prev.profiles_1d.electron.density_drho or derivate(ne0, rho)

        diff = np.zeros(shape=rho.shape)
        vconv = np.zeros(shape=rho.shape)

        for model in transports.model:
            # c1[imodel] = transport["C1"][imodel]
            diff += model.electron.particles.d
            vconv += model.electron.particles.v

        se_exp = np.zeros(shape=rho.shape)
        se_imp = np.zeros(shape=rho.shape)

        for source in sources.source:
            si_imp += source.electron.particles_decomposed.implicit_part
            si_exp += source.electron.particles_decomposed.explicit_part

        # Coefficients for electron diffusion equation in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff)  # AF 25.Apr.2016, 22.Aug.2016

        a = vpr
        b = vprm
        c = 1.
        d = vpr*gm3*(diff+diff_hyper)  # AF 25.Apr.2016
        e = vpr*gm3*(vconv+diff_hyper*ne0p/ne0) - B0dot/2./B0*rho*vpr
        f = vpr*se_exp
        g = vpr*se_imp

        h = tau

        # Boundary conditions for electron diffusion equation in form:
        #
        #     V*Y' + U*Y =W
        #
        # On axis:
        #       dNi/drho(rho=0)=0: #AF 25.Apr.2016 - this is Ne, not Ni
        if(solver_type != 4):
            v[0] = 1.
            u[0] = 0.
        else:
            #       IF (DIFF[0]>1.0E-6) :
            if((diff[0]+diff_hyper) > 1.0e-6):  # AF 25.Apr.2016
                #         V[0] = -DIFF[0]
                v[0] = -diff[0]-diff_hyper  # AF 25.Apr.2016
            else:
                v[0] = -1.0e-6

            #       U[0] = VCONV[0]
                u[0] = vconv[0]+diff_hyper*ne0p[0]/ne0[0]  # AF 25.Apr.2016

            w[0] = 0.

        # At the edge:
        #       FIXED Ne
        if(boundary_condition.type == 1):
            v[1] = 0.
            u[1] = 1.
            w[1] = ne_bnd[1, 0]
        #       FIXED grad_Ne
        elif(boundary_condition.type == 2):
            v[1] = 1.
            u[1] = 0.
            w[1] = -ne_bnd[1, 0]

        #       FIXED L_Ne
        elif(boundary_condition.type == 3):
            v[1] = ne_bnd[1, 0]
            u[1] = 1.
            w[1] = 0.

        #       FIXED Flux_Ne
        elif(boundary_condition.type == 4):
            v[1] = -vpr[-1]*gm3[-1]*diff[-1]
            u[1] = vpr[-1]*gm3[-1]*vconv[-1]
            w[1] = ne_bnd[1, 0]

        #       Generic boundary condition
        elif(boundary_condition.type == 5):
            v[1] = ne_bnd[1, 0]
            u[1] = ne_bnd(2, 2)
            w[1] = ne_bnd(2, 3)

        # Density equation is not solved:
        elif(boundary_condition.type == 0):

            dy = derivate(y, rho)

            flag = 0

            a.fill(1.0)
            b.fill(1.0)
            c.fill(1.0)
            d.fill(0.0)
            e.fill(0.0)
            f.fill(0.0)
            g.fill(0.0)

            v[1] = 0.0
            u[1] = 1.0
            w[1] = y[-1]

        # Solution of density diffusion equation:
        self.solve_eq(a, b, c, d, e, f, g, h)

        # dy check for nans in solution, return if true)
        if(any(np.isnan(solver["y"]))):
            raise RuntimeError('Error in the electron density equation: nans in the solution, stop')

        try:
            sol = self.solve_general_form(rho, ne0, ne0p, a, b, c, d, e, f, g, h, u, v, w)
        except RuntimeError as error:
            raise RuntimeError(f"Fail to solve ion density transport equation! \n {error} ")
        else:
            ne1 = sol.y[0]
            ne1p = sol.yp[0]

        intfun1 = integral(vpr*se_exp + vprm*ne0/tau - ne0*vpr*(1./tau+se_imp), rho)

        local_fun1_s4 = (se_exp[0] + ne0[0]/tau - ne[0]*(1./tau+se_imp[0]))/gm3[0]
        local_intfun1_s4 = local_fun1_s4*rho[0]/2.

        int_source = intfun1 + B0dot/2./B0*rho*vpr*ne
        flux = vpr*gm3 * (y*vconv - dy*diff)

        # Contribution to electron energy transport:
        flux_ne_conv = 0.

        # for imodel in range(nmodel):
        #     flux_ne_conv = flux_ne_conv + c1[imodel]*vpr*gm3 * (y*vconv_mod[imodel] - dy*diff_mod[imodel])

        # If equation is not solved, flux is determined
        #     by the integral of kwargs:
        if(boundary_condition.type == 0):
            diff = 1.e-6
            flux = int_source
            flux_ne_conv = 1.5*int_source
            if(vpr*gm3 != 0.0):
                diff = - flux / dy / (vpr*gm3)
            if (abs(diff) >= nine_diff_limit):
                diff = sign(nine_diff_limit, diff)
                vconv = 0.0
            if(diff <= 1.e-6):
                diff = 1.e-6
                vconv = (flux / (max(abs(vpr), 1.e-6)*gm3) + dy*diff) / y

        # Return new ion density and flux profiles to the work flow:

        core_profiles_iter.electron.density = ne0
        core_profiles_iter.electron.ddensity_rho = ne0p
        # core_profiles_iter.electron.transport.d = diff
        # core_profiles_iter.electron.transport.v = vconv
        # core_profiles_iter.electron.transport.flux = flux
        # # core_profiles_iter.ion[iion].transport.flux_conv = flux_ni_conv
        # core_profiles_iter.electron.source = si_exp - si_imp * ne0
        # core_profiles_iter.electron.int_source = integral(ne0*vpr, rho)

    def _quasi_neutrality(self,
                          core_profiles_iter,
                          core_profiles_prev,
                          *args,
                          transports=None,
                          sources=None,
                          hyper_diff=[0, 0],
                          **kwargs):

        rho = core_profiles_iter.solver_1d[_last_].grid.rho

        ne0 = core_profiles_prev.profiles_1d.electron.density

        ne0p = core_profiles_prev.profiles_1d.electron.density_drho or derivate(ne0, rho)

        flux_ne = profiles["FLUX_NE"]
        flux_ne_conv = profiles["FLUX_NE_CONV"]

        ni = profiles["NI"]
        dni = profiles["DNI"]
        flux_ni = profiles["FLUX_NI"]
        flux_ni_conv = profiles["FLUX_NI_CONV"]
        zion = profiles["ZION"]
        zion2 = profiles["ZION2"]

        nz = impurity["NZ"]
        flux_nz = impurity["FLUX_NZ"]
        zimp = impurity["ZIMP"]
        zimp2 = impurity["ZIMP2"]

        # Quasineutrality condition:

        if(QUASI_NEUT == 0):
            # ELECTRON density \ flux oB0ained from QN
            profiles["NE"] = 0.0
            profiles["DNE"] = 0.0  # AF
            profiles["FLUX_NE"] = 0.0
            profiles["CONTRIB_2_ENERGY_FLUX_NE"] = 0.0

            for iion in range(nion):
                profiles["NE"] = profiles["NE"] + profiles["ZION"][iion] * \
                    (profiles["NI"][iion] + profiles["NI_FAST"][iion])
                profiles["FLUX_NE"] = profiles["FLUX_NE"] + profiles["ZION"][iion]*profiles["FLUX_NI"][iion]
                profiles["CONTRIB_2_ENERGY_FLUX_NE"] = profiles["CONTRIB_2_ENERGY_FLUX_NE"] + \
                    profiles["ZION"][iion]*profiles["CONTRIB_2_ENERGY_FLUX_NI"][iion]

            for iimp in range(nimp):
                for izimp in range(nzimp):
                    profiles["NE"] = profiles["NE"] + impurity["ZIMP"][iimp, izimp]*impurity["NZ"][iimp, izimp]
                    profiles["FLUX_NE"] = profiles["FLUX_NE"] + \
                        impurity["ZIMP"][iimp, izimp]*impurity["FLUX_NZ"][iimp, izimp]

            profiles["NE"] = profiles["NE"] - profiles["NE_FAST"]

            #
            #        PROFILES["ZEFF(IRHO)          = NZ2(IRHO)/NE(IRHO)
            #

            profiles["DNE"] = derivate(ne, rho)

            local_flux_ne_conv_s4 = 0.0
            for iion in range(nion):
                local_flux_ne_conv_s4 = local_flux_ne_conv_s4 + profiles["ZION"][iion]*local_flux_ni_conv_s4[iion]

        elif(QUASI_NEUT == 1 or QUASI_NEUT == 2):
            # TOTAL ION density \ flux oB0ained from QN

            ni_tot = profiles["NE"] + profiles["NE_FAST"]
            flux_ni_tot = profiles["FLUX_NE"]
            contrib_2_energy_flux_ni_tot = profiles["CONTRIB_2_ENERGY_FLUX_NE"]

            for iion in range(nion):
                ni_tot = ni_tot - profiles["ZION"][iion]*profiles["NI_FAST"][iion]
                if(profiles["NI_BND_TYPE"][iion] > 0.5):
                    ni_tot = ni_tot - profiles["ZION"][iion]*profiles["NI"][iion]
                    flux_ni_tot = flux_ni_tot - profiles["ZION"][iion]*profiles["FLUX_NI"][iion]
                    contrib_2_energy_flux_ni_tot = contrib_2_energy_flux_ni_tot - \
                        profiles["ZION"][iion] * profiles["CONTRIB_2_ENERGY_FLUX_NI"][iion]

            for iimp in range(nimp):
                for izimp in range(nzimp):
                    ni_tot = ni_tot - impurity["ZIMP"][iimp, izimp]*impurity["NZ"][iimp, izimp]
                    flux_ni_tot = flux_ni_tot - impurity["ZIMP"][iimp, izimp]*impurity["FLUX_NZ"][iimp, izimp]

            # Separation for individual ion species
            nion_qn = 0
            ni_qn = 0.0
            zion_av = 0.0

            for iion in range(nion):
                if(profiles["NI_BND_TYPE"][iion] == 0):
                    ni_qn = ni_qn + profiles["NI_BND"](1, iion)
                    nion_qn = nion_qn + 1

            ion_fraction = profiles["NI_BND"][0]/ni_qn

            for iion in range(nion):
                if(profiles["NI_BND_TYPE"][iion] == 0):
                    zion_av = zion_av + (ion_fraction[iion]*profiles["ZION"][iion])

            if(QUASI_NEUT == 1):
                for iion in range(nion):
                    if(profiles["NI_BND_TYPE"][iion] == 0):
                        profiles["NI"][iion] = ni_tot / zion_av*ion_fraction[iion]
                        profiles["FLUX_NI"][iion] = flux_ni_tot / zion_av*ion_fraction[iion]
                        profiles["CONTRIB_2_ENERGY_FLUX_NI"][iion] = contrib_2_energy_flux_ni_tot / \
                            zion_av*ion_fraction[iion]

            if(QUASI_NEUT == 2):
                for iion in range(nion):
                    if(profiles["NI_BND_TYPE"][iion] == 0):
                        profiles["NI"][iion] = ni_tot / nion_qn/profiles["ZION"][iion]
                        profiles["FLUX_NI"][iion] = flux_ni_tot / nion_qn/profiles["ZION"][iion]
                        profiles["CONTRIB_2_ENERGY_FLUX_NI"][iion] = contrib_2_energy_flux_ni_tot / \
                            nion_qn/profiles["ZION"][iion]

            for iion in range(nion):
                aux1 = profiles["NI"][iion]
                aux2 = 0.0
                aux2 = derivate(aux1,  kwargs["RHO"])
                profiles["DNI"][iion] = aux2

            # AF - End

            local_flux_ni_conv_s4 = 0.0
            for iion in range(nion):
                if(profiles["NI_BND_TYPE"][iion] == 0):
                    if(QUASI_NEUT == 1):
                        # LOCAL_FLUX_NI_CONV_S4 = LOCAL_FLUX_NE_CONV_S4/ZTOT*FION(IION)
                        local_flux_ni_conv_s4 = local_flux_ne_conv_s4/zion_av*ion_fraction[iion]
                    if(QUASI_NEUT == 2):
                        local_flux_ni_conv_s4 = local_flux_ne_conv_s4/nion/zion[iion]

        # Plasma effective charge:
        ni_z2 = 0.0
        for iion in range(nion):
            ni_z2 = ni_z2 + profiles["ZION2"][iion]*(profiles["NI"][iion] + profiles["NI_FAST"][iion])
        for iimp in range(nimp):
            for izimp in range(nzimp):
                ni_z2 = ni_z2 + impurity["ZIMP2"][iimp, izimp]*impurity["NZ"][iimp, izimp]

        profiles["ZEFF"] = ni_z2 / (profiles["NE"] + profiles["NE_FAST"])

        # dy case when ne is not computed at all
        # dy just check that
        if (QUASI_NEUT == -1):

            if any(ne <= 0.0):
                raise RuntimeError('Error in the density equation: on-axis electron density is negative, stop')
            # else:
                # ne = ne(irho-1)

            profiles["ne"] = ne

    # HEAT TRANSPORT EQUATIONS
    def _temperatures(self,
                      core_profiles_iter,
                      core_profiles_prev,
                      *args,
                      collisions=None,
                      transports=None,
                      sources=None,
                      solve_type=0,
                      hyper_diff=[0, 0],
                      **kwargs):

        hyper_diff_exp = hyper_diff[0]  # AF 22.Aug.2016
        hyper_diff_imp = hyper_diff[1]  # AF 22.Aug.2016

        ########################################
        # -----------------------------------------------------------
        # time step                                         [s]
        tau = core_profiles_new.time - core_profiles.time

        # $R_0$ characteristic major radius of the device   [m]
        R0 = core_profiles_new.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = core_profiles_new.vacuum_toroidal_field.b0

        # $B_0^-$ previous time steps$B_0$,                 [T]
        B0m = core_profiles.vacuum_toroidal_field.b0

        # $\dot{B}_{0}$ time derivative or $B_0$,           [T/s]
        B0dot = (B0 - B0m)/tau

        # -----------------------------------------------------------
        # Grid
        # $rho$ not  normalised minor radius                [m]
        rho = core_profiles_iter.grid.rho
        # $\Psi$ flux function from current                 [Wb]
        psi0 = core_profiles_iter.grid.psi
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        psi0p = core_profiles_iter.grid.dpsi or derivate(psi0, rho)
        # normalized psi  [0,1]                            [-]
        psi_norm = core_profiles_prev.profiles_1d.grid.psi_norm

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = equilibrium.profiles_1d.interpolate("f")(rho_tor_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = equilibrium.profiles_1d.interpolate("dvolume_dpsi")(rho_tor_norm)
        dvpr = derivate(vpr, rho)  # Derivation of V'

        # $\left\langle \left|\nabla\rho\right|^{2}\right\rangle $  [-]
        gm3 = equilibrium.profiles_1d.interpolate("gm3")(rho_tor_norm)

        # -----------------------------------------------------------
        # Profile

        # $q$ safety factor                                 [-]
        qsf = core_profiles_prev.profiles_1d.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        sigma = core_profiles_prev.profiles_1d.conductivity_parallel

        # Energy exchange terms due to collisions
        #     (defined from previous iteration):
        self.plasma_collisions(core_profiles, collisions)

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
            ti0p = ion.dtemperature_drho or derivate(ti0, rho)
            ni0 = ion.density
            ni0p = ion.ddensity_drho or derivate(ni0, rho)

            ni1 = core_profiles_iter.ion[iion].density
            ni1p = core_profiles_iter.ion[iion].ddensity_drho or derivate(ni1, rho)

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

            diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff_ti)  # AF 25.Apr.2016, 22.Aug.2016

            a[iion] = 1.5*vpr*ni
            b[iion] = 1.5*vprm**fivethird/vpr**twothird*nim
            c[iion] = 1.0
            d[iion] = vpr*gm3*ni*(diff_ti+diff_hyper)  # AF 25.Apr.2016
            e[iion] = vpr*gm3*ni*(vconv_ti+diff_hyper*ti0p/ti0) + flux_ni - 1.5*B0dot/2./B0*rho*ni*vpr
            f[iion] = vpr * (qi_exp + qei + qzi + qgi)
            g[iion] = vpr*(qi_imp + vei + vzi) - B0dot/2./B0*rho*ni*dvpr

            # Boundary conditions for ion heat transport equation in form:
            #     V*Y' + U*Y =W

            # On axis:
            #       dTi/drho(rho=0)=0
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
                dy = derivate(y, rho)  # temperature gradient
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

        ne0 = core_profiles_prev.profiles_1d.electron.density
        ne0p = core_profiles_prev.profiles_1d.electron.density_drho or derivate(ne0, rho)

        te0 = core_profiles_prev.profiles_1d.electron.temperature
        te0p = core_profiles_prev.profiles_1d.electron.dtemperature_drho or derivate(te0, rho)

        diff_te = transport.electron.particles.d
        vconv_te = transport.electron.particles.v
        flux_ne = transports.electron.particles.flux

        qgi = 0.
        for iion in range(nion):
            qgi = qgi + transport["QGI"][iion]

        qe_exp = kwargs["QOH"] / itm_ev
        qe_imp = 0.

        qe_exp = qe_exp + transports.electron.energy_decomposed.explicit_part
        qe_imp = qe_imp + transports.electron.energy_decomposed.implicit_part

        qie = collisions["QIE"]
        vie = collisions["VIE"]

        # Set up boundary conditions for electron heat transport equation:
        te_bnd_type = profiles["TE_BND_TYPE"]
        te_bnd = profiles["TE_BND"]

        # Coefficients for electron heat transport equation in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff_te)  # AF 25.Apr.2016, 22.Aug.2016

        a = 1.5*vpr*ne
        # DPC temporary "fix" for NaNs

        b = 1.5*vprm**fivethird/vpr**twothird*ne0
        if vpr <= 0.0:
            b[0] = 0.0

        # DPC end of temporary fix
        c = 1.
        d = vpr * gm3 * ne * (diff_te+diff_hyper)
        e = vpr * gm3 * ne * (vconv_te+diff_hyper*dtem / tem) + flux_ne - 1.5*B0dot/2./B0*rho*ne*vpr
        f = vpr * (qe_exp + qie - qgi)
        g = vpr * (qe_imp + vie) - B0dot/2./B0*rho*ne*dvpr

        h = tau

        # Boundary conditions for electron heat
        #     transport equation in form:
        #
        #     V*Y' + U*Y =W

        # On axis
        #     dTe/drho(rho=0)=0:
        if(solver_type != 4):
            v[0] = 1.
            u[0] = 0.
        else:  # - Zero flux instead of zero gradient at the axis for solver 4
            #    IF (DIFF_TE[0]>1.0E-6) : #AF 19.Mar.2012 - To avoid problems with the axis boundary condition
            # AF 19.Mar.2012 - To avoid problems with the axis boundary condition #AF 25.Apr.2016
            if((diff_te[0]+diff_hyper) > 1.0e-6):
                #      V[0] = -DIFF_TE[0]*NE[0]
                v[0] = -(diff_te[0]+diff_hyper)*ne[0]  # AF 25.Apr.2016
            else:
                v[0] = -1.0e-6*ne[0]

                #    U[0] = VCONV_TE[0]*NE[0]+LOCAL_FLUX_NE_CONV_S4
                u[0] = (vconv_te[0]+diff_hyper*dtem[0]/tem[0])*ne[0]+local_flux_ne_conv_s4  # AF 25.Apr.2016
                w[0] = 0.

        # At the edge:

        #     FIXED Te
        if(te_bnd_type[1] == 1):
            v[1] = 0.
            u[1] = 1.
            w[1] = te_bnd[1, 0]

        #     FIXED grad_Te
        if(te_bnd_type[1] == 2):
            v[1] = 1.
            u[1] = 0.
            w[1] = -te_bnd[1, 0]

        #     FIXED L_Te
        if(te_bnd_type[1] == 3):
            v[1] = te_bnd[1, 0]
            u[1] = 1.
            w[1] = 0.

        #    FIXED Flux_Te
        if(te_bnd_type[1] == 4):
            #     V[1] = -G(NRHO)*DIFF(NRHO)*NE(NRHO)
            #     U[1] = G(NRHO)*VCONV(NRHO)*NE(NRHO)+FLUX_NE(NRHO)
            v[1] = -vpr[-1]*gm3[-1]*diff_te[-1]*ne[-1]
            u[1] = vpr[-1]*gm3[-1]*vconv_te[-1]*ne[-1]+flux_ne[-1]
            w[1] = te_bnd[1, 0]

        #    Generic boundary condition
        if(te_bnd_type[1] == 5):
            v[1] = te_bnd[1, 0]
            u[1] = te_bnd(2, 2)
            w[1] = te_bnd(2, 3)

        # Temperature equation is not solved:
        if(te_bnd_type[1] == 0):

            dy = derivate(y, rho)  # temperature gradient

            flag = 0

            a.fill(1.0)
            b.fill(1.0)
            c.fill(1.0)
            d.fill(0.0)
            e.fill(0.0)
            f.fill(0.0)
            g.fill(0.0)

            v[1] = 0.0
            u[1] = 1.0
            w[1] = y[-1]

        # Defining coefficients for numerical solver:
        solver["TYPE"] = solver_type
        solver["EQ_FLAG"][ndim] = flag
        solver["NDIM"] = ndim
        solver["NRHO"] = rho.shape[0]
        solver["AMIX"] = amix
        solver["DERIVATIVE_FLAG"][0] = 0

        solver["RHO"] = rho

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

        solver["V"][ndim, 1] = v[1]
        solver["U"][ndim, 1] = u[1]
        solver["W"][ndim, 1] = w[1]
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

            y = solver["Y"](iion, irho)
            dy = solver["DY"](iion, irho)

            if(ti_bnd_type[iion] == 0):
                y = profiles["TI"][iion]
                dy = derivate(y, rho)

            ti = y
            dti = dy
            if any(ti < 0.0):
                raise RuntimeError('Error in the temperature equation: on-axis ion temperature is negative, stop')

            else:
                # write(*, *) 'warning, temperature for ion ', iion, ' and irho ', irho, 'is negative, set it to the value at the previous irho'
                ti[1:] = ti[0:-1]
                dti[1:] = dti(irho-1)
                # end if
            # end if
            if(rho != 0.):  # FIXME
                fun1 = vpr/rho *                                           \
                    ((1.5*nim*ti0/tau*(vprm/vpr)**fivethird
                      + qi_exp + qei + qzi + qgi)
                     - (1.5*ni/tau + qi_imp + vei + vzi
                        - B0dot/2./B0*rho*ni*dvpr) * y)
            else:
                fun1 = ((1.5*nim*ti0/tau
                         + qi_exp + qei + qzi + qgi)
                        - (1.5*ni/tau + qi_imp + vei + vzi
                           - B0dot/2./B0*rho*ni*dvpr) * y)

            intfun1 = integral(fun1, rho)  # Integral source

            for irho in range(rho.shape[0]):
                flux_ti_conv = y*flux_ni

                flux_ti_cond = vpr*gm3*ni                                     \
                    * (y*vconv_ti - dy*diff_ti)

                flux_ti = flux_ti_conv + flux_ti_cond

                int_source = intfun1 + y * 1.5*B0dot/2./B0*rho*ni*vpr

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
                profiles["DTI"][iion] = dti  # AF, 25.Sep.2014
                profiles["DIFF_TI"][iion] = diff_ti
                profiles["VCONV_TI"][iion] = vconv_ti
                profiles["FLUX_TI_CONV"][iion] = flux_ti_conv
                profiles["FLUX_TI_COND"][iion] = flux_ti_cond
                profiles["FLUX_TI"][iion] = flux_ti
                profiles["SOURCE_TI"][iion] = qi_exp + qei + qzi + qgi - (qi_imp + vei + vzi) * ti
                profiles["INT_SOURCE_TI"][iion] = int_source
                profiles["QEI_OUT"] = profiles["QEI_OUT"]+qei

            fun1 = profiles["SOURCE_TI"][iion]*vpr
            intfun1 = integral(fun1, rho)
            profiles["INT_SOURCE_TI"][iion] = intfun1

        #-------------------------------------------------------#
        #     ELECTRONS:                                        #
        #-------------------------------------------------------#

        # dpc 2011-08-11: I think we need most of the following
        qgi = 0.
        for iion in range(nion):
            qgi = qgi + transport["QGI"][iion]
        # dpc end

        y = solver["Y"][ndim]
        dy = solver["DY"][ndim]

        if(te_bnd_type[1] == 0):
            y = profiles["TE"]
            dy = derivate(y, rho)

        te = y
        dte = dy

        if any(te < 0.0):
            raise RuntimeError('Error in the temperature equation: on-axis electron temperature is negative, stop')
        else:
            te[1:] = te[0:-1]
            dte[1:] = dte[0:-1]

        if(rho != 0.):  # FIXME
            fun2 = vpr/rho *                                         \
                (1.5*ne0*tem/tau*(vprm/vpr)**fivethird
                 + qe_exp + qie - qgi
                 - y * (1.5*ne/tau + qe_imp + vie
                        - B0dot/2./B0*rho*ne*dvpr/vpr))
        else:
            fun2 = (1.5*ne0*tem/tau
                    + qe_exp + qie - qgi
                    - y * (1.5*ne/tau + qe_imp + vie
                           - B0dot/2./B0*ne*dvpr))

        intfun2 = integral(fun2, rho)  # Integral source

        flux_te_conv = y*flux_ne

        flux_te_cond = vpr*gm3*ne * (y*vconv_te - dy*diff_te)

        flux_te = flux_te_conv + flux_te_cond

        int_source = intfun2 + y * 1.5*B0dot/2./B0*rho*ne*vpr

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
        profiles["DTE"] = dte  # AF, 25.Sep.2014
        profiles["DIFF_TE"] = diff_te
        profiles["VCONV_TE"] = vconv_te
        profiles["FLUX_TE"] = flux_te
        profiles["FLUX_TE_CONV"] = flux_te_conv
        profiles["FLUX_TE_COND"] = flux_te_cond
        profiles["SOURCE_TE"] = qe_exp + qie - qgi - (qe_imp + vie) * te
        profiles["INT_SOURCE_TE"] = int_source

        fun1 = profiles["SOURCE_TE"]*vpr
        intfun1 = integral(fun1, rho)
        profiles["INT_SOURCE_TE"] = intfun1

    #  ROTATION TRANSPORT EQUATIONS
    def _rotation(self,
                  core_profiles_iter,
                  core_profiles_prev,
                  *args,
                  transports=None,
                  sources=None,
                  boundary_condition=None,
                  hyper_diff=[0, 0],
                  **kwargs):

        # Allocate types for interface with PLASMA_COLLISIONS:
        self.allocate_collisionality(nrho, nion, collisions, ifail)

        # Allocate types for interface with numerical solver:
        self. allocate_numerics(ndim, nrho, solver, ifail)

        # Set up local variables:
        amix = control["AMIX"]
        tau = control["TAU"]
        solver_type = control["SOLVER_TYPE"]

        B0 = kwargs["BGEO"]
        B0m = kwargs["BTM"]
        B0dot = (B0-B0m)/tau

        # Flux surface averaged R {dynamic} [m]
        gm8 = equilibrium.profiles_1d.interpolate("gm8")(rho_tor_norm)
        gm3 = equilibrium.profiles_1d.interpolate("gm3")(rho_tor_norm)

        rho = kwargs["RHO"]
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
        solver["NRHO"] = nrho
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
            e = (vpr*gm3*ni*vconv + flux_ni - B0dot/2./B0*rho*ni*vpr) * gm8*mion
            f = vpr*(ui_exp + uzi)
            g = vpr*(ui_imp + wzi)

            h = tau

            # Boundary conditions for numerical solver in form:
            #
            #     V*Y' + U*Y =W

            # On axis
            #     dVtor,i/drho(rho=0)=0: # AF - 24.Jun.2010, replaces "#     dTi/drho(rho=0)=0:"
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
                v[1] = 0.
                u[1] = 1.
                w[1] = vtor_bnd[1, 0]

            #     FIXED grad_Vtor,i
            if(vtor_bnd_type[1] == 2):
                v[1] = 1.
                u[1] = 0.
                w[1] = -vtor_bnd[1, 0]

            #     FIXED L_Vtor,i
            if(vtor_bnd_type[1] == 3):
                v[1] = 1.
                u[1] = 1./vtor_bnd[1, 0]
                w[1] = 0.

            #     FIXED Flux_Mtor,i
            if(vtor_bnd_type[1] == 4):
                v[1] = -vpr[-1]*gm3[-1]*gm8[-1]*diff[-1]*ni[-1]*mion
                u[1] = vpr[-1]*gm3[-1]*gm8[-1]*vconv[-1] * \
                    ni[-1]*mion + gm8[-1]*flux_ni[-1]*mion
                w[1] = vtor_bnd[1, 0]

            #     Generic boundary condition
            if(vtor_bnd_type[1] == 5):
                v[1] = vtor_bnd[1, 0]
                u[1] = vtor_bnd(2, 2)
                w[1] = vtor_bnd(2, 3)

            # Rotation equation is not solved:
            if(vtor_bnd_type[1] == 0):
                dy = derivate(y, rho)
                flag = 0
                for irho in range(rho.shape[0]):
                    a = 1.0
                    b = 1.0
                    c = 1.0
                    d = 0.0
                    e = 0.0
                    f = 0.0
                    g = 0.0

                v[1] = 0.0
                u[1] = 1.0
                w[1] = y[-1]

            # Defining coefficients for numerical solver:
            solver["EQ_FLAG"][idim] = flag

            solver["RHO"] = rho

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
                solver["CM1"][iion, zion] = wii[irho, zion]

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
            y = solver["Y"](idim, irho)
            dy = solver["DY"](idim, irho)

            if(vtor_bnd_type[1] == 0):
                y = profiles["VTOR"][iion]
                dy = derivate(y, rho)

            # New rotation velocity and momentum flux:
            vtor = y
            dvtor = dy
            # dy 2017-10-06        WTOR(IRHO)           = Y(IRHO)/G2(NRHO)
            wtor = y/gm8

            mtor = gm8*ni*mion*y
            mtor_tot = mtor_tot + mtor

            if(rho != 0.):  # FIXME
                fun1 = vpr/rho * (ui_exp + uzi + (wzi + gm8*mion*ni/tau - ui_imp) * y)
            else:
                fun1 = (ui_exp + uzi + (wzi + gm8*mion*ni/tau - ui_imp) * y)

            intfun1 = integral(fun1, rho)  # Integral source

            flux_mtor_conv = gm8*mion*flux_ni*y

            flux_mtor_cond = vpr*gm3*gm8*mion*ni                          \
                * (y*vconv - dy*diff)

            flux_mtor = flux_mtor_conv + flux_mtor_cond

            int_source = intfun1 + vpr*gm8*B0dot/2./B0*rho*mion*ni*y

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
                profiles["DVTOR"][iion] = dvtor  # AF, 25.Sep.2014
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
            intfun1 = integral(fun1, rho)
            profiles["INT_SOURCE_MTOR"][iion] = intfun1

        for irho in range(rho.shape[0]):
            profiles["MTOR_TOT"] = mtor_tot
            profiles["FLUX_MTOR_TOT"] = flux_mtor_tot


if __name__ == "__main__":
    nrho = 128
    transport = Transport(
        rho=np.linspace(1.0/(nrho+1), 1, nrho, dtype=float),
        R0=1.0,
        Bt=1.0,
        Btm=1.0,
        fpol=np.ones(nrho, dtype=float),
        vprime=np.zeros(nrho, dtype=float),
        gm2=np.zeros(nrho, dtype=float),
        sigma=np.ones(nrho, dtype=float),
    )
    transport.current(
        psi=np.zeros(nrho, dtype=float),
        psim=np.zeros(nrho, dtype=float),
        dpsi=np.zeros(nrho, dtype=float),
        dpsim=np.zeros(nrho, dtype=float),
        qsf=np.ones(nrho, dtype=float)
    )


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
