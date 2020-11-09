import collections
import copy

import numpy as np
import scipy
from scipy import constants
from spdm.util.logger import logger
from spdm.util.AttributeTree import AttributeTree
from spdm.util.sp_export import sp_find_module
from spdm.util.Interpolate import integral, derivate, Interpolate1D
EPSILON = 1.0e-5

_SOLVER_COEFFICIENTS = collections.namedtuple("_SOLVER_COEFFICIENTS", "a b c d e f g h u v w")


class Transport:
    """
        Solve transport equation s

        Refs:
            - Hinton/Hazeltine, Rev. Mod. Phys. vol. 48 (1976), pp.239-308
            - David P. Coster, Vincent Basiuk, Grigori Pereverzev, Denis Kalupin, Roman Zagórksi, Roman Stankiewicz, Philippe Huynh, and Fréd…,
              "The European Transport Solver", IEEE Transactions on Plasma Science 38, 9 PART 1 (2010), pp. 2085--2092.
            - G V Pereverzev, P N Yushmanov, and Eta, "ASTRA–Automated System for Transport Analysis in a Tokamak",
              Max-Planck-Institut für Plasmaphysik (1991), 147.

            $  \\rho\\equiv\\sqrt{\\frac{\\Phi}{\\pi B_{0}}} $
            $  gm2 \\euqiv \\left\\langle \\left|\\frac{\\nabla\\rho}{R}\\right|^{2}\\right\\rangle $
    """
    @staticmethod
    def __new__(cls,  *args,   backend=None, **kwargs):
        if cls is not Transport:
            return super(Transport, cls).__new__(cls)

        if backend is not None:
            plugin_name = f"{__package__}.plugins.transport.Plugin{backend}"

            n_cls = sp_find_module(plugin_name, fragment=f"Transport{backend}")

            if n_cls is None:
                raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#Transport{backend}")
        else:
            n_cls = cls

        return object.__new__(n_cls)

    def __init__(self, *args,  **kwargs):
        pass

    def solve(self, core_profiles, equilibrium, enable_quasi_neutrality=True,
              psi_boundary=None,
              **kwargs):
        """Solve transport equations
            calculate new core_profiles at time = equilibrium.profiles_1d.time

            input:
                core_profiles   :=> imas_dd://core_profiles
                equilibrium     :=> imas_dd://equilibrium.time_slice
                transports      :=> imas_dd://core_transports
                sources,        :=> imas_dd://core_sources
        """
        core_profiles_new = AttributeTree()

        core_profiles_new.profiles_1d.grid = core_profiles.profiles_1d.grid

        core_profiles_new.time = equilibrium.time

        core_profiles_new.vacuum_toroidal_field = equilibrium.vacuum_toroidal_field

        if core_profiles_new.grid.dpsi is None or len(core_profiles_new.grid.dpsi) == 0:
            core_profiles_new.grid.dpsi = derivate(core_profiles_new.grid.psi, core_profiles_new.grid.rho)

        # current density profile:
        self.current(core_profiles, core_profiles_new,  equilibrium, boundary_condition=psi_boundary, **kwargs)

        # # ion density profiles:
        # self.ion_density(core_profiles,   core_profiles_new, equilibrium, **kwargs)

        # # electron density profile:
        # if enable_quasi_neutrality:
        #     self.electron_density(core_profiles, core_profiles_new,equilibrium,  **kwargs)

        # # electron/ion density profile from quasi-neutrality:
        # self.quasi_neutrality(core_profiles, core_profiles_new,equilibrium,  **kwargs)

        # # ion temperature profiles:
        # self.temperatures(core_profiles, core_profiles_new,equilibrium,  **kwargs)

        # # toroidal rotation profiles:
        # self.rotation(core_profiles, core_profiles_new,equilibrium,  **kwargs)

        self.update_global_quantities(core_profiles_new)

        return core_profiles_new

    def update_global_quantities(self, core_profiles):
        core_profiles.global_quantities = NotImplemented
        return core_profiles

    def solve_general_form(self, x,
                           y0, yp0,                 #
                           tau,                     # time step
                           a, b, c, d, e, f, g,     # coefficients, which are  fuction of x
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
            yp0 = derivate(y0, x)

        dd = derivate(d, x)
        de = derivate(e, x)

        A = (-dd+e)/d
        B = (c*a/tau+de+c*g)/d
        C = -c*b*y0/(tau*d)-c*f/d

        a = Interpolate1D(x, A)
        b = Interpolate1D(x, B)
        c = Interpolate1D(x, C)

        def fun(x, Y,  a=A, b=B, c=C):
            """
            Y=[y,y']

            $ \\frac{\\partial y}{\\partial\\rho}&=Y^{\\prime}  \\\\
              \\frac{\\partial y^{\\prime}}{\\partial\\rho}&=A\\cdot y^{\\prime}+B\\cdot y+C
            $
            dy/dx =y'
            dy'/dx=A*y'+B*Y+C
            """
            y, yp = Y
            dy = yp
            ddy = a(x)*yp+b(x)*y+c
            return [dy, ddy]

        def bc(Ya, Yb,   u=u, v=v, w=w):
            """
            u *y'(b)+v*y(b)=w

            Ya=[y(a),y'(a)]
            Yb=[y(b),y'(b)]
            P=[[u(a),u(b)],[v(a),v(b)],[w(a),w(b)]]

            """
            Ybc = [Ya, Yb]

            for k in range(2):
                if v[k] is None or v[k] == 1.0e-8:
                    Ybc[k][0] = -w[k]/u[k]
                else:
                    Ybc[k][1] = (w[k] - Ybc[k][0]*u[k])/v[k]

            return np.array([Ybc[0], Ybc[1]])

        solution = scipy.integrate.solve_bvp(fun, bc, x, [y0, yp0],  ** kwargs)

        if not solution.success:
            raise RuntimeError(solution.message)

        return solution.sol

    ############################################################################################
    #  TRANSPORT EQUATIONS:
    #  CURRENT TRANSPORT EQUATION
    def current(self,
                core_profiles,
                core_profiles_new,
                equilibrium,
                transports=None,
                sources=None,
                boundary_condition=None,
                **kwargs):

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
        B0prime = (B0 - B0m)/tau

        # -----------------------------------------------------------
        # Grid
        # $rho$ not  normalised minor radius                [m]
        rho = core_profiles_new.profiles_1d.grid.rho
        # $\Psi$ flux function from current                 [Wb]
        psi = core_profiles_new.profiles_1d.grid.psi
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        dpsi = core_profiles_new.profiles_1d.grid.dpsi
        # normalized psi  [0,1]                            [-]
        psi_norm = core_profiles.profiles_1d.grid.psi_norm

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fdia = equilibrium.profiles_1d.f(psi_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = equilibrium.profiles_1d.dvolume_dpsi(psi_norm)

        # $gm2 \euqiv \left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $  [m^-2]
        gm2 = equilibrium.profiles_1d.gm2(psi_norm)

        # -----------------------------------------------------------
        # Profile

        # $q$ safety factor                                 [-]
        qsf = core_profiles.profiles_1d.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        sigma = core_profiles.profiles_1d.conductivity_parallel

        # -----------------------------------------------------------
        # Sources
        # total non inductive current, PSI independent component,          [A/m^2]
        j_ni_exp = sources.profiles_1d.j_parallel if sources is not None else 0.0
        # total non inductive current, component proportional to PSI,      [A/m^2/V/s]
        j_ni_imp = sources.j_ni_imp if sources is not None else 0.0   # can not find data in imas dd

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
        c = constants.mu_0*B0*rho / fdia**2            # $\frac{\mu_{0}B_{0}\rho}{F^{2}}$
        d = vpr/(4.0*(constants.pi**2)*fdia)*gm2       # $\frac{V^{\prime}g_{3}}{4\pi^{2}F}$

        # $fun1=\frac{\sigma_{\parallel}\mu_{0}\rho^{2}}{F^{2}}$
        fun1 = sigma*constants.mu_0*(rho/fdia)**2

        # $dfun1=\frac{\partial}{\partial\rho}\frac{\sigma_{\parallel}\mu_{0}\rho^{2}}{F^{2}}$
        dfun1 = derivate(fun1, rho)

        # $e=-\text{FUN1}\frac{\dot{\Phi}_{b}}{2\Phi_{b}}$
        e = -fun1 * B0prime/2.0

        # $f=&-\frac{V^{\prime}}{2\pi\rho}j_{ni,exp}$
        f = -vpr/(2.0*constants.pi*rho) * j_ni_exp

        # $g=-\frac{V^{\prime}}{2\pi\rho}j_{ni,imp}+\sigma_{\parallel}\frac{\partial}{\partial\rho}\left(\frac{\sigma_{\parallel}\mu_{0}\rho^{2}}{F^{2}}\right)\cdot\frac{\dot{\Phi}_{b}}{2\Phi_{b}}$
        g = vpr/(2.0*constants.pi*rho) * j_ni_imp + B0prime/2.0*sigma*dfun1

        # $\frac{V^{\prime}}{4\pi^{2}}\left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $
        fun2 = vpr*gm2/(4.0*constants.pi**2)

        # for irho in range(nrho):  # fix magnetic axis
        #     if abs(rho[irho]) < EPSILON:
        #         f[irho] = 1.0/(2.0*constants.pi)*j_ni_exp[irho]
        #         g[irho] = 1.0/(2.0*constants.pi)*j_ni_imp[irho] + B0prime/2.0*sigma[irho]*dfun1[irho]

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
        if boundary_condition is None or boundary_condition.type == 0:          # Current equation is not solved:
            #  Interpretative value of safety factor should be given
            # if any(qsf != 0.0):  # FIXME
            dy = 2.0*constants.pi*B0*rho/qsf
            fun3 = 2.0*constants.pi*B0/qsf
            y = integral(fun3, rho)

            a[-1] = 1.0
            b[-1] = 1.0
            c[-1] = 1.0
            d[-1] = 0.0
            e[-1] = 0.0
            f[-1] = 0.0
            g[-1] = 0.0

            v[1] = 0.0
            u[1] = 1.0
            w[1] = y[-1]

        elif boundary_condition.type == 1:            # FIXED psi
            v[1] = 0.0
            u[1] = 1.0
            w[1] = boundary_condition.value[1][1]
        elif boundary_condition.type == 2:          # FIXED total current

            v[1] = 1.0
            u[1] = 0.0
            w[1] = - constants.mu_0/fun2[-1]*boundary_condition.value[1][0]
        elif boundary_condition.type == 3:          # FIXED loop voltage
            v[1] = 0.0
            u[1] = 1.0
            w[1] = tau*boundary_condition.value[1][0]+psi[-1]
        elif boundary_condition.type == 4:          # Generic boundary condition
            v[1], u[1], w[1] = boundary_condition.value[1]

        # Solution of current diffusion equation:
        try:
            sol = self.solve_general_form(rho, psi, dpsi, a, b, c, d, e, f, g, h, u, v, w)
        except RuntimeError as error:
            raise RuntimeError(f"Fail to solve current transport equation! \n {error} ")
        else:
            y = sol.y[0]
            yp = sol.yp[0]

        # New magnetic flux function and current density:
        # dpc2 = (c*((a*y-b*ym)/h + g*y - f) + e*y) / d
        intfun7 = integral(c*((a*y-b*psi)/h + g*y - f), rho)

        dy = intfun7/d + e/d*y

        # fun4 = fun2*dy
        # fun5 = fun2*dy*R0*B0/fdia

        dfun4 = derivate(fun2*dy, rho)  # Derivation of function FUN4
        dfun5 = derivate(fun2*dy*R0*B0/fdia, rho)  # Derivation of function FUN5

        # New profiles of plasma parameters obtained
        #     from current diffusion equation:

        core_profiles_new.profiles_1d.psi = y
        core_profiles_new.profiles_1d.dpsi = yp
        core_profiles_new.profiles_1d.q = 2.0*constants.pi*B0*rho/dpsi

        j_tor = - 2.0*constants.pi*R0/constants.mu_0/vpr * dfun4
        j_par = - 2.0*constants.pi/R0/constants.mu_0/vpr * (fdia/B0)**2*dfun5

        # $E_\parallel$  parallel electric field,,          [V/m]
        e_par = (j_par - j_ni_exp - j_ni_imp*y) / sigma
        core_profiles_new.profiles_1d.e_field.parallel = e_par

        # Total Ohmic currents                              [A]
        # fun7 = vpr * j_par / 2.0e0 / constants.pi * B0 / fdia**2
        intfun7 = integral(vpr * j_par / 2.0e0 / constants.pi * B0 / fdia**2, rho)
        core_profiles_new.profiles_1d.j_ohmic = intfun7[-1] * fdia[-1]

        # current density, toroidal,                        [A/m^2]
        core_profiles_new.profiles_1d.j_tor = j_tor

        # Total non-inductive currents                       [A]
        # fun7 = vpr * (j_ni_exp + j_ni_imp * psi) / (2.0e0 * constants.pi) * B0 / fdia**2
        intfun7 = integral(vpr * (j_ni_exp + j_ni_imp * psi) / (2.0e0 * constants.pi) * B0 / fdia**2, rho)
        core_profiles_new.profiles_1d.j_non_inductive = intfun7[-1] * fdia[-1]

        return True

    #  PARTICLE TRANSPORT EQUATIONS
    def ion_density(self,
                    core_profiles,
                    core_profiles_new,
                    equilibrium,
                    transports=None,
                    sources=None,
                    boundary_condition=None,
                    **kwargs):

        #-------------------------------------------------------#
        #     This subroutine solves ion particle transport     #
        #     equations for ion components from 1 to NION,      #
        #     and provides: density and flux of ion components  #
        #     from 1 to NION                                    #
        #-------------------------------------------------------#

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
        B0prime = (B0 - B0m)/tau

        # -----------------------------------------------------------
        # Grid
        # $rho$ not  normalised minor radius                [m]
        rho = core_profiles_new.profiles_1d.grid.rho
        # $\Psi$ flux function from current                 [Wb]
        psi = core_profiles_new.profiles_1d.grid.psi
        # $\frac{\partial\Psi}{\partial\rho}$               [Wb/m]
        dpsi = core_profiles_new.profiles_1d.grid.dpsi
        # normalized psi  [0,1]                            [-]
        psi_norm = core_profiles.profiles_1d.grid.psi_norm

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fdia = equilibrium.profiles_1d.f(psi_norm)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = equilibrium.profiles_1d.dvolume_dpsi(psi_norm)

        # $gm2 \euqiv \left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle $  [m^-2]
        gm2 = equilibrium.profiles_1d.gm2(psi_norm)

        # -----------------------------------------------------------
        # Profile

        # $q$ safety factor                                 [-]
        qsf = core_profiles.profiles_1d.q

        # plasma parallel conductivity,                     [(Ohm*m)^-1]
        sigma = core_profiles.profiles_1d.conductivity_parallel
        # hyper_diff_exp = hyper_diff[0]
        # hyper_diff_imp = self.hyper_diff[1]

        #    solution of particle transport equation for
        #    individual ion species

        # Set up boundary conditions for particular ion type:
        # ni_bnd_type = profiles["NI_BND_TYPE"][iion]
        # ni_bnd = profiles["NI_BND"][iion]

        # rho_bnd = profiles["NI_BND_RHO"][iion]

        # Set up local variables for particular ion type:

        ni = [ion.density for ion in core_profiles.profiles_1d.ion]
        dni = [ion.ddensity_drho or derivate(ion.density, rho) for ion in core_profiles.profiles_1d.ion]
        nim = kwargs["NIM"][iion]
        dnim = kwargs["DNIM"][iion]

        diff = 0.
        vconv = 0.

        for imodel in range(nmodel):
            c1[imodel] = transport["C1"][imodel]

            diff_mod[imodel] = transport["DIFF_NI"][iion, imodel]
            vconv_mod[imodel] = transport["VCONV_NI"][iion, imodel]

            diff = diff + diff_mod[imodel]
            vconv = vconv + vconv_mod[imodel]

        si_exp = 0.
        si_imp = 0.

        si_exp = si_exp + kwargs["SI_EXP"][iion]
        si_imp = si_imp + kwargs["SI_IMP"][iion]

        # Coefficients for ion diffusion equation  in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff)  # AF 25.Apr.2016, 22.Aug.2016

        y = ni
        dy = dni
        y0 = nim
        dy0 = dnim

        a = vpr
        b = vprm
        c = 1.
        #        D(IRHO)           = VPR(IRHO)*G1(IRHO)*DIFF(IRHO)
        d = vpr*g1*(diff+diff_hyper)  # AF 25.Apr.2016
        #        E(IRHO)           = VPR(IRHO)*G1(IRHO)*VCONV(IRHO)
        e = vpr*g1*(vconv+diff_hyper*dnim/nim) - \
            B0prime/2./B0*rho*vpr
        f = vpr*si_exp
        g = vpr*si_imp
        h = tau

        # Boundary conditions for ion diffusion equation in form:
        #
        #     V*Y' + U*Y =W
        #
        # On axis:
        #       dNi/drho(rho=0)=0:
        if(solver_type != 4):  # AF 11.Oct.2011
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
        if(ni_bnd_type[1] == 1):
            v[1] = 0.
            u[1] = 1.
            w[1] = ni_bnd[1, 0]

        #       FIXED grad_Ni
        elif(ni_bnd_type[1] == 2):
            v[1] = 1.
            u[1] = 0.
            w[1] = -ni_bnd[1, 0]

        #       FIXED L_Ni
        elif(ni_bnd_type[1] == 3):
            v[1] = ni_bnd[1, 0]
            u[1] = 1.
            w[1] = 0.

        #       FIXED Flux_Ni
        elif(ni_bnd_type[1] == 4):
            #        V[1] = -G(NRHO)*DIFF(NRHO)
            #        U[1] = G(NRHO)*VCONV(NRHO)
            v[1] = -vpr[-1]*g1[-1]*diff[-1]
            u[1] = vpr[-1]*g1[-1]*vconv[-1]
            w[1] = ni_bnd[1, 0]

        #       Generic boundary condition
        elif(ni_bnd_type[1] == 5):
            v[1] = ni_bnd[0, 1]
            u[1] = ni_bnd[1, 1]
            w[1] = ni_bnd[2, 1]

        # Density equation is not solved:
        if(ni_bnd_type[1] == 0):

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
        # Solution of current diffusion equation:
        sol = self.solve_general_form(rho, ni, dni, a, b, c, d, e, f, g, h, u, v, w)
        y = sol.y
        yp = sol.yp

        # dy check for nans in solution, return if true)
        if(any(np.isnan(y))):
            raise RuntimeError('Error in the ion density equation, nans in the solution, stop')

        # New ion density:

        if(ni_bnd_type[1] == 0):
            y[:] = profiles["NI"][:, iion]
            dy = derivate(y, rho)

        # New profiles of ion density flux and integral source:
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

        ni = y
        dni = dy
        if any([n <= 0.0 for n in i]):  # FIXME!!
            raise RuntimeError('Error in the density equation: on-axis ion density is negative, stop')
        # else:
        #     logger.debug('warning, density for ion {iion} and irho {irho} is negative, set it to the value at the previous irho')
        #     # ni[0] = ni(irho-1)
        #     # dni[0] = dni(irho-1)

        fun1 = (vpr*si_exp + vprm*nim/tau - ni*vpr*(1./tau+si_imp))

        #     INTEGR(NRHO,RHO,FUN1,INTFUN1)                                  #Integral source  #AF 11.Oct.2011
        # Integral source  #AF 11.Oct.2011 - this routine simply integrates the function, not the function times rho as INTEGR was doing
        intfun1 = integral(fun1, rho)

        local_fun1_s4 = (si_exp[0] + nim[0]/tau - ni[0]*(1./tau+si_imp[0]))/g1[0]  # stripping G1
        # stripped integral for the axis - from the INTEGR2 routine, first point only
        local_intfun1_s4 = local_fun1_s4*rho[0]/2.0

        int_source = intfun1 + B0prime/2./B0*rho*vpr*ni
        flux = vpr*g1 * (y*vconv - dy*diff)

        # Contribution to ion energy transport:
        flux_ni_conv = 0.

        for imodel in range(nmodel):
            flux_ni_conv = flux_ni_conv + c1[imodel]*vpr*g1 * (y*vconv_mod[imodel] - dy*diff_mod[imodel])

        # If equation is not solved, flux is determined
        #     by the integral of kwargs and transport coefficients
        #     are updated with effective values:
        if(ni_bnd_type[1] == 0):
            diff = 1.e-6
            flux = int_source
            flux_ni_conv = 1.5*int_source
            if((vpr*g1 != 0.0) and (dy != 0.0)):  # FIXME
                diff = - flux / dy / (vpr*g1)
            if (abs(diff) >= nine_diff_limit):
                diff = sign(nine_diff_limit, diff)
                vconv = 0.0
            if(diff <= 1.e-6 .AND. vpr*g1 != 0):
                diff = 1.e-6
                vconv = (flux / (max(abs(vpr), 1.e-6)*g1) + dy*diff) / max(abs(y), 1.e-6)

        # Return new ion density and flux profiles to the work flow:
        profiles["NI"][iion] = ni
        profiles["DNI"][iion] = dni
        profiles["DIFF_NI"][iion] = diff
        profiles["VCONV_NI"][iion] = vconv
        profiles["FLUX_NI"][iion] = flux
        profiles["FLUX_NI_CONV"][iion] = flux_ni_conv
        # dy 2017-10-06        PROFILES["SOURCE_NI(IRHO,IION)       = SI_EXP(IRHO) + SI_IMP(IRHO) * NI(IRHO)
        profiles["SOURCE_NI"][iion] = si_exp - si_imp * ni
        profiles["INT_SOURCE_NI"][iion] = int_source

        fun1 = profiles["SOURCE_NI"][iion]*vpr
        intfun1 = integral(fun1, rho)
        profiles["INT_SOURCE_NI"][iion] = intfun1

        # AF 11.Oct.2011 - local flux information for the axis boundary conditions in temperature and rotation equations

        local_int_source_s4[iion] = local_intfun1_s4 + B0prime/2./B0*rho[0]*ni[0]/g1[0]
        local_flux_ni_s4[iion] = (y[0]*vconv[0] - dy[0]*diff[0])

        local_flux_ni_conv_s4[iion] = 0.0
        for imodel in range(nmodel):
            local_flux_ni_conv_s4[iion] = local_flux_ni_conv_s4[iion] + \
                c1[imodel] * (y[0]*vconv_mod[1, imodel] - dy[0]*diff_mod[1, imodel])

        # If equation is not solved, flux is determined
        #     by the integral of kwargs:
        #        IF (NI_BND_TYPE[1] == 0) :
        if(1 == 0):
            local_flux_ni_s4[iion] = local_int_source_s4[iion]
            local_flux_ni_conv_s4[iion] = 1.5*local_int_source_s4[iion]

    def electron_density(self, *args, **kwargs):

        #-------------------------------------------------------#
        #     This subroutine solves electron transport         #
        #     equation and provides:                            #
        #     density and flux of electrons.                    #
        #-------------------------------------------------------#

        hyper_diff_exp = hyper_diff[0]  # AF 22.Aug.2016
        hyper_diff_imp = hyper_diff[1]  # AF 22.Aug.2016

        # Set up local variables:
        amix = control["AMIX"]
        tau = control["TAU"]
        solver_type = control["SOLVER_TYPE"]

        B0 = kwargs["BGEO"]
        B0m = kwargs["BTM"]
        B0prime = (B0-B0m)/tau

        rho = kwargs["RHO"]
        vpr = kwargs["VPR"]
        vprm = kwargs["VPRM"]
        g1 = kwargs["G1"]

        #    solution of particle transport equation

        # Set equation to 'predictive' and all coefficients to zero:
        flag = 1

        # Set up boundary conditions for particular ion type:
        ne_bnd_type[1] = profiles["NE_BND_TYPE"]
        ne_bnd = [[0, 0, 0], profiles["NE_BND"]]

        # Set up local variables for particular ion type:
        ne = profiles["NE"]
        dne = profiles["DNE"]
        nem = kwargs["NEM"]
        dnem = kwargs["DNEM"]

        diff.fill(0.0)
        vconv.fill(0.0)

        for imodel in range(nmodel):
            c1[imodel] = transport["C1"][imodel]

            diff_mod[imodel] = transport["DIFF_NE"][imodel]
            vconv_mod[imodel] = transport["VCONV_NE"][imodel]

            diff = diff + diff_mod[imodel]
            vconv = vconv + vconv_mod[imodel]

        se_exp = kwargs["SE_EXP"]
        se_imp = kwargs["SE_IMP"]

        # Coefficients for electron diffusion equation in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff)  # AF 25.Apr.2016, 22.Aug.2016

        y = ne
        dy = dne
        ym = nem
        dym = dnem

        a = vpr
        b = vprm
        c = 1.
        #        D(IRHO)           = VPR(IRHO)*G1(IRHO)*DIFF(IRHO)
        d = vpr*g1*(diff+diff_hyper)  # AF 25.Apr.2016
        #        E(IRHO)           = VPR(IRHO)*G1(IRHO)*VCONV(IRHO)                    \
        e = vpr*g1*(vconv+diff_hyper*dnem/nem) - B0prime/2./B0*rho*vpr
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
                u[0] = vconv[0]+diff_hyper*dnem[0]/nem[0]  # AF 25.Apr.2016

            w[0] = 0.

        # At the edge:
        #       FIXED Ne
        if(ne_bnd_type[1] == 1):
            v[1] = 0.
            u[1] = 1.
            w[1] = ne_bnd[1, 0]
        #       FIXED grad_Ne
        elif(ne_bnd_type[1] == 2):
            v[1] = 1.
            u[1] = 0.
            w[1] = -ne_bnd[1, 0]

        #       FIXED L_Ne
        elif(ne_bnd_type[1] == 3):
            v[1] = ne_bnd[1, 0]
            u[1] = 1.
            w[1] = 0.

        #       FIXED Flux_Ne
        elif(ne_bnd_type[1] == 4):
            v[1] = -vpr[-1]*g1[-1]*diff[-1]
            u[1] = vpr[-1]*g1[-1]*vconv[-1]
            w[1] = ne_bnd[1, 0]

        #       Generic boundary condition
        elif(ne_bnd_type[1] == 5):
            v[1] = ne_bnd[1, 0]
            u[1] = ne_bnd(2, 2)
            w[1] = ne_bnd(2, 3)

        # Density equation is not solved:
        elif(ne_bnd_type[1] == 0):

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

        # Defining coefficients for numerical solver:
        solver["TYPE"] = solver_type
        solver["EQ_FLAG"][ndim] = flag
        solver["NDIM"] = ndim
        solver["NRHO"] = nrho
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

        solver["H"] = h

        solver["V"][ndim] = v
        solver["U"][ndim] = u
        solver["W"][ndim] = w

        # Solution of density diffusion equation:
        self.solve_eq(a, b, c, d, e, f, g, h)

        # dy check for nans in solution, return if true)
        if(any(np.isnan(solver["y"]))):
            raise RuntimeError('Error in the electron density equation: nans in the solution, stop')

        # New electron density:
        y = solver["Y"][ndim]
        dy = solver["DY"][ndim]

        if(ne_bnd_type[1] == 0):
            y = profiles["NE"]
            dy = derivate(y, rho)

        # New profiles of electron density flux and integral source:
        ne = y
        dne = dy
        if any(ne <= 0.0):
            raise RuntimeError('Error in the density equation: on-axis electron density is negative, stop')
        else:
            ne = ne(irho-1)
            dne = dne(irho-1)

        fun1 = vpr*se_exp + vprm*nem/tau - ne*vpr*(1./tau+se_imp)

        intfun1 = integral(fun1, rho)

        local_fun1_s4 = (se_exp[0] + nem[0]/tau - ne[0]*(1./tau+se_imp[0]))/g1[0]
        local_intfun1_s4 = local_fun1_s4*rho[0]/2.

        int_source = intfun1 + B0prime/2./B0*rho*vpr*ne
        flux = vpr*g1 * (y*vconv - dy*diff)

        # Contribution to electron energy transport:
        flux_ne_conv = 0.

        for imodel in range(nmodel):
            flux_ne_conv = flux_ne_conv + c1[imodel]*vpr*g1 * (y*vconv_mod[imodel] - dy*diff_mod[imodel])

        # If equation is not solved, flux is determined
        #     by the integral of kwargs:
        if(ne_bnd_type[1] == 0):
            diff = 1.e-6
            flux = int_source
            flux_ne_conv = 1.5*int_source
            if(vpr*g1 != 0.0):
                diff = - flux / dy / (vpr*g1)
            if (abs(diff) >= nine_diff_limit):
                diff = sign(nine_diff_limit, diff)
                vconv = 0.0
            if(diff <= 1.e-6):
                diff = 1.e-6
                vconv = (flux / (max(abs(vpr), 1.e-6)*g1) + dy*diff) / y

        # Return new ion density and flux profiles to the work flow:
        profiles["NE"] = ne
        profiles["DNE"] = dne
        profiles["DIFF_NE"] = diff
        profiles["VCONV_NE"] = vconv
        profiles["FLUX_NE"] = flux
        profiles["FLUX_NE_CONV"] = flux_ne_conv
        # dy 2017-10-06        PROFILES["SOURCE_NE(IRHO)       = SE_EXP(IRHO) + SE_IMP(IRHO) * NE(IRHO)
        profiles["SOURCE_NE"] = se_exp - se_imp * ne
        profiles["INT_SOURCE_NE"] = int_source

        fun1 = profiles["SOURCE_NE"]*vpr
        intfun1 = integral(fun1, rho)
        profiles["INT_SOURCE_NE"] = intfun1

        local_int_source_s4 = local_intfun1_s4 + B0prime/2./B0*rho[0]*ne[0]/g1[0]
        local_flux_ne_s4 = (y[0]*vconv[0] - dy[0]*diff[0])

        local_flux_ne_conv_s4 = 0.
        for imodel in range(nmodel):
            local_flux_ne_conv_s4 = local_flux_ne_conv_s4 + c1[imodel] * \
                (y[0]*vconv_mod(1, imodel) - dy[0]*diff_mod(1, imodel))

        if(1 == 0):
            local_flux_ne_s4 = local_int_source_s4
            local_flux_ne_conv_s4 = 1.5*local_int_source_s4

    def quasi_neutrality(self, *args, **kwargs):

        # DIAG)

        #-------------------------------------------------------#
        #     This subroutine calculates electron density,      #
        #     electron flux plasma effective charge and         #
        #     convective contribution to electron energy        #
        #     transport from density and flux of background     #
        #     ions (all ion components computed by the ETS)     #
        #     and impurity ions (all ion components computed    #
        #     by separate impurity routine)                     #
        #     using quasi-neutrality condition                  #
        #-------------------------------------------------------#

        # Set up local variables:

        rho = kwargs["RHO"]

        ne = profiles["NE"]
        dne = profiles["DNE"]  # AF, 25.Sep.2014
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

        if(self.QUASI_NEUT == 0):
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

        elif(self.QUASI_NEUT == 1 or self.QUASI_NEUT == 2):
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

            if(self.QUASI_NEUT == 1):
                for iion in range(nion):
                    if(profiles["NI_BND_TYPE"][iion] == 0):
                        profiles["NI"][iion] = ni_tot / zion_av*ion_fraction[iion]
                        profiles["FLUX_NI"][iion] = flux_ni_tot / zion_av*ion_fraction[iion]
                        profiles["CONTRIB_2_ENERGY_FLUX_NI"][iion] = contrib_2_energy_flux_ni_tot / \
                            zion_av*ion_fraction[iion]

            if(self.QUASI_NEUT == 2):
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
                    if(self.QUASI_NEUT == 1):
                        # LOCAL_FLUX_NI_CONV_S4 = LOCAL_FLUX_NE_CONV_S4/ZTOT*FION(IION)
                        local_flux_ni_conv_s4 = local_flux_ne_conv_s4/zion_av*ion_fraction[iion]
                    if(self.QUASI_NEUT == 2):
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
        if (self.QUASI_NEUT == -1):

            if any(ne <= 0.0):
                raise RuntimeError('Error in the density equation: on-axis electron density is negative, stop')
            # else:
                # ne = ne(irho-1)

            profiles["ne"] = ne

    # HEAT TRANSPORT EQUATIONS
    def temperatures(self, *args, **kwargs):
        #-------------------------------------------------------#
        #     This subroutine solves transport equations        #
        #     for ion components from 1 to NION and electrons,  #
        #     and provides: temperatures, heat fluxes and       #
        #     its convective and conductive components          #
        #-------------------------------------------------------#

        hyper_diff_exp = hyper_diff[0]  # AF 22.Aug.2016
        hyper_diff_imp = hyper_diff[1]  # AF 22.Aug.2016

        dvpr = derivate(vpr, rho)  # Derivation of V'

        # Energy exchange terms due to collisions
        #     (defined from previous iteration):
        self.plasma_collisions(kwargs, profiles, collisions, ifail)

        vie = collisions["VIE"]  # DPC 2009-01-19

        # Parameters for numerical solver
        #     (common for all components):
        solver["TYPE"] = solver_type
        solver["NDIM"] = ndim
        solver["NRHO"] = nrho
        solver["AMIX"] = amix
        solver["DERIVATIVE_FLAG"][0] = 0

        solver["RHO"] = rho

        #-------------------------------------------------------#
        #                                                       #
        #                  ION HEAT TRANSPORT:                  #
        #                                                       #
        #-------------------------------------------------------#

        for iion in range(nion):

            # Set equation to 'predictive' and all coefficients to zero:
            flag = 1
            y = 0.0
            dy = 0.0
            ym = 0.0
            dym = 0.0
            a = 0.0
            b = 0.0
            c = 0.0
            d = 0.0
            e = 0.0
            f = 0.0
            g = 0.0
            h = 0.0
            v = 0.0
            u = 0.0
            w = 0.0

            # Set up boundary conditions for particular ion type:
            ti_bnd_type[iion] = [[0, 0, 0], profiles["TI_BND_TYPE"][iion]]
            ti_bnd = [[0, 0, 0], profiles["TI_BND"][iion]]

            # Set up local variables for particular ion type:
            ti = profiles["TI"][iion]
            dti = profiles["DTI"][iion]
            tim = kwargs["TIM"][iion]
            dtim = kwargs["DTIM"][iion]
            ni = profiles["NI"][iion]
            dni = profiles["DNI"][iion]
            nim = kwargs["NIM"][iion]
            dnim = kwargs["DNIM"][iion]

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

            for zion in range(nion):
                # DPC was           VII(IRHO,IION)   = COLLISIONS["VII(IRHO,IION,ZION)   ### check if the following is what was intended
                vii[zion] = collisions["VII"][iion, zion]

            # Coefficients for ion heat transport equation in form:
            #
            #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

            diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff_ti)  # AF 25.Apr.2016, 22.Aug.2016

            y = ti
            dy = dti
            ym = tim
            dym = dtim

            a = 1.5*vpr*ni
            # DPC temporary "fix" for NaNs
            if(vpr <= 0.0 and irho == 1):
                b = 0.0
            else:
                b = 1.5*vprm**fivethird/vpr**twothird*nim

            # DPC end of temporary fix
            c = 1.0
            #        D(IRHO)   = VPR(IRHO)*G1(IRHO)*NI(IRHO)*DIFF_TI(IRHO)
            d = vpr*g1*ni*(diff_ti+diff_hyper)  # AF 25.Apr.2016
            #       E(IRHO)   = VPR(IRHO)*G1(IRHO)*NI(IRHO)*VCONV_TI(IRHO)
            e = vpr*g1*ni*(vconv_ti+diff_hyper*dtim/tim) + flux_ni - 1.5*B0prime/2./B0*rho*ni*vpr
            f = vpr * (qi_exp + qei + qzi + qgi)
            g = vpr*(qi_imp + vei + vzi) - B0prime/2./B0*rho*ni*dvpr

            h = tau

            # Boundary conditions for ion heat transport equation in form:
            #
            #     V*Y' + U*Y =W

            # On axis:

            #       dTi/drho(rho=0)=0
            if(solver_type != 4):
                v[0] = 1.
                u[0] = 0.
            else:
                if((diff_ti[0]+diff_hyper) > 1.0e-6):
                    v[0] = -(diff_ti[0]+diff_hyper)*ni[0]
                else:
                    v[0] = -1.0e-6*ni[0]

                u[0] = (vconv_ti[0]+diff_hyper*dtim[0]/tim[0])*ni[0]+local_flux_ni_conv_s4[iion]
            w[0] = 0.0

            # At the edge:

            #       FIXED Ti
            if(ti_bnd_type(2, iion) == 1):
                v[1] = 0.
                u[1] = 1.
                w[1] = ti_bnd[1, 0]

            #       FIXED grad_Ti
            elif(ti_bnd_type(2, iion) == 2):
                v[1] = 1.
                u[1] = 0.
                w[1] = -ti_bnd[1, 0]

            #       FIXED L_Ti
            if(ti_bnd_type(2, iion) == 3):
                v[1] = ti_bnd[1, 0]
                u[1] = 1.
                w[1] = 0.

            #       FIXED Flux_Ti
            elif(ti_bnd_type(2, iion) == 4):
                v[1] = -vpr[-1]*g1[-1]*diff_ti[-1]*ni[-1]
                u[1] = vpr[-1]*g1[-1]*vconv_ti[-1]*ni[-1]+flux_ni[-1]
                w[1] = ti_bnd[1, 0]

            #       Generic boundary condition
            elif(ti_bnd_type(2, iion) == 5):
                v[1] = ti_bnd[1, 0]
                u[1] = ti_bnd(2, 2)
                w[1] = ti_bnd(2, 3)

            # Temperature equation is not solved:
            elif(ti_bnd_type(2, iion) == 0):
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
            solver["EQ_FLAG"][iion] = flag

            solver["Y"][iion] = y
            solver["DY"][iion] = dy
            solver["YM"][iion] = ym

            solver["A"][iion] = a
            solver["B"][iion] = b
            solver["C"][iion] = c
            solver["D"][iion] = d
            solver["E"][iion] = e
            solver["F"][iion] = f
            solver["G"][iion] = g

            solver["CM1"][ndim, iion] = vie
            solver["CM1"][iion, ndim] = vie

            for zion in range(nion):
                solver["CM1"][iion, zion] = vii[zion]

            solver["H"] = h
            solver["V"][iion] = v
            solver["U"][iion] = u
            solver["W"][iion] = w

        #-------------------------------------------------------#
        #                                                       #
        #               ELECTRON HEAT TRANSPORT:                #
        #                                                       #
        #-------------------------------------------------------#

        # Set equation to 'predictive' and all coefficients to zero:
        flag = 1
        y = 0.0
        dy = 0.0
        ym = 0.0
        dym = 0.0
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0
        e = 0.0
        f = 0.0
        g = 0.0
        h = 0.0
        v = 0.0
        u = 0.0
        w = 0.0

        # Set up local variables for electron heat transport equation:
        te = profiles["TE"]
        dte = profiles["DTE"]
        ne = profiles["NE"]
        dne = profiles["DNE"]
        tem = kwargs["TEM"]
        dtem = kwargs["DTEM"]
        nem = kwargs["NEM"]
        dnem = kwargs["DNEM"]

        flux_ne = profiles["FLUX_NE_CONV"]

        diff_te = transport["DIFF_TE"]
        vconv_te = transport["VCONV_TE"]
        qgi = 0.
        for iion in range(nion):
            qgi = qgi + transport["QGI"][iion]

        qe_exp = kwargs["QOH"] / itm_ev
        qe_imp = 0.

        qe_exp = qe_exp + kwargs["QE_EXP"]
        qe_imp = qe_imp + kwargs["QE_IMP"]

        qie = collisions["QIE"]
        vie = collisions["VIE"]

        # Set up boundary conditions for electron heat transport equation:
        te_bnd_type = profiles["TE_BND_TYPE"]
        te_bnd = profiles["TE_BND"]

        # Coefficients for electron heat transport equation in form:
        #
        #     (A*Y-B*Y(t-1))/H + 1/C * (-D*Y' + E*Y) = F - G*Y

        diff_hyper = hyper_diff_exp + hyper_diff_imp*maxval(diff_te)  # AF 25.Apr.2016, 22.Aug.2016

        for irho in range(rho.shape[0], 1, -1):
            y = te
            dy = dte
            ym = tem
            dym = dtem

            a = 1.5*vpr*ne
            # DPC temporary "fix" for NaNs
            if(vpr <= 0.0 and irho == 1):
                #        #write(*,*) B(2:5)
                b = 0.0
            else:
                b = 1.5*vprm**fivethird/vpr**twothird*nem

            # DPC end of temporary fix
            c = 1.
            #     D(IRHO)   = VPR(IRHO)*G1(IRHO)*NE(IRHO)*DIFF_TE(IRHO)
            d = vpr*g1*ne*(diff_te+diff_hyper)
            #     E(IRHO)   = VPR(IRHO)*G1(IRHO)*NE(IRHO)*VCONV_TE(IRHO)
            e = vpr*g1*ne*(vconv_te+diff_hyper*dtem /
                           tem) + flux_ne - 1.5*B0prime/2./B0*rho*ne*vpr
            f = vpr * (qe_exp + qie - qgi)
            g = vpr * (qe_imp + vie) - B0prime/2./B0*rho*ne*dvpr

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
            v[1] = -vpr[-1]*g1[-1]*diff_te[-1]*ne[-1]
            u[1] = vpr[-1]*g1[-1]*vconv_te[-1]*ne[-1]+flux_ne[-1]
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
            tim = kwargs["TIM"][iion]
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
                # write(*, *) 'warning, temperatuire for ion ', iion, ' and irho ', irho, 'is negative, set it to the value at the previous irho'
                ti[1:] = ti[0:-1]
                dti[1:] = dti(irho-1)
                # end if
            # end if
            if(rho != 0.):  # FIXME
                fun1 = vpr/rho *                                           \
                    ((1.5*nim*tim/tau*(vprm/vpr)**fivethird
                      + qi_exp + qei + qzi + qgi)
                     - (1.5*ni/tau + qi_imp + vei + vzi
                        - B0prime/2./B0*rho*ni*dvpr) * y)
            else:
                fun1 = ((1.5*nim*tim/tau
                         + qi_exp + qei + qzi + qgi)
                        - (1.5*ni/tau + qi_imp + vei + vzi
                           - B0prime/2./B0*rho*ni*dvpr) * y)

            intfun1 = integral(fun1, rho)  # Integral source

            for irho in range(rho.shape[0]):
                flux_ti_conv = y*flux_ni

                flux_ti_cond = vpr*g1*ni                                     \
                    * (y*vconv_ti - dy*diff_ti)

                flux_ti = flux_ti_conv + flux_ti_cond

                int_source = intfun1 + y * 1.5*B0prime/2./B0*rho*ni*vpr

                # If equation is not solved, total and conductive ion heat flux
                #     are determined from the integral of kwargs:
                if(ti_bnd_type(2, iion) == 0):

                    diff_ti = 1.e-6
                    flux_ti = int_source
                    flux_ti_cond = int_source - flux_ni*y

                    if((vpr*g1 != 0.0)):
                        # dy limit also DY if less than sqrt(epsilon(1.0))
                        diff_ti = - flux_ti_cond / sign(max(abs(dy), sqrt(epsilon(1.0))), dy) / (vpr*g1*ni)
                        # dy further limit diff_ti
                    if (abs(diff_ti) >= tite_diff_limit):
                        diff_ti = sign(tite_diff_limit, diff_ti)
                        vconv_ti = 0.0
                    if(diff_ti <= 1.e-6):
                        diff_ti = 1.e-6
                        vconv_ti = (flux_ti_cond / (max(abs(vpr), 1.e-6)*g1*ni) + dy*diff_ti) / max(abs(y), 1.e-6)

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
                (1.5*nem*tem/tau*(vprm/vpr)**fivethird
                 + qe_exp + qie - qgi
                 - y * (1.5*ne/tau + qe_imp + vie
                        - B0prime/2./B0*rho*ne*dvpr/vpr))
        else:
            fun2 = (1.5*nem*tem/tau
                    + qe_exp + qie - qgi
                    - y * (1.5*ne/tau + qe_imp + vie
                           - B0prime/2./B0*ne*dvpr))

        intfun2 = integral(fun2, rho)  # Integral source

        flux_te_conv = y*flux_ne

        flux_te_cond = vpr*g1*ne * (y*vconv_te - dy*diff_te)

        flux_te = flux_te_conv + flux_te_cond

        int_source = intfun2 + y * 1.5*B0prime/2./B0*rho*ne*vpr

        # If equation is not solved, conductive component of electron heat flux
        #     is determined from the integral of kwargs:
        if(te_bnd_type[1] == 0):

            diff_te = 1.e-6
            flux_te = int_source

            flux_te_cond = int_source - flux_ne*y

            if(vpr*g1 != 0.0):  # FIXME !!
                # dy limit also DY if less than sqrt(epsilon(1.0))
                diff_te = - flux_te_cond / sign(max(abs(dy), sqrt(epsilon(1.0))), dy) / (vpr*g1*ne)
                # dy further limit diff_ti
            if (abs(diff_te) >= tite_diff_limit):
                diff_te = sign(tite_diff_limit, diff_te)
                vconv_te = 0.0
            if(diff_te <= 1.e-6):
                diff_te = 1.e-6
                vconv_te = (flux_te_cond / (max(abs(vpr), 1.e-6)*g1*ne) + dy*diff_te) / y

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
    def rotation(self, *args, **kwargs):

        #-------------------------------------------------------#
        #     This subroutine solves the momentum transport     #
        #     equations for ion components fron 1 to NION,      #
        #     and provides: ion toroidal rotation velocity,ion  #
        #     angular velocity, ion momentum (total and         #
        #     individual per ion component), ion momentum flux  #
        #     (total and individual per ion component),         #
        #-------------------------------------------------------#

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
        B0prime = (B0-B0m)/tau

        for irho in range(rho.shape[0]):
            rho = kwargs["RHO"]
            vpr = kwargs["VPR"]
            vprm = kwargs["VPRM"]
            g1 = kwargs["G1"]
            g2 = kwargs["G2"]
            g2m = kwargs["G2M"]

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

            a = vpr*g2*ni*mion
            b = vprm*g2m*nim*mion
            c = 1.
            # AF, 14.May.2011 - multipication by G2, which in analytics is 1
            d = vpr*g1*ni*mion*diff*g2
            e = (vpr*g1*ni*vconv + flux_ni - B0prime/2./B0*rho*ni*vpr) * g2*mion
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
                v[1] = -vpr[-1]*g1[-1]*g2[-1]*diff[-1]*ni[-1]*mion
                u[1] = vpr[-1]*g1[-1]*g2[-1]*vconv[-1] * \
                    ni[-1]*mion + g2[-1]*flux_ni[-1]*mion
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
            wtor = y/g2

            mtor = g2*ni*mion*y
            mtor_tot = mtor_tot + mtor

            if(rho != 0.):  # FIXME
                fun1 = vpr/rho * (ui_exp + uzi + (wzi + g2*mion*ni/tau - ui_imp) * y)
            else:
                fun1 = (ui_exp + uzi + (wzi + g2*mion*ni/tau - ui_imp) * y)

            intfun1 = integral(fun1, rho)  # Integral source

            flux_mtor_conv = g2*mion*flux_ni*y

            flux_mtor_cond = vpr*g1*g2*mion*ni                          \
                * (y*vconv - dy*diff)

            flux_mtor = flux_mtor_conv + flux_mtor_cond

            int_source = intfun1 + vpr*g2*B0prime/2./B0*rho*mion*ni*y

            # if equation is not solved, conductive component of electron heat flux is determined from the integral of kwargs
            if(vtor_bnd_type[1] == 0):

                diff = 1.e-6
                flux_mtor = int_source

                flux_mtor_cond = int_source - g2*mion*flux_ni*y

                if((vpr*g1*g2*mion*ni != 0.0) and (dy != 0.0)):
                    diff = - flux_mtor_cond / dy / (vpr*g1*g2*mion*ni)
                    vconv = 0.0
                if(diff <= 1.e-6):
                    diff = 1.e-6
                    vconv = (flux_mtor_cond / (max(abs(vpr), 1.e-6)*g1*g2
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

    #  MATHEMATICAL defS

    def f_axis(self,   r, f):
        """
              This subroutine finds
              f(r_1=0) from f(r_2), f(r_3) and f(r_4)
        """
        h = np.zeros(r.shape)
        for i in range(1):
            h[i] = r[i+1]-r[i]

        f[0] = ((f[1]*r[3]/h[1]+f[3]*r[1]/h[2])*r[2]-f[2]*(r[1]/h[1]+r[3]/h[2])*r[1]*r[3]/r[2]) / (r[3]-r[1])

    def f_par_axis(self,  r, f):
        if n < 3:
            raise RuntimeError('n too small in F_par_AXIS')
        d1 = r[1]-r[0]
        if d1 < 0.0:
            raise RuntimeError('d1 <= 0 in F_par_AXIS')
        d2 = r[2]-r[1]
        if d2 < 0:
            raise RuntimeError('d2 <= 0 in F_par_AXIS')
        f[0] = f[1]-d1**2*(f[2]-f[1])/(d2**2+2*d1*d2)


if __name__ == "__main__":
    nrho = 128
    transport = Transport(
        rho=np.linspace(1.0/(nrho+1), 1, nrho, dtype=float),
        R0=1.0,
        Bt=1.0,
        Btm=1.0,
        fdia=np.ones(nrho, dtype=float),
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
