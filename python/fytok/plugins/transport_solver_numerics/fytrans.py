import typing
import numpy as np
import scipy.constants
from spdm.data.Expression import Variable, Expression
from spdm.data.Function import Function
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_
from spdm.numlib.bvp import solve_bvp

from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.atoms import atoms
from fytok.utils.logger import logger


@TransportSolverNumerics.register(["fytrans"])
class FyTrans(TransportSolverNumerics):
    r"""
        Solve transport equations $\rho=\sqrt{ \Phi/\pi B_{0}}$
        See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`

            Solve transport equations

            Current Equation

            Args:
                core_profiles       : profiles at :math:`t-1`
                equilibrium         : Equilibrium
                transports          : CoreTransport
                sources             : CoreSources
                boundary_condition  :

            Note:
                .. $$  \sigma_{\parallel}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho} \right) \psi= \
                            \frac{F^{2}}{\mu_{0}B_{0}\rho}\frac{\partial}{\partial\rho}\left[\frac{V^{\prime}}{4\pi^{2}}\left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle \
                            \frac{1}{F}\frac{\partial\psi}{\partial\rho}\right]-\frac{V^{\prime}}{2\pi\rho}\left(j_{ni,exp}+j_{ni,imp}\psi\right)
                    $$


                if $\psi$ is not solved, then

                .. $$  \psi =\int_{0}^{\rho}\frac{2\pi B_{0}}{q}\rho d\rho$$

            Particle Transport
            Note:

                ..$$
                    \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                    \left(V^{\prime}n_{s}\right)+\frac{\partial}{\partial\rho}\Gamma_{s}=\
                    V^{\prime}\left(S_{s,exp}-S_{s,imp}\cdot n_{s}\right)
                 $$

                ..$$
                    \Gamma_{s}\equiv-D_{s}\cdot\frac{\partial n_{s}}{\partial\rho}+v_{s}^{pinch}\cdot n_{s}
                  $$

            Heat transport equations

            Note:

                ion

                .. $$ \frac{3}{2}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                            \left(n_{i}T_{i}V^{\prime\frac{5}{3}}\right)+V^{\prime\frac{2}{3}}\frac{\partial}{\partial\rho}\left(q_{i}+T_{i}\gamma_{i}\right)=\
                            V^{\prime\frac{5}{3}}\left[Q_{i,exp}-Q_{i,imp}\cdot T_{i}+Q_{ei}+Q_{zi}+Q_{\gamma i}\right]
                   $$

                electron

                ..$$\frac{3}{2}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                            \left(n_{e}T_{e}V^{\prime\frac{5}{3}}\right)+V^{\prime\frac{2}{3}}\frac{\partial}{\partial\rho}\left(q_{e}+T_{e}\gamma_{e}\right)=
                            V^{\prime\frac{5}{3}}\left[Q_{e,exp}-Q_{e,imp}\cdot T_{e}+Q_{ei}-Q_{\gamma i}\right]
                  $$ transport_electron_temperature
        """

    def _update_coefficient(
        self,
        current: TransportSolverNumerics.TimeSlice,
        previous: TransportSolverNumerics.TimeSlice | None,
        *args,
        equilibrium: Equilibrium,
        core_transport: CoreTransport,
        core_sources: CoreSources,
        **kwargs,
    ) -> typing.Tuple[TransportSolverNumerics.TimeSlice.Solver1D, typing.List[Variable], float]:
        solver_1d = current.solver_1d

        if previous is not None and previous is not _not_found_:
            dt = current.time - previous.time
            one_over_dt = 1.0 / dt
            solver_1d_previous = previous.solver_1d
        else:
            dt = 0
            one_over_dt = 0
            solver_1d_previous = None

        hyper_diff = self.code.parameters.get("hyper_diff", 0.001)

        x = Variable(0, "x")

        vars = {"x": x}

        equilibrium_ = equilibrium.time_slice.current

        equilibrium_prev = equilibrium.time_slice.previous

        eq_1d = equilibrium.time_slice.current.profiles_1d

        eq_1d_prev = (
            equilibrium.time_slice.previous.profiles_1d if equilibrium.time_slice.previous is not _not_found_ else None
        )

        idx = 0
        for equ in solver_1d.equation:
            name = equ.primary_quantity.identifier
            vars[name] = Variable(idx := idx + 1, name, label=equ.primary_quantity.label or name)
            vars[name + "_flux"] = Variable(idx := idx + 1, name + "_flux", label=name + "_flux")

        psi = Function(solver_1d.grid.psi, solver_1d.grid.rho_tor_norm, label="psi")(x)

        # $R_0$ characteristic major radius of the device   [m]
        R0 = equilibrium_.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = equilibrium_.vacuum_toroidal_field.b0

        B0m = equilibrium_prev.vacuum_toroidal_field.b0 if equilibrium_prev is not _not_found_ else np.nan

        k_B = (B0 - B0m) / (B0 + B0m) * 2.0 * one_over_dt

        # Mesh
        rho_tor_boundary = eq_1d.grid.rho_tor_boundary

        rho_tor_boundary_m = eq_1d_prev.grid.rho_tor_boundary if eq_1d_prev is not None else 0

        k_rho_bdry = (
            (rho_tor_boundary - rho_tor_boundary_m) / (rho_tor_boundary + rho_tor_boundary_m) * 2.0 * one_over_dt
        )

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary * x

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = eq_1d.f(psi)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]

        vpr = eq_1d.dvolume_drho_tor(psi)

        vprm = eq_1d_prev.dvolume_drho_tor(psi) if eq_1d_prev is not None else np.nan

        inv_vpr23 = vpr ** (-2 / 3)

        fpol = eq_1d.f(psi)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = eq_1d.q(psi)
        gm1 = eq_1d.gm1(psi)  # <1/R^2>
        gm2 = eq_1d.gm2(psi)  # <|grad_rho_tor|^2/R^2>
        gm3 = eq_1d.gm3(psi)  # <|grad_rho_tor|^2>
        gm8 = eq_1d.gm8(psi)  # <R>
        # Qimp_k_ns = (3*k_rho_bdry - k_phi * Function(vpr, x).d())

        density = {}
        # quasi_neutrality_condition
        if "electons.density_thermal" not in vars:
            ne = 0
            for k, v in vars.items():
                spec = k.removesuffix("/density_thermal")
                if k != spec:
                    density[spec] = v
                    ne -= atoms[spec.removeprefix("ion/")].z * v

        else:
            ne = vars["electons/density_thermal"]
            z_of_ions = 0
            for k, v in vars.items():
                spec = k.removesuffix("/energy")
                if k != spec:
                    density[spec] = None
                    z_of_ions += atoms[spec.removeprefix("ion/")].z

            for k in density:
                density[k] = -ne / z_of_ions

        density["electrons"] = ne

        density_m = {k: None for k in density.keys()}

        if solver_1d_previous is not None:
            for equ in solver_1d_previous.equation:
                identifier = equ.primary_quantity.identifier
                spec = identifier.removesuffix("/density_thermal")
                if spec != identifier:
                    density_m[spec] = equ.primary_quantity.profile

        for idx, equ in enumerate(solver_1d.equation):
            var_name = equ.primary_quantity.identifier.split("/")

            quantity_name = var_name[-1]
            spec = "/".join(var_name[:-1])

            bc = [[0, 0, 0], [0, 0, 0]]

            match quantity_name:
                case "psi":
                    conductivity_parallel = 0

                    j_parallel = 0

                    j_parallel_imp = 0

                    if core_sources is not None:
                        for source in core_sources.source:
                            core_source_1d = source.time_slice.current.profiles_1d
                            conductivity_parallel += core_source_1d.conductivity_parallel or 0
                            j_parallel += core_source_1d.j_parallel or 0
                            j_parallel_imp += core_source_1d.j_parallel_imp or 0

                        if isinstance(conductivity_parallel, Expression):
                            conductivity_parallel = conductivity_parallel(x)
                        if isinstance(j_parallel, Expression):
                            j_parallel = j_parallel(x)
                        if isinstance(j_parallel_imp, Expression):
                            j_parallel_imp = j_parallel_imp(x)

                    a = conductivity_parallel

                    b = conductivity_parallel

                    c = (scipy.constants.mu_0 * B0 * rho_tor * rho_tor_boundary) / fpol2

                    d = vpr * gm2 / (fpol * rho_tor_boundary) / ((2.0 * scipy.constants.pi) ** 2)

                    e = 0

                    f = -vpr * (j_parallel) / (2.0 * scipy.constants.pi)

                    g = -vpr * (j_parallel_imp) / (2.0 * scipy.constants.pi)

                    for i in range(2):
                        bc_ = equ.boundary_condition[i]

                        match bc_.identifier.index:
                            case 1:  # poloidal flux;
                                u = 1
                                v = 0
                                w = bc_.value[0]
                            case 2:  # ip, total current inside x=1
                                Ip = bc_.value[0]
                                u = 0
                                v = 1
                                w = scipy.constants.mu_0 * Ip / fpol
                            case 3:  # loop voltage;
                                Uloop_bdry = bc_.value[0]
                                u = 0
                                v = 1
                                w = (dt * Uloop_bdry + ym) * d
                            case 5:  # generic boundary condition y expressed as a1y' + a2y=a3;
                                u = bc_.value[1]
                                v = bc_.value[0]
                                w = bc_.value[2]
                            case 6:  # equation not solved;
                                raise NotImplementedError(bc_.identifier.index)
                            case _:
                                u, v, w = 0, 0, 0
                                if bc_.identifier.index is not _not_found_:
                                    logger.warning(f"ignore {bc_.identifier.index}")
                        bc[i] = [u, v, w]

                case "density_thermal":
                    transp_d = 0
                    transp_v = 0
                    transp_flux = 0
                    Sexpl = 0
                    Simpl = 0

                    if core_transport is not None:
                        for model in core_transport.model:
                            core_transp_1d = model.time_slice.current.profiles_1d
                            transp_d += core_transp_1d.get(f"{spec}/particles/d", 0)
                            transp_v += core_transp_1d.get(f"{spec}/particles/v", 0)
                            transp_flux += core_transp_1d.get(f"{spec}/particles/flux", 0)
                        if isinstance(transp_d, Expression):
                            transp_d = transp_d(x)
                        if isinstance(transp_v, Expression):
                            transp_v = transp_v(x)
                        if isinstance(transp_flux, Expression):
                            transp_flux = transp_flux(x)

                    if core_sources is not None:
                        for source in core_sources.source:
                            core_source_1d = source.time_slice.current.profiles_1d
                            S = core_source_1d.get(f"{spec}/particles", None)
                            if S is not None:
                                Sexpl += S
                            else:
                                Sexpl += core_source_1d.get(f"{spec}/particles_decomposed/explicit_part", 0)
                                Simpl += core_source_1d.get(f"{spec}/particles_decomposed/implicit_part", 0)
                        if isinstance(Sexpl, Expression):
                            Sexpl = Sexpl(x)
                        if isinstance(Simpl, Expression):
                            Simpl = Simpl(x)

                    a = vpr

                    b = vprm

                    c = rho_tor_boundary

                    d = vpr * gm3 * transp_d / rho_tor_boundary

                    e = vpr * gm3 * (transp_v - rho_tor * k_phi)

                    f = vpr * Sexpl

                    g = vpr * (Simpl + k_phi)

                    for i in range(2):
                        bc_ = equ.boundary_condition[i]

                        match bc_.identifier.index:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = bc_.value[0]
                            # 2: radial derivative of the field (-dy/drho_tor);
                            case 2:
                                u = -e / d
                                v = 1.0 / d
                                w = bc_.value[0]
                            # 3: scale length of the field y/(-dy/drho_tor);
                            case 3:
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")
                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = bc_.value[0]
                            # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                            case 5:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")
                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc[i] = [u, v, w]

                case "temperature":
                    nuz_Tz = 0  # $\sum_{z\neq s}\nu_{zs}T_{z}$
                    nu_z = 0  # $\sum_{z\neq s}\nu_{zs}$

                    energy_diff = 0
                    energy_vcon = 0
                    energy_flux = 0
                    Qexpl = 0
                    Qimpl = 0
                    flux_multiplier = 0

                    # fmt:off

                    if core_transport is not None:
                        for model in core_transport.model:
                            flux_multiplier += model.time_slice.current.flux_multiplier 
                            core_transp_1d=model.time_slice.current.profiles_1d
                            energy_diff              += core_transp_1d.get(f"{spec}/energy/d",0)
                            energy_vcon              += core_transp_1d.get(f"{spec}/energy/v",0)
                            energy_flux              += core_transp_1d.get(f"{spec}/energy/flux",0)

                        if isinstance(energy_diff,Expression):
                            energy_diff = energy_diff(x)
                        if isinstance(energy_vcon,Expression):
                            energy_vcon = energy_vcon(x)
                        if isinstance(energy_flux,Expression):
                            energy_flux = energy_flux(x)

                    if core_sources is not None:
                        for source in core_sources.source:
                            core_source_1d = source.time_slice.current.profiles_1d
                            Q=core_source_1d.get(f"{spec}/energy", None)

                            if Q is not None:
                                Qexpl         += Q
                            else:
                                Qexpl                    += core_source_1d.get(f"{spec}/energy_decomposed/explicit_part")
                                Qimpl                    += core_source_1d.get(f"{spec}/energy_decomposed/implicit_part")
                        
                        if isinstance(Qexpl,Expression):
                            Qexpl = Qexpl(x)
                        if isinstance(Qimpl,Expression):
                            Qimpl = Qimpl(x)
                            
                    ns                                   = density[spec]

                    ns_m                    = solver_1d_previous.equation.get(f"{spec}/density_thermal/profile",0) if solver_1d_previous is not None else 0

                    ns_flux   = flux_multiplier*vars.get(f"{spec}/density_thermal_flux",0)

                    # fmt:on

                    a = (3 / 2) * (vpr ** (5 / 3)) * ns

                    b = (3 / 2) * (vprm ** (5 / 3)) * ns_m

                    c = rho_tor_boundary * inv_vpr23

                    d = vpr * gm3 * ns * energy_diff / rho_tor_boundary

                    e = vpr * gm3 * ns * energy_vcon + flux_multiplier * ns_flux

                    f = vpr ** (5 / 3)  # * (Qexpl + nuz_Tz * ns)

                    g = (vpr ** (5 / 3)) * (Qimpl + nu_z * ns)

                    for i in range(2):
                        bc_ = equ.boundary_condition[i]

                        match bc_.identifier.index:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = bc_.value[0]

                            case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                                u = -e / d
                                v = 1.0 / d
                                w = bc_.value[0]

                            case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")

                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = bc_.value[0]

                            case 5:  # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")
                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc[i] = [u, v, w]

                case "momentum":
                    # $\sum_{z\neq s}\frac{m_{z}}{\tau_{zs}}n_{z}u_{z,\varphi}$
                    n_u_z = 0
                    n_z = 0  # $\sum_{z\neq s}\frac{1}{\tau_{zs}}m_{z}n_{z}$

                    ms = atoms[spec]["mass"]

                    ns = density.get(spec, 0)

                    ns_m = (
                        solver_1d_previous.equation.get(f"{spec}/density_thermal/profile", 0)
                        if solver_1d_previous is not None
                        else 0
                    )

                    ns_flux = vars.get(f"{spec}/density_thermal_flux")

                    chi_u = 0
                    Uexpl = 0
                    Uimpl = 0

                    if core_transport is not None:
                        for model in core_transport.model:
                            core_transp_1d += model.time_slice.current.profiles_1d
                            chi_u += core_transp_1d.get(f"{spec}/momentum/d", 0)

                        if isinstance(chi_u, Expression):
                            chi_u = chi_u(x)

                    if core_sources is not None:
                        for source in core_sources.source:
                            core_source_1d = source.time_slice.current.profiles_1d
                            Uexpl = core_source_1d.get(f"{spec}/momentum/toroidal_decomposed/explicit_part", 0)
                            Uimpl = core_source_1d.get(f"{spec}/momentum/toroidal_decomposed/implicit_part", 0)

                        if isinstance(Uexpl, Expression):
                            Uexpl = Uexpl(x)
                        if isinstance(Uimpl, Expression):
                            Uimpl = Uimpl(x)

                    a = (vpr ** (5 / 3)) * ms * ns

                    b = (vprm ** (5 / 3)) * ms * ns_m

                    c = rho_tor_boundary

                    d = vpr * gm3 * ms * gm8 * ns * chi_u / rho_tor_boundary

                    e = vpr * gm3 * ms * gm8 * ns + ms * gm8 * ns_flux - ms * gm8 * vpr * rho_tor * k_phi * ns

                    f = vpr * (Uexpl + gm8 * n_u_z)

                    g = vpr * (Uimpl + gm8 * (n_z + ms * ns * k_rho_bdry))

                    for i in range(2):
                        bc_ = equ.boundary_condition[i]

                        match bc_.identifier.index:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = bc_.value[0]
                            # 2: radial derivative of the field (-dy/drho_tor);
                            case 2:
                                u = -e / d
                                v = 1.0 / d
                                w = bc_.value[0]
                            # 3: scale length of the field y/(-dy/drho_tor);
                            case 3:
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")
                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = bc_.value[0]
                            # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                            case 5:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")
                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc[i] = [u, v, w]

                case _:
                    raise RuntimeError(f"Unknown equation of {equ.primary_quantity.identifier}!")

            equ["coefficient"] = [a, b, c, d, e, f, g] + bc

            a, b, c, d, e, f, g, (u0, v0, w0), (u1, v1, w1) = equ.coefficient

            y = vars[equ.primary_quantity.identifier]

            flux = vars[equ.primary_quantity.identifier + "_flux"]

            ym = solver_1d_previous.equation[idx].primary_quantity.profile if solver_1d_previous is not None else 0

            equ.primary_quantity["d_dr"] = (-flux + e * y + hyper_diff * y.d) / (d + hyper_diff)

            equ.primary_quantity["dflux_dr"] = (f - g * y - (a * y - b * ym) * one_over_dt + hyper_diff * flux.d) / (
                1.0 / c + hyper_diff
            )

            equ.boundary_condition[0]["func"] = u0 * y + v0 * flux - w0
            equ.boundary_condition[1]["func"] = u1 * y + v1 * flux - w1

        return solver_1d, vars

    def solve(
        self,
        current: TransportSolverNumerics.TimeSlice,
        previous: TransportSolverNumerics.TimeSlice | None,
        *args,
        **kwargs,
    ):
        solver_1d, vars = self._update_coefficient(current, previous, *args, **kwargs)

        logger.info(
            f"Solve transport equations [{len(solver_1d.equation)}] : {','.join([equ.primary_quantity.identifier for equ in solver_1d.equation])}"
        )

        x = solver_1d.grid.rho_tor_norm

        Y0 = np.zeros([len(vars) - 1, len(x)])

        for idx, equ in enumerate(solver_1d.equation):
            y = equ.primary_quantity.profile
            if callable(y):
                y = y(x)
            Y0[idx * 2] = np.full_like(x, y)

        for idx, equ in enumerate(solver_1d.equation):
            a, b, c, d, e, f, g, *_ = equ.coefficient
            y = Y0[idx * 2]
            Y0[idx * 2 + 1] = -Function(y, x).d()(x) * d(x, *Y0) + y * e(x, *Y0)

            if np.any(np.isnan(Y0[idx * 2 + 1])):
                logger.error((equ.primary_quantity.identifier, d))

        def func(x: array_type, y: array_type, *args) -> array_type:
            # TODO: 需要加速

            res = []
            for equ in solver_1d.equation:
                try:
                    dydr = equ.primary_quantity.d_dr
                    if callable(dydr):
                        dydr = dydr(x, *y, *args)
                    else:
                        dydr = np.full_like(x, dydr)
                except Exception as error:
                    raise RuntimeError(
                        f"Error when apply  dydr={equ.primary_quantity.d_dr.__repr__()}  x={x} args={(y)} !"
                    ) from error
                else:
                    if np.any(np.isnan(dydr)):
                        raise RuntimeError((equ.primary_quantity.identifier, equ.primary_quantity.d_dr))
                    else:
                        res.append(dydr)

                try:
                    dfluxdr = equ.primary_quantity.dflux_dr
                    if callable(dfluxdr):
                        dfluxdr = dfluxdr(x, *y, *args)
                    else:
                        dfluxdr = np.full_like(x, dfluxdr)
                except Exception as error:
                    raise RuntimeError(
                        f"Error when apply  dflux_dr={equ.primary_quantity.dflux_dr.__repr__()} x={x} args={(y)} !"
                    ) from error
                else:
                    if np.any(np.isnan(dfluxdr)):
                        a, b, c, d, e, f, g, *_ = equ.coefficient
                        raise RuntimeError(
                            (
                                equ.primary_quantity.identifier,
                                equ.primary_quantity.dflux_dr,
                                d(x, *y),
                                e(x, *y),
                                f(x, *y),
                                # g(x,*y),
                            )
                        )
                    else:
                        res.append(dfluxdr)

            res = np.stack(res)
            return res

        def bc(ya: array_type, yb: array_type, *args) -> array_type:
            res = []
            for eq in solver_1d.equation:
                res.append(eq.boundary_condition[0].func(x[0], *ya))
                res.append(eq.boundary_condition[1].func(x[-1], *yb))
            return np.array(res)

        sol = solve_bvp(
            func,
            bc,
            x,
            Y0,
            bvp_rms_mask=self.code.parameters.get("bvp_rms_mask", []),
            tolerance=self.code.parameters.get("tolerance", 1.0e-3),
            max_nodes=self.code.parameters.get("max_nodes", 250),
            verbose=self.code.parameters.get("verbose", 2),
        )

        solver_1d.grid.remesh(rho_tor_norm=sol.x)

        for idx, equ in enumerate(solver_1d.equation):
            equ.primary_quantity["profile"] = sol.y[2 * idx]
            equ.primary_quantity["d_dr"] = sol.yp[2 * idx]
            equ.primary_quantity["flux"] = sol.y[2 * idx + 1]
            equ.primary_quantity["dflux_dr"] = sol.yp[2 * idx + 1]

        if not sol.success:
            logger.error(f"Solve BVP failed: {sol.message} , {sol.niter} iterations")
        else:
            logger.debug(f"Solve BVP success: {sol.message} , {sol.niter} iterations")

        return sol.status
