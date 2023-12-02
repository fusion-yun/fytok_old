import typing
import numpy as np
import scipy.constants
from copy import copy

from spdm.data.Expression import Variable, Expression
from spdm.data.Function import Function
from spdm.data.sp_property import sp_tree
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_
from spdm.numlib.bvp import solve_bvp

from fytok.modules.Utilities import Code
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics

from fytok.utils.atoms import atoms
from fytok.utils.logger import logger
from fytok.utils.envs import FY_DEBUG


@TransportSolverNumerics.register(["fy_trans"])
@sp_tree
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

    code: Code = {"name": "fy_trans"}

    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)

        current = self.time_slice.current

        previous = self.time_slice.previous

        equilibrium: Equilibrium = self.inputs.get_source("equilibrium")

        core_transport: CoreTransport = self.inputs.get_source("core_transport")

        core_sources: CoreSources = self.inputs.get_source("core_sources")

        core_profiles: CoreProfiles = self.inputs.get_source("core_profiles")

        core_profiles_1d = core_profiles.time_slice.current.profiles_1d

        # 声明变量
        x = self.primary_coordinate

        vars = {"x": x}

        for equ in self.equations:
            vars[equ.profile.name] = equ.profile
            vars[equ.flux.name] = equ.flux

        vars_m = {}

        if not isinstance(previous, TransportSolverNumerics.TimeSlice):
            one_over_dt = 0

        else:
            dt = current.time - previous.time

            if np.isclose(dt, 0.0) or dt < 0:
                raise RuntimeError(f"dt={dt}<=0")
            else:
                one_over_dt = 1.0 / dt

            for equ in previous.equation:
                identifier = equ.primary_quantity.identifier
                vars_m[identifier] = equ.primary_quantity.profile
                vars_m[f"{identifier}_flux"] = equ.primary_quantity.flux

        # 设定全局参数
        hyper_diff = self.code.parameters.get("hyper_diff", 0.001)

        psi = Function(current.grid.rho_tor_norm, current.grid.psi, label="psi")(x)

        # $R_0$ characteristic major radius of the device   [m]
        R0 = equilibrium.time_slice.current.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = equilibrium.time_slice.current.vacuum_toroidal_field.b0

        eq_1d = equilibrium.time_slice.current.profiles_1d

        rho_tor_boundary = eq_1d.grid.rho_tor_boundary

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = eq_1d.dvolume_drho_tor(psi)

        eq_m = equilibrium.time_slice.previous

        if eq_m is not _not_found_:
            B0m = eq_m.vacuum_toroidal_field.b0

            rho_tor_boundary_m = eq_m.profiles_1d.grid.rho_tor_boundary

            vprm = eq_m.profiles_1d.dvolume_drho_tor(psi)

        else:
            B0m = B0

            rho_tor_boundary_m = rho_tor_boundary

            vprm = vpr

        k_B = (B0 - B0m) / (B0 + B0m) * one_over_dt

        k_rho_bdry = (rho_tor_boundary - rho_tor_boundary_m) / (rho_tor_boundary + rho_tor_boundary_m) * one_over_dt

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary * x

        inv_vpr23 = vpr ** (-2 / 3)

        k_vppr = 0  # (3 / 2) * k_rho_bdry - k_phi *　x * vpr(psi).dln()

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = eq_1d.f(psi)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = eq_1d.q(psi)

        gm1 = eq_1d.gm1(psi)  # <1/R^2>
        gm2 = eq_1d.gm2(psi)  # <|grad_rho_tor|^2/R^2>
        gm3 = eq_1d.gm3(psi)  # <|grad_rho_tor|^2>
        gm8 = eq_1d.gm8(psi)  # <R>

        species: typing.List[str] = []

        for equ in self.equations:
            identifier = equ.identifier
            if identifier == "psi" or identifier.endswith("/momentum"):
                continue
            species.append("/".join(identifier.split("/")[:-1]))

        coeff: typing.Dict[str, Expression] = {
            k: copy(
                {
                    "transp_D": 0.0,
                    "transp_V": 0.0,
                    "transp_F": 0.0,
                    "energy_D": 0.0,
                    "energy_V": 0.0,
                    "energy_F": 0.0,
                    "chi_u": 0.0,
                    "S": 0.0,
                    "Q": 0.0,
                    "U": 0.0,
                }
            )
            for k in species
        }

        # quasi_neutrality_condition
        if "electrons/density" not in vars:
            ne = 0
            ne_flux = 0
            for spec in species:
                if spec == "electrons":
                    continue
                z = atoms[spec.removeprefix("ion/")].z
                ne += z * vars.get(f"{spec}/density", 0.0)
                ne_flux += z * vars.get(f"{spec}/density_flux", 0.0)

            for ion in core_profiles_1d.ion:
                if f"ion/{ion.label}/density" not in vars:
                    ne += ion.density(x)

            vars["electrons/density"] = ne
            vars["electrons/density_flux"] = ne_flux
            # S_ne_explicit = 0
            # S_ne_implicit = 0
            # for source in core_sources.source:
            #     electrons = source.time_slice.current.profiles_1d.electrons
            #     S_ne_explicit += electrons.get("particles", 0)
            #     S_ne_explicit += electrons.particles_decomposed.get("explicit_part", 0)
            #     S_ne_implicit += electrons.particles_decomposed.get("implicit_part", 0)
            # if isinstance(S_ne_explicit, Expression):
            #     S_ne_explicit = S_ne_explicit(x)
            # if isinstance(S_ne_implicit, Expression):
            #     S_ne_implicit = S_ne_implicit(x)

            # vars["electrons/density_flux"] = rho_tor_boundary * (vpr * (S_ne_explicit + ne * S_ne_implicit)).I

        else:
            ne = vars["electrons/density"]
            ne_flux = vars["electrons/density_flux"]

            z_of_ions = 0
            for spec in species:
                if spec == "electrons":
                    continue

                vars[f"{spec}/density"] = None
                z_of_ions += atoms[spec.removeprefix("ion/")].z

            for k in vars:
                vars[k] = -ne / z_of_ions(x)

        if core_transport is not None:
            flux_multiplier = sum(
                [
                    model.time_slice.current.flux_multiplier
                    for model in core_transport.model
                    if model.time_slice.current.flux_multiplier is not _not_found_
                ],
                0,
            )

        else:
            flux_multiplier = 1

        if core_transport is not None:
            for model in core_transport.model:
                trans_1d = model.fetch(**vars).profiles_1d

                for spec, d in coeff.items():
                    d["transp_D"] += trans_1d.get(f"{spec}/particles/d", 0)
                    d["transp_V"] += trans_1d.get(f"{spec}/particles/v", 0)
                    d["transp_F"] += trans_1d.get(f"{spec}/particles/flux", 0)
                    d["energy_D"] += trans_1d.get(f"{spec}/energy/d", 0)
                    d["energy_V"] += trans_1d.get(f"{spec}/energy/v", 0)
                    d["energy_F"] += trans_1d.get(f"{spec}/energy/flux", 0)
                    d["chi_u"] += trans_1d.get(f"{spec}/momentum/d", 0)

        if core_sources is not None:
            for source in core_sources.source:
                source_1d = source.fetch(**vars).profiles_1d
                for spec, d in coeff.items():
                    d["S"] += source_1d.get(f"{spec}/particles", 0)
                    d["Q"] += source_1d.get(f"{spec}/energy", 0)
                    d["U"] += source_1d.get(f"{spec}/momentum/toroidal", 0)

        for equ in current.equation:
            identifier = equ.primary_quantity.identifier

            y = equ.primary_quantity.profile

            flux = equ.primary_quantity.flux

            var_name = identifier.split("/")

            quantity_name = var_name[-1]

            spec = "/".join(var_name[:-1])

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

                    for bc in equ.boundary_condition:
                        match bc.identifier:
                            case 1:  # poloidal flux;
                                u = 1
                                v = 0
                                w = bc.value[0]
                            case 2:  # ip, total current inside x=1
                                Ip = bc.value[0]
                                u = 0
                                v = 1
                                w = scipy.constants.mu_0 * Ip / fpol
                            case 3:  # loop voltage;
                                Uloop_bdry = bc.value[0]
                                u = 0
                                v = 1
                                w = (dt * Uloop_bdry + ym) * d
                            case 5:  # generic boundary condition y expressed as a1y' + a2y=a3;
                                u = bc.value[0]
                                v = bc.value[1]
                                w = bc.value[2]
                            case 6:  # equation not solved;
                                raise NotImplementedError(bc.identifier)
                            case _:
                                u, v, w = 0, 0, 0

                        bc.func = u * y + v * flux - w

                        equ.coefficient += [[u, v, w]]

                case "density":
                    transp_D = coeff[spec].get("transp_D", 0)
                    transp_V = coeff[spec].get("transp_V", 0)
                    S = coeff[spec].get("S", 0)

                    a = vpr

                    b = vprm

                    c = rho_tor_boundary

                    d = vpr * gm3 * transp_D / rho_tor_boundary

                    e = vpr * gm3 * (transp_V - rho_tor * k_phi)

                    f = vpr * S

                    g = vpr * k_phi

                    for bc in equ.boundary_condition:
                        match bc.identifier:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = bc.value[0]

                            case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                                u = e / d
                                v = -1.0 / d
                                w = bc.value[0]

                            case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")

                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = bc.value[0]

                            case 5:  # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")

                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")

                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc.func = u * y + v * flux - w

                        equ.coefficient += [[u, v, w]]

                case "temperature":
                    energy_D = coeff[spec].get("energy_D", 0)
                    energy_V = coeff[spec].get("energy_V", 0)
                    Q = coeff[spec].get("Q", 0)

                    ns = vars.get(f"{spec}/density", 0)

                    ns_m = vars_m.get(f"{spec}/density", 0)

                    ns_flux = flux_multiplier * vars.get(f"{spec}/density_flux", 0)

                    a = (3 / 2) * (vpr ** (5 / 3)) * ns

                    b = (3 / 2) * (vprm ** (5 / 3)) * ns_m

                    c = rho_tor_boundary * inv_vpr23

                    d = vpr * gm3 * ns * energy_D / rho_tor_boundary

                    e = vpr * gm3 * ns * energy_V + ns_flux
                    # - vpr*(3/2*k_phi)*rho_tor_boundary*x*ns

                    f = (vpr ** (5 / 3)) * Q

                    g = k_vppr * ns

                    for bc in equ.boundary_condition:
                        match bc.identifier:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = bc.value[0]

                            case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                                u = e / d
                                v = -1 / d
                                w = bc.value[0]

                            case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")

                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = bc.value[0]

                            case 5:  # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                                u = bc.value[0]
                                v = bc.value[1]
                                w = bc.value[2]

                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")

                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc.func = u * y + v * flux - w
                        equ.coefficient += [[u, v, w]]

                case "momentum":
                    ms = atoms.get(f"{spec}/mass", np.nan)

                    ns = vars.get(f"{spec}/density", 0)

                    ns_flux = vars.get(f"{spec}/density_flux", 0)

                    ns_m = vars_m.get(f"{spec}/density", 0)

                    chi_u = coeff[spec].get("chi_u", 0)

                    U = coeff[spec].get("U", 0)

                    a = (vpr ** (5 / 3)) * ms * ns

                    b = (vprm ** (5 / 3)) * ms * ns_m

                    c = rho_tor_boundary

                    d = vpr * gm3 * ms * gm8 * ns * chi_u / rho_tor_boundary

                    e = vpr * gm3 * ms * gm8 * ns + ms * gm8 * ns_flux - ms * gm8 * vpr * rho_tor * k_phi * ns

                    f = vpr * (U + gm8 * n_u_z)

                    g = vpr * gm8 * (n_z + ms * ns * k_rho_bdry)

                    for bc in equ.boundary_condition:
                        match bc.identifier:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = bc.value[0]
                            # 2: radial derivative of the field (-dy/drho_tor);
                            case 2:
                                u = -e / d
                                v = 1.0 / d
                                w = bc.value[0]
                            # 3: scale length of the field y/(-dy/drho_tor);
                            case 3:
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")
                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = bc.value[0]
                            # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                            case 5:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")
                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc.func = u * y + v * flux - w
                        equ.coefficient += [[u, v, w]]

                case _:
                    raise RuntimeError(f"Unknown equation of {equ.primary_quantity.identifier}!")

            equ.coefficient += [a, b, c, d, e, f, g]

            ym = vars_m.get(identifier, 0)

            equ.primary_quantity.d_dr = (-flux + e * y + hyper_diff * y.d) / (d + hyper_diff)

            equ.primary_quantity.d_dt = (dy_dt := (a * y - b * ym) * one_over_dt)

            equ.primary_quantity.dflux_dr = (f - g * y - dy_dt + hyper_diff * flux.d) / (1.0 / c + hyper_diff)

    def execute(
        self,
        current: TransportSolverNumerics.TimeSlice,
        previous: TransportSolverNumerics.TimeSlice | None,
        *args,
        **kwargs,
    ):
        super().execute(current, previous, *args, **kwargs)

        logger.info(
            f"Solve transport equations : {','.join([equ.primary_quantity.identifier for equ in current.equation])}"
        )

        x = current.grid.rho_tor_norm

        rho_tor_norm_axis = x[0]
        rho_tor_norm_bdry = x[-1]

        num_of_equation = len(current.equation)

        Y0 = np.zeros([num_of_equation * 2, len(x)])

        initial_value = self.inputs.get_source("initial_value")

        if isinstance(initial_value, array_type) and initial_value.shape == (num_of_equation * 2, x.size):
            Y0 = initial_value

        elif isinstance(initial_value, list):
            if len(initial_value) < num_of_equation:
                raise RuntimeError(f"{ len(initial_value)} != {num_of_equation}")

            for idx in range(num_of_equation):
                Y0[idx * 2, :] = np.full_like(x, initial_value[idx])

        elif isinstance((core_profiles := self.inputs.get_source("core_profiles")), CoreProfiles):
            core_profiles_1d = core_profiles.time_slice.current.profiles_1d

            # 计算 y 和 dydr
            for idx, equ in enumerate(current.equation):
                y: Function = core_profiles_1d.get(equ.primary_quantity.identifier)
                Y0[idx * 2] = y(x)
                Y0[idx * 2 + 1] = y.d(x)

            # 将 dydr 换成 flux
            for idx, equ in enumerate(current.equation):
                y = Y0[idx * 2]
                yp = Y0[idx * 2 + 1]

                *_, (a, b, c, d, e, f, g) = equ.coefficient

                Y0[idx * 2 + 1] = -yp * d(x, *Y0) + y * e(x, *Y0)

            logger.debug(f"Load initial value from core_profiles!")

        else:
            raise TypeError(f"{type(initial_value)}")

        # if np.any(Y0):
        #     ValueError(f"Found nan in initial value! \n {Y0}")

        equ_list = []
        bc_list = []
        for equ in current.equation:
            equ_list.append(equ.primary_quantity.d_dr)
            equ_list.append(equ.primary_quantity.dflux_dr)
            bc_list.append(equ.boundary_condition[0].func)
            bc_list.append(equ.boundary_condition[1].func)

        if True:

            def func(x: array_type, y: array_type, *args) -> array_type:
                res = np.zeros([len(current.equation) * 2, x.size])

                for idx, equ in enumerate(current.equation):
                    try:
                        res[idx * 2] = equ.primary_quantity.d_dr(x, *y, *args)
                        res[idx * 2 + 1] = equ.primary_quantity.dflux_dr(x, *y, *args)
                    except Exception as error:
                        # a, b, c, d, e, f, g, *_ = equ.coefficient
                        logger.error(
                            (
                                equ.primary_quantity.identifier,
                                equ.primary_quantity.d_dr,
                                equ.primary_quantity.dflux_dr,
                                # d(x, *y) if callable(d) else d,
                                # e(x, *y) if callable(e) else e,
                                # f(x, *y) if callable(f) else f,
                                # g(x, *y) if callable(g) else g,
                                # x,
                                # *y,
                            )
                        )

                        raise RuntimeError(
                            f"Failure to calculate the equation of {current.equation[int(idx/2)].primary_quantity.identifier}{'_flux' if int(idx/2)*2!=idx else ''} {equ}   !"
                        ) from error
                if np.any(np.isnan(res)):
                    logger.debug(res)
                return res

        else:

            def func(x: array_type, y: array_type, *args) -> array_type:
                return np.array([equ(x, *y, *args) for equ in equ_list])

        def bc(ya: array_type, yb: array_type, *args) -> array_type:
            res = np.array(
                [
                    func(rho_tor_norm_axis, *ya, *args) if idx % 2 == 0 else func(rho_tor_norm_bdry, *yb, *args)
                    for idx, func in enumerate(bc_list)
                ]
            )
            return res

        sol = solve_bvp(
            func,
            bc,
            x,
            Y0,
            bvp_rms_mask=current.control_parameters.bvp_rms_mask,
            tolerance=current.control_parameters.tolerance,
            max_nodes=current.control_parameters.max_nodes,
            verbose=current.control_parameters.verbose,
        )

        x = sol.x

        current.grid.remesh(x)

        for idx, equ in enumerate(current.equation):
            # assert np.all(vars.pop(f"{equ.primary_quantity.identifier}", 0) == sol.y[2 * idx])
            # assert np.all(vars.pop(f"{equ.primary_quantity.identifier}_flux", 0) == sol.y[2 * idx + 1])
            equ.primary_quantity["profile"] = sol.y[2 * idx]
            equ.primary_quantity["d_dr"] = sol.yp[2 * idx]
            equ.primary_quantity["flux"] = sol.y[2 * idx + 1]
            equ.primary_quantity["dflux_dr"] = sol.yp[2 * idx + 1]

        logger.debug(f"Solve BVP { 'success' if   sol.success else 'failed'}: {sol.message} , {sol.niter} iterations")

        return sol.status
