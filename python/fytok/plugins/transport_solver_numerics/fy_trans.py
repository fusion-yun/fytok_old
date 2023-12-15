# from __future__ import annotations
# @NOTE：
#   在插件中 from __future__ import annotations 会导致插件无法加载，
#   故障点是：typing.get_type_hints() 找不到类型， i.e. Code,TimeSeriesAoS


import typing
import numpy as np
import scipy.constants
from copy import copy

from spdm.utils.typing import array_type, array_like
from spdm.utils.tags import _not_found_
from spdm.data.Expression import Variable, Expression, Scalar, one, zero, derivative
from spdm.data.Function import Function
from spdm.data.sp_property import sp_tree
from spdm.data.AoS import AoS
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice

from fytok.modules.Utilities import *
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics, TransportSolverNumericsTimeSlice

from fytok.utils.atoms import atoms

from fytok.utils.logger import logger
from fytok.utils.envs import FY_DEBUG

from spdm.numlib.bvp import solve_bvp


class FyTransTimeSlice(TransportSolverNumericsTimeSlice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variables: typing.OrderedDict[str, Expression | array_type] = {}

    def setup(
        self,
        *previous,
        x: Variable,
        variables: typing.Dict[str, Expression],
        equilibrium: Equilibrium,
        core_transport: CoreTransport,
        core_sources: CoreSources,
        core_profiles: CoreProfiles,
        **kwargs,
    ):
        current = self

        previous: TransportSolverNumericsTimeSlice = previous[0] if len(previous) > 0 else None

        psi_norm = Function(current.grid.rho_tor_norm, current.grid.psi_norm, label=r"\bar{\psi}")(x)

        # 设定全局参数
        eq_1d = equilibrium.time_slice.current.profiles_1d
        # $R_0$ characteristic major radius of the device   [m]
        R0 = equilibrium.time_slice.current.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = equilibrium.time_slice.current.vacuum_toroidal_field.b0

        rho_tor_boundary = Scalar(eq_1d.grid.rho_tor_boundary)
        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = eq_1d.dvolume_drho_tor(psi_norm)

        if not isinstance(previous, TransportSolverNumerics.TimeSlice):
            one_over_dt = zero
            B0m = B0
            rho_tor_boundary_m = rho_tor_boundary
            vprm = vpr
            variables_m = {}
        else:
            dt = current.time - previous.time

            if np.isclose(dt, 0.0) or dt < 0:
                raise RuntimeError(f"dt={dt}<=0")
            else:
                one_over_dt = one / dt

            variables_m: typing.OrderedDict[str, Expression | array_type] = getattr(previous, "variables", {})

            eq_m = next(equilibrium.time_slice.previous)

            if eq_m is _not_found_:
                raise RuntimeError(f"Can not get equilibrium at previous time slice !")

            B0m = eq_m.vacuum_toroidal_field.b0

            rho_tor_boundary_m = eq_m.profiles_1d.grid.rho_tor_boundary

            vprm = eq_m.profiles_1d.dvolume_drho_tor(psi_norm)

        k_B = (B0 - B0m) / (B0 + B0m) * one_over_dt

        k_rho_bdry = (rho_tor_boundary - rho_tor_boundary_m) / (rho_tor_boundary + rho_tor_boundary_m) * one_over_dt

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary * x

        inv_vpr23 = vpr ** (-2 / 3)

        k_vppr = 0  # (3 / 2) * k_rho_bdry - k_phi *　x * vpr(psi).dln()

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = eq_1d.f(psi_norm)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = eq_1d.q(psi_norm)

        gm1 = eq_1d.gm1(psi_norm)  # <1/R^2>
        gm2 = eq_1d.gm2(psi_norm)  # <|grad_rho_tor|^2/R^2>
        gm3 = eq_1d.gm3(psi_norm)  # <|grad_rho_tor|^2>
        gm8 = eq_1d.gm8(psi_norm)  # <R>

        rho_tor_norm = self.grid.rho_tor_norm

        bc_pos = [rho_tor_norm[0], rho_tor_norm[-1]]

        trans_models: typing.List[CoreTransport.Model.TimeSlice] = [
            model.fetch(x, **variables) for model in core_transport.model
        ]

        trans_sources: typing.List[CoreSources.Source.TimeSlice] = [
            src.fetch(x, **variables) for src in core_sources.source
        ]

        for idx, equ in enumerate(self.equations):
            identifier = equ.identifier

            y = variables.get(identifier, zero)
            ym = variables_m.get(identifier, zero)

            var_name = identifier.split("/")

            quantity_name = var_name[-1]

            spec = "/".join(var_name[:-1])

            bc_value = []

            match quantity_name:
                case "psi":
                    conductivity_parallel = 0

                    j_parallel = 0

                    j_parallel_imp = 0

                    if core_sources is not None:
                        for source in trans_sources:
                            core_source_1d = source.profiles_1d
                            conductivity_parallel += core_source_1d.conductivity_parallel
                            j_parallel += core_source_1d.j_parallel
                            j_parallel_imp += core_source_1d.get("j_parallel_imp", 0)

                        if isinstance(conductivity_parallel, Expression):
                            conductivity_parallel = conductivity_parallel(x)
                        if isinstance(j_parallel, Expression):
                            j_parallel = j_parallel(x)
                        if isinstance(j_parallel_imp, Expression):
                            j_parallel_imp = j_parallel_imp(x)

                    a = conductivity_parallel(x) * one_over_dt

                    b = conductivity_parallel * one_over_dt

                    c = (scipy.constants.mu_0 * B0 * rho_tor * rho_tor_boundary) / fpol2

                    d = vpr * gm2 / (fpol * rho_tor_boundary) / ((2.0 * scipy.constants.pi) ** 2)

                    e = 0

                    f = -vpr * (j_parallel(x)) / (2.0 * scipy.constants.pi)

                    g = -vpr * (j_parallel_imp(x)) / (2.0 * scipy.constants.pi)

                    for jdx, bc in enumerate(equ.boundary_condition_type):
                        match bc:
                            case 1:  # poloidal flux;
                                u = 1
                                v = 0
                                w = equ.boundary_value[jdx][0]
                            case 2:  # ip, total current inside x=1
                                Ip = equ.boundary_value[jdx][0]
                                u = 0
                                v = 1
                                w = scipy.constants.mu_0 * Ip / fpol
                            case 3:  # loop voltage;
                                Uloop_bdry = equ.boundary_value[jdx][0]
                                u = 0
                                v = 1
                                w = (dt * Uloop_bdry + ym) * d
                            case 5:  # generic boundary condition y expressed as a1y' + a2y=a3;
                                u = equ.boundary_value[jdx][0]
                                v = equ.boundary_value[jdx][1]
                                w = equ.boundary_value[jdx][2]
                            case 6:  # equation not solved;
                                raise NotImplementedError(bc)
                            case _:
                                u, v, w = 0, 0, 0

                        bc_value += [[u, v, w]]

                case "density":
                    transp_D = zero
                    transp_V = zero

                    for model in trans_models:
                        core_transp_1d = model.profiles_1d

                        transp_D += core_transp_1d.get(f"{spec}/particles/d", zero)
                        transp_V += core_transp_1d.get(f"{spec}/particles/v", zero)
                        # transp_F += core_transp_1d.get(f"{spec}/particles/flux", 0)

                    S = zero
                    for source in trans_sources:
                        source_1d = source.profiles_1d
                        S += source_1d.get(f"{spec}/particles", zero)

                    a = vpr * one_over_dt

                    b = vprm * one_over_dt

                    c = rho_tor_boundary

                    d = vpr * gm3 * transp_D(x) / rho_tor_boundary

                    e = vpr * gm3 * (transp_V(x) - rho_tor * k_phi)

                    f = vpr * S(x)

                    g = vpr * k_phi

                    for jdx, bc in enumerate(equ.boundary_condition_type):
                        match bc:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = equ.boundary_value[jdx][0]

                            case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                                u = e
                                v = -1.0
                                w = equ.boundary_value[jdx][0] * d

                            case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")

                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = equ.boundary_value[jdx][0]

                            case 5:  # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")

                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")

                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc_value += [[u, v, w]]

                case "temperature":
                    ns = variables.get(f"{spec}/density", zero)
                    ns_m = variables_m.get(f"{spec}/density", zero)

                    energy_D = zero
                    energy_V = zero
                    energy_F = zero

                    flux_multiplier = zero

                    for model in trans_models:
                        flux_multiplier += model.flux_multiplier

                        trans_1d = model.profiles_1d

                        energy_D += trans_1d.get(f"{spec}/energy/d", zero)
                        energy_V += trans_1d.get(f"{spec}/energy/v", zero)
                        energy_F += trans_1d.get(f"{spec}/energy/flux", zero)

                    if flux_multiplier is zero:
                        flux_multiplier = one

                    Q = zero
                    for source in trans_sources:
                        source_1d = source.profiles_1d
                        Q += source_1d.get(f"{spec}/energy", zero)

                    ns_flux = variables[f"{spec}/density_flux"]  # * flux_multiplier

                    a = (3 / 2) * (vpr ** (5 / 3)) * ns * one_over_dt

                    b = (3 / 2) * (vprm ** (5 / 3)) * ns_m * one_over_dt

                    c = inv_vpr23 * rho_tor_boundary

                    d = vpr * gm3 * ns * energy_D(x) / rho_tor_boundary

                    e = vpr * gm3 * ns * energy_V(x) + ns_flux - vpr * (3 / 2 * k_phi) * rho_tor_boundary * x * ns

                    f = (vpr ** (5 / 3)) * Q(x)

                    g = k_vppr * ns

                    for jdx, bc in enumerate(equ.boundary_condition_type):
                        match bc:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = equ.boundary_value[jdx][0]

                            case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                                u = e
                                v = -1
                                w = equ.boundary_value[jdx][0] * d

                            case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")

                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = equ.boundary_value[jdx][0]

                            case 5:  # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                                u = equ.boundary_value[jdx][0]
                                v = equ.boundary_value[jdx][1]
                                w = equ.boundary_value[jdx][2]

                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")

                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc_value += [[u, v, w]]

                case "momentum":
                    chi_u = zero
                    for model in trans_models:
                        trans_1d = model.profiles_1d
                        chi_u += trans_1d.get(f"{spec}/momentum/d", zero)

                    U = zero
                    for source in trans_sources:
                        source_1d = source.profiles_1d
                        U += source_1d.get(f"{spec}/momentum/toroidal", zero)

                    ms = atoms.get(f"{spec}/mass", np.nan)

                    ns = variables.get(f"{spec}/density", zero)

                    ns_flux = variables.get(f"{spec}/density_flux", zero)

                    ns_m = variables_m.get(f"{spec}/density", zero)

                    a = (vpr ** (5 / 3)) * ms * ns * one_over_dt

                    b = (vprm ** (5 / 3)) * ms * ns_m * one_over_dt

                    c = rho_tor_boundary

                    d = vpr * gm3 * ms * gm8 * ns * chi_u / rho_tor_boundary

                    e = vpr * gm3 * ms * gm8 * ns + ms * gm8 * ns_flux - ms * gm8 * vpr * rho_tor * k_phi * ns

                    f = vpr * (U + gm8 * n_u_z)

                    g = vpr * gm8 * (n_z + ms * ns * k_rho_bdry)

                    for jdx, bc in enumerate(equ.boundary_condition_type):
                        match bc:
                            case 1:  # 1: value of the field y;
                                u = 1
                                v = 0
                                w = equ.boundary_value[jdx][0]
                            # 2: radial derivative of the field (-dy/drho_tor);
                            case 2:
                                u = -e
                                v = 1.0
                                w = equ.boundary_value[jdx][0] * d
                            # 3: scale length of the field y/(-dy/drho_tor);
                            case 3:
                                raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")
                            case 4:  # 4: flux;
                                u = 0
                                v = 1
                                w = equ.boundary_value[jdx][0]
                            # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                            case 5:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            case 6:  # 6: equation not solved;
                                raise NotImplementedError(f"6: equation not solved;")
                            case _:
                                u, v, w = 0, 0, 0
                                # raise NotImplementedError(bc_.identifier)

                        bc_value += [[u, v, w]]

                case _:
                    raise RuntimeError(f"Unknown equation of {equ.identifier}!")

            equ["coefficient"] = [*bc_value, a, b, c, d, e, f, g, ym]

    def func(self, x: array_type, Y: array_type, *args) -> array_type:
        res = np.zeros([len(self.equations) * 2, x.size])
        hyper_diff = self.control_parameters.hyper_diff or 0.001

        for idx, equ in enumerate(self.equations):
            y = Y[idx * 2]
            flux = Y[idx * 2 + 1]
            try:
                bc0, bc1, a, b, c, d, e, f, g, ym = equ.coefficient

                a = a(x, *Y, *args)
                b = b(x, *Y, *args)
                c = c(x, *Y, *args)
                d = d(x, *Y, *args)
                e = e(x, *Y, *args)
                f = f(x, *Y, *args)
                g = g(x, *Y, *args)
                ym = ym(x)

            except Exception as error:
                # a, b, c, d, e, f, g, *_ = equ.coefficient

                logger.error(equ.identifier)
                logger.error((x, Y))
                # logger.error(equ.d_dr)
                # logger.error(equ.dflux_dr)

                raise RuntimeError(
                    f"Failure to calculate the equation of {self.equations[int(idx/2)].identifier}{'_flux' if int(idx/2)*2!=idx else ''} {equ}   !"
                ) from error

            d_dr = (-flux + e * y + hyper_diff * derivative(y, x)) / (d + hyper_diff)

            dy_dt = a * y - b * ym

            dflux_dr = (f - g * y - dy_dt + hyper_diff * derivative(flux, x)) / (1.0 / c + hyper_diff)

            res[idx * 2] = d_dr
            res[idx * 2 + 1] = dflux_dr

            if np.any(np.isnan(d_dr)) or np.any(np.isnan(dflux_dr)):
                logger.exception(f"Error! {equ.identifier} {( a, b, c, d, e, f, g)} ")

        return res

    def bc(self, ya: array_type, yb: array_type, *args) -> array_type:
        current = self

        res = []
        for idx, equ in enumerate(current.equations):
            x0 = self.grid.rho_tor_norm[0]
            x1 = self.grid.rho_tor_norm[-1]

            [u0, v0, w0], [u1, v1, w1], *_ = equ.coefficient

            try:
                u0 = u0(x0, *ya, *args) if isinstance(u0, Expression) else u0
                v0 = v0(x0, *ya, *args) if isinstance(v0, Expression) else v0
                w0 = w0(x0, *ya, *args) if isinstance(w0, Expression) else w0
                u1 = u0(x1, *yb, *args) if isinstance(u1, Expression) else u1
                v1 = v0(x1, *yb, *args) if isinstance(v1, Expression) else v1
                w1 = w0(x1, *yb, *args) if isinstance(w1, Expression) else w1
            except Exception as error:
                logger.error(((u0, v0, w0), (u1, v1, w1)))
                raise RuntimeError(f"Boundary error of equation {equ.identifier}  ") from error

            y0 = ya[2 * idx]
            flux0 = ya[2 * idx + 1]

            y1 = yb[2 * idx]
            flux1 = yb[2 * idx + 1]

            res.extend([u0 * y0 + v0 * flux0 - w0, u1 * y1 + v1 * flux1 - w1])

        return np.array(res)

    def solve(self, variables: typing.Dict[str, Expression]):
        current = self

        x = self.grid.rho_tor_norm

        num_of_equation = len(self.equations)

        Y0 = np.zeros([num_of_equation * 2, len(x)])

        for idx, equ in enumerate(current.equations):
            Y0[idx * 2] = y = array_like(x, equ.profile)

        for idx, equ in enumerate(current.equations):
            bc0, bc1, a, b, c, d, e, f, g, ym = equ.coefficient
            y = Y0[idx * 2]
            Y0[idx * 2 + 1] = -d(x, *Y0) * derivative(y, x) + e(x, *Y0)

        sol = solve_bvp(
            current.func,
            current.bc,
            x,
            Y0,
            bvp_rms_mask=current.control_parameters.bvp_rms_mask,
            tolerance=current.control_parameters.tolerance,
            max_nodes=current.control_parameters.max_nodes or 0,
            verbose=current.control_parameters.verbose or 0,
        )

        current["grid"] = current.grid.remesh(sol.x)

        for idx, equ in enumerate(current.equations):
            equ["profile"] = sol.y[2 * idx]
            equ["d_dr"] = sol.yp[2 * idx]
            equ["flux"] = sol.y[2 * idx + 1]
            equ["dflux_dr"] = sol.yp[2 * idx + 1]

        for k in list(variables.keys()):
            current.variables[k] = variables[k](sol.x, *sol.y)

        logger.debug(f"Solve BVP { 'success' if sol.success else 'failed'}: {sol.message} , {sol.niter} iterations")

        return sol.status


@sp_tree
class FyTrans(TransportSolverNumerics):
    r"""
    Solve transport equations $\rho=\sqrt{ \Phi/\pi B_{0}}$
    See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`"""

    code: Code = {"name": "fy_trans", "copyright": "fytok"}

    primary_coordinate: str | Variable = "rho_tor_norm"

    TimeSlice = FyTransTimeSlice

    time_slice: TimeSeriesAoS[FyTransTimeSlice]

    def execute(self, current: FyTransTimeSlice, *previous: FyTransTimeSlice) -> FyTransTimeSlice:
        current = super().execute(current, *previous)
         
        current.setup(
            x=self.primary_coordinate,
            variables=self.variables,
            equilibrium=self.inputs.get_source("equilibrium"),
            core_transport=self.inputs.get_source("core_transport"),
            core_sources=self.inputs.get_source("core_sources"),
            core_profiles=self.inputs.get_source("core_profiles"),
        )

        status = current.solve(self.variables)

        if status == 0:
            logger.info(f"Solve transport equations success!")
        else:
            logger.error(f"Solve transport equations failed!")
        return current


TransportSolverNumerics.register(["fy_trans"], FyTrans)
# # 声明变量 Variable
# def guess_label(name):
#     s = name.split("/")
#     if len(s) >= 2:
#         if s[-2] == "electrons":
#             s[-2] = "e"

#     if s[-1] == "psi":
#         label = r"\psi"
#     elif s[-1] == "psi_flux":
#         label = r"\psi"
#     elif s[-1] == "density":
#         label = f"n_{{{s[-2]}}}"
#     elif s[-1] == "density_flux":
#         label = rf"\Gamma_{{{s[-2]}}}"
#     elif s[-1] == "temperature":
#         label = f"T_{{{s[-2]}}}"
#     elif s[-1] == "temperature_flux":
#         label = f"H_{{{s[-2]}}}"
#     else:
#         label = name

#     return label

# variables["x"] =
# x = Variable((index := 0), "x", label=r"\bar{\rho}_{tor}")

# for idx, equ in enumerate(self.equations):
#     variables[equ.identifier] = Variable(
#         (index := index + 1),
#         equ.identifier,
#         label=guess_label(equ.identifier),
#     )
#     variables[f"{equ.identifier}_flux"] = Variable(
#         (index := index + 1),
#         f"{equ.identifier}_flux",
#         label=guess_label(f"{equ.identifier}_flux"),
#     )
