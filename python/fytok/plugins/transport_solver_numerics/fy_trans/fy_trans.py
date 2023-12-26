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

from fytok.modules.Utilities import *
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics, TransportSolverNumericsTimeSlice

from fytok.utils.atoms import atoms, nuclear_reaction

from fytok.utils.logger import logger
from fytok.utils.envs import FY_DEBUG


from .bvp import solve_bvp

EPSILON = 1.0e-32


@sp_tree
class FyTrans(TransportSolverNumerics):
    r"""
    Solve transport equations $\rho=\sqrt{ \Phi/\pi B_{0}}$
    See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`"""

    code: Code = {"name": "fy_trans"}

    solver: str = "fy_trans_bvp_solver"

    primary_coordinate: str | Variable = "rho_tor_norm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyper_diff = self.code.parameters.hyper_diff or 0.001

        self.rho_tor_norm = self.code.parameters.rho_tor_norm

        if self.rho_tor_norm is _not_found_:
            self.rho_tor_norm = np.linspace(0.001, 0.995, 128)

        enable_momentum = self.code.parameters.enable_momentum

        ######################################################################################
        # 定义  primary_coordinate

        if self.primary_coordinate == "rho_tor_norm":
            x = Variable((i := 0), self.primary_coordinate, label=r"\bar{\rho}_{tor}")

        else:
            x = Variable((i := 0), self.primary_coordinate)

        # 定义  variables 和 equations

        variables = {}
        equations: typing.List[typing.Dict[str, typing.Any]] = []

        bc_type = self.cache_get("boundary_condition_type", {})

        if self.code.parameters.current_diffusion is not False:
            variables["psi"] = Variable((i := i + 1), "psi", label=r"\psi")
            variables["psi_flux"] = Variable((i := i + 1), "psi", label=r"\Gamma_{\psi}")
            equations.append({"identifier": "psi", "boundary_condition_type": bc_type.get("psi", 1)})

        variables["electrons/temperature"] = Te = Variable(
            (i := i + 1),
            "electrons/temperature",
            label=r"T_{e}",
        )

        variables["electrons/temperature_flux"] = Variable(
            (i := i + 1),
            "electrons/temperature",
            label=r"H_{Te}",
        )

        equations.append(
            {
                "identifier": "electrons/temperature",
                "boundary_condition_type": bc_type.get("electrons/temperature", None)
                or bc_type.get(f"*/temperature", 1),
            }
        )

        # 计算总离子密度
        t_i_average = zero
        n_i_total = zero
        n_i_flux_totoal = zero

        for s in self.ion_thermal:
            variables[f"ion/{s}/density"] = ns = Variable(
                (i := i + 1),
                f"ion/{s}/density",
                label=rf"n_{s}",
            )

            variables[f"ion/{s}/density_flux"] = ns_flux = Variable(
                (i := i + 1),
                f"ion/{s}/density_flux",
                label=rf"\Gamma_{s}",
            )

            equations.append(
                {
                    "identifier": f"ion/{s}/density",
                    "boundary_condition_type": bc_type.get(f"ion/{s}/density", None) or bc_type.get(f"*/density", 1),
                }
            )

            variables[f"ion/{s}/temperature"] = Ts = Variable(
                (i := i + 1),
                f"ion/{s}/temperature",
                label=rf"T_{{{s}}}",
            )

            variables[f"ion/{s}/temperature_flux"] = Variable(
                (i := i + 1),
                f"ion/{s}/temperature_flux",
                label=rf"H_{{{s}}}",
            )

            equations.append(
                {
                    "identifier": f"ion/{s}/temperature",
                    "boundary_condition_type": bc_type.get(f"ion/{s}/temperature", None)
                    or bc_type.get(f"*/temperature", 1),
                }
            )

            if enable_momentum:
                variables[f"ion/{s}/momentum"] = ns = Variable(
                    (i := i + 1),
                    f"ion/{s}/momentum",
                    label=rf"u_{s}",
                )

                variables[f"ion/{s}/momentum_flux"] = ns_flux = Variable(
                    (i := i + 1),
                    f"ion/{s}/momentum_flux",
                    label=rf"U_{s}",
                )

                equations.append(
                    {
                        "identifier": f"ion/{s}/momentum",
                        "boundary_condition_type": bc_type.get(f"ion/{s}/momentum", None)
                        or bc_type.get(f"*/momentum", 1),
                    }
                )

            z = atoms[s].z
            t_i_average += z * ns * Ts
            n_i_total += z * ns
            n_i_flux_totoal += z * ns_flux

        # 平均离子温度, 没有计入反应产物 alpha 和 He ash
        t_i_average /= n_i_total

        variables["t_i_average"] = t_i_average

        for s in self.ion_non_thermal:
            variables[f"ion/{s}/density"] = ns = Variable(
                (i := i + 1),
                f"ion/{s}/density",
                label=rf"n_{s}",
            )

            variables[f"ion/{s}/density_flux"] = ns_flux = Variable(
                (i := i + 1),
                f"ion/{s}/density_flux",
                label=rf"\Gamma_{s}",
            )

            equations.append(
                {
                    "identifier": f"ion/{s}/density",
                    "boundary_condition_type": bc_type.get(f"ion/{s}/density", None) or bc_type.get(f"*/density", 1),
                }
            )

            z = atoms[s].z
            n_i_total += z * ns
            n_i_flux_totoal += z * ns_flux

        # for ash in fusion_ash:
        #     # 令 He ash 的温度等于平均离子温度
        #     variables[f"ion/{ash}/temperature"] = t_i_average

        variables["n_i_total"] = n_i_total
        variables["n_i_flux_totoal"] = n_i_flux_totoal

        self.primary_coordinate = x
        self._cache["variables"] = variables
        self._cache["equations"] = equations

        normalize_units = self.code.parameters.normalize_units or {}

        self.normalize_factor = []

        for equ in self.equations:
            identifier = equ.identifier
            quantity_name = identifier.split("/")[-1]
            self.normalize_factor.extend(
                [
                    (normalize_units.get(identifier, None) or normalize_units.get(f"*|{quantity_name}", 1.0)),
                    (
                        normalize_units.get(f"{identifier}_flux", None)
                        or normalize_units.get(f"*|{quantity_name}_flux", 1.0)
                    ),
                ]
            )

        self.normalize_factor = np.array(self.normalize_factor)

        logger.debug(f" Variables : {[k for k,v in self.variables.items() if isinstance(v,Variable)]}")

    def preprocess(self, *args, initial_value=None, boundary_value=None, **kwargs) -> TransportSolverNumericsTimeSlice:
        """准本设定本次迭代
        - 方程 from self.equations
        - 初值 from initial_value
        - 边界值 from boundary_value
        """
        current: TransportSolverNumericsTimeSlice = super().preprocess(*args, **kwargs)

        equilibrium: Equilibrium = self.inputs.get_source("equilibrium")
        core_transport: CoreTransport = self.inputs.get_source("core_transport")
        core_sources: CoreSources = self.inputs.get_source("core_sources")
        core_profiles: CoreProfiles = self.inputs.get_source("core_profiles")

        core_profiles_1d = core_profiles.time_slice.current.profiles_1d

        eq_1d: Equilibrium.TimeSlice.Profiles1D = equilibrium.time_slice.current.profiles_1d

        grid = current.cache_get("grid", _not_found_)

        if not isinstance(grid, CoreRadialGrid):
            # TODO: 根据时间获取时间片, 例如：
            # assert math.isclose(equilibrium.time, self.time), f"{equilibrium.time} != {current.time}"
            #   eq:Equilibrium.TimeSlice= equilibrium.time_slice.get(self.time)

            grid = eq_1d.grid.remesh(kwargs.get("rho_tor_norm", self.code.parameters.get("rho_tor_norm", _not_found_)))
            current["grid"] = grid

        x = self.primary_coordinate
        variables = self.variables

        # 从 core_profiles 获得杂质分布

        n_imp = zero
        n_imp_flux = zero

        for s in self.impurities:
            z_ion_1d = core_profiles_1d.ion[s].z_ion_1d
            n_imp += (core_profiles_1d.ion[s].density * z_ion_1d)(x)
            # n_imp_flux += (core_profiles_1d.ion[s].density_flux * z_ion_1d)(x)

        n_e = n_imp + self.variables["n_i_total"]
        n_e_flux = n_imp_flux + self.variables["n_i_flux_totoal"]

        # quasi neutrality condition
        n_e._metadata["name"] = "ne"
        n_e._metadata["label"] = r"n_{e}"
        n_e_flux._metadata["label"] = r"\Gamma_{e}"

        variables["electrons/density"] = n_e
        variables["electrons/density_flux"] = n_e_flux

        psi_norm = Function(current.grid.rho_tor_norm, current.grid.psi_norm, label=r"\bar{\psi}_{norm}")(x)

        # 设定全局参数
        # $R_0$ characteristic major radius of the device   [m]
        R0 = equilibrium.time_slice.current.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = equilibrium.time_slice.current.vacuum_toroidal_field.b0

        rho_tor_boundary = Scalar(eq_1d.grid.rho_tor_boundary)
        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = eq_1d.dvolume_drho_tor(psi_norm)

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = eq_1d.f(psi_norm)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = eq_1d.q(psi_norm)

        gm1 = eq_1d.gm1(psi_norm)  # <1/R^2>
        gm2 = eq_1d.gm2(psi_norm)  # <|grad_rho_tor|^2/R^2>
        gm3 = eq_1d.gm3(psi_norm)  # <|grad_rho_tor|^2>
        gm8 = eq_1d.gm8(psi_norm)  # <R>

        core_profiles_m = next(core_profiles.time_slice.previous)

        if core_profiles_m is _not_found_:
            core_profiles_1d_m = None
        else:
            core_profiles_1d_m = core_profiles_m.profiles_1d

        equilibrium_m = next(equilibrium.time_slice.previous)

        if equilibrium_m is _not_found_:
            one_over_dt = 0
            B0m = B0
            rho_tor_boundary_m = rho_tor_boundary
            vpr_m = vpr
            gm8_m = gm8
        else:
            dt = equilibrium.time - equilibrium_m.time

            if np.isclose(dt, 0.0) or dt < 0:
                raise RuntimeError(f"dt={dt}<=0")
            else:
                one_over_dt = one / dt

            B0m = equilibrium_m.vacuum_toroidal_field.b0
            rho_tor_boundary_m = equilibrium_m.profiles_1d.grid.rho_tor_boundary
            vpr_m = equilibrium_m.profiles_1d.dvolume_drho_tor(psi_norm)
            gm8_m = equilibrium_m.profiles_1d.gm8(psi_norm)

        k_B = (B0 - B0m) / (B0 + B0m) * one_over_dt

        k_rho_bdry = (rho_tor_boundary - rho_tor_boundary_m) / (rho_tor_boundary + rho_tor_boundary_m) * one_over_dt

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary * x

        inv_vpr23 = vpr ** (-2 / 3)

        k_vppr = 0  # (3 / 2) * k_rho_bdry - k_phi *　x * vpr(psi).dln()

        # Sij = {}
        # Qij = {}
        # Uij = {}

        # # 粒子交换，核反应
        # Sij, Qij = fusion_sources(Sij, Qij, variables, fusion_reactions=self.fusion_reactions)

        # # 碰撞 - 能量交换
        # Qij = collisional_sources(Qij, variables)

        # # 环向转动动量交换 momentum exchange with other ion components:
        # Uij = momentum_sources(Uij, variables)

        trans_models: typing.List[CoreTransport.Model.TimeSlice] = [
            model.fetch(x, **variables) for model in core_transport.model
        ]

        trans_sources: typing.List[CoreSources.Source.TimeSlice] = [
            src.fetch(x, **variables) for src in core_sources.source
        ]

        if boundary_value is None:
            boundary_value = {}

        for idx, equ in enumerate(self.equations):
            identifier = equ.identifier

            pth = identifier.split("/")

            quantity_name = pth[-1]

            label = pth[-2] if len(pth) >= 2 else pth[0]

            spec = "/".join(pth[:-1])

            bc_value = boundary_value.get(equ.identifier, None)

            match quantity_name:
                case "psi":
                    psi = variables.get(f"psi", zero)
                    psi_m = Path(f"psi").get(core_profiles_1d_m, zero)(x)

                    conductivity_parallel = zero

                    j_parallel = zero

                    if core_sources is not None:
                        for source in trans_sources:
                            core_source_1d = source.profiles_1d
                            conductivity_parallel += core_source_1d.conductivity_parallel or zero
                            j_parallel += core_source_1d.j_parallel or zero
                            # j_parallel_imp += core_source_1d.j_parallel_imp or zero

                    if j_parallel is zero:
                        j_parallel = eq_1d.j_tor(psi_norm)  # FIXME:这里应该是 j_parallel

                    c = (scipy.constants.mu_0 * B0 * x * (rho_tor_boundary**2)) / fpol2

                    d_dt = one_over_dt * conductivity_parallel * (psi - psi_m) * c

                    D = (
                        vpr * gm2 / fpol / (-(2.0 * scipy.constants.pi))
                    ) / rho_tor_boundary  # FIXME: 检查 psi 的定义，符号，2Pi系数

                    V = -k_phi * x * conductivity_parallel * c

                    R = (
                        -vpr * (j_parallel) / (2.0 * scipy.constants.pi * x * rho_tor_boundary)
                        - k_phi * conductivity_parallel * (2 - 2 * x * fpol.dln + x * conductivity_parallel.dln) * psi
                    ) * c

                    if bc_value is None:
                        bc_value = current.grid.psi_boundary

                    # at axis x=0 , dpsi_dx=0
                    bc = [[0, 1 / self.normalize_factor[idx * 2 + 1], 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        # poloidal flux;
                        case 1:
                            u = self.normalize_factor[idx * 2 + 1] / self.normalize_factor[idx * 2]
                            v = 0
                            w = bc_value * self.normalize_factor[idx * 2 + 1] / self.normalize_factor[idx * 2]

                        # ip, total current inside x=1
                        case 2:
                            Ip = bc_value
                            u = 0
                            v = 1
                            w = scipy.constants.mu_0 * Ip / fpol

                        # loop voltage;
                        case 3:
                            Uloop_bdry = bc_value
                            u = 0
                            v = 1
                            w = (dt * Uloop_bdry + psi_m) * (D - self.hyper_diff)

                        #  generic boundary condition y expressed as a1y'+a2y=a3.
                        case _:
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    u /= self.normalize_factor[idx * 2 + 1]
                    v /= self.normalize_factor[idx * 2 + 1]
                    w /= self.normalize_factor[idx * 2 + 1]

                    bc += [[u, v, w]]

                case "density":
                    ns = variables.get(f"{spec}/density", zero)
                    ns_m = Path(f"{spec}/density").get(core_profiles_1d_m, zero)(x)

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

                    d_dt = one_over_dt * (vpr * ns - vpr_m * ns_m) * rho_tor_boundary

                    D = vpr * gm3 * transp_D / rho_tor_boundary

                    V = vpr * gm3 * (transp_V - rho_tor * k_phi)

                    R = vpr * (S - k_phi * ns) * rho_tor_boundary

                    # at axis x=0 , flux=0
                    bc = [[0, 1.0 / self.normalize_factor[idx * 2 + 1], 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = self.normalize_factor[idx * 2 + 1] / self.normalize_factor[idx * 2]
                            v = 0
                            w = bc_value * self.normalize_factor[idx * 2 + 1] / self.normalize_factor[idx * 2]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - self.hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - self.hyper_diff) / L
                            v = 1.0
                            w = 0
                        case 4:  # 4: flux;
                            u = 0
                            v = 1
                            w = bc_value
                        # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                        case _:
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    u /= self.normalize_factor[idx * 2 + 1]
                    v /= self.normalize_factor[idx * 2 + 1]
                    w /= self.normalize_factor[idx * 2 + 1]

                    bc += [[u, v, w]]

                case "temperature":
                    ns = variables.get(f"{spec}/density", zero)
                    Ts = variables.get(f"{spec}/temperature", zero)

                    ns_m = Path(f"{spec}/density").get(core_profiles_1d_m, zero)(x)
                    Ts_m = Path(f"{spec}/temperature").get(core_profiles_1d_m, zero)(x)

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

                    gamma_s = variables[f"{spec}/density_flux"] * flux_multiplier

                    d_dt = (
                        one_over_dt
                        * (3 / 2)
                        * (vpr * ns * Ts - (vpr_m ** (5 / 3)) * ns_m * Ts_m * inv_vpr23)
                        * rho_tor_boundary
                    )

                    D = vpr * gm3 * ns * energy_D / rho_tor_boundary

                    V = vpr * gm3 * ns * energy_V + gamma_s - (3 / 2) * k_phi * vpr * rho_tor * ns

                    R = vpr * (Q - k_vppr * ns * Ts) * rho_tor_boundary

                    # at axis x=0, dT_dx=0
                    bc = [[V / self.normalize_factor[idx * 2 + 1], -1 / self.normalize_factor[idx * 2 + 1], 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = self.normalize_factor[idx * 2 + 1] / self.normalize_factor[idx * 2]
                            v = 0
                            w = bc_value * self.normalize_factor[idx * 2 + 1] / self.normalize_factor[idx * 2]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - self.hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - self.hyper_diff) / L
                            v = 1.0
                            w = 0
                        case 4:  # 4: flux;
                            u = 0
                            v = 1
                            w = bc_value

                        case _:  # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    u /= self.normalize_factor[idx * 2 + 1]
                    v /= self.normalize_factor[idx * 2 + 1]
                    w /= self.normalize_factor[idx * 2 + 1]

                    bc += [[u, v, w]]

                case "momentum":
                    us = variables.get(f"{spec}/momentum", zero)
                    ns = variables.get(f"{spec}/density", zero)

                    gamma_s = variables.get(f"{spec}/density_flux", zero)

                    us_m = Path(f"{spec}/momentum").get(core_profiles_1d_m, zero)(x)
                    ns_m = Path(f"{spec}/density").get(core_profiles_1d_m, zero)(x)

                    chi_u = zero
                    V_u_pinch = zero

                    for model in trans_models:
                        trans_1d = model.profiles_1d
                        chi_u += trans_1d.get(f"{spec}/momentum/d", zero)
                        V_u_pinch += trans_1d.get(f"{spec}/momentum/v", zero)

                    U = zero

                    for source in trans_sources:
                        source_1d = source.profiles_1d
                        U += source_1d.get(f"{spec}/momentum/toroidal", zero)

                    U *= gm8

                    ms = atoms[spec].a

                    d_dt = one_over_dt * ms * (vpr * gm8 * ns * us - vpr_m * gm8_m * ns_m * us_m) * rho_tor_boundary

                    D = vpr * gm3 * gm8 * ms * ns * chi_u / rho_tor_boundary

                    V = (vpr * gm3 * ns * V_u_pinch + gamma_s - k_phi * vpr * rho_tor * ns) * gm8 * ms

                    R = vpr * (U - k_rho_bdry * ms * ns * us) * rho_tor_boundary

                    # at axis x=0, du_dx=0
                    bc = [[V / self.normalize_factor[idx * 2 + 1], -1 / self.normalize_factor[idx * 2 + 1], 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = self.normalize_factor[idx * 2 + 1] / self.normalize_factor[idx * 2]
                            v = 0
                            w = bc_value * self.normalize_factor[idx * 2 + 1] / self.normalize_factor[idx * 2]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - self.hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - self.hyper_diff) / L
                            v = 1.0
                            w = 0
                        case 4:  # 4: flux;
                            u = 0
                            v = 1
                            w = bc_value

                        # 5: generic boundary condition y expressed as u y + v y'=w.
                        case _:
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    u /= self.normalize_factor[idx * 2 + 1]
                    v /= self.normalize_factor[idx * 2 + 1]
                    w /= self.normalize_factor[idx * 2 + 1]

                    bc += [[u, v, w]]

                case _:
                    raise RuntimeError(f"Unknown equation of {equ.identifier}!")

            equ["coefficient"] = [d_dt, D, V, R]

            equ["boundary_condition_value"] = bc

        if isinstance(initial_value, dict):
            X = current.grid.rho_tor_norm
            Y = np.zeros([len(self.equations) * 2, X.size])

            for idx, equ in enumerate(self.equations):
                profile = initial_value.get(equ.identifier, 0)
                profile = profile(X) if isinstance(profile, Expression) else profile
                Y[idx * 2] = np.full_like(X, profile)

            for idx, equ in enumerate(self.equations):
                d_dt, D, V, R = equ.coefficient
                y = Y[idx * 2]
                Y[idx * 2 + 1] = -D(X, *Y) * derivative(y, X) + V(X, *Y) * y

            Y /= self.normalize_factor.reshape(-1, 1)

            current.Y = Y

        return current

    def func(self, X: array_type, Y: array_type, *args) -> array_type:
        res = np.zeros([len(self.equations) * 2, X.size])

        hyper_diff = self.hyper_diff

        # 添加量纲和归一化系数，复原为物理量
        Y = Y * self.normalize_factor.reshape(-1, 1)

        # 将负值置为0
        Y[2:] = np.where(Y[2:] < 0, 1.0e3, Y[2:])

        for idx, equ in enumerate(self.equations):
            y = Y[idx * 2]
            flux = Y[idx * 2 + 1]

            d_dt, D, V, R = equ.coefficient

            try:
                d_dt = d_dt(X, *Y, *args) if isinstance(d_dt, Expression) else d_dt
                D = D(X, *Y, *args) if isinstance(D, Expression) else D
                V = V(X, *Y, *args) if isinstance(V, Expression) else V
                R = R(X, *Y, *args) if isinstance(R, Expression) else R
            except RuntimeError as error:
                raise RuntimeError(f"Error when calcuate {equ.identifier}") from error

            yp = derivative(y, X)

            d_dr = (-flux + V * y + hyper_diff * yp) / (D + hyper_diff)

            fluxp = derivative(flux, X)

            dflux_dr = (R - d_dt + hyper_diff * fluxp) / (1.0 + hyper_diff)

            # if np.any(np.isnan(d_dr)):
            #     # logger.warning(f"NaN in {equ.identifier}! {D} {V}  ")
            #     d_dr = np.nan_to_num(d_dr)
            # elif np.any(np.isnan(dflux_dr)):
            #     # logger.warning(f"NaN in {equ.identifier}! {R} {equ.coefficient[3]}")
            #     dflux_dr =np.nan_to_num(dflux_dr)

            # 无量纲，归一化
            res[idx * 2] = d_dr / self.normalize_factor[idx * 2]
            res[idx * 2 + 1] = dflux_dr / self.normalize_factor[idx * 2 + 1]

        return res

    def bc(self, ya: array_type, yb: array_type, *args) -> array_type:
        x0 = self.rho_tor_norm[0]
        x1 = self.rho_tor_norm[-1]

        bc = []

        ya = ya * self.normalize_factor
        yb = yb * self.normalize_factor
        for idx, equ in enumerate(self.equations):
            [u0, v0, w0], [u1, v1, w1] = equ.boundary_condition_value

            try:
                u0 = u0(x0, *ya, *args) if isinstance(u0, Expression) else u0
                v0 = v0(x0, *ya, *args) if isinstance(v0, Expression) else v0
                w0 = w0(x0, *ya, *args) if isinstance(w0, Expression) else w0
                u1 = u1(x1, *yb, *args) if isinstance(u1, Expression) else u1
                v1 = v1(x1, *yb, *args) if isinstance(v1, Expression) else v1
                w1 = w1(x1, *yb, *args) if isinstance(w1, Expression) else w1
            except Exception as error:
                logger.error(((u0, v0, w0), (u1, v1, w1)))
                raise RuntimeError(f"Boundary error of equation {equ.identifier}  ") from error

            y0 = ya[2 * idx]
            flux0 = ya[2 * idx + 1]

            y1 = yb[2 * idx]
            flux1 = yb[2 * idx + 1]

            # NOTE: 要求边界值是无量纲的，以 self.normalize_factor 归一化
            bc.extend([(u0 * y0 + v0 * flux0 - w0), (u1 * y1 + v1 * flux1 - w1)])

        bc = np.array(bc)
        return bc

    def execute(
        self, current: TransportSolverNumericsTimeSlice, *previous: TransportSolverNumericsTimeSlice
    ) -> TransportSolverNumericsTimeSlice:
        current = super().execute(current, *previous)

        X = current.grid.rho_tor_norm
        Y = getattr(current, "Y", None)

        # 设定初值
        if Y is None:
            Y = np.zeros([len(self.equations) * 2, len(X)])

        # if np.count_nonzero(np.isnan(Y)) != 0:
        #     raise RuntimeError(f"Initial value has nan! {np.count_nonzero(np.isnan(Y))}")

        sol = solve_bvp(
            self.func,
            self.bc,
            X,
            Y,
            bvp_rms_mask=self.code.parameters.bvp_rms_mask,
            tol=self.code.parameters.tolerance or 1.0e-3,
            bc_tol=self.code.parameters.bc_tol or 1e6,
            max_nodes=self.code.parameters.max_nodes or 1000,
            verbose=self.code.parameters.verbose or 0,
        )

        current["grid"] = current.grid.remesh(sol.x)

        current.Y = sol.y
        equations = []
        for idx, equ in enumerate(self.equations):
            equations.append(
                {
                    "identifier": equ.identifier,
                    "boundary_condition_type": equ.boundary_condition_type,
                    "boundary_condition_value": equ.boundary_condition_value,
                    "profile": sol.y[2 * idx] * self.normalize_factor[2 * idx],
                    "flux": sol.y[2 * idx + 1] * self.normalize_factor[2 * idx + 1],
                    "d_dr": sol.yp[2 * idx] * self.normalize_factor[2 * idx],
                    "dflux_dr": sol.yp[2 * idx + 1] * self.normalize_factor[2 * idx + 1],
                }
            )

        current._cache["equations"] = equations

        logger.debug(f"Solve BVP { 'success' if sol.success else 'failed'}: {sol.message} , {sol.niter} iterations")

        if sol.status == 0:
            logger.info(f"Solve transport equations success!")
        else:
            logger.error(f"Solve transport equations failed!")
        return current


TransportSolverNumerics.register(["fy_trans"], FyTrans)
