# from __future__ import annotations
# @NOTE：
#   在插件中 from __future__ import annotations 会导致插件无法加载，
#   故障点是：typing.get_type_hints() 找不到类型， i.e. Code,TimeSeriesAoS

import typing
import numpy as np
import scipy.constants

from spdm.data.Expression import Variable, Expression, Scalar, one, zero, derivative
from spdm.data.sp_property import sp_tree
from spdm.data.Path import as_path

from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_


from fytok.modules.Utilities import *
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics, TransportSolverNumericsTimeSlice

from fytok.utils.logger import logger
from fytok.utils.atoms import atoms

from .bvp import solve_bvp

EPSILON = 1.0e-32


@sp_tree
class FyTrans(TransportSolverNumerics):
    r"""
    Solve transport equations $\rho=\sqrt{ \Phi/\pi B_{0}}$
    See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`"""

    code: Code = {"name": "fy_trans", "copyright": "FyTok"}

    solver: str = "fy_trans_bvp_solver"

    primary_coordinate: str | Variable = "rho_tor_norm"

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        enable_momentum = self.code.parameters.enable_momentum or False
        enable_impurity = self.code.parameters.enable_impurity or False

        ######################################################################################
        # 确定待求未知量

        unknowns = self.code.parameters.unknowns or list()

        # 极向磁通
        unknowns.append("psi")

        # 电子
        # - 电子密度由准中性条件给出
        # - 电子温度，求解
        # - 电子转动，跟随离子，不求解
        unknowns.append("electrons/temperature")

        for s in self.ion_thermal:
            # 热化离子组份
            # - 密度 density,可求解，
            # - 温度 temperature,可求解，
            # - 环向转动 velocity/totoridal，可求解，

            unknowns.append(f"ion/{s}/density")
            unknowns.append(f"ion/{s}/temperature")
            if enable_momentum:
                unknowns.append(f"ion/{s}/velocity/toroidal")

        for s in self.ion_non_thermal:
            # 非热化离子组份，
            # - 密度 density ，可求解
            # - 温度 temperature, 无统一定义不求解，
            #    - He ash 温度与离子温度一致，alpha粒子满足慢化分布
            # - 环向转动 velocity/totoridal，无统一定义不求解
            #
            unknowns.append(f"ion/{s}/density")

        if enable_impurity:
            # enable 杂质输运
            for s in self.impurities:
                unknowns.append(f"ion/{s}/density")
                unknowns.append(f"ion/{s}/temperature")
                if enable_momentum:
                    unknowns.append(f"ion/{s}/velocity/toroidal")
        else:
            # 杂质分布需要由外部 （core_profiles）给定
            pass

        ######################################################################################
        # 声明主磁面坐标 primary_coordinate
        # 默认为 x=\bar{\rho}_{tor}=\sqrt{\frac{\Phi}{\Phi_{bdry}}}
        # \rho_{tor}= \sqrt{\frac{\Phi}{B_0 \pi}}

        x = Variable((i := 0), self.primary_coordinate)

        if self.primary_coordinate == "rho_tor_norm":
            x._metadata["label"] = r"\bar{\rho}_{tor}"

        ######################################################################################
        # 声明  variables 和 equations
        # profiles_1d = {self.primary_coordinate: x}
        # equations: typing.List[typing.Dict[str, typing.Any]] = []

        # 在 x=0 处边界条件唯一， flux=0 (n,T,u) or \farc{d \psi}{dx}=0 ( for psi )
        # 在 \rho_{bdry} 处边界条件类型可由参数指定
        bc_type = self.find_cache("boundary_condition_type", {})

        # 归一化/无量纲化单位
        # 在放入标准求解器前，系数矩阵需要无量纲、归一化
        units = self.code.parameters.units
        if units is _not_found_:
            units = {}

        else:
            units = units.__value__

        profiles_1d = self.profiles_1d

        ni = sum([ion.z * ion.density for ion in profiles_1d.ion], zero)
        ni_flux = sum([ion.z * ion.get("density_flux") for ion in profiles_1d.ion], zero)

        profiles_1d.electrons["density"] = ni
        profiles_1d.electrons["density_flux"] = ni_flux

        equations = []

        profiles_1d[self.primary_coordinate] = x

        for s in unknowns:
            pth = Path(s)

            if pth[0] == "psi":
                label_p = r"\psi"
                label_f = r"\Psi"
                bc = bc_type.get(s, 1)

            if pth[-1] == "density":
                label_p = "n"
                label_f = r"\Gamma"
                bc = bc_type.get(s, None) or bc_type.get(f"*/density", 1)

            if pth[-1] == "temperature":
                label_p = "T"
                label_f = "H"
                bc = bc_type.get(s, None) or bc_type.get(f"*/temperature", 1)

            if pth[-1] == "toroidal":
                label_p = "u"
                label_f = r"\Phi"
                bc = bc_type.get(s, None) or bc_type.get(f"*/velocity/toroidal", 1)

            if pth[0] == "electrons":
                label_p += "_{e}"
                label_f += "_{e}"

            if pth[0] == "ion":
                label_p += f"_{{{pth[1]}}}"
                label_f += f"_{{{pth[1]}}}"

            profiles_1d[s] = Variable((i := i + 1), s, label=label_p)

            profiles_1d[f"{s}_flux"] = Variable((i := i + 1), f"{s}_flux", label=label_f)

            unit_profile = units.get(s, None) or units.get(f"*/{pth[-1]}", 1)

            unit_flux = units.get(f"{s}_flux", None) or units.get(f"*/{pth[-1]}_flux", 1)

            equations.append(
                {
                    "@name": s,
                    "identifier": s,
                    "units": (unit_profile, unit_flux),
                    "boundary_condition_type": bc,
                }
            )

        self["equations"] = equations

        ###################################################################################################
        # 赋值属性
        # self.profiles_1d.update(profiles_1d)
        # self.equations = equations
        ##################################################################################################
        # 定义内部控制参数

        self._hyper_diff = self.code.parameters.hyper_diff or 0.001

        logger.debug([equ.identifier for equ in self.equations])

    def preprocess(self, *args, boundary_value=None, **kwargs) -> TransportSolverNumericsTimeSlice:
        """准备迭代求解
        - 方程 from self.equations
        - 初值 from initial_value
        - 边界值 from boundary_value
        """
        current: TransportSolverNumericsTimeSlice = super().preprocess(*args, **kwargs)

        profiles = self.profiles_1d

        x: Expression = profiles[self.primary_coordinate]

        if self.primary_coordinate != "rho_tor_norm":
            raise NotImplementedError(self.primary_coordinate)

        else:
            rho_tor_norm = x

            psi_norm = Function(
                current.grid.rho_tor_norm,
                current.grid.psi_norm,
                name="psi_norm",
                label=r"\bar{\psi}",
            )(rho_tor_norm)

        # fmt:off
            
        eq: Equilibrium.TimeSlice.Profiles1D = self.inports["equilibrium/time_slice/0/profiles_1d"].fetch()

        eq_m: Equilibrium.TimeSlice.Profiles1D = self.inports["equilibrium/time_slice/-1/profiles_1d"].fetch()

        core: CoreProfiles.TimeSlice.Profiles1D = self.inports["core_profiles/time_slice/0/profiles_1d"].fetch()

        core_m: CoreProfiles.TimeSlice.Profiles1D = self.inports["core_profiles/time_slice/-1/profiles_1d"].fetch()

        tranport: AoS[CoreTransport.Model.TimeSlice.Profiles1D] = self.inports["core_transport/model/*/time_slice/0/profiles_1d"].fetch()

        sources: AoS[CoreSources.Source.TimeSlice.Profiles1D] = self.inports["core_sources/source/*/time_slice/0/profiles_1d"].fetch()

        # fmt:on
        for s in self.impurities:
            identifier = as_path(f"ion/{s}/density")
            n_imp = identifier.get(core, zero)(rho_tor_norm)
            profiles[identifier] = n_imp
        # 设定全局参数
        # $R_0$ characteristic major radius of the device   [m]
        R0 = eq._root.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = eq._root.vacuum_toroidal_field.b0

        rho_tor_boundary = Scalar(current.grid.rho_tor_boundary)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = eq.dvolume_drho_tor(psi_norm)

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = eq.f(psi_norm)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = eq.q(psi_norm)

        gm1 = eq.gm1(psi_norm)  # <1/R^2>
        gm2 = eq.gm2(psi_norm)  # <|grad_rho_tor|^2/R^2>
        gm3 = eq.gm3(psi_norm)  # <|grad_rho_tor|^2>
        gm8 = eq.gm8(psi_norm)  # <R>

        if eq_m is _not_found_ or eq_m is None:
            one_over_dt = 0
            B0m = B0
            rho_tor_boundary_m = rho_tor_boundary
            vpr_m = vpr
            gm8_m = gm8
        else:
            dt = eq._root.time - eq_m._root.time

            if dt < 0:
                raise RuntimeError(f"dt={dt}<=0")
            elif np.isclose(dt, 0.0):
                one_over_dt = 0.0
            else:
                one_over_dt = one / dt

            B0m = eq_m._root.vacuum_toroidal_field.b0
            rho_tor_boundary_m = eq_m.grid.rho_tor_boundary
            vpr_m = eq_m.dvolume_drho_tor(psi_norm)
            gm8_m = eq_m.gm8(psi_norm)

        k_B = (B0 - B0m) / (B0 + B0m) * one_over_dt

        k_rho_bdry = (rho_tor_boundary - rho_tor_boundary_m) / (rho_tor_boundary + rho_tor_boundary_m) * one_over_dt

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary * rho_tor_norm

        inv_vpr23 = vpr ** (-2 / 3)

        k_vppr = 0  # (3 / 2) * k_rho_bdry - k_phi *　x * vpr(psi).dln()

        if boundary_value is None:
            boundary_value = {}

        hyper_diff = self._hyper_diff

        for idx, equ in enumerate(self.equations):
            identifier = as_path(equ.identifier)
            path = identifier.parent
            quantity = identifier[-1]

            if quantity == "toroidal":
                quantity = "velocity/toroidal"
                path = path.parent

            bc_value = boundary_value.get(equ.identifier, None)
            match quantity:
                case "psi":
                    psi = profiles.psi

                    psi_m = identifier.get(core_m, zero)(rho_tor_norm)

                    conductivity_parallel: Expression = zero

                    j_parallel: Expression = zero

                    for source_1d in sources:
                        conductivity_parallel += source_1d.conductivity_parallel(rho_tor_norm)
                        j_parallel += source_1d.j_parallel(rho_tor_norm)

                    c = fpol2 / (scipy.constants.mu_0 * B0 * rho_tor * (rho_tor_boundary))

                    d_dt = one_over_dt * conductivity_parallel * (psi - psi_m) / c

                    D = vpr * gm2 / (fpol * rho_tor_boundary * 2.0 * scipy.constants.pi)

                    V = -k_phi * rho_tor_norm * conductivity_parallel

                    R = (
                        -vpr * (j_parallel) / (2.0 * scipy.constants.pi * rho_tor)
                        - k_phi
                        * conductivity_parallel
                        * (2 - 2 * rho_tor_norm * fpol.dln + rho_tor_norm * conductivity_parallel.dln)
                        * psi
                    ) / c

                    if bc_value is None:
                        bc_value = current.grid.psi_boundary

                    # at axis x=0 , dpsi_dx=0
                    bc = [[0, 1, 0]]

                    if bc_value is None:
                        assert equ.boundary_condition_type == 1
                        bc_value = current.grid.psi_boundary

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        # poloidal flux;
                        case 1:
                            u = equ.units[1] / equ.units[0]
                            v = 0
                            w = bc_value * equ.units[1] / equ.units[0]

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
                            w = (dt * Uloop_bdry + psi_m) * (D - hyper_diff)

                        #  generic boundary condition y expressed as a1y'+a2y=a3.
                        case _:
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    bc += [[u, v, w]]

                case "density":
                    ns = (path / "density").get(profiles, zero)
                    ns_m = (path / "density").get(core_m, zero)(rho_tor_norm)

                    transp_D = zero
                    transp_V = zero

                    for transp_1d in tranport:
                        transp_D += (path / "particles/d").get(transp_1d, zero)(profiles)
                        transp_V += (path / "particles/v").get(transp_1d, zero)(profiles)
                        # transp_F += (pth / "particles/flux").get(core_transp_1d, zero)

                    S = zero

                    for source_1d in sources:
                        S += (path / "particles").get(source_1d, zero)(profiles)

                    d_dt = one_over_dt * (vpr * ns - vpr_m * ns_m) * rho_tor_boundary

                    D = vpr * gm3 * transp_D / rho_tor_boundary  #

                    V = vpr * gm3 * (transp_V - rho_tor * k_phi)

                    R = vpr * (S - k_phi * ns) * rho_tor_boundary

                    # at axis x=0 , flux=0
                    bc = [[0, 1, 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = equ.units[1] / equ.units[0]
                            v = 0
                            w = bc_value * equ.units[1] / equ.units[0]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - hyper_diff) / L
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

                    bc += [[u, v, w]]

                case "temperature":
                    ns = (path / "density").get(profiles, zero)
                    Gs = (path / "density_flux").get(profiles, zero)
                    Ts = (path / "temperature").get(profiles, zero)

                    ns_m = (path / "density").get(core_m, zero)(rho_tor_norm)
                    Ts_m = (path / "temperature").get(core_m, zero)(rho_tor_norm)

                    energy_D = zero
                    energy_V = zero
                    energy_F = zero

                    flux_multiplier = zero

                    for trans_1d in tranport:
                        flux_multiplier += trans_1d._parent.flux_multiplier

                        energy_D += (path / "energy/d").get(trans_1d, zero)(rho_tor_norm)
                        energy_V += (path / "energy/v").get(trans_1d, zero)(rho_tor_norm)
                        energy_F += (path / "energy/flux").get(trans_1d, zero)(rho_tor_norm)

                    if flux_multiplier is zero:
                        flux_multiplier = one

                    Q = zero

                    for source_1d in sources:
                        Q += (path / "energy").get(source_1d, zero)(rho_tor_norm)

                    d_dt = (
                        one_over_dt
                        * (3 / 2)
                        * (vpr * ns * Ts - (vpr_m ** (5 / 3)) * ns_m * Ts_m * inv_vpr23)
                        * rho_tor_boundary
                    )

                    D = vpr * gm3 * ns * energy_D  #

                    D = D / rho_tor_boundary

                    V = vpr * gm3 * ns * energy_V + Gs * flux_multiplier - (3 / 2) * k_phi * vpr * rho_tor * ns

                    R = vpr * (Q - k_vppr * ns * Ts) * rho_tor_boundary

                    # at axis x=0, dH_dx=0
                    bc = [[0, 1, 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = equ.units[1] / equ.units[0]
                            v = 0
                            w = bc_value * equ.units[1] / equ.units[0]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - hyper_diff) / L
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

                    bc += [[u, v, w]]

                case "velocity/toroidal":
                    us = (path / "velocity/toroidal").get(profiles, zero)
                    ns = (path / "density").get(profiles, zero)
                    Gs = (path / "density_flux").get(profiles, zero)

                    us_m = (path / "velocity/toroidal").get(core_m, zero)(rho_tor_norm)
                    ns_m = (path / "density").get(core_m, zero)(rho_tor_norm)

                    chi_u = zero
                    V_u_pinch = zero

                    for trans_1d in tranport:
                        chi_u += (path / "momentum/toroidal/d").get(trans_1d, zero)(rho_tor_norm)
                        V_u_pinch += (path / "momentum/toroidal/v").get(trans_1d, zero)(rho_tor_norm)

                    U = zero

                    for source_1d in sources:
                        U += (identifier / "../../momentum/toroidal").get(source_1d, zero)(rho_tor_norm)

                    U *= gm8

                    ms = identifier.get(atoms).a

                    d_dt = one_over_dt * ms * (vpr * gm8 * ns * us - vpr_m * gm8_m * ns_m * us_m) * rho_tor_boundary

                    D = vpr * gm3 * gm8 * ms * ns * chi_u / rho_tor_boundary

                    V = (vpr * gm3 * ns * V_u_pinch + Gs - k_phi * vpr * rho_tor * ns) * gm8 * ms

                    R = vpr * (U - k_rho_bdry * ms * ns * us) * rho_tor_boundary

                    # at axis x=0, du_dx=0
                    bc = [[0, 1, 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = equ.units[1]
                            v = 0
                            w = bc_value * equ.units[1]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - hyper_diff) / L
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

                    bc += [[u, v, w]]

                case _:
                    raise RuntimeError(f"Unknown equation of {equ.identifier}!")

            equ["coefficient"] = [d_dt, D, V, R]

            equ["boundary_condition_value"] = bc

        self._units = np.array(sum([equ.units for equ in self.equations], tuple()))

        X = current.grid.rho_tor_norm
        Y = np.zeros([len(self.equations) * 2, X.size])

        if (initial_value := kwargs.get("initial_value", _not_found_)) is not _not_found_:
            for idx, equ in enumerate(self.equations):
                profile = initial_value.get(equ.identifier, 0)
                profile = profile(X) if isinstance(profile, Expression) else profile
                Y[idx * 2] = np.full_like(X, profile)

            for idx, equ in enumerate(self.equations):
                d_dt, D, V, R = equ.coefficient
                y = Y[idx * 2]
                yp = derivative(y, X)
                Y[idx * 2 + 1] = -D(X, *Y) * yp + V(X, *Y) * y
                # Y[idx * 2 + 1] = -D(X, *Y) * derivative(y, X) + V(X, *Y) * y

            Y /= self._units.reshape(-1, 1)

        current.X = X
        current.Y = Y

        return current

    def func(self, X: array_type, Y: array_type, *args) -> array_type:
        res = np.zeros([len(self.equations) * 2, X.size])

        hyper_diff = self._hyper_diff

        # 添加量纲和归一化系数，复原为物理量
        Y = Y * self._units.reshape(-1, 1)

        for idx, equ in enumerate(self.equations):
            y = Y[idx * 2]
            flux = Y[idx * 2 + 1]

            _d_dt, _D, _V, _R = equ.coefficient

            try:
                d_dt = _d_dt(X, *Y, *args) if isinstance(_d_dt, Expression) else _d_dt
                D = _D(X, *Y, *args) if isinstance(_D, Expression) else _D
                V = _V(X, *Y, *args) if isinstance(_V, Expression) else _V
                R = _R(X, *Y, *args) if isinstance(_R, Expression) else _R
            except RuntimeError as error:
                raise RuntimeError(f"Error when calcuate {equ.identifier} {_R}") from error

            yp = derivative(y, X)

            d_dr = (-flux + V * y + hyper_diff * yp) / (D + hyper_diff)

            fluxp = derivative(flux, X)

            dflux_dr = (R - d_dt + hyper_diff * fluxp) / (1.0 + hyper_diff)

            if np.any(np.isnan(dflux_dr)):
                logger.warning(f"Error: {equ.identifier} nan in dflux_dr {_R._render_latex_()} {dflux_dr}")
            elif np.any(np.isnan(d_dr)):
                logger.warning(f"Error: {equ.identifier} nan in d_dr")

            # 无量纲，归一化
            res[idx * 2] = d_dr
            res[idx * 2 + 1] = dflux_dr

        res /= self._units.reshape(-1, 1)

        return res

    def bc(self, ya: array_type, yb: array_type, *args) -> array_type:
        x0 = self._rho_tor_norm[0]
        x1 = self._rho_tor_norm[-1]

        bc = []

        ya = ya * self._units
        yb = yb * self._units
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

            # NOTE: 边界值量纲为 flux 通量，以 equ.units[1] 归一化
            bc.extend([(u0 * y0 + v0 * flux0 - w0) / equ.units[1], (u1 * y1 + v1 * flux1 - w1) / equ.units[1]])

        bc = np.array(bc)
        return bc

    def execute(
        self, current: TransportSolverNumericsTimeSlice, *previous: TransportSolverNumericsTimeSlice
    ) -> TransportSolverNumericsTimeSlice:
        current = super().execute(current, *previous)

        X = current.X
        Y = current.Y

        # 设定初值
        if Y is None:
            Y = np.zeros([len(self.equations) * 2, len(X)])

            for idx, equ in enumerate(self.equations):
                Y[2 * idx + 0] = (
                    equ.profile(X)
                    if isinstance(equ.profile, Expression)
                    else np.full_like(X, equ.profile if equ.profile is not _not_found_ else 0)
                )
                Y[2 * idx + 1] = (
                    equ.flux(X)
                    if isinstance(equ.flux, Expression)
                    else np.full_like(X, equ.flux if equ.flux is not _not_found_ else 0)
                )

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

        current.X = sol.x
        current.Y = sol.y
        current.Yp = sol.yp

        current["grid"] = current.grid.remesh(sol.x)

        equations = []

        for idx, equ in enumerate(self.equations):
            d_dt, D, V, R = equ.coefficient

            equations.append(
                {
                    "identifier": equ.identifier,
                    "boundary_condition_type": equ.boundary_condition_type,
                    "boundary_condition_value": equ.boundary_condition_value,
                    "profile": sol.y[2 * idx] * self._units[2 * idx],
                    "flux": sol.y[2 * idx + 1] * self._units[2 * idx + 1],
                    "d_dr": sol.yp[2 * idx] * self._units[2 * idx],
                    "dflux_dr": sol.yp[2 * idx + 1] * self._units[2 * idx + 1],
                    "coefficient": [d_dt(sol.x, *sol.y), D(sol.x, *sol.y), V(sol.x, *sol.y), R(sol.x, *sol.y)],
                }
            )

        current.equations = equations

        logger.debug(f"Solve BVP { 'success' if sol.success else 'failed'}: {sol.message} , {sol.niter} iterations")

        if sol.status == 0:
            logger.info(f"Solve transport equations success!")
        else:
            logger.error(f"Solve transport equations failed!")

        return current


TransportSolverNumerics.register(["fy_trans"], FyTrans)
