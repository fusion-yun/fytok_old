import collections
from math import isclose
import typing
import numpy as np
import scipy.constants

from spdm.data.Expression import Expression, Variable
from spdm.data.Function import Function, function_like
from spdm.data.Path import Path
from spdm.utils.typing import array_type
from spdm.numlib.bvp import solve_bvp
from spdm.numlib.misc import array_like
from spdm.utils.tags import _not_found_

from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.logger import logger
from fytok.utils.atoms import atoms

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWO_PI = 2.0 * scipy.constants.pi


@TransportSolverNumerics.register(["fytrans"])
class FyTrans(TransportSolverNumerics):

    def refresh(self, *args,
                core_profiles: CoreProfiles,
                equilibrium: Equilibrium = None,
                core_transport: CoreTransport = None,
                core_sources: CoreSources = None,
                **kwargs):

        super().refresh(*args,
                        core_profiles=core_profiles,
                        equilibrium=equilibrium,
                        core_transport=core_transport,
                        core_sources=core_sources,
                        **kwargs)

        self._solve(
            tau=0,
            core_profiles=core_profiles,
            equilibrium=equilibrium,
            core_transport=core_transport,
            core_sources=core_sources,
        )

    def advance(self, *args,
                core_profiles: CoreProfiles,
                equilibrium: Equilibrium = None,
                core_transport: CoreTransport = None,
                core_sources: CoreSources = None,
                **kwargs):

        super().advance(*args, **kwargs)

        self._solve(
            tau=equilibrium.time_slice.current.time-equilibrium.time_slice.previous.time,
            core_profiles=core_profiles,
            equilibrium=equilibrium,
            core_transport=core_transport,
            core_sources=core_sources,
        )

    def _update_solver(self,
                       tau: float,
                       equilibrium: Equilibrium,
                       core_profiles: CoreProfiles,
                       core_transport: CoreTransport,
                       core_sources: CoreSources,
                       **kwargs
                       ):

        solver_1d = self.time_slice.current.solver_1d

        x = Variable(0, "x")

        vars = {"x": x}

        core_profiles_next = core_profiles.time_slice.current

        core_profiles_prev = core_profiles.time_slice.previous

        core_profiles_1d_prev = core_profiles_prev.profiles_1d if core_profiles_prev is not _not_found_ else _not_found_

        core_profiles_1d_next = core_profiles_next.profiles_1d

        equilibrium_next = equilibrium.time_slice.current

        equilibrium_prev = equilibrium.time_slice.previous

        eq_1d_next = equilibrium_next.profiles_1d

        eq_1d_prev = equilibrium_prev.profiles_1d if equilibrium_prev is not _not_found_ else _not_found_

        inv_tau = 1.0/tau if tau > 0 else 0

        for idx, eq in enumerate(solver_1d.equation):
            name = eq.primary_quantity.identifier

            vars[name] = Variable(idx*2+1, name, label=eq.primary_quantity.label or name)
            vars[name + "_flux"] = Variable(idx*2+2, name+"_flux",
                                            label=rf"\Gamma_{{{eq.primary_quantity.label}}}" or name+"_flux")

        psi = Function(solver_1d.grid.psi, solver_1d.grid.rho_tor_norm, label="psi")(x)

        # $R_0$ characteristic major radius of the device   [m]
        R0 = equilibrium_next.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = equilibrium_next.vacuum_toroidal_field.b0

        B0m = equilibrium_prev.vacuum_toroidal_field.b0 if equilibrium_prev is not _not_found_ else np.nan

        k_B = (B0 - B0m) / (B0 + B0m) * 2.0/tau if tau > 0 else 0

        # Mesh
        rho_tor_boundary = eq_1d_next.grid.rho_tor_boundary

        rho_tor_boundary_m = eq_1d_prev.grid.rho_tor_boundary if eq_1d_prev is not _not_found_ else _not_found_

        k_rho_bdry = (rho_tor_boundary - rho_tor_boundary_m) / \
            (rho_tor_boundary + rho_tor_boundary_m)*2.0/tau if tau > 0 else 0

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary*x

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = eq_1d_next.f(psi)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]

        vpr = eq_1d_next.dvolume_drho_tor(psi)

        vprm = eq_1d_prev.dvolume_drho_tor(psi) if eq_1d_prev is not _not_found_ else np.nan

        inv_vpr23 = vpr**(-2/3)

        fpol = eq_1d_next.f(psi)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = eq_1d_next.q(psi)
        gm1 = eq_1d_next.gm1(psi)  # <1/R^2>
        gm2 = eq_1d_next.gm2(psi)  # <|grad_rho_tor|^2/R^2>
        gm3 = eq_1d_next.gm3(psi)  # <|grad_rho_tor|^2>
        gm8 = eq_1d_next.gm8(psi)  # <R>
        # Qimp_k_ns = (3*k_rho_bdry - k_phi * Function(vpr, x).d())

        # quasi_neutrality_condition
        if "electons.density_thermal" not in vars:
            ns = 0
            for k, ni in vars.items():
                if k.endswith("density_thermal") or k.endswith("density_fast") and ni is not _not_found_:
                    continue
                ns += ni

            vars["electons.density_thermal"] = ne = -1 * ns
        else:
            ne = vars["electons.density_thermal"]

            num_of_ion = sum([1 for ion in core_profiles_1d_next.ion if not ion.is_impurity])

            nz = sum([ion.z_ion_1d*ion.density for ion in core_profiles_1d_next.ion if ion.is_impurity])

            n_i_prop = (ne-nz) / num_of_ion

            for ion in core_profiles_1d_next.ion:
                if not ion.is_impurity:
                    ion["density"] = n_i_prop/ion.z

        for idx, eq in enumerate(solver_1d.equation):

            var_name = eq.primary_quantity.identifier

            bc = [[0, 0, 0], [0, 0, 0]]

            if var_name == "psi":

                conductivity_parallel = 0

                j_parallel = 0

                j_parallel_imp = 0

                if core_sources is not None:
                    for source in core_sources.source:
                        core_source_1d = source.time_slice.current.profiles_1d
                        conductivity_parallel += core_source_1d.conductivity_parallel or 0
                        j_parallel += core_source_1d.j_parallel or 0
                        j_parallel_imp += core_source_1d.j_parallel_imp or 0

                a = conductivity_parallel

                b = conductivity_parallel

                c = (scipy.constants.mu_0 * B0*rho_tor * rho_tor_boundary)/fpol2

                d = vpr * gm2 / (fpol * rho_tor_boundary)/(TWO_PI**2)

                e = 0

                f = - vpr * (j_parallel)/TWO_PI

                g = - vpr * (j_parallel_imp)/TWO_PI

                for i in range(2):
                    bc_ = eq.boundary_condition[i]
                    x = bc_.rho_tor_norm
                    match bc_.identifier.index:
                        case 1:  # poloidal flux;
                            u = 1
                            v = 0
                            w = bc_.value[0]
                        case 2:  # ip, total current inside x=1
                            Ip = bc_.value[0]
                            u = 0
                            v = 1
                            w = scipy.constants.mu_0 * Ip / fpol[-1]
                        case 3:  # loop voltage;
                            Uloop_bdry = bc_.value[0]
                            u = 0
                            v = 1
                            w = (tau*Uloop_bdry + psi_prev(x))*d(x)
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

            elif var_name.endswith("density_thermal"):

                spec = var_name[:-len("density_thermal")-1]

                transp_d = 0
                transp_v = 0
                transp_flux = 0
                Sexpl = 0
                Simpl = 0

                # fmt:off

                if core_transport is not None:
                    for model in core_transport.model:
                        core_transp_1d=model.time_slice.current.profiles_1d
                        transp_d      += core_transp_1d.get(f"{spec}/particles/d"   ,0)
                        transp_v      += core_transp_1d.get(f"{spec}/particles/v"   ,0)
                        transp_flux   += core_transp_1d.get(f"{spec}/particles/flux",0)

                if core_sources is not None:
                    for source in core_sources.source:
                        core_source_1d = source.time_slice.current.profiles_1d
                        S=core_source_1d.get(f"{spec}/particles", None)
                        if S is not None:
                            Sexpl         += S
                        else:
                            Sexpl         += core_source_1d.get(f"{spec}/particles_decomposed/explicit_part", 0)
                            Simpl         += core_source_1d.get(f"{spec}/particles_decomposed/implicit_part", 0)
                        # fmt:on

                a = vpr

                b = vprm

                c = rho_tor_boundary

                d = vpr * gm3 * transp_d / rho_tor_boundary

                e = vpr * gm3 * transp_v - rho_tor*k_phi

                f = vpr * Sexpl

                g = vpr * (Simpl+k_phi)

                for i in range(2):
                    bc_ = eq.boundary_condition[i]

                    x = bc_.rho_tor_norm or (0.0 if i == 0 else 1.0)

                    match bc_.identifier.index:
                        case 1:   # 1: value of the field y;
                            u = 1
                            v = 0
                            w = bc_.value[0]
                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = -e(x)/d(x)
                            v = 1.0/d(x)
                            w = bc_.value[0]
                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")
                        case 4:   # 4: flux;
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

            elif var_name.endswith("temperature"):

                spec = var_name[:-len("temperature")-1]

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
                        energy_diff              += core_transp_1d.get(f"{spec}/energy/d")
                        energy_vcon              += core_transp_1d.get(f"{spec}/energy/v")
                        energy_flux              += core_transp_1d.get(f"{spec}/energy/flux")

                if core_sources is not None:
                    for source in core_sources.source:
                        core_source_1d = source.time_slice.current.profiles_1d
                        Q=core_source_1d.get(f"{spec}/energy", None)

                        if Q is not None:
                            Qexpl         += Q
                        else:
                            Qexpl                    += core_source_1d.get(f"{spec}/energy_decomposed/explicit_part")
                            Qimpl                    += core_source_1d.get(f"{spec}/energy_decomposed/implicit_part")

                ns               = vars.get(f"{spec}/density",0)

                ns_flux  = flux_multiplier*vars.get(f"{spec}/density_flux",0)

                # fmt:on

                a = (3/2) * (vpr**(5/3))

                b = (3/2) * (vprm**(5/3))

                c = rho_tor_boundary * inv_vpr23

                d = vpr * gm3 * ns * energy_diff / rho_tor_boundary

                e = vpr * gm3 * ns * energy_vcon + flux_multiplier * ns_flux

                f = (vpr**(5/3)) * (Qexpl + nuz_Tz*ns)

                g = (vpr**(5/3)) * (Qimpl + nu_z*ns)

                for i in range(2):
                    bc_ = eq.boundary_condition[i]

                    x = bc_.rho_tor_norm or (0.0 if i == 0 else 1.0)

                    match bc_.identifier.index:
                        case 1:   # 1: value of the field y;
                            u = 1
                            v = 0
                            w = bc_.value[0]
                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = -e(x)/d(x)
                            v = 1.0/d(x)
                            w = bc_.value[0]
                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")
                        case 4:   # 4: flux;
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

            elif var_name.endswith("momentum"):

                spec = var_name[:-len("momentum")-1]

                n_u_z = 0  # $\sum_{z\neq s}\frac{m_{z}}{\tau_{zs}}n_{z}u_{z,\varphi}$
                n_z = 0    # $\sum_{z\neq s}\frac{1}{\tau_{zs}}m_{z}n_{z}$

                # fmt:off
                ms   = atoms.get(f"{spec}/mass",0.0) 

                ns          = vars.get(f"{spec}/density_thermal")

                ns_m       = core_profiles_prev.get(f"profiles_1d/{spec}/density",0) if core_profiles_prev is not None else 0

                ns_flux     = vars.get(f"{spec}/density_thermal_flux")

                chi_u=0
                Uexpl=0
                Uimpl=0

                if core_transport is not None:
                    for model in core_transport.model:
                        core_transp_1d += model.time_slice.current.profiles_1d 
                        chi_u      += core_transp_1d.get(f"{spec}/momentum/d",0)

                if core_sources is not None:
                    for source in core_sources.source:
                        core_source_1d = source.time_slice.current.profiles_1d
                        Uexpl               = core_source_1d.get(f"{spec}/momentum.toroidal_decomposed.explicit_part",0)
                        Uimpl               = core_source_1d.get(f"{spec}/momentum.toroidal_decomposed.implicit_part",0)
                # fmt:on

                a = (vpr ** (5/3)) * ms * ns

                b = (vprm ** (5/3)) * ms * ns_m

                c = rho_tor_boundary

                d = vpr*gm3*ms*gm8*ns*chi_u/rho_tor_boundary

                e = vpr*gm3*ms*gm8*ns + ms*gm8*ns_flux - ms*gm8 * vpr * rho_tor*k_phi*ns

                f = vpr * (Uexpl+gm8*n_u_z)

                g = vpr * (Uimpl+gm8*(n_z+ms*ns*k_rho_bdry))

                for i in range(2):
                    bc_ = eq.boundary_condition[i]

                    x = bc_.rho_tor_norm or (0.0 if i == 0 else 1.0)

                    match bc_.identifier.index:
                        case 1:   # 1: value of the field y;
                            u = 1
                            v = 0
                            w = bc_.value[0]
                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = -e(x)/d(x)
                            v = 1.0/d(x)
                            w = bc_.value[0]
                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            raise NotImplementedError(f" # 3: scale length of the field y/(-dy/drho_tor);")
                        case 4:   # 4: flux;
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

            else:
                raise RuntimeError(f"Unknown equation of {var_name}!")

            eq["coefficient"] = [a, b, c, d, e, f, g]

            eq["boundary_condition"] = [{"value": bc[0]}, {"value": bc[1]}]

            # logger.debug((var_name, a, b, c, d, e, f, g, bc))
        return vars

    def _solve(self, *args, tau, core_profiles: CoreProfiles, **kwargs):

        hyper_diff = kwargs.get("hyper_diff", None) or\
            self.code.parameters.get("hyper_diff", None) or\
            self._metadata.get("hyper_diff", None) or\
            0.001

        vars = self._update_solver(tau, core_profiles=core_profiles, **kwargs)

        inv_tau = 1.0/tau if tau > 0 else 0

        core_profiles_prev = core_profiles.time_slice.previous

        core_profiles_1d = core_profiles.time_slice.current.profiles_1d

        solver_1d = self.time_slice.current.solver_1d

        equ_s = []

        for equ in solver_1d.equation:

            var_name = equ.primary_quantity.identifier

            Y = vars[var_name]

            Ym = core_profiles_prev.get(f"profiles_1d/{var_name}", 0) if core_profiles_prev is not _not_found_ else 0

            G = vars[var_name+"_flux"]

            a, b, c, d, e, f, g, *_ = equ.coefficient

            dY = (-G + e * Y + hyper_diff * Y.d)/(d + hyper_diff)

            dG = c*(f - g * Y-(a*Y-b*Ym)*inv_tau)

            equ_s.append([Y, dY,  equ.boundary_condition[0].value])

            equ_s.append([G, dG,  equ.boundary_condition[1].value])

        def func(x: array_type, y: array_type, *args) -> array_type:
            # TODO: 需要加速

            res = []
            for var, eq, bc in equ_s:
                if isinstance(eq, (int, float)):
                    res.append(np.full_like(x, eq))
                else:
                    try:
                        eq_res = eq(x, *y[:], *args)
                    except Exception as error:
                        raise RuntimeError(f"Error when apply  op={eq.__repr__()} x={x} args={(y)} !") from error
                    else:
                        # logger.debug((var.__label__, bc, eq_res[:5]))
                        res.append(eq_res)

            res = np.stack(res)
            # logger.debug(res[:, :5])
            return res

        def bc(ya: array_type, yb: array_type, *args) -> array_type:
            res = np.stack([((u*ya[idx]+v*ya[idx+1]-w) if int(idx/2)*2 == idx else (u*yb[idx]+v*yb[idx-1]-w))
                           for idx, (_, _, (u, v, w)) in enumerate(equ_s)])
            # logger.debug(res)
            return res

        x = solver_1d.grid.rho_tor_norm

        Y0 = np.zeros([len(equ_s), len(x)])

        for idx, equ in enumerate(solver_1d.equation):

            a, b, c, d, e, f, g = equ.coefficient

            y = core_profiles_1d[equ.primary_quantity.identifier].__array__()

            Y0[idx*2] = y

            Y0[idx*2+1] = -Function(y, x).d()(x)*d(x) + y*e(x)

        sol = solve_bvp(
            func,
            bc,
            x, Y0,
            bvp_rms_mask=self.code.parameters.get("bvp_rms_mask",  []),
            tolerance=self.code.parameters.get("tolerance", 1.0e-3),
            max_nodes=self.code.parameters.get("max_nodes", 250),
            verbose=self.code.parameters.get("verbose", 2)
        )

        solver_1d.grid.remesh(rho_tor_norm=sol.x)

        for idx, equ in enumerate(solver_1d.equation):

            equ.primary_quantity["profile"] = sol.y[2*idx]
            equ.primary_quantity["flux"] = sol.y[2*idx+1]

        if not sol.success:
            raise RuntimeError(sol.message)
        else:
            logger.debug(f"Solve BVP success: {sol.message} , {sol.niter} iterations")

        solver_1d.grid.remesh(rho_tor_norm=sol.x, psi=sol.y[0])

        for idx, equ in enumerate(solver_1d.equation):
            equ.primary_quantity.profile = sol.y[2*idx]
