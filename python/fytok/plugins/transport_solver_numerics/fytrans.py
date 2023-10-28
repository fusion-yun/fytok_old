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

from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources, CoreSourcesSourceParticles, CoreSourcesSourceEnergy, CoreSourcesSourceMomentum
from fytok.modules.CoreTransport import CoreTransport, CoreTransportModelParticles, CoreTransportModelEnergy, CoreTransportModelMomentum
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.logger import logger

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWO_PI = 2.0 * scipy.constants.pi


@TransportSolverNumerics.register(["fytrans"])
class FyTrans(TransportSolverNumerics):

    def refresh(self, *args,
                core_profiles: CoreProfiles.TimeSlice,
                core_transport: CoreTransport.Model.TimeSlice = None,
                core_sources: CoreSources.Source.TimeSlice = None,
                equilibrium: Equilibrium = None, **kwargs):

        super().refresh(*args,
                        core_profiles=core_profiles,
                        core_transport=core_transport,
                        core_sources=core_sources,
                        equilibrium=equilibrium, **kwargs)

        self.solve(*args,
                   core_profiles=core_profiles,
                   core_transport=core_transport,
                   core_sources=core_sources,
                   equilibrium=equilibrium,
                   **kwargs
                   )

        solver_1d = self.time_slice.current.solver_1d

        core_profiles_1d = core_profiles.time_slice.current.profiles_1d

        for eq in solver_1d.equation:
            core_profiles_1d[eq.primary_quantity.identifier.name] = eq.primary_quantity.profile.__array__()

    def _update_solver(self, *args,
                       solver_1d: TransportSolverNumerics.TimeSlice.Solver1D,
                       core_transport: CoreTransport.Model.TimeSlice,
                       core_sources: CoreSources.Source.TimeSlice,
                       core_profiles: CoreProfiles,
                       equilibrium: Equilibrium, **kwargs):
        # solver_1d = self.time_slice.current.solver_1d

        x = Variable(0, "x")
        vars = {"x": x}

        for idx, eq in enumerate(solver_1d.equation):
            name = var_name
            vars[name] = Variable(idx*2+1, name),
            vars[name + "_flux"] = Variable(idx*2+2, name+"_flux")

        psi = Function(solver_1d.grid.psi, solver_1d.grid.rho_tor_norm)

        rho_tor = rho_tor_boundary*x

        core_profiles_1d = core_profiles.time_slice.current.profiles_1d
        core_profiles_1d_prev = core_profiles.time_slice.previous.profiles_1d

        equilibrium_1d = equilibrium.time_slice.current.profiles_1d
        equilibrium_1d_prev = equilibrium.time_slice.previous.profiles_1d

        core_transp_1d = core_transport.profiles_1d
        core_source_1d = core_sources.profiles_1d

        tau = kwargs.pop("dt",   equilibrium.time_slice.current.time-equilibrium.time_slice.previous.time)

        inv_tau = 0.0 if isclose(tau, 0.0) else 1.0/tau

        flux_multiplier = core_transport.flux_multiplier or 3/2

        # $R_0$ characteristic major radius of the device   [m]
        R0 = equilibrium.time_slice.current.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = equilibrium.time_slice.current.vacuum_toroidal_field.b0

        B0m = equilibrium.time_slice.previous.vacuum_toroidal_field.b0

        # Mesh
        rho_tor_boundary = equilibrium.time_slice.current.profiles_1d.rho_tor(
            equilibrium.time_slice.current.boundary.psi)

        rho_tor_boundary_m = equilibrium.time_slice.previous.profiles_1d.rho_tor(
            equilibrium.time_slice.previous.boundary.psi)

        k_B = (B0 - B0m) / (B0 + B0m) * inv_tau * 2.0

        k_rho_bdry = (rho_tor_boundary - rho_tor_boundary_m) / \
            (rho_tor_boundary + rho_tor_boundary_m)*inv_tau*2.0

        k_phi = k_B + k_rho_bdry

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = equilibrium_1d.profiles_1d.fpol

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]

        vpr = equilibrium.time_slice.current.profiles_1d.dvolume_drho_tor(psi)

        vprm = equilibrium.time_slice.previous.profiles_1d.dvolume_drho_tor(psi)

        inv_vpr23 = vpr**(-2/3)

        fpol = equilibrium_1d.f(psi)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = equilibrium_1d.q(psi)
        gm1 = equilibrium_1d.gm1(psi)  # <1/R^2>
        gm2 = equilibrium_1d.gm2(psi)  # <|grad_rho_tor|^2/R^2>
        gm3 = equilibrium_1d.gm3(psi)  # <|grad_rho_tor|^2>
        gm8 = equilibrium_1d.gm8(psi)  # <R>
        Qimp_k_ns = (3*k_rho_bdry - k_phi * vpr.d())

        conductivity_parallel = core_source_1d.conductivity_parallel

        j_parallel = core_source_1d.j_parallel
        j_parallel_imp = core_source_1d.j_parallel_imp or 0

        # quasi_neutrality_condition
        if "electons.density_thermal" not in vars:

            vars["electons.density_thermal"] = -1 * sum(
                [v for k, v in vars.items() if k.endswith("density_thermal") or k.endswith("density_fast")], 0)
        else:
            ne = vars["electons.density_thermal"]

            num_of_ion = sum([1 for ion in core_profiles_1d.ion if not ion.is_impurity])

            density_imp = sum([ion.z_ion_1d*ion.density for ion in core_profiles_1d.ion if ion.is_impurity])

            n_i_prop = (ne-density_imp) / num_of_ion

            for ion in core_profiles_1d.ion:
                if not ion.is_impurity:
                    ion["density"] = n_i_prop/ion.z

        for idx, eq in enumerate(solver_1d.equation):

            var_name = eq.primary_quantity.identifier.name

            bc = [[0, 0, 0], [0, 0, 0]]
            a = 0
            b = 0

            if var_name == "psi":
                # FIXME: not complete

                a = conductivity_parallel

                b = conductivity_parallel

                c = (scipy.constants.mu_0 * B0*rho_tor * rho_tor_boundary)/fpol2

                d = vpr * gm2 / (fpol * rho_tor_boundary)/(TWO_PI**2)

                e = 0

                f = -  vpr * (j_parallel)/TWO_PI

                g = - vpr * (j_parallel_imp)/TWO_PI

                for i in range(2):
                    bc_ = eq.boundary_condition[i]

                    match bc_.identifier.index:
                        case 1:  # poloidal flux;
                            u = 1
                            v = 0
                            w = bc_.value[0]
                        case 2:  # ip, total current inside x=1
                            Ip = bc_.value[0]
                            u = 0
                            v = 1
                            w = scipy.constants.mu_0 * Ip / self._fpol[-1]
                        case 3:  # loop voltage;
                            Uloop_bdry = bc_.value[0]
                            u = 0
                            v = 1
                            w = (tau*Uloop_bdry + core_profiles_1d_prev.psi[-1])*d(1.0)
                        case 5:  # generic boundary condition y expressed as a1y' + a2y=a3;
                            u = bc_.value[1]
                            v = bc_.value[0]
                            w = bc_.value[2]
                        case 6:  # equation not solved;
                            raise NotImplementedError(bc_.identifier.index)
                        case _:
                            raise NotImplementedError(bc_)
                    bc[i] = [u, v, w]

            elif var_name.endswith("density_thermal"):
                spec = var_name[:-len("density_thermal")]

                # fmt:off
                transp_d                                = core_transp_1d.get(f"{spec}/particles/d")
                transp_v                                = core_transp_1d.get(f"{spec}/particles/v")
                transp_flux                             = core_transp_1d.get(f"{spec}/particles/flux")
                Sexpl                             = core_source_1d.get(f"{spec}/particles_decomposed/explicit_part")
                Simpl                             = core_source_1d.get(f"{spec}/particles_decomposed/implicit_part")
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

                    match bc_.identifier.index:
                        case 1:
                            u = 1
                            v = 0
                            w = bc.value[0]
                        case _:
                            raise NotImplementedError(bc.identifier)
                    bc[i] = [u, v, w]

            elif var_name.endswith("temperature"):
                spec = var_name[:-len("temperature")]

                nuz_Tz = 0  # $\sum_{z\neq s}\nu_{zs}T_{z}$
                nu_z = 0  # $\sum_{z\neq s}\nu_{zs}$

                # fmt:off
                energy_diff                               = core_transp_1d.get(f"{spec}/energy/d")
                energy_vcon                               = core_transp_1d.get(f"{spec}/energy/v")
                energy_flux                               = core_transp_1d.get(f"{spec}/energy/flux")
                Qexpl                                     = core_source_1d.get(f"{spec}/energy_decomposed/explicit_part")
                Qimpl                                     = core_source_1d.get(f"{spec}/energy_decomposed/implicit_part")

                ns                     = vars.get(f"{spec}/density"),
                ns_flux                = vars.get(f"{spec}/density_flux"),
                # fmt:on

                if inv_tau != 0:
                    a = (3/2) * (vpr**(5/3))
                    b = (3/2) * (vprm**(5/3))

                c = rho_tor_boundary * inv_vpr23

                d = vpr * gm3 * ns * energy_diff / rho_tor_boundary

                e = vpr * gm3 * ns * energy_vcon + flux_multiplier * ns_flux

                f = (vpr**(5/3)) * (Qexpl + nuz_Tz*ns)

                g = (vpr**(5/3)) * (Qimpl + nu_z*ns)

                for i in range(2):
                    bc_ = eq.boundary_condition[i]

                    match bc_.identifier.index:
                        case 1:
                            u = 1
                            v = 0
                            w = bc_.value[0]
                        case _:
                            raise NotImplementedError(bc.identifier)
                    bc[i] = [u, v, w]

            elif var_name.endswith("momentum"):
                spec = var_name[:-len("momentum")]

                n_u_z = 0  # $\sum_{z\neq s}\frac{m_{z}}{\tau_{zs}}n_{z}u_{z,\varphi}$
                n_z = 0    # $\sum_{z\neq s}\frac{1}{\tau_{zs}}m_{z}n_{z}$
                # fmt:off
                ms                  = core_profiles_1d.get(f"{spec}/mass")
                ns      = vars.get(f"{spec}/density")
                ns_flux = vars.get(f"{spec}/density_flux")
                chi_u               = core_transp_1d.get(f"{spec}/momentum/d")
                Uexpl               = core_source_1d.get(f"{spec}/momentum.toroidal_decomposed.explicit_part")
                Uimpl               = core_source_1d.get(f"{spec}/momentum.toroidal_decomposed.implicit_part")
                # fmt:on
                if inv_tau != 0:
                    ns_m = core_profiles_1d_prev.get(f"{spec}/density")
                    a = (vpr**(5/3))*ms * ns
                    b = (vprm**(5/3))*ms * ns_m

                c = rho_tor_boundary

                d = vpr*gm3*ms*gm8*ns*chi_u/rho_tor_boundary

                e = vpr*gm3*ms*gm8*ns + ms*gm8*ns_flux - ms*gm8 * vpr * rho_tor*k_phi*ns

                f = vpr * (Uexpl+gm8*n_u_z)

                g = vpr * (Uimpl+gm8*(n_z+ms*ns*k_rho_bdry))

                for i in range(2):
                    bc_ = eq.boundary_condition[i]

                    match bc_.identifier.index:
                        case 1:
                            u = 1
                            v = 0
                            w = bc_.value[0]
                        case _:
                            raise NotImplementedError(bc.identifier)
                    bc[i] = [u, v, w]
            else:
                raise RuntimeError(f"Unknown equation of {var_name}!")

            eq.coefficient = [a, b, c, d, e, f, g]

            eq.boundary_condition = bc

        return solver_1d, vars, inv_tau

    def solve(self, *args,
              core_transport: CoreTransport.Model.TimeSlice,
              core_sources: CoreSources.Source.TimeSlice,
              core_profiles: CoreProfiles,
              equilibrium: Equilibrium,
              **kwargs):

        solver_1d, vars, inv_tau = self._update_solver(self.time_slice.current.solver_1d, *args, **kwargs)

        hyper_diff = kwargs.get("hyper_diff", None) or \
            self.code.parameters.get("hyper_diff", None) or \
            self._metadata.get("hyper_diff", None)

        core_profiles_1d_prev = core_profiles.time_slice.previous.profiles_1d

        equ_s = []

        for eq in solver_1d.equation:
            var_name = eq.primary_quantity.identifier.name
            Y = vars[var_name]
            G = vars[var_name+"_flux"]

            a, b, c, d, e, f, g = eq.coefficient

            dG = c*(f - g * Y)

            dY = (-G + e * Y + hyper_diff * Y.d())/(d + hyper_diff)

            if inv_tau > 0:
                Ym = core_profiles_1d_prev.get(var_name, 0)
                dG -= inv_tau*c*(a*Y-b*Ym)
                equ_s.append([Y, dY, bc])
                equ_s.append([G, dG, 0])

        def func(x: array_type, y: array_type, *args) -> array_type:
            # TODO: 需要加速
            return np.stack([eq(x, *y[:], *args) for _, eq, _ in equ_s])

        def bc(ya: array_type, yb: array_type, *args) -> array_type:
            # TODO: 需要加速
            return np.stack([bc(ya, yb, *args) for _, _, bc in equ_s])

        X0 = self.time_slice.current.solver_1d.grid.rho_tor_norm

        Y0 = np.zeros([len(vars), len(X0)])

        for idx, k in enumerate(vars.keys()):
            Y0[idx] = vars[k]

        sol = solve_bvp(
            func,
            bc,
            X0, Y0,
            bvp_rms_mask=self._parameters.get("bvp_rms_mask", []),
            tolerance=self._parameters.get("tolerance", 1.0e-3),
            max_nodes=self._parameters.get("max_nodes", 250),
            verbose=self._parameters.get("verbose", 0)
        )

    def rotation_bc(self):
        pass
