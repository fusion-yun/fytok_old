import collections
from math import isclose
from typing import (Tuple)

import numpy as np
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics
from fytok.modules.Equilibrium import Equilibrium
from scipy import constants
from spdm.data.Function import function_like
from spdm.data.Path import Path
from spdm.utils.typing import array_type
from spdm.numlib.bvp import solve_bvp
from spdm.numlib.misc import array_like
from fytok.utils.logger import logger

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWO_PI = 2.0 * constants.pi


@TransportSolverNumerics.register(["fytrans"])
class TransportSolverBVP(TransportSolverNumerics):

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

        self._update_coeff(*args,
                           core_profiles=core_profiles,
                           core_transport=core_transport,
                           core_sources=core_sources,
                           equilibrium=equilibrium, **kwargs
                           )

    def _update_coeff(self, *args,
                      core_profiles: CoreProfiles.TimeSlice,
                      core_transport: CoreTransport.Model.TimeSlice = None,
                      core_sources: CoreSources.Source.TimeSlice = None,
                      equilibrium: Equilibrium = None, **kwargs):

        solver_1d = self.time_slice.current.solver_1d

        psi = solver_1d.grid.psi

        rho_tor_norm = solver_1d.grid.rho_tor_norm

        core_profiles_1d = core_profiles.profiles_1d
        equilibrium_1d = equilibrium.time_slice.current.profiles_1d
        equilibrium_1d_prev = equilibrium.time_slice.previous.profiles_1d
        core_transport_1d = core_transport.profiles_1d
        core_sources_1d = core_sources.profiles_1d

        tau = kwargs.pop("dt",   equilibrium.time_slice.current.time-equilibrium.time_slice.previous.time)

        inv_tau = 0.0 if isclose(tau, 0.0) else 1.0/tau

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
        fpol = equilibrium_1d.profiles_1d.fpol(psi)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]

        vpr = equilibrium.time_slice.current.profiles_1d.dvolume_drho_tor(psi)

        vprm = equilibrium.time_slice.previous.profiles_1d.dvolume_drho_tor(psi)

        vpr5_3 = np.abs(vpr)**(5/3)

        vpr5_3m = np.abs(vprm)**(5/3)

        if np.isclose(vpr[0], 0.0):
            inv_vpr23 = function_like(vpr[1:]**(-2/3), rho_tor_norm[1:])(rho_tor_norm)
        else:
            inv_vpr23 = function_like(vpr**(-2/3), rho_tor_norm)(rho_tor_norm)

        # $q$ safety factor                                 [-]
        qsf = equilibrium.time_slice.current.profiles_1d.q(psi)
        gm1 = equilibrium.time_slice.current.profiles_1d.gm1(psi)
        gm2 = equilibrium.time_slice.current.profiles_1d.gm2(psi)
        gm3 = equilibrium.time_slice.current.profiles_1d.gm3(psi)

        Qimp_k_ns = (3*k_rho_bdry - k_phi * vpr.d()(psi))

        for eq in solver_1d.equation:
            if eq.primary_quantity.identifier.name == "current":
                eq.coefficient = self._current_coeff(
                    rho_tor_norm, eq.primary_quantity.profile, eq.primary_quantity.d_dr, eq.primary_quantity.d_dt,
                    conductivity_parallel=core_transport_1d.conductivity_parallel(rho_tor_norm),
                    j_parallel=core_sources_1d.j_parallel(rho_tor_norm),
                    vpr=vpr, gm2=gm2
                )
            elif eq.primary_quantity.identifier.name == "electrons.particle":
                ele_transp = core_transport_1d.electrons
                ele_source = core_sources_1d.electrons
                # fmt:off
                eq.coefficient = self._electrons_particle_coeff(
                    rho_tor_norm, eq.primary_quantity.profile, eq.primary_quantity.d_dr, eq.primary_quantity.d_dt,
                    d                   = ele_transp.particles.d(rho_tor_norm),
                    v                   = ele_transp.particles.v(rho_tor_norm),
                    flux                = ele_transp.particles.flux(rho_tor_norm),
                    s                   = ele_source.particles(rho_tor_norm),

                    vpr                 = vpr,
                    vprm                = vprm,
                    gm3                 = gm3
                )
                # fmt:on
            elif eq.primary_quantity.identifier.name == "electrons.temperature":
                ele_transp = core_transport_1d.electrons
                ele_source = core_sources_1d.electrons
                density = core_profiles_1d.electrons.density(rho_tor_norm)
                density_flux = core_profiles_1d.electrons.get("density_flux", 0)(rho_tor_norm)
                # fmt:off
                eq.coefficient = self._electrons_energy_coeff(
                    rho_tor_norm, eq.primary_quantity.profile, eq.primary_quantity.d_dr, eq.primary_quantity.d_dt,
                    d                   = ele_transp.energy.d(rho_tor_norm),
                    v                   = ele_transp.energy.v(rho_tor_norm),
                    flux                = ele_transp.energy.flux(rho_tor_norm),
                    s                   = ele_source.energy(rho_tor_norm),

                    density             = density,
                    density_flux        = density_flux,
                    rho_tor_boundary    = rho_tor_boundary,
                    k_phi               = k_phi,
                    k_rho_bdry          = k_rho_bdry,
                    vpr                 = vpr,
                    vprm                = vprm,
                    vpr5_3              = vpr5_3,
                    vpr5_3m             = vpr5_3m,
                    inv_vpr23           = inv_vpr23,
                    Qimp_k_ns           = Qimp_k_ns,
                    gm3                 = gm3,
                )
                # fmt:on
            elif eq.primary_quantity.identifier.name.startswith("ion") and eq.primary_quantity.identifier.name.endswith("density_thermal"):
                ion_transp = core_transport_1d.ion[eq.primary_quantity.identifier.index]
                ion_source = core_sources_1d.ion[eq.primary_quantity.identifier.index]
                # fmt:off
                eq.coefficient = self._ion_particle_coeff(
                    rho_tor_norm, eq.primary_quantity.profile, eq.primary_quantity.d_dr, eq.primary_quantity.d_dt,
                    d                   = ion_transp.particles.d(rho_tor_norm),
                    v                   = ion_transp.particles.v(rho_tor_norm),
                    flux                = ion_transp.particles.flux(rho_tor_norm),
                    s                   = ion_source.particles(rho_tor_norm),
                    vpr                 = vpr,
                    vprm                = vprm,
                    gm3                 = gm3
                )
                # fmt:on
            elif eq.primary_quantity.identifier.name.startswith("ion") and eq.primary_quantity.identifier.name.endswith("energy"):
                ion_transp = core_transport_1d.ion[eq.primary_quantity.identifier.index]
                ion_source = core_sources_1d.ion[eq.primary_quantity.identifier.index]
                density = core_profiles_1d.ion[eq.primary_quantity.identifier.index].density(rho_tor_norm)
                density_flux = core_profiles_1d.ion[eq.primary_quantity.identifier.index].get(
                    "density_flux", 0)(rho_tor_norm)
                # fmt:off

                eq.coefficient = self._ion_particle_coeff(
                    rho_tor_norm, eq.primary_quantity.profile, eq.primary_quantity.d_dr, eq.primary_quantity.d_dt,
                    d                   = ion_transp.energy.d(rho_tor_norm),
                    v                   = ion_transp.energy.v(rho_tor_norm),
                    flux                = ion_transp.energy.flux(rho_tor_norm),
                    s                   = ion_source.energy(rho_tor_norm),

                    density             = density,
                    density_flux        = density_flux,
                    rho_tor_boundary    = self._rho_tor_boundary,
                    k_phi               = k_phi,
                    k_rho_bdry          = k_rho_bdry,
                    vpr                 = vpr,
                    vprm                = vprm,
                    vpr5_3              = vpr5_3,
                    vpr5_3m             = vpr5_3m,
                    inv_vpr23           = inv_vpr23,
                    Qimp_k_ns           = Qimp_k_ns,
                    gm3                 = gm3,
                )
                # fmt:on

    ###########################################################################################################################################

    def _current_coeff(self, x: np.ndarray,  core_profiles_1d: CoreProfiles.TimeSlice.Profiles1D, *args,
                       conductivity_parallel: array_type,
                       j_parallel: array_type,
                       vpr: array_type,
                       gm2: array_type,
                       **kwargs):

        ym = array_like(x, self._core_profiles_prev.profiles_1d.get("psi", 0))

        y = array_like(x, core_profiles_1d.get("psi", 0))

        flux = array_like(x, core_profiles_1d.get("psi_flux", 0))

        yp = function_like(y, x).derivative(x)

        hyper_diff = self._hyper_diff

        # conductivity_parallel = self._core_transport_1d.conductivity_parallel(x)

        # j_parallel = self._core_sources_1d.j_parallel(x)

        inv_tau = self._inv_tau

        B0 = self._B0

        fpol = self._fpol(x)

        fpol2 = fpol**2

        vpr = self._vpr(x)

        gm2 = self._gm2(x)

        rho_tor_boundary = self._rho_tor_boundary

        a = conductivity_parallel * x * rho_tor_boundary

        b = conductivity_parallel * x * rho_tor_boundary

        c = (constants.mu_0 * B0 * rho_tor_boundary)/fpol2

        d = vpr * gm2 / fpol / (rho_tor_boundary)/(TWO_PI)

        e = 0

        S = -  vpr * (j_parallel)/TWO_PI

        dy = (-flux + e * y + hyper_diff * yp)/(d + hyper_diff)

        dg = S

        if not np.isclose(inv_tau, 0.0):
            k_phi = self._k_phi
            Qimp_k_ns = self._Qimp_k_ns(x)
            C = (constants.mu_0 * B0 * k_phi) * \
                (conductivity_parallel * (x*rho_tor_boundary)**2/fpol2)
            dg = dg - (a * y - b * ym) * inv_tau
            dg = dg + conductivity_parallel*Qimp_k_ns*y
            dg = dg + function_like(C, x).derivative(x)*y + C*dy

        dg = dg*c

        dy = array_like(x, dy)

        dy[0] = 0

        dg = array_like(x, dg)

        return dy, dg

    def _current_bc(self, ya: float, ga: float, yb: float, gb: float, var):
        bc = self.boundary_conditions_1d.current
        # -----------------------------------------------------------
        # boundary condition, value
        # bc: TransportSolverNumerics.BoundaryConditions1D.BoundaryConditions = self.boundary_conditions_1d.current
        # axis
        ua, va, wa = 0, 1, 0

        # Identifier of the boundary condition type.
        #   ID =    1: poloidal flux;
        #           2: ip;
        #           3: loop voltage;
        #           4: undefined;
        #           5: generic boundary condition y expressed as a1y'+a2y=a3.
        #           6: equation not solved; [eV]

        if bc.identifier.index == 1:  # poloidal flux;
            ub = 1
            vb = 0
            wb = bc.value[0]
        elif bc.identifier.index == 2:  # ip, total current inside x=1
            Ip = bc.value[0]
            ub = 0
            vb = 1
            wb = constants.mu_0 * Ip / self._fpol[-1]
        elif bc.identifier.index == 3:  # loop voltage;
            Uloop_bdry = bc.value[0]
            ub = 0
            vb = 1
            wb = (self._tau*Uloop_bdry +
                  self._core_profiles_prev["psi"][-1])*d(1.0)
        elif bc.identifier.index == 5:  # generic boundary condition y expressed as a1y' + a2y=a3;
            ub = bc.value[1]
            vb = bc.value[0]
            wb = bc.value[2]
        elif bc.identifier.index == 6:  # equation not solved;
            raise NotImplementedError(bc.identifier.index)
        else:
            raise NotImplementedError(bc)

        return float((ua * ya + va * ga - wa)), float((ub * yb + vb * gb - wb))

    def _electrons_particle_bc(self, ya: float, ga: float, yb: float, gb: float, *args):
        return self.particle_bc(ya, ga, yb, gb, self.boundary_conditions_1d.electrons.particles)

    def electrons_energy_bc(self, ya: float, ga: float, yb: float, gb: float, *args):
        return self.energy_bc(ya, ga, yb, gb, self.boundary_conditions_1d.electrons.energy)

    def ion_particle_fast_coeff(self, x: np.ndarray, core_profiles_1d: CoreProfiles.TimeSlice.Profiles1D, var) -> Tuple[np.ndarray, np.ndarray]:

        ym = array_like(x, self._core_profiles_prev.profiles_1d.ion[var[1]].density_fast)

        y = array_like(x, core_profiles_1d.ion[var[1]].density_fast)

        flux = array_like(x, core_profiles_1d.ion[var[1]].get("density_fast_flux", 0))

        ion_src: CoreSources.Source.Profiles1D.Ion = self._core_sources_1d.ion[var[1]]

        ion_transp: TransportCoeff = self._core_transport_1d.ion[var[1]].particles

        D = ion_transp.d(x)*ion_transp.d_fast_factor(x)

        V = D*ion_transp.v_fast_factor(x)

        S = ion_src.particles_fast(x)

        return self.particle_coeff(x, y, flux, ym, D, V, S)

    def ion_particle_fast_bc(self, ya: float, ga: float, yb: float, gb: float, var):
        return self.particle_bc(ya, ga, yb, gb, self.boundary_conditions_1d.ion[var[1]].particles_fast)

    def ion_particle_bc(self, ya: float, ga: float, yb: float, gb: float, var):
        return self.particle_bc(ya, ga, yb, gb, self.boundary_conditions_1d.ion[var[1]].particles)

    def ion_energy_coeff(self, x: np.ndarray, core_profiles_1d: CoreProfiles.TimeSlice.Profiles1D, var) -> Tuple[np.ndarray, np.ndarray]:
        ym = array_like(x, self._core_profiles_prev.profiles_1d.ion[var[1]].temperature)

        y = array_like(x, core_profiles_1d.ion[var[1]].temperature)

        flux = array_like(x, core_profiles_1d.ion[var[1]].get("temperature_flux", 0))

        density = array_like(x, core_profiles_1d.ion[var[1]].density)

        density_flux = array_like(x, core_profiles_1d.ion[var[1]].get("density_flux", 0))

        return self.energy_coeff(x, y, flux, ym,
                                 self._core_transport_1d.ion[var[1]].energy.d(x),
                                 self._core_transport_1d.ion[var[1]].energy.v(x),
                                 self._core_sources_1d.ion[var[1]].energy(x),
                                 density,
                                 density_flux)

    def ion_energy_bc(self, ya: float, ga: float, yb: float, gb: float, var):
        return self.energy_bc(ya, ga, yb, gb, self.boundary_conditions_1d.ion[var[1]].energy)

    ###########################################################################################################################################

    def particle_coeff(self,
                       x: array_type,
                       y: np.ndarray,
                       dydr: np.ndarray,
                       dydt: np.ndarray,
                       ym: np.ndarray,
                       flux: np.ndarray,
                       transp_d: np.ndarray,
                       transp_v: np.ndarray,
                       source: np.ndarray,

                       hyper_diff: float,
                       inv_tau: float,
                       k_phi: float,
                       k_rho_bdry: float,
                       vpr: array_type,
                       vprm: array_type,
                       gm3: array_type,
                       rho_tor_boundary,
                       ) -> Tuple[np.ndarray, np.ndarray]:

        a = vpr

        b = vprm

        c = rho_tor_boundary

        d = vpr * gm3 * transp_d / rho_tor_boundary

        e = vpr * gm3 * transp_v

        S = vpr * source

        dy = (-flux + e * y + hyper_diff * dydr)/(d + hyper_diff)

        dg = S

        if not isclose(inv_tau, 0.0):
            dg = dg - (a * y - b * ym)*inv_tau + vpr * k_rho_bdry
            dg = dg + function_like(vpr * x * k_phi, x).d()(x)*y
            dg = dg + vpr * x * k_phi * dy

        dg = dg*c
        dy = array_like(x, dy)
        dg = array_like(x, dg)
        return dy, dg

    def particle_bc(self, ya: float, ga: float, yb: float, gb: float, bc: TransportSolverNumerics.BoundaryConditions1D):

        # axis
        u0, v0, w0 = 0, 1, 0

        if bc.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        else:
            raise NotImplementedError(bc.identifier)

        return float((u0 * ya + v0 * ga - w0)), float((u1 * yb + v1 * gb - w1))

    def energy_coeff(self,
                     x: array_type,
                     y: np.ndarray,
                     ym: np.ndarray,
                     dydr: np.ndarray,
                     dydt: np.ndarray,
                     flux: np.ndarray,
                     heat_diff:  np.ndarray,
                     heat_vconv:  np.ndarray,
                     heat_src:  np.ndarray,
                     density: np.ndarray,
                     density_flux: np.ndarray,
                     hyper_diff: float,
                     inv_daut: float,
                     rho_tor_boundary: float,
                     k_phi: float,
                     k_rho_bdry: float,
                     vpr: array_type,
                     vprm: array_type,
                     vpr5_3: array_type,
                     vpr5_3m: array_type,
                     inv_vpr23: array_type,
                     Qimp_k_ns: array_type,
                     gm3: array_type,
                     ) -> Tuple[np.ndarray, np.ndarray]:

        a = (3/2) * vpr5_3 * y

        b = (3/2) * vpr5_3m * ym

        c = rho_tor_boundary * inv_vpr23

        d = vpr * gm3 * density * heat_diff / rho_tor_boundary

        e = vpr * gm3 * density * heat_vconv + 3/2 * density_flux

        S = vpr5_3 * heat_src

        dy = (-flux + e * y + hyper_diff * dydr)/(d + hyper_diff)

        dg = S

        if not isclose(inv_tau, 0.0):
            dg = dg - (a * y - b * ym)*inv_tau
            dg = dg + vpr5_3 * Qimp_k_ns * y
            dg = dg + function_like(vpr * (3/4)*k_phi * x * density, x).d()(x) * y
            dg = dg + vpr * (3/4)*k_phi * x * density*dy

        dg = dg*c

        dy = array_like(x, dy)
        dg = array_like(x, dg)
        dy[0] = 0
        return dy, dg

    def energy_bc(self, ya: float, ga: float, yb: float, gb: float, bc: TransportSolverNumerics.BoundaryConditions1D,):

        # ----------------------------------------------
        # Boundary Condition
        # bc: TransportSolverNumerics.BoundaryConditions1D.BoundaryConditions = \
        #     self.boundary_conditions_1d.fetch(var_id).energy

        # axis
        u0, v0, w0 = 0, 1, 0

        if bc.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        else:
            raise NotImplementedError(bc)

        return float((u0 * ya + v0 * ga - w0)), float((u1 * yb + v1 * gb - w1))

    def rotation_coeff(self, x: np.ndarray,  y: np.ndarray, flux: np.ndarray, **kwargs):
        r"""
            Rotation Transport
            .. math::  \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime\frac{5}{3}}\left\langle R\right\rangle \
                        m_{i}n_{i}u_{i,\varphi}\right)+\frac{\partial}{\partial\rho}\Phi_{i}=V^{\prime}\left[U_{i,\varphi,exp}-U_{i,\varphi}\cdot u_{i,\varphi}+U_{zi,\varphi}\right]
                :label: transport_rotation
        """
        logger.warning(f"TODO: Rotation Transport is not implemented!")
        return 0.0

    def quasi_neutrality_condition(self, core_profiles_1d: CoreProfiles.TimeSlice.Profiles1D) -> None:

        rho_tor_norm = core_profiles_1d.grid.rho_tor_norm

        density_imp = sum([ion.z_ion_1d(rho_tor_norm)*ion.density(rho_tor_norm)
                          for ion in core_profiles_1d.ion if ion.is_impurity])

        density_flux_imp = sum([ion.z_ion_1d(rho_tor_norm)*array_like(rho_tor_norm, ion.get("density_flux", 0))
                                for ion in core_profiles_1d.ion if ion.is_impurity])

        if self._particle_solver == "electrons":
            n_e = core_profiles_1d.electrons.density(rho_tor_norm)

            num_of_ion = sum([1 for ion in core_profiles_1d.ion if not ion.is_impurity])

            n_i_prop = (n_e-density_imp) / num_of_ion

            for ion in core_profiles_1d.ion:
                if not ion.is_impurity:
                    ion["density"] = n_i_prop/ion.z

        else:

            density_ion = sum([ion.z*ion.density(rho_tor_norm)
                               for ion in core_profiles_1d.ion if not ion.is_impurity])
            density_flux_ion = sum([ion.z*ion.density_flux(rho_tor_norm)
                                    for ion in core_profiles_1d.ion if not ion.is_impurity])

            core_profiles_1d.electrons["density"] = density_ion + density_imp
            core_profiles_1d.electrons["density_flux"] = density_flux_ion + density_flux_imp

        # core_profiles_next. profiles_1d.grid.rho_tor_norm

        # profiles_1d_next["conductivity_parallel"] = core_transport_1d.conductivity_parallel(rho_tor_norm)

        # profiles_1d_next["j_total"] = rho_tor_norm, core_transport_1d.j_parallel(rho_tor_norm)

        # for idx, var_id in enumerate(var_list):
        #     profiles_1d_next[var_id] =  sol.y[idx*2]
        #     profiles_1d_next[var_id[:-1]+[f"{var_id[-1]}_flux"]] =   sol.y[idx*2+1]
