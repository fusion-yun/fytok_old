"""

"""

import collections
import enum
from itertools import chain
from math import isclose, log
from typing import (Any, Callable, Iterator, Mapping, Optional, Sequence,
                    Tuple, Type, Union)

import numpy as np
from fytok.IDS import IDS
from fytok.common.Misc import Identifier, VacuumToroidalField
from fytok.constants.Atoms import atoms
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport, TransportCoeff
from fytok.transport.CoreTransportSolver import CoreTransportSolver
from fytok.transport.Equilibrium import Equilibrium
from fytok.transport.MagneticCoordSystem import RadialGrid
from spdm.numlib.bvp import BVPResult, solve_bvp
from spdm.numlib.misc import array_like
from numpy.core.defchararray import array
from scipy import constants
from spdm.data import (Dict, Function, List, Path, Query, function_like,
                       sp_property)
from spdm.util.logger import logger
from spdm.common.tags import _not_found_
from spdm.util.misc import convert_to_named_tuple

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWO_PI = 2.0 * constants.pi


class CoreTransportSolverBVPNonlinear(CoreTransportSolver):
    r"""
        Solve transport equations :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
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
                .. math ::  \sigma_{\parallel}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho} \right) \psi= \
                            \frac{F^{2}}{\mu_{0}B_{0}\rho}\frac{\partial}{\partial\rho}\left[\frac{V^{\prime}}{4\pi^{2}}\left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle \
                            \frac{1}{F}\frac{\partial\psi}{\partial\rho}\right]-\frac{V^{\prime}}{2\pi\rho}\left(j_{ni,exp}+j_{ni,imp}\psi\right)
                    :label: transport_current


                if :math:`\psi` is not solved, then

                ..  math ::  \psi =\int_{0}^{\rho}\frac{2\pi B_{0}}{q}\rho d\rho

            Particle Transport
            Note:

                .. math::
                    \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                    \left(V^{\prime}n_{s}\right)+\frac{\partial}{\partial\rho}\Gamma_{s}=\
                    V^{\prime}\left(S_{s,exp}-S_{s,imp}\cdot n_{s}\right)
                    :label: particle_density_transport

                .. math::
                    \Gamma_{s}\equiv-D_{s}\cdot\frac{\partial n_{s}}{\partial\rho}+v_{s}^{pinch}\cdot n_{s}
                    :label: particle_density_gamma

            Heat transport equations

            Note:

                ion

                .. math:: \frac{3}{2}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                            \left(n_{i}T_{i}V^{\prime\frac{5}{3}}\right)+V^{\prime\frac{2}{3}}\frac{\partial}{\partial\rho}\left(q_{i}+T_{i}\gamma_{i}\right)=\
                            V^{\prime\frac{5}{3}}\left[Q_{i,exp}-Q_{i,imp}\cdot T_{i}+Q_{ei}+Q_{zi}+Q_{\gamma i}\right]
                    :label: transport_ion_temperature

                electron

                .. math:: \frac{3}{2}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                            \left(n_{e}T_{e}V^{\prime\frac{5}{3}}\right)+V^{\prime\frac{2}{3}}\frac{\partial}{\partial\rho}\left(q_{e}+T_{e}\gamma_{e}\right)=
                            V^{\prime\frac{5}{3}}\left[Q_{e,exp}-Q_{e,imp}\cdot T_{e}+Q_{ei}-Q_{\gamma i}\right]
                    :label: transport_electron_temperature
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._core_transport: CoreTransport = None
        # self._core_sources: CoreSources = None
        # self._equilibrium: Equilibrium = None
        # self._var_list = []
        self._enable_nonlinear = True

    ###########################################################################################################################################

    def transp_current(self, x: np.ndarray,  core_profiles_1d: CoreProfiles.Profiles1D, *args, **kwargs):

        ym = array_like(x, self._core_profiles_prev.profiles_1d.get("psi", 0))

        y = array_like(x, core_profiles_1d.get("psi", 0))

        flux = array_like(x, core_profiles_1d.get("psi_flux", 0))

        yp = Function(x, y).derivative(x)

        hyper_diff = self._hyper_diff

        conductivity_parallel = self._core_transport_1d.conductivity_parallel(x)

        j_parallel = self._core_sources_1d.j_parallel(x)

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
            dg = dg + Function(x, C).derivative(x)*y + C*dy

        dg = dg*c

        dy = array_like(x, dy)

        dy[0] = 0

        dg = array_like(x, dg)

        return dy, dg

    def bc_current(self, ya: float, ga: float, yb: float, gb: float, var):
        bc = self.boundary_conditions_1d.current
        # -----------------------------------------------------------
        # boundary condition, value
        # bc: CoreTransportSolver.BoundaryConditions1D.BoundaryConditions = self.boundary_conditions_1d.current
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

    def transp_electrons_particle(self, x: np.ndarray, core_profiles_1d: CoreProfiles.Profiles1D, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        ym = self._core_profiles_prev.profiles_1d.electrons.density(x)

        y = core_profiles_1d.electrons.density(x)

        flux = array_like(x, core_profiles_1d.electrons.get("density_flux", 0))

        return self.transp_particle(x, y,  flux, ym,
                                    self._core_transport_1d.electrons.particles.d(x),
                                    self._core_transport_1d.electrons.particles.v(x),
                                    self._core_sources_1d.electrons.particles(x))

    def bc_electrons_particle(self, ya: float, ga: float, yb: float, gb: float, *args):
        return self.bc_particle(ya, ga, yb, gb, self.boundary_conditions_1d.electrons.particles)

    def transp_electrons_energy(self, x: np.ndarray, core_profiles_1d: CoreProfiles.Profiles1D, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        ym = array_like(x, self._core_profiles_prev.profiles_1d.electrons.temperature)

        y = array_like(x, core_profiles_1d.electrons.temperature)

        flux = array_like(x, core_profiles_1d.electrons.get("temperature_flux", 0))

        density = array_like(x, core_profiles_1d.electrons.density)

        density_flux = array_like(x, core_profiles_1d.electrons.get("density_flux", 0))

        return self.transp_energy(x, y, flux, ym,
                                  self._core_transport_1d.electrons.energy.d(x),
                                  self._core_transport_1d.electrons.energy.v(x),
                                  self._core_sources_1d.electrons.energy(x),
                                  density,
                                  density_flux)

    def bc_electrons_energy(self, ya: float, ga: float, yb: float, gb: float, *args):
        return self.bc_energy(ya, ga, yb, gb, self.boundary_conditions_1d.electrons.energy)

    def transp_ion_particle(self, x: np.ndarray, core_profiles_1d: CoreProfiles.Profiles1D, var) -> Tuple[np.ndarray, np.ndarray]:

        ym = array_like(x, self._core_profiles_prev.profiles_1d.ion[var[1]].density)

        y = array_like(x, core_profiles_1d.ion[var[1]].density)

        flux = array_like(x, core_profiles_1d.ion[var[1]].get("density_flux", 0))

        ion_src: CoreSources.Source.Profiles1D.Ion = self._core_sources_1d.ion[var[1]]

        ion_transp: CoreTransport.Model.Profiles1D.Ion = self._core_transport_1d.ion[var[1]]

        return self.transp_particle(x, y, flux, ym,
                                    ion_transp.particles.d(x),
                                    ion_transp.particles.v(x),
                                    ion_src.particles(x))

    def transp_ion_particle_thermal(self, x: np.ndarray, core_profiles_1d: CoreProfiles.Profiles1D, var) -> Tuple[np.ndarray, np.ndarray]:

        ym = array_like(x, self._core_profiles_prev.profiles_1d.ion[var[1]].density_thermal)

        y = array_like(x, core_profiles_1d.ion[var[1]].density_thermal)

        flux = array_like(x, core_profiles_1d.ion[var[1]].get("density_thermal_flux", 0))

        ion_src: CoreSources.Source.Profiles1D.Ion = self._core_sources_1d.ion[var[1]]

        ion_transp: CoreTransport.Model.Profiles1D.Ion = self._core_transport_1d.ion[var[1]]

        return self.transp_particle(x, y, flux, ym,
                                    ion_transp.particles.d(x),
                                    ion_transp.particles.v(x),
                                    ion_src.particles(x))

    def transp_ion_particle_fast(self, x: np.ndarray, core_profiles_1d: CoreProfiles.Profiles1D, var) -> Tuple[np.ndarray, np.ndarray]:

        ym = array_like(x, self._core_profiles_prev.profiles_1d.ion[var[1]].density_fast)

        y = array_like(x, core_profiles_1d.ion[var[1]].density_fast)

        flux = array_like(x, core_profiles_1d.ion[var[1]].get("density_fast_flux", 0))

        ion_src: CoreSources.Source.Profiles1D.Ion = self._core_sources_1d.ion[var[1]]

        ion_transp: TransportCoeff = self._core_transport_1d.ion[var[1]].particles

        D = ion_transp.d(x)*ion_transp.d_fast_factor(x)

        V = D*ion_transp.v_fast_factor(x)

        S = ion_src.particles_fast(x)

        return self.transp_particle(x, y, flux, ym, D, V, S)

    def bc_ion_particle_fast(self, ya: float, ga: float, yb: float, gb: float, var):
        return self.bc_particle(ya, ga, yb, gb, self.boundary_conditions_1d.ion[var[1]].particles_fast)

    def bc_ion_particle(self, ya: float, ga: float, yb: float, gb: float, var):
        return self.bc_particle(ya, ga, yb, gb, self.boundary_conditions_1d.ion[var[1]].particles)

    def transp_ion_energy(self, x: np.ndarray, core_profiles_1d: CoreProfiles.Profiles1D, var) -> Tuple[np.ndarray, np.ndarray]:
        ym = array_like(x, self._core_profiles_prev.profiles_1d.ion[var[1]].temperature)

        y = array_like(x, core_profiles_1d.ion[var[1]].temperature)

        flux = array_like(x, core_profiles_1d.ion[var[1]].get("temperature_flux", 0))

        density = array_like(x, core_profiles_1d.ion[var[1]].density)

        density_flux = array_like(x, core_profiles_1d.ion[var[1]].get("density_flux", 0))

        return self.transp_energy(x, y, flux, ym,
                                  self._core_transport_1d.ion[var[1]].energy.d(x),
                                  self._core_transport_1d.ion[var[1]].energy.v(x),
                                  self._core_sources_1d.ion[var[1]].energy(x),
                                  density,
                                  density_flux)

    def bc_ion_energy(self, ya: float, ga: float, yb: float, gb: float, var):
        return self.bc_energy(ya, ga, yb, gb, self.boundary_conditions_1d.ion[var[1]].energy)

    ###########################################################################################################################################

    def transp_particle(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        flux: np.ndarray,
                        ym: np.ndarray,
                        transp_d: np.ndarray,
                        transp_v: np.ndarray,
                        source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        hyper_diff = self._hyper_diff

        yp = Function(x, y).derivative(x)

        inv_tau = self._inv_tau

        rho_tor_boundary = self._rho_tor_boundary

        k_phi = self._k_phi

        k_rho_bdry = self._k_rho_bdry

        vpr = self._vpr(x)

        vprm = self._vprm(x)

        gm3 = self._gm3(x)

        a = vpr

        b = vprm

        c = rho_tor_boundary

        d = vpr * gm3 * transp_d / rho_tor_boundary

        e = vpr * gm3 * transp_v

        S = vpr * source

        dy = (-flux + e * y + hyper_diff * yp)/(d + hyper_diff)

        dg = S

        if not isclose(inv_tau, 0.0):
            dg = dg - (a * y - b * ym)*inv_tau + vpr * k_rho_bdry
            dg = dg + Function(x, vpr * x * k_phi).derivative(x)*y
            dg = dg + vpr * x * k_phi * dy

        dg = dg*c
        dy = array_like(x, dy)
        dg = array_like(x, dg)
        return dy, dg

    def bc_particle(self, ya: float, ga: float, yb: float, gb: float, bc: CoreTransportSolver.BoundaryConditions1D.BoundaryConditions):

        # axis
        u0, v0, w0 = 0, 1, 0

        if bc.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        else:
            raise NotImplementedError(bc.identifier)

        return float((u0 * ya + v0 * ga - w0)), float((u1 * yb + v1 * gb - w1))

    def transp_energy(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      flux: np.ndarray,
                      ym: np.ndarray,
                      heat_diff:  np.ndarray,
                      heat_vconv:  np.ndarray,
                      heat_src:  np.ndarray,
                      density: np.ndarray,
                      density_flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        hyper_diff = self._hyper_diff

        inv_tau = self._inv_tau

        rho_tor_boundary = self._rho_tor_boundary

        k_phi = self._k_phi

        k_rho_bdry = self._k_rho_bdry

        vpr = self._vpr(x)

        vprm = self._vprm(x)

        vpr5_3 = vpr**(5/3)

        vpr5_3m = vprm**(5/3)

        inv_vpr23 = self._inv_vpr23(x)

        Qimp_k_ns = self._Qimp_k_ns(x)

        gm3 = self._gm3(x)

        yp = Function(x, y).derivative(x)

        a = (3/2) * vpr5_3 * y

        b = (3/2) * vpr5_3m * ym

        c = rho_tor_boundary * inv_vpr23

        d = vpr * gm3 * density * heat_diff / rho_tor_boundary

        e = vpr * gm3 * density * heat_vconv + 3/2 * density_flux

        S = vpr5_3 * heat_src

        dy = (-flux + e * y + hyper_diff * yp)/(d + hyper_diff)

        dg = S

        if not isclose(inv_tau, 0.0):
            dg = dg - (a * y - b * ym)*inv_tau
            dg = dg + vpr5_3 * Qimp_k_ns * y
            dg = dg + Function(x,  vpr * (3/4)*k_phi * x * density).derivative(x) * y
            dg = dg + vpr * (3/4)*k_phi * x * density*dy

        dg = dg*c

        dy = array_like(x, dy)
        dg = array_like(x, dg)
        dy[0] = 0
        return dy, dg

    def bc_energy(self, ya: float, ga: float, yb: float, gb: float, bc: CoreTransportSolver.BoundaryConditions1D.BoundaryConditions,):

        # ----------------------------------------------
        # Boundary Condition
        # bc: CoreTransportSolver.BoundaryConditions1D.BoundaryConditions = \
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

    def transp_rotation(self, x: np.ndarray,  y: np.ndarray, flux: np.ndarray, **kwargs):
        r"""
            Rotation Transport
            .. math::  \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime\frac{5}{3}}\left\langle R\right\rangle \
                        m_{i}n_{i}u_{i,\varphi}\right)+\frac{\partial}{\partial\rho}\Phi_{i}=V^{\prime}\left[U_{i,\varphi,exp}-U_{i,\varphi}\cdot u_{i,\varphi}+U_{zi,\varphi}\right]
                :label: transport_rotation
        """
        logger.warning(f"TODO: Rotation Transport is not implemented!")
        return 0.0

    def quasi_neutrality_condition(self, core_profiles_1d: CoreProfiles.Profiles1D) -> None:

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

        # profiles_1d_next["conductivity_parallel"] = function_like(
        #     rho_tor_norm, core_transport_1d.conductivity_parallel(rho_tor_norm))

        # profiles_1d_next["j_total"] = function_like(
        #     rho_tor_norm, core_transport_1d.j_parallel(rho_tor_norm))

        # for idx, var_id in enumerate(var_list):
        #     profiles_1d_next[var_id] = Function(rho_tor_norm, sol.y[idx*2])
        #     profiles_1d_next[var_id[:-1]+[f"{var_id[-1]}_flux"]] = Function(rho_tor_norm, sol.y[idx*2+1])

    def _gather(self, x: np.ndarray, core_profiles_1d: CoreProfiles.Profiles1D) -> np.ndarray:
        assert(len(self._var_list) > 0)
        y_list = [
            [array_like(x, core_profiles_1d.get(Path(var), 0)),
             array_like(x, core_profiles_1d.get(Path(var[:-1]+[var[-1]+"_flux"]), 0))]
            for var, *_ in self._var_list
        ]
        return np.vstack(sum(y_list, []))

    def _dispatch(self, x: np.ndarray, y: np.ndarray):

        core_profiles = CoreProfiles({
            "profiles_1d": {
                "grid": self._radial_grid.remesh("rho_tor_norm", x),
                "psi": array_like(x, self._core_profiles_prev.profiles_1d.get("psi", None)),
                "electrons": {
                    "label": "electron",
                    "density": self._core_profiles_prev.profiles_1d.electrons.density(x),
                    "temperature": self._core_profiles_prev.profiles_1d.electrons.temperature(x),
                },
                "ion": [
                    {
                        "label": ion.label,
                        "is_impurity": ion.is_impurity,
                        "has_fast_particle": ion.has_fast_particle,
                        "density": ion.density(x),  # if ion.is_impurity else None,
                        "temperature": ion.temperature(x)  # if ion.is_impurity else None,
                    } for ion in self._core_profiles_prev.profiles_1d.ion
                ]
            }
        })

        core_profiles_1d = core_profiles.profiles_1d

        for idx, (var, *_) in enumerate(self._var_list):
            core_profiles_1d[Path(var)] = Function(x, y[idx*2])
            core_profiles_1d[Path(var[:-1]+[var[-1]+"_flux"])] = Function(x, y[idx*2+1])

        self.quasi_neutrality_condition(core_profiles.profiles_1d)

        return core_profiles

    def _dy(self, x: np.ndarray, Y: np.ndarray, p: np.ndarray = None) -> np.ndarray:

        self._core_profiles_iter = self._dispatch(x, Y)

        # logger.debug(core_profiles.profiles_1d.ion[{"label": "He"}].density_fast(x))

        # if self._enable_nonlinear:
        self._core_transport.refresh(core_profiles=self._core_profiles_iter, equilibrium=self._eq_next)
        self._core_sources.refresh(core_profiles=self._core_profiles_iter, equilibrium=self._eq_next)

        self._core_transport_1d = self._core_transport.model_combiner.profiles_1d
        self._core_sources_1d = self._core_sources.source_combiner.profiles_1d

        # equations
        dy_list = np.vstack(sum([list(eq(x, self._core_profiles_iter.profiles_1d, Path(var)))
                            for var, eq, *_ in self._var_list], []))

        return dy_list

    def _bc(self, Ya: np.ndarray, Yb: np.ndarray, p: np.ndarray = None) -> np.ndarray:
        res_list = sum([list(bc(Ya[idx*2], Ya[idx*2+1],  Yb[idx*2], Yb[idx*2+1], Path(var)))
                        for idx, (var, eq, bc) in enumerate(self._var_list)], [])
        return res_list

    def _update_context(self, /,
                        core_profiles_prev: CoreProfiles,
                        core_sources: CoreSources,
                        core_transport: CoreTransport,
                        equilibrium_prev: Equilibrium,
                        equilibrium_next: Equilibrium = None,
                        dt: float = None,
                        var_list=None,
                        **kwargs
                        ):
        """
            quasi_neutral_condition:
                = electrons : n_e= sum(n_i*z_i)
                = ion       : n_i0*z_i0=n_i1*z_i1 ...
        """

        self._parameters = collections.ChainMap(kwargs, self.get("code.parameters", {}).dump())

        self._core_sources = core_sources

        self._core_transport = core_transport

        self._core_transport_1d = self._core_transport.model_combiner.profiles_1d

        self._core_sources_1d = self._core_sources.source_combiner.profiles_1d

        self._eq_prev = equilibrium_prev

        self._eq_next = equilibrium_next if equilibrium_next is not None else self._eq_prev

        # -----------------------------------------------------------
        # Setup common variables
        #

        self._radial_grid = self._eq_next.radial_grid.remesh("rho_tor_norm")

        rho_tor_norm = self._radial_grid.rho_tor_norm

        psi_norm = self._radial_grid.psi_norm

        # geometry

        self._tau = dt if dt is not None else self._eq_next.time-self._eq_prev.time

        self._inv_tau = 0.0 if isclose(self._tau, 0.0) else 1.0/self._tau

        # $R_0$ characteristic major radius of the device   [m]
        self._R0 = self._eq_next.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        self._B0 = self._eq_next.vacuum_toroidal_field.b0

        self._B0m = self._eq_prev.vacuum_toroidal_field.b0

        # Grid
        self._rho_tor_boundary = self._eq_next.profiles_1d.rho_tor[-1]

        self._rho_tor_boundary_m = self._eq_prev.profiles_1d.rho_tor[-1]

        self._k_B = (self._B0 - self._B0m) / (self._B0 + self._B0m) * self._inv_tau * 2.0

        self._k_rho_bdry = (self._rho_tor_boundary - self._rho_tor_boundary_m) / \
            (self._rho_tor_boundary + self._rho_tor_boundary_m)*self._inv_tau*2.0

        self._k_phi = self._k_B + self._k_rho_bdry

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        self._fpol = Function(rho_tor_norm,  self._eq_next.profiles_1d.fpol(psi_norm))

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]

        self._vpr = Function(rho_tor_norm, self._eq_next.profiles_1d.dvolume_drho_tor(psi_norm))

        self._vprm = Function(rho_tor_norm,  self._eq_prev.profiles_1d.dvolume_drho_tor(psi_norm))

        self._vpr5_3 = np.abs(self._vpr)**(5/3)

        self._vpr5_3m = np.abs(self._vprm)**(5/3)

        if np.isclose(self._eq_next.profiles_1d.dvolume_drho_tor(psi_norm[0]), 0.0):
            self._inv_vpr23 = Function(
                rho_tor_norm[1:], self._eq_next.profiles_1d.dvolume_drho_tor(psi_norm[1:])**(-2/3))
        else:
            self._inv_vpr23 = Function(rho_tor_norm,   self._eq_next.profiles_1d.dvolume_drho_tor(psi_norm)**(-2/3))

        # $q$ safety factor                                 [-]
        self._qsf = Function(rho_tor_norm,   self._eq_next.profiles_1d.q(psi_norm))
        self._gm1 = Function(rho_tor_norm,   self._eq_next.profiles_1d.gm1(psi_norm))
        self._gm2 = Function(rho_tor_norm,   self._eq_next.profiles_1d.gm2(psi_norm))
        self._gm3 = Function(rho_tor_norm,   self._eq_next.profiles_1d.gm3(psi_norm))

        self._Qimp_k_ns = (3*self._k_rho_bdry - self._k_phi * self._vpr.derivative())

        ###############################################################################
        # options

        self._particle_solver: str = self._parameters.get("particle_solver", None)
        self._hyper_diff: float = self._parameters.get("hyper_diff", 1.0e-4)
        self._fusion_reaction = self._parameters.get("fusion_reaction", [])
        self._enable_impurity: bool = self._parameters.get("enable_impurity", False)
        self._enable_ion: bool = self._parameters.get("enable_ion", False) \
            or self._enable_impurity \
            or len(self._fusion_reaction) > 0

        ###############################################################################
        self._core_profiles_prev = core_profiles_prev

        # update var list

        core_profiles_1d = self._core_profiles_prev.profiles_1d

        self._var_list = []

        # self._var_list.append((["psi"],   self.transp_current, self.bc_current))

        if self._particle_solver == "electrons":

            self._var_list.append((["electrons", "density"],
                                   self.transp_electrons_particle, self.bc_electrons_particle))
            self._var_list.append((["electrons", "temperature"],
                                   self.transp_electrons_energy, self.bc_electrons_energy))

            for ion in core_profiles_1d.ion:
                if ion.is_impurity:
                    # TODO: impurity  transport
                    continue
                else:
                    self._var_list.append((["ion", Query({"label": ion.label}), "temperature"],
                                           self.transp_ion_energy, self.bc_ion_energy))
        else:
            # self._var_list.append((["electrons", "temperature"],
            #                        self.transp_electrons_energy, self.bc_electrons_energy))
            # self._var_list.append((["ion", {"label": "He"}, "density_fast"],
            #                        self.transp_ion_particle_fast, self.bc_ion_particle_fast))
            # self._var_list.append((["ion", {"label": "He"}, "density_thermal"],
            #                        self.transp_ion_particle_thermal, self.bc_ion_particle))
            for ion in core_profiles_1d.ion:
                if ion.is_impurity:
                    # TODO: impurity  transport
                    continue
                elif ion.has_fast_particle:
                    self._var_list.append((["ion", Query({"label": ion.label}), "density_fast"],
                                           self.transp_ion_particle_fast, self.bc_ion_particle_fast))
                    self._var_list.append((["ion", Query({"label": ion.label}), "density_thermal"],
                                           self.transp_ion_particle_thermal, self.bc_ion_particle))
                    # self._var_list.append((["ion", {"label": ion.label}, "temperature"],
                    #                        self.transp_ion_energy, self.bc_ion_energy))
                else:
                    self._var_list.append((["ion", Query({"label": ion.label}), "density"],
                                           self.transp_ion_particle, self.bc_ion_particle))
                    # self._var_list.append((["ion", {"label": ion.label}, "temperature"],
                    #                        self.transp_ion_energy, self.bc_ion_energy))

    def solve(self, /,
              core_profiles_prev: CoreProfiles,
              core_sources: CoreSources,
              core_transport: CoreTransport,
              equilibrium_prev: Equilibrium,
              equilibrium_next: Equilibrium = None,
              var_list=None,
              **kwargs) -> CoreProfiles:

        self._update_context(
            core_profiles_prev=core_profiles_prev,
            core_sources=core_sources,
            core_transport=core_transport,
            equilibrium_prev=equilibrium_prev,
            equilibrium_next=equilibrium_next,
            var_list=var_list,
            **kwargs)

        # --------------------------------------------------------------------------------------------
        # Solve equation group
        #
        x0 = self._radial_grid.rho_tor_norm

        if core_profiles_prev is not None:
            Y0 = self._gather(x0, core_profiles_prev.profiles_1d)

        else:
            Y0 = np.zeros([len(self._var_list), len(x0)])

        sol = solve_bvp(
            self._dy,
            self._bc,
            x0, Y0,
            bvp_rms_mask=self._parameters.get("bvp_rms_mask", []),
            tolerance=self._parameters.get("tolerance", 1.0e-3),
            max_nodes=self._parameters.get("max_nodes", 250),
            verbose=self._parameters.get("verbose", 0)
        )

        # --------------------------------------------------------------------------------------------
        # Update result
        # core_profiles_next = CoreProfiles({
        #     "profiles_1d": {
        #         "grid": self._radial_grid.remesh("rho_tor_norm", sol.x),
        #         "electrons": {
        #             "label": "electron",
        #             # "density_fast": self._core_profiles_prev.profiles_1d.electrons.get("density_fast", None),
        #             # "density_thermal": self._core_profiles_prev.profiles_1d.electrons.get("density_thermal", None),
        #             # "density": self._core_profiles_prev.profiles_1d.electrons.get("density"),
        #             # "temperature": self._core_profiles_prev.profiles_1d.electrons.get("temperature", 0),
        #         },
        #         "ion": [
        #             {
        #                 "label": ion.label,
        #                 "is_impurity": ion.is_impurity,
        #                 "has_fast_particle": ion.has_fast_particle,
        #                 "density_fast": ion.get("density_fast", None) if ion.is_impurity else None,
        #                 "density_thermal": ion.get("density_thermal", None) if ion.is_impurity else None,
        #                 "density": ion.get("density") if ion.is_impurity else None,
        #                 "temperature": ion.get("temperature", 0) if ion.is_impurity else None,
        #             } for ion in core_profiles_prev.profiles_1d.ion
        #         ]
        #     }
        # })
        # core_profiles_1d = core_profiles_next.profiles_1d

        # for idx, (var, *_) in enumerate(self._var_list):
        #     core_profiles_1d[var] = Function(sol.x, sol.y[idx*2])
        #     core_profiles_1d[var[:-1]+[var[-1]+"_flux"]] = Function(sol.x, sol.y[idx*2+1])

        if sol.success:
            core_profiles_next = self._dispatch(sol.x, sol.y)
        else:
            core_profiles_next = self._core_profiles_iter

        core_profiles_1d = core_profiles_next.profiles_1d

        # self.quasi_neutrality_condition(core_profiles_1d)

        core_profiles_1d["rms_residuals"] = Function((sol.x[:-1] + sol.x[1:])*0.5, sol.rms_residuals)

        logger.info(
            f"""Solve transport equations [{'Success' if sol.success else 'Failed'}] :
                    solver          : [{self.__class__.__name__}]
                    max residual    : {np.max(sol.rms_residuals)}
                    variable list   : {[var for var, *_ in self._var_list]}
                    enable_impurity : {self._enable_impurity}
                    enable_ion      : {self._enable_ion}
                    fusion reaction : {self._fusion_reaction}
                """)

        return core_profiles_next


__SP_EXPORT__ = CoreTransportSolverBVPNonlinear
