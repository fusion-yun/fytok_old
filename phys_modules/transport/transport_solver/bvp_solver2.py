"""

"""

import collections
import enum
from itertools import chain
from math import log
from typing import Callable, Iterator, Mapping, Optional, Type, Union, Sequence, Any, Tuple

from matplotlib.pyplot import loglog

from fytok.common.IDS import IDS
from fytok.common.Misc import Identifier, VacuumToroidalField
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport
from fytok.transport.Equilibrium import Equilibrium
from fytok.transport.MagneticCoordSystem import RadialGrid
from fytok.transport.TransportSolver import TransportSolver
from numpy.core.fromnumeric import var
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, _not_found_, sp_property
from spdm.numlib import constants, np
from spdm.numlib.bvp import solve_bvp, BVPResult
from spdm.numlib.misc import array_like
from spdm.util.logger import logger
from spdm.util.utilities import convert_to_named_tuple

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class TransportSolverBVP2(TransportSolver):
    r"""
        Solve transport equations :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
        See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

        self._tau = self._core_profiles.time - self._core_profiles.previous_state.time

        self._core_profiles_next = self._core_profiles.profiles_1d
        self._core_profiles_prev = self._core_profiles.previous_state.profiles_1d
        self._c_transp = self._core_transport.model
        self._c_source = self._core_sources.source
        self._eq = self._equilibrium.time_slice.profiles_1d

        self._inv_tau = 0 if abs(self._tau) < EPSILON else 1.0/self._tau

        # $R_0$ characteristic major radius of the device   [m]
        self._R0 = self._equilibrium.time_slice.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        self._B0 = self._equilibrium.time_slice.vacuum_toroidal_field.b0

        self._B0m = self._equilibrium.previous_state.time_slice.vacuum_toroidal_field.b0
        # $rho_tor_{norm}$ normalized minor radius                [-]
        self._rho_tor_norm = self._core_profiles_next.grid.rho_tor_norm

        self._psi_norm = self._core_profiles_next.grid.psi_norm

        # Grid
        # $rho_tor$ not  normalized minor radius                [m]
        self._rho_tor = Function(self._rho_tor_norm, self._core_profiles_next.grid.rho_tor)

        self._rho_tor_boundary = self._core_profiles_next.grid.rho_tor[-1]

        self._rho_tor_boundary_m = self._core_profiles_prev.grid.rho_tor[-1]

        self._k_B = (self._B0 - self._B0m) / (self._B0 + self._B0m) * self._inv_tau * 2.0

        self._k_rho_bdry = (self._rho_tor_boundary - self._rho_tor_boundary_m) / \
            (self._rho_tor_boundary + self._rho_tor_boundary_m)*self._inv_tau*2.0

        self._k_phi = self._k_B + self._k_rho_bdry

        # -----------------------------------------------------------
        # Equilibrium
        # diamagnetic function,$F=R B_\phi$                 [T*m]
        self._fpol = Function(self._rho_tor_norm, self._eq.fpol(self._psi_norm))

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        self._vpr = Function(self._rho_tor_norm, self._eq.dvolume_drho_tor(self._psi_norm))

        self._vprm = Function(self._rho_tor_norm,
                              self._equilibrium.previous_state.time_slice.profiles_1d.dvolume_drho_tor(self._psi_norm))

        self._vpr35 = self._vpr**(5/3)
        self._vpr35m = self._vprm**(5/3)

        if np.isclose(self._eq.dvolume_drho_tor(self._psi_norm[0]), 0.0):
            self._inv_vpr23 = Function(self._rho_tor_norm[1:], self._eq.dvolume_drho_tor(self._psi_norm[1:])**(-2/3))
        else:
            self._inv_vpr23 = Function(self._rho_tor_norm, self._eq.dvolume_drho_tor(self._psi_norm)**(-2/3))

        # $q$ safety factor                                 [-]
        self._qsf = Function(self._rho_tor_norm, self._eq.q(self._psi_norm))
        self._gm1 = Function(self._rho_tor_norm, self._eq.gm1(self._psi_norm))
        self._gm2 = Function(self._rho_tor_norm, self._eq.gm2(self._psi_norm))
        self._gm3 = Function(self._rho_tor_norm, self._eq.gm3(self._psi_norm))

        self._Qimp_k_ns = (3*self._k_rho_bdry - self._k_phi * self._vpr.derivative())

    def f_current(self,  var_idx: int, var_id: Sequence, x0: np.ndarray, Y0: np.ndarray,
                  inv_tau=0, hyper_diff=1e-4, **kwargs):

        # -----------------------------------------------------------------
        # Transport
        # plasma parallel conductivity,                                 [(Ohm*m)^-1]
        # j_exp = self._c_source.j_parallel + self._c_source.get("j_decomposed.explicit_part", Function(0))
        # j_imp = self._c_source.get("j_decomposed.implicit_part", Function(0))

        ym = Function(x0, Y0[var_idx*2])

        def func(x: np.ndarray,  Y: np.ndarray,
                 transp: CoreTransport.Model.Profiles1D,
                 source: CoreSources.Source.Profiles1D,) -> Tuple[np.ndarray, np.ndarray]:

            conductivity_parallel = transp.conductivity_parallel

            y = Y[var_idx*2]
            g = Y[var_idx*2+1]
            yp = Function(x, y).derivative(x)

            a = conductivity_parallel*self._rho_tor

            b = conductivity_parallel*self._rho_tor

            c = (constants.mu_0 * self._B0 * self._rho_tor_boundary)/(self._fpol**2)

            d = self._vpr * self._gm2 / self._fpol / (self._rho_tor_boundary)/(TWOPI)

            e = (- constants.mu_0 * self._B0 * self._k_phi) * \
                (conductivity_parallel * self._rho_tor**2/self._fpol**2)

            S = -self._vpr * (source.j_parallel)/TWOPI

            dy = (-g + e * y + hyper_diff * yp)/(d + hyper_diff)

            dg = (S - (a * y - b * ym)*inv_tau)*c

            return array_like(x, dy), array_like(x, dg)

        # -----------------------------------------------------------
        # boundary condition, value
        bc: TransportSolver.BoundaryConditions1D.BoundaryConditions = self.boundary_conditions_1d.current
        # axis
        u0, v0, w0 = 0, 1, 0

        # Identifier of the boundary condition type.
        #   ID =    1: poloidal flux;
        #           2: ip;
        #           3: loop voltage;
        #           4: undefined;
        #           5: generic boundary condition y expressed as a1y'+a2y=a3.
        #           6: equation not solved; [eV]

        if bc.identifier.index == 1:  # poloidal flux;
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        elif bc.identifier.index == 2:  # ip, total current inside x=1
            Ip = bc.value[0]
            u1 = 0
            v1 = 1
            w1 = constants.mu_0 * Ip / self._fpol[-1]
        elif bc.identifier.index == 3:  # loop voltage;
            Uloop_bdry = bc.value[0]
            u1 = 0
            v1 = 1
            w1 = (self._tau*Uloop_bdry + self._core_profiles_prev.psi[-1])*d(1.0)
        elif bc.identifier.index == 5:  # generic boundary condition y expressed as a1y' + a2y=a3;
            u1 = bc.value[1]
            v1 = bc.value[0]
            w1 = bc.value[2]
        elif bc.identifier.index == 6:  # equation not solved;
            raise NotImplementedError(bc.identifier.index)
        else:
            raise NotImplementedError(bc)

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p: Any = None, /, _bc: Sequence = (u0, v0, w0, u1, v1, w1)) -> Tuple[float, float]:
            u0, v0, w0,  u1, v1, w1 = _bc
            return float((u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0)), float((u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1))

        return func, bc_func

    def f_particle(self, var_idx: int, var_id: Sequence, x0: np.ndarray, Y0: np.ndarray,
                   inv_tau=0,  hyper_diff=1e-4, ** kwargs):

        ym = Function(x0, Y0[var_idx*2])

        def func(x: np.ndarray,  Y: np.ndarray,
                 transp: CoreTransport.Model.Profiles1D,
                 source: CoreSources.Source.Profiles1D,) -> Tuple[np.ndarray, np.ndarray]:
            _transp: Union[CoreTransport.Model.Profiles1D.Ion,
                           CoreTransport.Model.Profiles1D.Electrons] = transp.fetch(var_id[:-1], NotImplemented)

            _source: Union[CoreSources.Source.Profiles1D.Ion,
                           CoreSources.Source.Profiles1D.Electrons] = source.fetch(var_id[:-1], NotImplemented)
            y = Y[var_idx*2]
            g = Y[var_idx*2+1]
            yp = Function(x, y).derivative(x)

            a = self._vpr
            b = self._vprm
            c = self._rho_tor_boundary
            d = self._vpr * self._gm3 * _transp.particles.d / self._rho_tor_boundary
            e = self._vpr * self._gm3 * _transp.particles.v - self._vpr * self._rho_tor * self._k_phi
            S = self._vpr * (_source.particles + self._k_rho_bdry)

            dy = (-g + e * y + hyper_diff * yp)/(d + hyper_diff)

            dg = (S - (a * y - b * ym)*inv_tau)*c

            return array_like(x, dy), array_like(x, dg)

        # -----------------------------------------------------------
        # boundary condition, value
        bc: TransportSolver.BoundaryConditions1D.BoundaryConditions =  \
            self.boundary_conditions_1d.fetch(var_id[:-1], NotImplemented).particles

        # axis
        u0, v0, w0 = 0, 1, 0

        if bc.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        else:
            raise NotImplementedError(bc.identifier)

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p: Any = None, /, _bc: Sequence = (u0, v0, w0, u1, v1, w1)) -> Tuple[float, float]:
            u0, v0, w0,  u1, v1, w1 = _bc
            return float(u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0), float(u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1)

        return func, bc_func

    def f_energy(self, var_idx: int, var_id: Sequence, x0: np.ndarray, Y0: np.ndarray,
                 density: Union[Callable, int],
                 inv_tau=0,  hyper_diff=1e-4, **kwargs):

        def _density(x, Y):
            if isinstance(density, int):
                ns = Y[density*2]
                gs = Y[density*2+1]
            elif callable(density):
                ns, gs = density(x, Y)

            if isinstance(x, np.ndarray) and isinstance(x, np.ndarray) and x.shape == ns.shape:
                return ns, gs
            elif isinstance(ns, np.ndarray) and isinstance(x, int):
                return ns[x], gs[x]
            else:
                return ns, gs

                # energy transport

        ym = Function(x0, Y0[var_idx*2])

        def func(x: np.ndarray,  Y: np.ndarray,
                 transp: CoreTransport.Model.Profiles1D,
                 source: CoreSources.Source.Profiles1D,) -> Tuple[np.ndarray, np.ndarray]:

            _transp: Union[CoreTransport.Model.Profiles1D.Ion,
                           CoreTransport.Model.Profiles1D.Electrons] = transp.fetch(var_id[:-1], NotImplemented)

            _source: Union[CoreSources.Source.Profiles1D.Ion,
                           CoreSources.Source.Profiles1D.Electrons] = source.fetch(var_id[: -1], NotImplemented)

            y = Y[var_idx*2]
            g = Y[var_idx*2+1]

            n, gamma = _density(x, Y)

            yp = Function(x, y).derivative(x)
            a = (3/2) * self._vpr35 * y
            b = (3/2) * self._vpr35m * ym
            c = self._rho_tor_boundary * self._inv_vpr23

            d = self._vpr * self._gm3 * n * _transp.energy.d / self._rho_tor_boundary - \
                self._vpr * (3/4)*self._k_phi * self._rho_tor * ym

            e = self._vpr * self._gm3 * n * _transp.energy.v + 3/2*gamma

            S = self._vpr35 * (_source.energy + self._Qimp_k_ns * y)

            dy = (-g + e * y + hyper_diff * yp)/(d + hyper_diff)

            dg = (S - (a * y - b * ym)*inv_tau)*c

            return array_like(x, dy), array_like(x, dg)

        # ----------------------------------------------
        # Boundary Condition
        bc: TransportSolver.BoundaryConditions1D.BoundaryConditions = self.boundary_conditions_1d.fetch(
            var_id[: -1], NotImplemented).energy

        # axis
        u0, v0, w0 = 0, 1, 0

        if bc.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        else:
            raise NotImplementedError(bc)

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p: Any = None, /, _bc: Sequence = (u0, v0, w0, u1, v1, w1)):
            u0, v0, w0, u1, v1, w1 = _bc
            return float(u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0), float(u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1)

        return func, bc_func

        # return path+["temperature"], (a, b, c, d, e, S), (u0, v0, w0,  u1, v1, w1)

    def rotation_transport(self, var_idx: int, var_id: Sequence, x0: np.ndarray, Y0: np.ndarray, inv_tau=None, hyper_diff=1e-4, **kwargs):
        r"""
            Rotation Transport
            .. math::  \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime\frac{5}{3}}\left\langle R\right\rangle \
                        m_{i}n_{i}u_{i,\varphi}\right)+\frac{\partial}{\partial\rho}\Phi_{i}=V^{\prime}\left[U_{i,\varphi,exp}-U_{i,\varphi}\cdot u_{i,\varphi}+U_{zi,\varphi}\right]
                :label: transport_rotation
        """
        logger.warning(f"TODO: Rotation Transport is not implemented!")
        return 0.0

    def quasi_neutral_condition_electron(self, var_id: Sequence, /,  ion_species=[],   ion_index: Iterator[int] = [], impurities: Sequence[Callable] = []):
        n_impurity, g_impurity = impurities

        def func(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return \
                np.sum([self._core_profiles_next["ion", {"label": ion_species[i]}].z*Y[ion_idx*2]
                        for i, ion_idx in enumerate(ion_index)])+n_impurity(x), \
                np.sum([self._core_profiles_next["ion", {"label": ion_species[i]}].z*Y[ion_idx*2+1]
                        for i, ion_idx in enumerate(ion_index)])+g_impurity(x)
        return func

    def quasi_neutral_condition_ion(self, var_id: Sequence, /,  ion_species=[],   electron_index: int = 1, impurities=[]):
        factor = self._core_profiles_next[var_id].z / \
            np.sum([self._core_profiles_next["ion", {"label": label}].z for label in ion_species])
        n_impurity, g_impurity = impurities

        def func(x: np.ndarray, Y: np.ndarray):
            return factor*(Y[electron_index*2]-array_like(x, n_impurity)),  factor * (Y[electron_index*2+1]-array_like(x, g_impurity))

        return func

    def _solve_equations(self, x0: np.ndarray, Y0: np.ndarray, eq_grp: Sequence, /,
                         tolerance=1.0e-3, max_nodes=250, **kwargs) -> BVPResult:

        eq_list = [eq(idx, var_id, x0, Y0, ** (_args[0] if len(_args) > 0 else {}))
                   for idx, (var_id, eq,  *_args) in enumerate(eq_grp)]

        var_list = [var_id for var_id, eq,  *_args in eq_grp]

        def func(x: np.ndarray, Y: np.ndarray, p=None, /, eq_list: Sequence[Tuple[Callable, Callable]] = eq_list) -> np.ndarray:
            core_profiles = self._convert_to_core_profiles(x, Y, var_list)
            self._c_transp.update(core_profiles=core_profiles)
            self._c_source.update(core_profiles=core_profiles)
            transp = self._c_transp.combine.profiles_1d
            source = self._c_source.combine.profiles_1d
            return np.vstack([array_like(x, d) for d in sum([list(func(x, Y, transp, source)) for func, bc in eq_list], [])])

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p=None, /, eq_list: Sequence[Tuple[Callable, Callable]] = eq_list) -> np.ndarray:
            return np.asarray(sum([list(bc(Ya, Yb, p)) for func, bc in eq_list], []))

        return solve_bvp(func, bc_func, x0, Y0, tolerance=tolerance, max_nodes=max_nodes, **kwargs)

    def _convert_to_core_profiles(self, x: np.ndarray, Y: np.ndarray, var_list=[]) -> CoreProfiles:
        new_grid = self._core_profiles_next.grid.remesh(x, "rho_tor_norm")
        core_profiles = CoreProfiles(grid=new_grid)

        profiles = core_profiles.profiles_1d

        for idx, path in enumerate(var_list):
            profiles[path] = Y[idx*2]
            profiles[path[:-1]+[f"{path[-1]}_flux"]] = Y[idx*2+1]

        return core_profiles

    def solve_core(self,  /,
                   enable_ion_particle_solver: bool = False,
                   ion_species: Sequence = None,
                   impurities: Sequence = [],

                   **kwargs) -> float:
        r"""
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

        if ion_species is None:
            ion_species = [ion.label for ion in self._core_profiles_prev.ion if ion.label not in impurities]
        # elif isinstance(ion_species, collections.abc.Sequence):
        #     ion_species_list = [["ion", {"label": label}] for label in ion_species if label not in impurities]

        # impurities_list = [["ion", {"label": label}] for label in impurities]

        if self._core_profiles_next.get("psi", None) is None:
            self._core_profiles_next["psi"] = self._core_profiles_next.grid.psi
        n_impurity = Function(0)
        g_impurity = Function(0)
        for label in impurities:
            ion = self._core_profiles_next["ion", {"label": label}]
            n_impurity = n_impurity + ion.z * ion.density
            g_impurity = g_impurity + ion.z * ion.get("density_flux", 0)

        if enable_ion_particle_solver is True:
            eq_grp = [
                (["psi"],                                        self.f_current,),
                (["electrons", "temperature"],                   self.f_energy,
                 {"density": self.quasi_neutral_condition_electron(["electrons"],
                                                                   ion_species=ion_species,
                                                                   impurities=(n_impurity, g_impurity),
                                                                   ion_index=range(2, 2+len(ion_species)))}),
                *[(["ion", {"label": label}, "density"],         self.f_particle,) for label in ion_species],
                *[(["ion", {"label": label}, "temperature"],     self.f_energy, {"density": 2+ion_idx})
                  for ion_idx, label in enumerate(ion_species)],
            ]
        else:
            eq_grp = [
                (["psi"],                                        self.f_current,),
                (["electrons", "density"],                       self.f_particle,),
                (["electrons", "temperature"],                   self.f_energy, {"density": 1}),
                *[(["ion", {"label": label}, "temperature"],     self.f_energy,
                   {"density": self.quasi_neutral_condition_ion(["ion", {"label": label}],
                                                                ion_species=ion_species,
                                                                impurities=(n_impurity, g_impurity),
                                                                electron_index=1)}) for ion_idx, label in enumerate(ion_species)],
            ]

        x = self._rho_tor_norm

        Y = np.vstack(sum([[array_like(x, self._core_profiles_prev.fetch(var_id, 0)), np.zeros_like(x)]
                           for var_id, *_ in eq_grp], []))

        sol = self._solve_equations(x, Y, eq_grp, **kwargs)

        rms_residuals = np.max(sol.rms_residuals)

        for idx, (var_id, * _) in enumerate(eq_grp):
            self._core_profiles_next[var_id] = Function(sol.x, sol.y[idx*2])

        self._core_profiles_next["rms_residuals"] = Function((sol.x[: -1]+sol.x[1:])*0.5, sol.rms_residuals)

        logger.info(
            f"Solve transport equations [{'Success' if sol.success else 'Failed'}] : max reduisal={rms_residuals} \n  {[var_id for var_id,*_ in eq_grp]}")

        return rms_residuals


__SP_EXPORT__ = TransportSolverBVP2
