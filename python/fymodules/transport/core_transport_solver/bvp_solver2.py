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
from fytok.transport.CoreTransportSolver import CoreTransportSolver
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


class CoreTransportSolverBVP2(CoreTransportSolver):
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

    def transp_current(self, var_id: Sequence,  inv_tau=0, hyper_diff=1e-4, **kwargs):

        # -----------------------------------------------------------------
        # Transport
        # plasma parallel conductivity,                                 [(Ohm*m)^-1]

        def func(var_idx,
                 x: np.ndarray,  Y: np.ndarray,
                 x0: np.ndarray, Y0: np.ndarray,
                 p=None,
                 _conductivity_parallel=self._core_transport.profiles_1d.conductivity_parallel,
                 _j_parallel=self._core_sources.profiles_1d.j_parallel,
                 _inv_tau=inv_tau,
                 _hyper_diff=hyper_diff) -> Tuple[np.ndarray, np.ndarray]:

            conductivity_parallel = _conductivity_parallel(x)

            j_parallel = _j_parallel(x)

            ym = Function(x0, Y0[var_idx*2])

            y = Y[var_idx*2]

            g = Y[var_idx*2+1]

            yp = Function(x, y).derivative(x)

            a = conductivity_parallel*self._rho_tor

            b = conductivity_parallel*self._rho_tor

            c = (constants.mu_0 * self._B0 * self._rho_tor_boundary)/(self._fpol**2)

            d = self._vpr * self._gm2 / self._fpol / (self._rho_tor_boundary)/(TWOPI)

            e = 0

            S = - self._vpr * (j_parallel)/TWOPI

            dy = (-g + e * y + _hyper_diff * yp)/(d + _hyper_diff)

            dg = S

            if not np.isclose(_inv_tau, 0.0):
                C = (constants.mu_0 * self._B0 * self._k_phi) * (conductivity_parallel * self._rho_tor**2/self._fpol**2)
                dg = dg - (a * y - b * ym)*_inv_tau
                dg = dg + conductivity_parallel*self._Qimp_k_ns*y
                dg = dg + Function(x, C).derivative(x)*y + C*dy

            dg = dg*c

            dy = array_like(x, dy)
            dg = array_like(x, dg)
            return dy, dg

        # -----------------------------------------------------------
        # boundary condition, value
        bc: CoreTransportSolver.BoundaryConditions1D.BoundaryConditions = self.boundary_conditions_1d.current
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
            w1 = (self._tau*Uloop_bdry + self._core_profiles_prev["psi"][-1])*d(1.0)
        elif bc.identifier.index == 5:  # generic boundary condition y expressed as a1y' + a2y=a3;
            u1 = bc.value[1]
            v1 = bc.value[0]
            w1 = bc.value[2]
        elif bc.identifier.index == 6:  # equation not solved;
            raise NotImplementedError(bc.identifier.index)
        else:
            raise NotImplementedError(bc)

        def bc_func(var_idx, Ya: np.ndarray, Yb: np.ndarray,  p: Any = None, /, _bc: Sequence = (u0, v0, w0, u1, v1, w1)) -> Tuple[float, float]:
            u0, v0, w0,  u1, v1, w1 = _bc
            return float((u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0)), float((u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1))

        return func, bc_func

    def transp_particle(self,  var_id: Sequence,  inv_tau=0,  hyper_diff=1e-4, ** kwargs):

        def func(var_idx,
                 x: np.ndarray,  Y: np.ndarray,
                 x0: np.ndarray, Y0: np.ndarray,
                 p=None,
                 ) -> Tuple[np.ndarray, np.ndarray]:

            _transp: Union[CoreTransport.Model.Profiles1D.Ion,
                           CoreTransport.Model.Profiles1D.Electrons] = self._core_transport.profiles_1d.fetch(var_id)
            _source: Union[CoreSources.Source.Profiles1D.Ion,
                           CoreSources.Source.Profiles1D.Electrons] = self._core_sources.profiles_1d.fetch(var_id)

            ym = Function(x0, Y0[var_idx*2])

            y = Y[var_idx*2]
            g = Y[var_idx*2+1]

            yp = Function(x, y).derivative(x)

            a = self._vpr
            b = self._vprm
            c = self._rho_tor_boundary
            d = self._vpr * self._gm3 * _transp.particles.d / self._rho_tor_boundary
            e = self._vpr * self._gm3 * _transp.particles.v
            S = self._vpr * _source.particles

            dy = (-g + e * y + hyper_diff * yp)/(d + hyper_diff)

            dg = S

            if not np.isclose(inv_tau, 0.0):
                dg = dg - (a * y - b * ym)*inv_tau + self._vpr * self._k_rho_bdry
                dg = dg + Function(x, self._vpr * x * self._k_phi).derivative(x)*y
                dg = dg + self._vpr * x * self._k_phi * dy

            dg = dg*c
            dy = array_like(x, dy)
            dg = array_like(x, dg)
            return dy, dg

        # -----------------------------------------------------------
        # boundary condition, value

        bc: CoreTransportSolver.BoundaryConditions1D.BoundaryConditions = \
            self.boundary_conditions_1d.fetch(var_id).particles
        # axis
        u0, v0, w0 = 0, 1, 0

        if bc.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        else:
            raise NotImplementedError(bc.identifier)

        def bc_func(var_idx, Ya: np.ndarray, Yb: np.ndarray, p: Any = None, /, _bc: Sequence = (u0, v0, w0, u1, v1, w1)) -> Tuple[float, float]:
            u0, v0, w0,  u1, v1, w1 = _bc
            return float(u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0), float(u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1)

        return func, bc_func

    def transp_energy(self, var_id: Sequence, inv_tau=0,  hyper_diff=1e-4, **kwargs):

        def func(var_idx,
                 x: np.ndarray,  Y: np.ndarray,
                 x0: np.ndarray, Y0: np.ndarray,
                 p=None,
                 ) -> Tuple[np.ndarray, np.ndarray]:

            _transp: Union[CoreTransport.Model.Profiles1D.Ion,
                           CoreTransport.Model.Profiles1D.Electrons] = self._core_transport.profiles_1d.fetch(var_id)

            _source: Union[CoreSources.Source.Profiles1D.Ion,
                           CoreSources.Source.Profiles1D.Electrons] = self._core_sources.profiles_1d.fetch(var_id)
            logger.debug(var_id+["density"])
            _density = self._core_profiles_next.get(var_id+["density"])

            _density_flux = self._core_profiles_next.get(var_id+["density_flux"])

            ym = Function(x0, Y0[var_idx*2])
            y = Y[var_idx*2]
            g = Y[var_idx*2+1]

            n = _density(x)

            gamma = _density_flux(x)

            yp = Function(x, y).derivative(x)

            a = (3/2) * self._vpr5_3 * y

            b = (3/2) * self._vpr5_3m * ym

            c = self._rho_tor_boundary * self._inv_vpr23

            d = self._vpr * self._gm3 * n * _transp.energy.d / self._rho_tor_boundary

            e = self._vpr * self._gm3 * n * _transp.energy.v + 3/2 * gamma

            S = self._vpr5_3 * _source.energy

            dy = (-g + e * y + hyper_diff * yp)/(d + hyper_diff)

            dg = S

            if not np.isclose(inv_tau, 0.0):
                dg = dg - (a * y - b * ym)*inv_tau
                dg = dg + self._vpr5_3 * self._Qimp_k_ns * y
                dg = dg + Function(x,  self._vpr * (3/4)*self._k_phi * x * n).derivative(x) * y
                dg = dg + self._vpr * (3/4)*self._k_phi * x * n*dy

            dg = dg*c

            dy = array_like(x, dy)
            dg = array_like(x, dg)
            dy[0] = 0
            return dy, dg

        # ----------------------------------------------
        # Boundary Condition
        bc: CoreTransportSolver.BoundaryConditions1D.BoundaryConditions = self.boundary_conditions_1d.fetch(
            var_id).energy

        # axis
        u0, v0, w0 = 0, 1, 0

        if bc.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        else:
            raise NotImplementedError(bc)

        def bc_func(var_idx, Ya: np.ndarray, Yb: np.ndarray, p: Any = None, /, _bc: Sequence = (u0, v0, w0, u1, v1, w1)):
            u0, v0, w0, u1, v1, w1 = _bc
            return float(u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0), float(u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1)

        return func, bc_func

        # return path+["temperature"], (a, b, c, d, e, S), (u0, v0, w0,  u1, v1, w1)

    def transp_rotation(self, var_idx: int, var_id: Sequence, x0: np.ndarray, Y0: np.ndarray,
                        inv_tau=None, hyper_diff=1e-4, **kwargs):
        r"""
            Rotation Transport
            .. math::  \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime\frac{5}{3}}\left\langle R\right\rangle \
                        m_{i}n_{i}u_{i,\varphi}\right)+\frac{\partial}{\partial\rho}\Phi_{i}=V^{\prime}\left[U_{i,\varphi,exp}-U_{i,\varphi}\cdot u_{i,\varphi}+U_{zi,\varphi}\right]
                :label: transport_rotation
        """
        logger.warning(f"TODO: Rotation Transport is not implemented!")
        return 0.0

    def quasi_neutral_condition(self, profiles: CoreProfiles.Profiles1D, quasi_neutral="electrons"):
        x = profiles.grid.rho_tor_norm
        n_imp = 0  # impurity
        Z_total = 0
        for ion in profiles.ion:
            if ion.is_impurity:
                n_imp = n_imp + ion.z*ion.density(x)
            else:
                Z_total += ion.z

        if quasi_neutral == "electrons":
            ni = profiles.electrons.density(x) - n_imp
            gi = profiles.electrons.get("density_flux", 0)
            for ion in profiles.ion:
                if ion.is_impurity:
                    continue
                profiles.ion[{"label": ion.label}, "density"] = ion.z/Z_total*ni
                profiles.ion[{"label": ion.label}, "density_flux"] = ion.z/Z_total*gi

        else:
            n_e = 0
            g_e = 0
            for ion in profiles.ion:
                n_e = n_e + ion.z*ion.density(x)
                g_e = g_e + ion.z*ion.get("density_flux", 0)

            profiles.electrons["density"] = n_e
            profiles.electrons["density_flux"] = g_e

    def _solve_equations(self, x0: np.ndarray, Y0: np.ndarray, eq_grp: Sequence, /,
                         quasi_neutral="electrons",
                         tolerance=1.0e-3, max_nodes=250, **kwargs) -> BVPResult:

        eq_list = [equ(idx, var_id, x0, Y0, ** (_args[0] if len(_args) > 0 else {}))
                   for idx, (var_id, equ,  *_args) in enumerate(eq_grp)]

        var_list = [var_id for var_id, eq,  *_args in eq_grp]

        def func(x: np.ndarray, Y: np.ndarray, p=None, /,
                 eq_list: Sequence[Tuple[Callable, Callable]] = eq_list) -> np.ndarray:

            # core_profiles_next.refresh(radial_grid=self._equilibrium_next.radial_grid.remesh("rho_tor_norm", x))

            profiles_1d: CoreProfiles = core_profiles_next.profiles_1d

            for idx, var_id in enumerate(var_list):
                profiles_1d[var_id] = Function(x, Y[idx*2])
                profiles_1d[var_id[:-1]+[f"{var_id[-1]}_flux"]] = Function(x, Y[idx*2+1])

            self.quasi_neutral_condition(profiles_1d, quasi_neutral=quasi_neutral)

            self._core_transport.refresh(equlibrium=self._equilibrium_next,  core_profiles=self._core_profiles_next)

            self._core_sources.refresh(equlibrium=self._equilibrium_next,  core_profiles=self._core_profiles_next)

            res = sum([list(func(x, Y, profiles_1d,
                                 self._core_transport.profiles_1d,
                                 self._core_sources.profiles_1d)) for func, bc in eq_list], [])

            return np.vstack([array_like(x, d) for d in res])

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p=None, /, eq_list: Sequence[Tuple[Callable, Callable]] = eq_list) -> np.ndarray:
            return np.asarray(sum([list(bc(Ya, Yb, p)) for func, bc in eq_list], []))

        return solve_bvp(func, bc_func, x0, Y0, tolerance=tolerance, max_nodes=max_nodes, **kwargs)

    def refresh(self, *args,  **kwargs):
        super().refresh(*args, **kwargs)

        r_grid = self._core_profiles_next.profiles_1d.grid

        self._tau = self._core_profiles_next.time - self._core_profiles_prev.time

        self._inv_tau = 0 if abs(self._tau) < EPSILON else 1.0/self._tau

        # $R_0$ characteristic major radius of the device   [m]
        self._R0 = self._equilibrium_next.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        self._B0 = self._equilibrium_next.vacuum_toroidal_field.b0

        self._B0m = self._equilibrium_prev.vacuum_toroidal_field.b0
        # $rho_tor_{norm}$ normalized minor radius                [-]
        self._rho_tor_norm = r_grid.rho_tor_norm

        self._psi_norm = r_grid.psi_norm

        # Grid
        # $rho_tor$ not  normalized minor radius                [m]
        self._rho_tor = Function(self._rho_tor_norm, r_grid.rho_tor)

        self._rho_tor_boundary = r_grid.rho_tor_boundary

        self._rho_tor_boundary_m = self._core_profiles_prev.profiles_1d.grid.rho_tor_boundary

        self._k_B = (self._B0 - self._B0m) / (self._B0 + self._B0m) * self._inv_tau * 2.0

        self._k_rho_bdry = (self._rho_tor_boundary - self._rho_tor_boundary_m) / \
            (self._rho_tor_boundary + self._rho_tor_boundary_m)*self._inv_tau*2.0

        self._k_phi = self._k_B + self._k_rho_bdry

        # -----------------------------------------------------------
        # Equilibrium
        eq_profile1d = self._equilibrium_next.profiles_1d

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        self._fpol = Function(self._rho_tor_norm, eq_profile1d.fpol(self._psi_norm))

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]

        self._vpr = Function(self._rho_tor_norm, eq_profile1d.dvolume_drho_tor(self._psi_norm))

        self._vprm = Function(self._rho_tor_norm,
                              self._equilibrium_prev.profiles_1d.dvolume_drho_tor(self._psi_norm))

        self._vpr5_3 = np.abs(self._vpr)**(5/3)
        self._vpr5_3m = np.abs(self._vprm)**(5/3)

        if np.isclose(eq_profile1d.dvolume_drho_tor(self._psi_norm[0]), 0.0):
            self._inv_vpr23 = Function(self._rho_tor_norm[1:],
                                       eq_profile1d.dvolume_drho_tor(self._psi_norm[1:])**(-2/3))
        else:
            self._inv_vpr23 = Function(self._rho_tor_norm,  eq_profile1d.dvolume_drho_tor(self._psi_norm)**(-2/3))

        # $q$ safety factor                                 [-]
        self._qsf = Function(self._rho_tor_norm,  eq_profile1d.q(self._psi_norm))
        self._gm1 = Function(self._rho_tor_norm,  eq_profile1d.gm1(self._psi_norm))
        self._gm2 = Function(self._rho_tor_norm,  eq_profile1d.gm2(self._psi_norm))
        self._gm3 = Function(self._rho_tor_norm,  eq_profile1d.gm3(self._psi_norm))

        self._Qimp_k_ns = (3*self._k_rho_bdry - self._k_phi * self._vpr.derivative())

    def solve(self,  /,   quasi_neutral: str = "electrons", tolerance=1.0e-3, max_nodes=250, **kwargs) -> Tuple[float, CoreProfiles]:

        residual, self._core_profiles_next = super().solve(**kwargs)
        self.refresh(**kwargs)
        profiles_1d = self._core_profiles_next.profiles_1d

        if quasi_neutral == "electrons":
            eq_grp = [
                (["psi"],                                               *self.transp_current(["psi"]),),
                (["electrons", "density"],                              *self.transp_particle(["electrons"]),),
                (["electrons", "temperature"],                          *self.transp_energy(["electrons"]), ),
                *[(["ion", {"label": ion.label}, "temperature"],        *self.transp_energy(["ion", {"label": ion.label}]), )
                  for ion in profiles_1d.ion if not ion.is_impurity],
            ]
        else:
            eq_grp = [
                (["psi"],                                               *self.transp_current(),),
                (["electrons", "temperature"],                          *self.transp_energy(["electrons"]),),
                *[(["ion", {"label": ion.label}, "density"],            *self.transp_particle(["ion", {"label": ion.label}]),)
                  for ion in profiles_1d.ion if not ion.is_impurity],
                *[(["ion", {"label": ion.label}, "temperature"],        *self.transp_energy(["ion", {"label": ion.label}]), )
                  for ion in profiles_1d.ion if not ion.is_impurity],
            ]

        x0 = profiles_1d.grid.rho_tor_norm

        Y0 = np.vstack(sum([[array_like(x0, self._core_profiles_prev.get(var_id, 0)), np.zeros_like(x0)]
                            for var_id, *_ in eq_grp], []))

        def func(x: np.ndarray, Y: np.ndarray, p=None, /,
                 _eq_grp: Sequence[Tuple[Callable, Callable]] = eq_grp) -> np.ndarray:

            for idx, (var_id, *_) in enumerate(_eq_grp):
                profiles_1d[var_id] = Function(x, Y[idx*2])
                profiles_1d[var_id[:-1]+[f"{var_id[-1]}_flux"]] = Function(x, Y[idx*2+1])

            self.quasi_neutral_condition(profiles_1d, quasi_neutral=quasi_neutral)

            # self._core_transport.refresh(equlibrium=equilibrium_next,  core_profiles=core_profiles_next)
            # self._core_sources.refresh(equlibrium=equilibrium_next,  core_profiles=core_profiles_next)

            res = sum([list(fc(idx, x, Y, x0, Y0, p)) for idx, (var_id,  fc, _) in enumerate(_eq_grp)], [])

            return np.vstack([array_like(x, d) for d in res])

        
        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p=None, /,
                    _eq_grp: Sequence[Tuple[Callable, Callable]] = eq_grp) -> np.ndarray:
            
            return np.asarray(sum([list(bc(idx, Ya, Yb, p)) for idx, (var_id,  fc,  bc) in enumerate(_eq_grp)], []))

        sol = solve_bvp(func, bc_func, x0, Y0, tolerance=tolerance, max_nodes=max_nodes, **kwargs)

        residual += np.max(sol.rms_residuals)

        for idx, var_id, * _ in enumerate(eq_grp):
            profiles_1d[var_id] = Function(sol.x, sol.y[idx*2])
            profiles_1d[var_id[:-1]+[f"{var_id[-1]}_flux"]] = Function(sol.x, sol.y[idx*2+1])

        logger.debug(self._core_profiles_next.profiles_1d.electrons.density)

        profiles_1d["rms_residuals"] = Function((sol.x[: -1]+sol.x[1:])*0.5, sol.rms_residuals)

        logger.info(
            f"Solve transport equations [{'Success' if sol.success else 'Failed'}] : max reduisal={residual} \n  {[var_id for var_id,*_ in eq_grp]}")

        return residual, self._core_profiles_next


__SP_EXPORT__ = CoreTransportSolverBVP2
