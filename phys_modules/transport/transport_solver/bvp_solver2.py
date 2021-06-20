"""

"""

import collections
import enum
from itertools import chain
from math import log
from typing import Callable, Mapping, Optional, Type, Union, Sequence, Any

from matplotlib.pyplot import loglog

from fytok.common.IDS import IDS
from fytok.common.Misc import Identifier
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
        self._c_transp = self._core_transport.model.combine.profiles_1d
        self._c_source = self._core_sources.source.combine.profiles_1d
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

    def _create_transport_eq(self,
                             coeff: Sequence = [],  # (i,x,y,gamma)
                             hyper_diff=0.0001,
                             **kwargs
                             ) -> BVPResult:

        def func(i: int, x: np.ndarray, y: np.ndarray, g: np.ndarray, /,
                 _coeff: Sequence[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = coeff,
                 _hyper_diff=hyper_diff):

            a, b, c, d, e, S, ym, h = _coeff

            yp = Function(x, y).derivative(x)

            dy = (-g + e(x) * y + _hyper_diff * yp)/(d(x) + _hyper_diff)

            dg = S(y)

            if h is not None and ym is not None:
                dg = dg - (a(x)*y - b(x)*ym(x))/h

            if c is not None:
                dg = dg*c

            return array_like(x, dy), array_like(x, dg)

        return func

    def f_current(self,  var_idx: int, var_id: Sequence, hyper_diff=0.0, inv_tau=None, **kwargs):

        # -----------------------------------------------------------------
        # Transport
        # plasma parallel conductivity,                                 [(Ohm*m)^-1]
        conductivity_parallel = self._c_transp.conductivity_parallel
        j_exp = self._c_source.j_parallel + self._c_source.get("j_decomposed.explicit_part", Function(0))
        j_imp = self._c_source.get("j_decomposed.implicit_part", Function(0))

        ym = Function(self._rho_tor_norm, self._core_profiles_prev.fetch("psi", None))

        def func(x: np.ndarray,  Y: np.ndarray, p: Any = None):
            y = Y[var_idx*2]
            g = Y[var_idx*2+1]
            yp = Function(x, y).derivative(x)

            a = conductivity_parallel*self._rho_tor
            b = conductivity_parallel*self._rho_tor
            c = (constants.mu_0 * self._B0 * self._rho_tor_boundary)/(self._fpol**2)
            d = self._vpr * self._gm2 / self._fpol / (self._rho_tor_boundary)/(TWOPI)
            e = (- constants.mu_0 * self._B0 * self._k_phi) * (conductivity_parallel * self._rho_tor**2/self._fpol**2)
            S = -self._vpr * (j_exp + j_imp*y)/TWOPI

            dy = (-g + e * y + hyper_diff * yp)/(d + hyper_diff)
            dg = (S - (a * y - b * ym)/inv_tau)*c

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

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p: Any = None, /, _bc: Sequence = (u0, v0, w0, u1, v1, w1)):
            u0, v0, w0,  u1, v1, w1 = _bc
            return (u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0), (u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1)

        return func, bc

    def f_particle(self, var_idx: int, var_id: Sequence, hyper_diff=0.0, inv_tau=None, ** kwargs):
        # Particle Transport
        transp: Union[CoreTransport.Model.Profiles1D.Ion,
                      CoreTransport.Model.Profiles1D.Electrons] = self._c_transp.fetch(var_id[:-1], NotImplemented)

        diff = transp.particles.d

        conv = transp.particles.v

        source: Union[CoreSources.Source.Profiles1D.Ion,
                      CoreSources.Source.Profiles1D.Electrons] = self._c_source.fetch(var_id[:-1], NotImplemented)

        se_exp = source.particles  # + source.fetch("particles_decomposed.explicit_part", 0)

        se_imp = 0  # source.fetch("particles_decomposed.implicit_part", 0)

        ym = self._core_profiles_prev.fetch(var_id, 0)

        def func(x: np.ndarray,  Y: np.ndarray, p: Any = None):
            y = Y[var_idx*2]
            g = Y[var_idx*2+1]
            yp = Function(x, y).derivative(x)

            a = self._vpr(x)
            b = self._vprm(x)
            c = self._rho_tor_boundary
            d = (self._vpr * self._gm3 * diff / self._rho_tor_boundary)(x)
            e = (self._vpr * self._gm3 * conv - self._vpr * self._rho_tor * self._k_phi)(x)
            S = (self._vpr * (se_exp + se_imp*y + self._k_rho_bdry))(x)

            dy = (-g + e * y + hyper_diff * yp)/(d + hyper_diff)
            dg = (S - (a * y - b * ym)/inv_tau)*c

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

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p: Any = None, /, _bc: Sequence = (u0, v0, w0, u1, v1, w1)):
            u0, v0, w0,  u1, v1, w1 = _bc
            return (u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0), (u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1)

        return func, bc_func

    def f_energy(self, var_idx: int, var_id: Sequence, hyper_diff=0.0, inv_tau=None, density_idx=1, **kwargs):

        # energy transport
        transp: CoreTransport.Model.Profiles1D.Ion = self._c_transp.fetch(var_id[:-1], NotImplemented)
        chi = transp.energy.d
        v_pinch = transp.energy.v

        source: Union[CoreSources.Source.Profiles1D.Ion,
                      CoreSources.Source.Profiles1D.Electrons] = self._c_source.fetch(var_id[:-1], NotImplemented)
        Qs_exp = source.energy + source.get("energy_decomposed.explicit_part", 0)
        Qs_imp = source.get("energy_decomposed.implicit_part", 0)

        ym = self._core_profiles_prev.fetch(var_id, 0)

        def func(x: np.ndarray,  Y: np.ndarray, p: Any = None):
            y = Y[var_idx*2]
            g = Y[var_idx*2+1]

            n = Y[density_idx*2]
            gamma = Y[density_idx*2+1]

            yp = Function(x, y).derivative(x)
            a = (3/2) * self._vpr35 * y
            b = (3/2) * self._vpr35m * ym
            c = self._rho_tor_boundary * self._inv_vpr23

            d = self._vpr * self._gm3 * n * chi / self._rho_tor_boundary -\
                self._vpr * (3/4)*self._k_phi * self._rho_tor * ym
            e = self._vpr * self._gm3 * n * v_pinch + 3/2*gamma

            S = self._vpr35 * (Qs_exp + (Qs_imp + self._Qimp_k_ns)*y)

            dy = (-g + e * y + hyper_diff * yp)/(d + hyper_diff)
            dg = (S - (a * y - b * ym)/inv_tau)*c

            return array_like(x, dy), array_like(x, dg)

        # ----------------------------------------------
        # Boundary Condition
        bc: TransportSolver.BoundaryConditions1D.BoundaryConditions =\
            self.boundary_conditions_1d.fetch(var_id[:-1], NotImplemented).energy

        # axis
        u0, v0, w0 = 0, 1, 0

        if bc.identifier.index == 1:
            u1 = 1
            v1 = 0
            w1 = bc.value[0]
        else:
            raise NotImplementedError(bc)

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p: Any = None, /, _bc: Sequence = (u1, v1, w1)):
            u1, v1, w1 = _bc
            na = Ya[density_idx*2]
            ga = Ya[density_idx*2+1]
            u0 = - (self._vpr * self._gm3 * na * v_pinch + 3/2*ga)
            v0 = 1
            w0 = 3/2*ga
            return (u0 * Ya[var_idx*2] + v0 * Ya[var_idx*2+1] - w0), (u1 * Yb[var_idx*2] + v1 * Yb[var_idx*2+1] - w1)

        return func, bc_func

        # return path+["temperature"], (a, b, c, d, e, S), (u0, v0, w0,  u1, v1, w1)

    def rotation_transport(self, var_idx: int, var_id: Sequence,  hyper_diff=0.0, inv_tau=None, **kwargs):
        r"""
            Rotation Transport
            .. math::  \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\left(V^{\prime\frac{5}{3}}\left\langle R\right\rangle \
                        m_{i}n_{i}u_{i,\varphi}\right)+\frac{\partial}{\partial\rho}\Phi_{i}=V^{\prime}\left[U_{i,\varphi,exp}-U_{i,\varphi}\cdot u_{i,\varphi}+U_{zi,\varphi}\right]
                :label: transport_rotation
        """
        logger.warning(f"TODO: Rotation Transport is not implemented!")
        return 0.0

    def neutral_condition(self, path, /, n_impurity: Function = 0, electron_index: int = 1, factor: float = 1.0):
        if path[0] == "electrons":
            raise NotImplementedError()
        else:
            def func(idx: int, x: np.ndarray, Y: np.ndarray, Ym: np.ndarray, G: np.ndarray, *args, n_impurity=n_impurity, factor=factor, electron_index=electron_index):
                ne = Y[electron_index]
                if n_impurity is not None:
                    ne = ne - n_impurity(x)
                return factor*Function(x, ne).derivative(x), factor*Function(x, G[electron_index]).derivative(x)

            def bc_func(idx: int, ya: float, ga: float, yb: float, gb: float, *args, factor=factor, electron_index=electron_index):
                return np.hstack([(yb[idx] - factor*yb[electron_index]), (gb[idx] - factor*gb[electron_index])])

        return path+["density"],  func, bc_func

    def _solve_equations(self, x0: np.ndarray, Y0: np.ndarray, eq_grp: Sequence, /,
                         tolerance=1.0e-3,
                         max_nodes=250, **kwargs) -> BVPResult:

        fc_list = [func(idx, var_id, *_args) for idx, (var_id, func,  *_args) in enumerate(eq_grp)]

        def func(x: np.ndarray, Y: np.ndarray, p=None, /, fc_list=fc_list):
            return np.vstack(map(array_like, sum([func(x, Y, p) for func, bc in fc_list], [])))

        def bc_func(Ya: np.ndarray, Yb: np.ndarray, p=None, /, fc_list=fc_list):
            return np.np.asarray(sum([bc(Ya, Yb, p) for func, bc in fc_list], []))

        return solve_bvp(func, bc_func, x0, Y0, tolerance=tolerance, max_nodes=max_nodes, **kwargs)

    def _convert_to_dict(Y: np.ndarray, var_list=[]):
        if Y is None:
            return None
        profiles = Dict()

        for idx, path in enumerate(var_list):
            if path[0] != 'ion':
                profiles[path] = Y[idx*2]
                profiles[path[:-1]+[f"{path[-1]}_flux"]] = Y[idx*2+1]
            else:
                profiles[path] = Y[idx*2]
                profiles[path[:-1]+[{"label": path[-1]['label']+"_flux"}]] = Y[idx*2+1]

        return profiles

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

        n_impurity = None
        # FIXME: Fuction do not support sum
        for ion in self._core_profiles_next.ion:
            if ion.label not in ion_species:
                if n_impurity is not None:
                    n_impurity = n_impurity + ion.z*ion.density
                else:
                    n_impurity = ion.z*ion.density

        if enable_ion_particle_solver is True:
            eq_grp = [
                (["psi"],                                        self.f_current,),
                *[(["ion", {"label": label}, "density"],         self.f_particle,) for label in ion_species],
                (["electrons", "temperature"],                   self.f_energy,),
                *[(["ion", {"label": label}, "temperature"],     self.f_energy,) for label in ion_species],
            ]
        else:

            eq_grp = [
                (["psi"],                                        self.f_current,),
                (["electrons", "density"],                       self.f_particle,),
                (["electrons", "temperature"],                   self.f_energy,),
                *[(["ion", {"label": label}, "temperature"],     self.f_energy,) for label in ion_species],
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
            f"Solve transport equations {'Success' if sol.success else 'Failed'}] max reduisal={rms_residuals}:\n  {[var_id for var_id, in eq_grp]}")

        return rms_residuals


__SP_EXPORT__ = TransportSolverBVP2
