
import collections
import functools
import inspect
import numbers
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.constants as constants
from fytok.util.Interpolate import (Interpolate1D, Interpolate2D, derivate,
                                    find_critical, find_root, integral,
                                    interpolate)
from fytok.util.Plot import plot_profiles
from fytok.util.RadialGrid import RadialGrid
from numpy import arctan2, cos, sin, sqrt
from scipy.interpolate import (RectBivariateSpline, SmoothBivariateSpline,
                               UnivariateSpline)
from spdm.data.PhysicalGraph import PhysicalGraph, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.data.Profile import Profile, Profiles
from spdm.util.sp_export import sp_find_module
from sympy import Point, Polygon


class CoreProfiles(PhysicalGraph):
    """CoreProfiles
    """
    IDS = "core_profiles"

    def __init__(self, cache=None, *args, time=None,  grid=None, ** kwargs):
        super().__init__(*args, ** kwargs)

        if isinstance(cache, LazyProxy) or isinstance(cache, PhysicalGraph):
            self.__dict__["_cache"] = cache
        else:
            self.__dict__["_cache"] = PhysicalGraph(cache)

        self.__dict__["_time"] = time or 0.0
        self.__dict__["_grid"] = grid

        # self.vacuum_toroidal_field = tokamak.vacuum_toroidal_field

    class Profiles1D(Profiles):
        def __init__(self, cache=None,  *args, parent=None, grid=None, **kwargs):
            if grid is None:
                grid = parent._grid
            super().__init__(cache, * args, axis=grid.rho_tor_norm, **kwargs)
            self.__dict__["_parent"] = parent
            self.__dict__["_grid"] = grid
            self.__dict__["_b0"] = parent.vacuum_toroidal_field.b0
            self.__dict__["_r0"] = parent.vacuum_toroidal_field.r0

        @property
        def grid(self):
            return self._grid

        # def __missing__(self, key):
        #     logger.debug(key)
        #     d = super().__missing__(key)
        #     if d is None:
        #         d = self._parent._tokamak.equilibrium.profiles_1d.mapping("rho_tor_norm", key, self.grid.rho_tor_norm)
        #     return d

        class TemperatureFit(PhysicalGraph):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class DensityFit(PhysicalGraph):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class Electrons(Profiles):
            def __init__(self, cache=None, *args, grid=None,  **kwargs):
                super().__init__(cache, *args, axis=grid.rho_tor_norm, **kwargs)
                self.__dict__['_grid'] = grid
                self |= {
                    "temperature_validity": 0,
                    "density_validity": 0
                }
            # @property
            # def temperature(self):
            #     """Temperature {dynamic} [eV]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def temperature_validity(self):
            #     """Indicator of the validity of the temperature profile.
            #     0: valid from automated processing,
            #     1: valid and certified by the RO;
            #     - 1 means problem identified in the data processing (request verification by the RO),
            #     -2: invalid data, should not be used {dynamic}"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            @cached_property
            def temperature_fit(self):
                """Information on the fit used to obtain the temperature profile [eV]  """
                return CoreProfiles.DensityFit(self._grid)

            # @property
            # def density(self):
            #     """Density (thermal+non-thermal) {dynamic} [m^-3]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_validity(self):
            #     """Indicator of the validity of the density profile.
            #     0: valid from automated processing,
            #     1: valid and certified by the RO;
            #     - 1 means problem identified in the data processing (request verification by the RO),
            #     -2: invalid data, should not be used {dynamic}"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            @cached_property
            def density_fit(self):
                """Information on the fit used to obtain the density profile [m^-3]"""
                return CoreProfiles.DensityFit(self._grid)

            # @property
            # def density_thermal(self):
            #     """Density of thermal particles {dynamic} [m^-3]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_fast(self):
            #     """Density of fast (non-thermal) particles {dynamic} [m^-3]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure(self):
            #     """Pressure(thermal+non-thermal) {dynamic}[Pa]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_thermal(self):
            #     """Pressure(thermal) associated with random motion ~average((v-average(v)) ^ 2) {dynamic}[Pa]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_perpendicular(self):
            #     """Fast(non-thermal) perpendicular pressure {dynamic}[Pa]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_parallel(self):
            #     """Fast(non-thermal) parallel pressure {dynamic}[Pa]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def collisionality_norm(self):
            #     """Collisionality normalised to the bounce frequency {dynamic}[-]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

        class Ion(Profiles):
            def __init__(self, cache=None,  *args, grid=None, z_ion=1, label=None, neutral_index=None,  **kwargs):
                super().__init__(cache, *args, axis=grid.rho_tor_norm, **kwargs)
                self.__dict__['_grid'] = grid
                self |= {
                    "z_ion": "z_ion",
                    "label": "label",
                    "neutral_index": "neutral_index",
                    "element": [],
                    "state": [],
                    "temperature_validity": 0,
                    "density_validity": 0
                }

            # @property
            # def z_ion(self):
            #     """Ion charge (of the dominant ionisation state; lumped ions are allowed),
            #     volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def label(self):
            #     """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def neutral_index(self):
            #     """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def z_ion_1d(self):
            #     """Average charge of the ion species (sum of states charge weighted by state density and
            #     divided by ion density) {dynamic} [-]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def z_ion_square_1d(self):
            #     """Average square charge of the ion species (sum of states square charge weighted by
            #     state density and divided by ion density) {dynamic} [-]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def temperature(self):
            #     """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def temperature_validity(self):
            #     """Indicator of the validity of the temperature profile.
            #     0: valid from automated processing,
            #     1: valid and certified by the RO;
            #     - 1 means problem identified in the data processing (request verification by the RO),
            #     -2: invalid data, should not be used {dynamic}    """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            @cached_property
            def temperature_fit(self):
                """Information on the fit used to obtain the temperature profile [eV]    """
                return CoreProfiles.TemperatureFit(self._grid)

            # @property
            # def density(self):
            #     """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_validity(self):
            #     """Indicator of the validity of the density profile.
            #      0: valid from automated processing,
            #      1: valid and certified by the RO;
            #      - 1 means problem identified in the data processing (request verification by the RO),
            #      -2: invalid data, should not be used {dynamic}    """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            @cached_property
            def density_fit(self):
                """Information on the fit used to obtain the density profile [m^-3]    """
                return CoreProfiles.DensityFit(self._grid)

            # @property
            # def density_thermal(self):
            #     """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_fast(self):
            #     """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure(self):
            #     """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_thermal(self):
            #     """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
            #     (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_perpendicular(self):
            #     """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_parallel(self):
            #     """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def rotation_frequency_tor(self):
            #     """Toroidal rotation frequency (i.e. toroidal velocity divided by the major radius at which the toroidal velocity is taken)
            #     (average over charge states when multiple charge states are considered) {dynamic} [rad.s^-1]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def velocity(self):
            #     """Velocity (average over charge states when multiple charge states are considered) at the position of maximum major
            #     radius on every flux surface [m.s^-1]    """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def multiple_states_flag(self):
            #     """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state  {dynamic}    """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def state(self):
            #     """Quantities related to the different states of the species (ionisation, energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

        class Neutral(Profiles):
            def __init__(self, cache=None,  *args, grid=None, label=None, ion_index=None, **kwargs):
                super().__init__(cache, *args, axis=grid.rho_tor_norm, **kwargs)
                self.__dict__['_grid'] = grid
                self |= {
                    "label": label,
                    "element": [],
                    "state": [],
                    "temperature_validity": 0,
                    "density_validity": 0
                }

            # @property
            # def element(self):
            #     """List of elements forming the atom or molecule  struct_array [max_size=unbounded]  1- 1...N"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def label(self):
            #     """String identifying the species (e.g. H, D, T, He, C, D2, DT, CD4, ...) {dynamic}  STR_0D  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def ion_index(self):
            #     """Index of the corresponding ion species in the ../../ion array {dynamic}  INT_0D  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def temperature(self):
            #     """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density(self):
            #     """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_thermal(self):
            #     """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_fast(self):
            #     """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure(self):
            #     """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_thermal(self):
            #     """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
            #     (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_perpendicular(self):
            #     """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_parallel(self):
            #     """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def multiple_states_flag(self):
            #     """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the
            #     state structure {dynamic}  INT_0D  """
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def state(self):
            #     """Quantities related to the different states of the species (energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

        @cached_property
        def ion(self):
            """Quantities related to the different ion species"""
            return PhysicalGraph(default_factory_array=lambda _holder=self: CoreProfiles.Profiles1D.Ion(None, parent=_holder, grid=_holder.grid))

        @cached_property
        def electrons(self):
            """Quantities related to the electrons"""
            return CoreProfiles.Profiles1D.Electrons(self._cache["electrons"], grid=self.grid)

        @cached_property
        def neutral(self):
            """Quantities related to the different neutral species"""
            return CoreProfiles.Profiles1D.Neutral(self._cache["neutral"], grid=self.grid)

        @cached_property
        def t_i_average(self):
            """Ion temperature(averaged on charge states and ion species) {dynamic}[eV]"""
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "t_i_average"})

        @cached_property
        def t_i_average_fit(self):
            """Information on the fit used to obtain the t_i_average profile[eV]"""
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "t_i_average_fit"})

        @cached_property
        def n_i_total(self):
            """ total ion density(sum over species and charge states)   (thermal+non-thermal) {dynamic}[-]"""
            res = Profile(None, axis=self.grid.rho_tor_norm,  description={"name": "n_i_total"})
            for ion in self.ion:
                logger.debug(ion)
                res += ion.z_ion*(ion.density_thermal+ion.density_fast)
            return res

        @cached_property
        def n_i_total_over_n_e(self):
            """Ratio of total ion density(sum over species and charge states) over electron density. (thermal+non-thermal) {dynamic}[-]"""
            return self.n_i_total/self.electrons.density

        @cached_property
        def n_i_thermal_total(self):
            """Total ion thermal density(sum over species and charge states) {dynamic}[m ^ -3]"""
            res = Profile(0.0, axis=self.grid.rho_tor_norm,  description={"name": "n_i_thermal_total"})
            for ion in self.ion:
                res += ion.z_ion * ion.density_thermal
            return res

        @cached_property
        def zeff(self):
            """Effective charge {dynamic}[-]"""
            res = Profile(0.0, axis=self.grid.rho_tor_norm,  description={"name": "zeff"})
            for ion in self.ion:
                res += ion.z_ion * ion.z_ion * ion.density
            return res/self.n_i_total

        @cached_property
        def zeff_fit(self):
            """Information on the fit used to obtain the zeff profile[-]  """
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "zeff_fit"})

        @cached_property
        def momentum_tor(self):
            """Total plasma toroidal momentum, summed over ion species and electrons weighted by their density and major radius,
             i.e. sum_over_species(n*R*m*Vphi) {dynamic}[kg.m ^ -1.s ^ -1]"""
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "momentum_tor"})

        @cached_property
        def pressure_ion_total(self):
            """Total(sum over ion species) thermal ion pressure {dynamic}[Pa]"""
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "pressure_ion_total"})

        @cached_property
        def pressure_thermal(self):
            """Thermal pressure(electrons+ions) {dynamic}[Pa]"""
            return NotImplemented

        @cached_property
        def pressure_perpendicular(self):
            """Total perpendicular pressure(electrons+ions, thermal+non-thermal) {dynamic}[Pa]"""
            return NotImplemented

        @cached_property
        def pressure_parallel(self):
            """Total parallel pressure(electrons+ions, thermal+non-thermal) {dynamic}[Pa]"""
            return NotImplemented

        @cached_property
        def j_total(self):
            """Total parallel current density = average(jtot.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "j_total"})

        # @property
        # def current_parallel_inside(self):
        #     """Parallel current driven inside the flux surface. Cumulative surface integral of j_total {dynamic}[A]"""
        #     return NotImplemented

        @cached_property
        def j_tor(self):
            """Total toroidal current density = average(J_Tor/R) / average(1/R) {dynamic}[A/m ^ 2]"""
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "j_tor"})

        @cached_property
        def j_ohmic(self):
            """Ohmic parallel current density = average(J_Ohmic.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
            return NotImplemented

        @cached_property
        def j_non_inductive(self):
            """Non-inductive(includes bootstrap) parallel current density = average(jni.B) / B0,
            where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
            return NotImplemented

        @cached_property
        def j_bootstrap(self):
            """Bootstrap current density = average(J_Bootstrap.B) / B0,
             where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
            return NotImplemented

        # @property
        # def conductivity_parallel(self):
        #     """Parallel conductivity {dynamic}[ohm ^ -1.m ^ -1]"""
        #     return NotImplemented

        class EField(Profiles):
            def __init__(self, cache=None,  *args, grid=None,    **kwargs):
                super().__init__(cache, *args, axis=grid.rho_tor_norm, **kwargs)
                self.__dict__['_grid'] = grid

        @cached_property
        def e_field(self):
            """Electric field, averaged on the magnetic surface. E.g for the parallel component, average(E.B) / B0,
             using core_profiles/vacuum_toroidal_field/b0[V.m ^ -1]  """
            return CoreProfiles.Profiles1D.EField(self._cache["e_field"], grid=self.grid)

        @cached_property
        def phi_potential(self):
            """Electrostatic potential, averaged on the magnetic flux surface {dynamic}[V]"""
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "phi_potential"})

        @cached_property
        def rotation_frequency_tor_sonic(self):
            """Derivative of the flux surface averaged electrostatic potential with respect to the poloidal flux, multiplied by - 1.
            This quantity is the toroidal angular rotation frequency due to the ExB drift, introduced in formula(43) of Hinton and Wong,
            Physics of Fluids 3082 (1985), also referred to as sonic flow in regimes in which the toroidal velocity is dominant over the
            poloidal velocity Click here for further documentation. {dynamic}[s ^ -1]"""
            return Profile(None, axis=self.grid.rho_tor_norm, description={"name": "rotation_frequency_tor_sonic"})

        # @cached_property
        # def q(self):
        #     """Safety factor(IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic}[-].
        #     This quantity is COCOS-dependent, with the following transformation: """
        #     # q = (constants.pi*2.0)*self._b0*self.dpsi_drho_tor.axis*self.grid.rho_tor[-1]/self.dpsi_drho_tor
        #     # q[0] = 2*q[1]-q[2]
        #     # return Profile(self.dpsi_drho_tor.axis, q)
        #     return Profile(self.grid.rho_tor_norm, 0, description={"name": "Safety factor"})

        # @cached_property
        # def dpsi_drho_tor(self):
        #     return Profile(self.grid.rho_tor_norm, 0, description={"name": "dpsi_drho_tor"})

        @cached_property
        def magnetic_shear(self):
            """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic}[-]"""
            return NotImplemented

    class GlobalQuantities(PhysicalGraph):
        def __init__(self, cache=None, *args, core_profiles=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__["_core_profiles"] = core_profiles

        @cached_property
        def p(self):
            """ Total plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A]. """
            return NotImplemented

        @cached_property
        def current_non_inductive(self):
            """Total non-inductive current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A]. """
            return NotImplemented

        @cached_property
        def current_bootstrap(self):
            """Bootstrap current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A]. """
            return NotImplemented

        @cached_property
        def v_loop(self):
            """ LCFS loop voltage (positive value drives positive ohmic current that flows anti-clockwise when viewed from above) {dynamic} [V]. """
            return NotImplemented

        @cached_property
        def li_3(self):
            """ Internal inductance. The li_3 definition is used, i.e. li_3 = 2/R0/mu0^2/Ip^2 * int(Bp^2 dV). {dynamic} [-]"""
            return NotImplemented

        @cached_property
        def beta_tor(self):
            """ Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2 {dynamic} [-]"""
            return NotImplemented

        @cached_property
        def beta_tor_norm(self):
            """Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] {dynamic} [-]"""
            return NotImplemented

        @cached_property
        def beta_pol(self):
            """ Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]"""
            return NotImplemented

        @cached_property
        def energy_diamagnetic(self):
            """Plasma energy content = 3/2 * integral over the plasma volume of the total perpendicular pressure {dynamic} [J]"""
            return NotImplemented

        @cached_property
        def z_eff_resistive(self):
            """Volume average plasma effective charge, estimated from the flux consumption in the ohmic phase {dynamic} [-]"""
            return NotImplemented

    @property
    def time(self):
        return self._time

    @cached_property
    def profiles_1d(self):
        return CoreProfiles.Profiles1D(self._cache.profiles_1d, parent=self)

    @cached_property
    def global_quantities(self):
        return CoreProfiles.GlobalQuantities(self._cache.global_quantities, core_profiles=self)

    def plot(self, profiles, fig_axis=None, axis=None):
        return plot_profiles(
            self,
            profiles,
            axis=axis or ("grid.rho_tor_norm", r"$\rho_{norm}$"),
            prefix="profiles_1d",
            fig_axis=fig_axis)
