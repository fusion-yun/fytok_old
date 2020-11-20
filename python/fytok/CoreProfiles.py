
import collections
import functools
import inspect
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arctan2, cos, sin, sqrt
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, SmoothBivariateSpline
from spdm.util.AttributeTree import AttributeTree, _next_
from spdm.util.Interpolate import (Interpolate1D, Interpolate2D, derivate,
                                   find_critical, find_root, integral,
                                   interpolate)
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from sympy import Point, Polygon

from .Plot import plot_profiles


class CoreProfiles(AttributeTree):
    """
        imas dd version 3.28

        ids = core_profiles.profiles_1d
    """

    def __init__(self, time=None, *args,  equilibrium=None, rho_tor_norm=None, ** kwargs):
        super().__init__(*args, ** kwargs)
        self.time = time or equilibrium.time
        self.vacuum_toroidal_field = equilibrium.vacuum_toroidal_field
        self._equilibrium = equilibrium
        self._rho_tor_norm = rho_tor_norm

    def update(self, config):
        if isinstance(config, LazyProxy):
            config = config()
        self.__update__(config)

    class Profiles1D(AttributeTree):
        def __init__(self,  *args, equilibrium=None, rho_tor_norm=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._eq = equilibrium
            self.grid = CoreProfiles.Profiles1D.Grid(equilibrium, rho_tor_norm=rho_tor_norm)

        def __missing__(self, key):
            d = self._eq.profiles_1d[key]
            if d is not NotImplemented and d is not None and len(d) > 0:
                return UnivariateSpline(self._eq.profiles_1d.rho_tor_norm, d)(self.grid.rho_tor_norm)
            else:
                return np.full(self.grid.rho_tor_norm.shape, np.nan)

        def interpolate(self, key):
            if not isinstance(key, str) and isinstance(key, collections.abc.Sequence):
                return {k: self.interpolate(k) for k in key}
            v = self.__getitem__(key)
            if v is None or v is NotImplemented:
                return None
            elif len(v) == len(self.grid.rho_tor_norm):
                return UnivariateSpline(self.grid.rho_tor_norm, v)

        class TemperatureFit(AttributeTree):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class DensityFit(AttributeTree):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class Grid(AttributeTree):
            def __init__(self, eq, *args, rho_tor_norm=None,  **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__['_eq'] = eq

                """Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm,
                is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of
                a fixed boundary equilibium calculation, see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]	"""
                if rho_tor_norm is None:
                    rho_tor_norm = 129
                if type(rho_tor_norm) is int:
                    self.rho_tor_norm = np.linspace(0, 1.0, rho_tor_norm)
                else:
                    self.rho_tor_norm = rho_tor_norm

                self |= {
                    "psi_magnetic_axis": self._eq.global_quantities.psi_axis,  # Value of the poloidal magnetic flux at the magnetic axis (useful to normalize the psi array values when the radial grid doesn't go from the magnetic axis to the plasma boundary) {dynamic} [Wb]	FLT_0D"""
                    "psi_boundary": self._eq.global_quantities.psi_boundary    # Value of the poloidal magnetic flux at the plasma boundary (useful to normalize the psi array values when the radial grid doesn't go from the magnetic axis to the plasma boundary) {dynamic} [Wb]	FLT_0D"""
                }

            def __missing__(self, key):
                path = key.split('.')
                x = self._eq.profiles_1d.rho_tor_norm
                y = self._eq.profiles_1d[key]
                if y is None or y is NotImplemented:
                    raise KeyError(path)
                elif not isinstance(y, np.ndarray) or len(y.shape) > 1:
                    return y
                else:
                    return UnivariateSpline(x, y)(self.rho_tor_norm)

            # def rho_tor(self):
            #     """Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m].
            #     The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""

            # def rho_pol_norm(self):
            #     """Normalised poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))) {dynamic} [-]"""
            #     return NotImplemented

            # def psi(self):
            #     """Poloidal magnetic flux {dynamic} [Wb]. This quantity is COCOS-dependent, with the following transformation :"""

            # def volume(self):
            #     """Volume enclosed inside the magnetic surface {dynamic} [m^3]"""

            # def area(self):
            #     """Cross-sectional area of the flux surface {dynamic} [m^2]"""

            # def surface(self):
            #     """Surface area of the toroidal flux surface {dynamic} [m^2]"""

        class Electons(AttributeTree):
            def __init__(self,  grid,  *args,  **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__['_grid'] = grid
                self |= {
                    "temperature_validity": 0,
                    "density_validity": 0
                }

            def __missing__(self, key):
                return np.full(self._grid.rho_tor_norm.shape, np.nan)

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
                """Information on the fit used to obtain the temperature profile [eV]	"""
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
            #     """	Fast(non-thermal) parallel pressure {dynamic}[Pa]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def collisionality_norm(self):
            #     """	Collisionality normalised to the bounce frequency {dynamic}[-]"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

        class Ion(AttributeTree):
            def __init__(self,  grid,  *args, z_ion=1, label=None, neutral_index=None, **kwargs):
                super().__init__(*args, z_ion=z_ion, label=label, neutral_index=neutral_index, **kwargs)
                self.__dict__['_grid'] = grid
                self |= {
                    "element": [],
                    "state": [],
                    "temperature_validity": 0,
                    "density_validity": 0
                }

            def __missing__(self, key):
                return np.full(self._grid.rho_tor_norm.shape, np.nan)

            # @property
            # def z_ion(self):
            #     """Ion charge (of the dominant ionisation state; lumped ions are allowed),
            #     volume averaged over plasma radius {dynamic} [Elementary Charge Unit]	FLT_0D	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def label(self):
            #     """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}		"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def neutral_index(self):
            #     """Index of the corresponding neutral species in the ../../neutral array {dynamic}		"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def z_ion_1d(self):
            #     """Average charge of the ion species (sum of states charge weighted by state density and
            #     divided by ion density) {dynamic} [-]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def z_ion_square_1d(self):
            #     """Average square charge of the ion species (sum of states square charge weighted by
            #     state density and divided by ion density) {dynamic} [-]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def temperature(self):
            #     """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def temperature_validity(self):
            #     """Indicator of the validity of the temperature profile.
            #     0: valid from automated processing,
            #     1: valid and certified by the RO;
            #     - 1 means problem identified in the data processing (request verification by the RO),
            #     -2: invalid data, should not be used {dynamic}		"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            @cached_property
            def temperature_fit(self):
                """Information on the fit used to obtain the temperature profile [eV]		"""
                return CoreProfiles.TemperatureFit(self._grid)

            # @property
            # def density(self):
            #     """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_validity(self):
            #     """Indicator of the validity of the density profile.
            #      0: valid from automated processing,
            #      1: valid and certified by the RO;
            #      - 1 means problem identified in the data processing (request verification by the RO),
            #      -2: invalid data, should not be used {dynamic}		"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            @cached_property
            def density_fit(self):
                """Information on the fit used to obtain the density profile [m^-3]		"""
                return CoreProfiles.DensityFit(self._grid)

            # @property
            # def density_thermal(self):
            #     """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_fast(self):
            #     """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure(self):
            #     """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_thermal(self):
            #     """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
            #     (sum over charge states when multiple charge states are considered) {dynamic} [Pa]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_perpendicular(self):
            #     """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_parallel(self):
            #     """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def rotation_frequency_tor(self):
            #     """Toroidal rotation frequency (i.e. toroidal velocity divided by the major radius at which the toroidal velocity is taken)
            #     (average over charge states when multiple charge states are considered) {dynamic} [rad.s^-1]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def velocity(self):
            #     """Velocity (average over charge states when multiple charge states are considered) at the position of maximum major
            #     radius on every flux surface [m.s^-1]		"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def multiple_states_flag(self):
            #     """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state  {dynamic}		"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def state(self):
            #     """Quantities related to the different states of the species (ionisation, energy, excitation, ...)	struct_array [max_size=unbounded]	1- 1...N"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

        class Neutral(AttributeTree):
            def __init__(self, grid, *args, label=None, ion_index=None, **kwargs):
                super().__init__(*args, **kwargs)
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
            #     """List of elements forming the atom or molecule	struct_array [max_size=unbounded]	1- 1...N"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def label(self):
            #     """String identifying the species (e.g. H, D, T, He, C, D2, DT, CD4, ...) {dynamic}	STR_0D	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def ion_index(self):
            #     """Index of the corresponding ion species in the ../../ion array {dynamic}	INT_0D	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def temperature(self):
            #     """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density(self):
            #     """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_thermal(self):
            #     """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def density_fast(self):
            #     """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure(self):
            #     """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_thermal(self):
            #     """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
            #     (sum over charge states when multiple charge states are considered) {dynamic} [Pa]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_perpendicular(self):
            #     """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def pressure_fast_parallel(self):
            #     """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def multiple_states_flag(self):
            #     """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the
            #     state structure {dynamic}	INT_0D	"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

            # @property
            # def state(self):
            #     """Quantities related to the different states of the species (energy, excitation, ...)	struct_array [max_size=unbounded]	1- 1...N"""
            #     return self._core_profiles.cache[self.__class__.__name__, inspect.currentframe().f_code.co_name]

        @cached_property
        def ion(self):
            """Quantities related to the different ion species"""
            return CoreProfiles.Ion(self)

        @cached_property
        def electons(self):
            """Quantities related to the electrons"""
            return CoreProfiles.Electons(self)

        @cached_property
        def neutral(self):
            """Quantities related to the different neutral species"""
            return CoreProfiles.Neutral(self)

        @property
        def pprime(self):
            return None

        @property
        def ffprime(self):
            return None

        # @property
        # def t_i_average(self):
        #     """	Ion temperature(averaged on charge states and ion species) {dynamic}[eV]"""
        #     return NotImplemented

        # @property
        # def t_i_average_fit(self):
        #     """Information on the fit used to obtain the t_i_average profile[eV]"""
        #     return NotImplemented

        # @property
        # def n_i_total_over_n_e(self):
        #     """	Ratio of total ion density(sum over species and charge states) over electron density. (thermal+non-thermal) {dynamic}[-]"""
        #     return NotImplemented

        # @property
        # def n_i_thermal_total(self):
        #     """	Total ion thermal density(sum over species and charge states) {dynamic}[m ^ -3]"""
        #     return NotImplemented

        # @property
        # def momentum_tor(self):
        #     """	Total plasma toroidal momentum, summed over ion species and electrons weighted by their density and major radius, i.e. sum_over_species(n*R*m*Vphi) {dynamic}[kg.m ^ -1.s ^ -1]"""
        #     return NotImplemented

        # @property
        # def zeff(self):
        #     """	Effective charge {dynamic}[-]"""
        #     return NotImplemented

        # @property
        # def zeff_fit(self):
        #     """Information on the fit used to obtain the zeff profile[-]	"""
        #     return NotImplemented

        # @property
        # def pressure_ion_total(self):
        #     """	Total(sum over ion species) thermal ion pressure {dynamic}[Pa]"""
        #     return NotImplemented

        # @property
        # def pressure_thermal(self):
        #     """	Thermal pressure(electrons+ions) {dynamic}[Pa]"""
        #     return NotImplemented

        # @property
        # def pressure_perpendicular(self):
        #     """	Total perpendicular pressure(electrons+ions, thermal+non-thermal) {dynamic}[Pa]"""
        #     return NotImplemented

        # @property
        # def pressure_parallel(self):
        #     """	Total parallel pressure(electrons+ions, thermal+non-thermal) {dynamic}[Pa]"""
        #     return NotImplemented

        # @property
        # def j_total(self):
        #     """	Total parallel current density = average(jtot.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        #     return NotImplemented

        # @property
        # def current_parallel_inside(self):
        #     """	Parallel current driven inside the flux surface. Cumulative surface integral of j_total {dynamic}[A]"""
        #     return NotImplemented

        # @property
        # def j_tor(self):
        #     """	Total toroidal current density = average(J_Tor/R) / average(1/R) {dynamic}[A/m ^ 2]"""
        #     return NotImplemented

        # @property
        # def j_ohmic(self):
        #     """	Ohmic parallel current density = average(J_Ohmic.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        #     return NotImplemented

        # @property
        # def j_non_inductive(self):
        #     """	Non-inductive(includes bootstrap) parallel current density = average(jni.B) / B0,
        #     where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        #     return NotImplemented

        # @property
        # def j_bootstrap(self):
        #     """	Bootstrap current density = average(J_Bootstrap.B) / B0,
        #      where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        #     return NotImplemented

        # @property
        # def conductivity_parallel(self):
        #     """	Parallel conductivity {dynamic}[ohm ^ -1.m ^ -1]"""
        #     return NotImplemented

        @property
        def e_field(self):
            """Electric field, averaged on the magnetic surface. E.g for the parallel component, average(E.B) / B0,
             using core_profiles/vacuum_toroidal_field/b0[V.m ^ -1]	"""
            return AttributeTree(
                parallel=self.grid.rho_tor_norm
            )

        # @property
        # def phi_potential(self):
        #     """	Electrostatic potential, averaged on the magnetic flux surface {dynamic}[V]"""
        #     return NotImplemented

        # @property
        # def rotation_frequency_tor_sonic(self):
        #     """	Derivative of the flux surface averaged electrostatic potential with respect to the poloidal flux, multiplied by - 1.
        #     This quantity is the toroidal angular rotation frequency due to the ExB drift, introduced in formula(43) of Hinton and Wong,
        #     Physics of Fluids 3082 (1985), also referred to as sonic flow in regimes in which the toroidal velocity is dominant over the
        #     poloidal velocity Click here for further documentation. {dynamic}[s ^ -1]"""
        #     return NotImplemented

        # @property
        # def q(self):
        #     """	Safety factor(IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic}[-].
        #     This quantity is COCOS-dependent, with the following transformation: """
        #     return NotImplemented

        # @property
        # def magnetic_shear(self):
        #     """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic}[-]"""
        #     return NotImplemented

    @cached_property
    def profiles_1d(self):
        return CoreProfiles.Profiles1D(None,
                                       equilibrium=self._equilibrium,
                                       rho_tor_norm=self._rho_tor_norm)

    def plot(self, profiles, axis=None):
        return plot_profiles(self, profiles, x_axis="grid.rho_tor_norm", prefix="profiles_1d", axis=axis)


    #     fig, axis = plt.subplots(ncols=1, nrows=len(profiles), sharex=True)

    #     if not isinstance(profiles, list):
    #         profiles = [profiles]

    #     x_axis = self.profiles_1d.grid.rho_tor_norm

    #     for idx, data in enumerate(profiles):
    #         ylabel = None
    #         opts = {}
    #         if isinstance(data, tuple):
    #             data, ylabel = data
    #         if isinstance(data, str):
    #             ylabel = data

    #         if not isinstance(data, list):
    #             data = [data]

        #    for d in data:
        #         value = self.profiles_1d[d]

        #         if value is not NotImplemented and value is not None and len(value) > 0:
        #             axis[idx].plot(x_axis, value, **opts)
        #         else:
        #             logger.error(f"Can not find profile '{d}'")

        #     axis[idx].legend(fontsize=6)

        #     if ylabel:
        #         axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
        #     axis[idx].labelsize = "media"
        #     axis[idx].tick_params(labelsize=6)

        # axis[-1].set_xlabel(r"$\rho_{norm}$", fontsize=6)

        # return fig
