import collections
from functools import cached_property

import numpy as np
import scipy.constants
from spdm.data.Function import Function
from spdm.data.Node import Dict, List
from spdm.data.Profiles import Profiles
from spdm.data.TimeSeries import TimeSeries
from spdm.util.logger import logger
from spdm.data.AoS import AoS
from ..utilities.IDS import IDS
from ..utilities.Misc import VacuumToroidalField
from .MagneticCoordSystem import TWOPI, RadialGrid
from .ParticleSpecies import Species


class CoreProfiles1D(Profiles):
    def __init__(self, *args, grid=None, time=None, parent=None, **kwargs):
        grid = grid or getattr(parent, "_grid", None)
        assert(grid is not None)
        super().__init__(*args, axis=grid.rho_tor_norm, parent=parent, **kwargs)
        self._grid = grid
        self._time = time or 0.0
        self._r0 = self._parent.vacuum_toroidal_field.r0
        self._b0 = self._parent.vacuum_toroidal_field.b0
        if callable(self._b0):
            self._b0 = self._b0(time)

    @property
    def time(self) -> float:
        return self._time

    @property
    def grid(self) -> RadialGrid:
        return self._grid

    class Electrons(Profiles):
        def __init__(self,   *args, axis=None, parent=None, **kwargs):
            super().__init__(*args, axis=axis if axis is not None else parent.grid.rho_tor_norm,  **kwargs)

        @cached_property
        def temperature(self):
            """Temperature {dynamic} [eV]"""
            return self["temperature"]

        # @property
        # def temperature_validity(self):
        #     """Indicator of the validity of the temperature profile.
        #     0: valid from automated processing,
        #     1: valid and certified by the RO;
        #     - 1 means problem identified in the data processing (request verification by the RO),
        #     -2: invalid data, should not be used {dynamic}"""
        #     return NotImplemented
        # @cached_property
        # def temperature_fit(self):
        #     """Information on the fit used to obtain the temperature profile [eV]  """
        #     return NotImplemented
        @cached_property
        def density(self):
            """Density (thermal+non-thermal) {dynamic} [m^-3]"""
            return self["density"]
        # @property
        # def density_validity(self):
        #     """Indicator of the validity of the density profile.
        #     0: valid from automated processing,
        #     1: valid and certified by the RO;
        #     - 1 means problem identified in the data processing (request verification by the RO),
        #     -2: invalid data, should not be used {dynamic}"""
        #     return NotImplemented
        # @cached_property
        # def density_fit(self):
        #     """Information on the fit used to obtain the density profile [m^-3]"""
        #     return NotImplemented
        # @property
        # def density_thermal(self):
        #     """Density of thermal particles {dynamic} [m^-3]"""
        #     return NotImplemented
        # @property
        # def density_fast(self):
        #     """Density of fast (non-thermal) particles {dynamic} [m^-3]"""
        #     return NotImplemented

        @cached_property
        def pressure(self):
            """Pressure(thermal+non-thermal) {dynamic}[Pa]"""
            if self.pressure_fast_perpendicular is not NotImplemented:
                return self.pressure_thermal+self.pressure_fast_perpendicular+self.pressure_fast_parallel
            else:
                return self.pressure_thermal

        @cached_property
        def pressure_thermal(self):
            """Pressure(thermal) associated with random motion ~average((v-average(v)) ^ 2) {dynamic}[Pa]"""
            return self.density*self.temperature

        @cached_property
        def pressure_fast_perpendicular(self):
            """Fast(non-thermal) perpendicular pressure {dynamic}[Pa]"""
            return NotImplemented

        @cached_property
        def pressure_fast_parallel(self):
            """Fast(non-thermal) parallel pressure {dynamic}[Pa]"""
            return NotImplemented

        @cached_property
        def collisionality_norm(self):
            """Collisionality normalised to the bounce frequency {dynamic}[-]"""
            return NotImplemented

    class Ion(Species):
        def __init__(self,   *args, axis=None, parent=None, **kwargs):
            super().__init__(*args, axis=axis if axis is not None else parent.grid.rho_tor_norm,  **kwargs)

        @cached_property
        def z_ion(self):
            """Ion charge (of the dominant ionisation state; lumped ions are allowed),
            volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """
            return self.__raw_get__("z_ion")

        @cached_property
        def neutral_index(self):
            """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
            return self.__raw_get__("neutral_index")

        @cached_property
        def z_ion_1d(self):
            """Average charge of the ion species (sum of states charge weighted by state density and
            divided by ion density) {dynamic} [-]  """
            return NotImplemented

        @cached_property
        def z_ion_square_1d(self):
            """Average square charge of the ion species (sum of states square charge weighted by
            state density and divided by ion density) {dynamic} [-]  """
            return NotImplemented

        @cached_property
        def temperature(self):
            """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]  """
            return self["temperature"]

        # @property
        # def temperature_validity(self):
        #     """Indicator of the validity of the temperature profile.
        #     0: valid from automated processing,
        #     1: valid and certified by the RO;
        #     - 1 means problem identified in the data processing (request verification by the RO),
        #     -2: invalid data, should not be used {dynamic}    """
        #     return NotImplemented

        # @cached_property
        # def temperature_fit(self):
        #     """Information on the fit used to obtain the temperature profile [eV]    """
        #     return NotImplemented

        @cached_property
        def density(self):
            """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            return self["density"]

        # @property
        # def density_validity(self):
        #     """Indicator of the validity of the density profile.
        #      0: valid from automated processing,
        #      1: valid and certified by the RO;
        #      - 1 means problem identified in the data processing (request verification by the RO),
        #      -2: invalid data, should not be used {dynamic}    """
        #     return NotImplemented

        # @cached_property
        # def density_fit(self):
        #     """Information on the fit used to obtain the density profile [m^-3]    """
        #     return NotImplemented

        @property
        def density_thermal(self):
            """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            return NotImplemented

        @property
        def density_fast(self):
            """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
            return NotImplemented

        @cached_property
        def pressure(self):
            """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            if self.pressure_fast_perpendicular is not NotImplemented:
                return self.pressure_thermal+self.pressure_fast_perpendicular+self.pressure_fast_parallel
            else:
                return self.pressure_thermal

        @cached_property
        def pressure_thermal(self):
            """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
            (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            return self.density*self.temperature

        @cached_property
        def pressure_fast_perpendicular(self):
            """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            return self["pressure_fast_perpendicular"]

        @cached_property
        def pressure_fast_parallel(self):
            """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
            return self["pressure_fast_parallel"]

        @property
        def rotation_frequency_tor(self):
            """Toroidal rotation frequency (i.e. toroidal velocity divided by the major radius at which the toroidal velocity is taken)
            (average over charge states when multiple charge states are considered) {dynamic} [rad.s^-1]  """
            return self["rotation_frequency_tor"]

        @property
        def velocity(self):
            """Velocity (average over charge states when multiple charge states are considered) at the position of maximum major
            radius on every flux surface [m.s^-1]    """
            return self["velocity"]

        @cached_property
        def multiple_states_flag(self):
            """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state  {dynamic}    """
            return self["multiple_states_flag"]

        @cached_property
        def state(self):
            """Quantities related to the different states of the species (ionisation, energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
            return self["state"]

    class Neutral(Species):
        def __init__(self,   *args, axis=None, parent=None, **kwargs):
            super().__init__(*args, axis=axis if axis is not None else parent.grid.rho_tor_norm,  **kwargs)

        @property
        def ion_index(self):
            """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
            return self.__raw_get__("ion_index")
        # @property
        # def element(self):
        #     """List of elements forming the atom or molecule  struct_array [max_size=unbounded]  1- 1...N"""
        #     return NotImplemented

        # @property
        # def label(self):
        #     """String identifying the species (e.g. H, D, T, He, C, D2, DT, CD4, ...) {dynamic}  STR_0D  """
        #     return NotImplemented

        # @property
        # def ion_index(self):
        #     """Index of the corresponding ion species in the ../../ion array {dynamic}  INT_0D  """
        #     return NotImplemented

        # @property
        # def temperature(self):
        #     """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]  """
        #     return NotImplemented

        # @property
        # def density(self):
        #     """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        #     return NotImplemented

        # @property
        # def density_thermal(self):
        #     """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        #     return NotImplemented

        # @property
        # def density_fast(self):
        #     """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        #     return NotImplemented

        # @property
        # def pressure(self):
        #     """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        #     return NotImplemented

        # @property
        # def pressure_thermal(self):
        #     """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
        #     (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        #     return NotImplemented

        # @property
        # def pressure_fast_perpendicular(self):
        #     """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        #     return NotImplemented

        # @property
        # def pressure_fast_parallel(self):
        #     """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        #     return NotImplemented

        # @property
        # def multiple_states_flag(self):
        #     """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the
        #     state structure {dynamic}  INT_0D  """
        #     return NotImplemented

        # @property
        # def state(self):
        #     """Quantities related to the different states of the species (energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
        #     return NotImplemented

    @cached_property
    def electrons(self) -> Electrons:
        """Quantities related to the electrons"""
        return CoreProfiles1D.Electrons(self["electrons"], axis=self._axis, parent=self)

    @cached_property
    def ion(self) -> List[Ion]:
        """Quantities related to the different ion species"""
        return List[CoreProfiles1D.Ion](self["ion"],  parent=self)

    @cached_property
    def neutral(self) -> List[Neutral]:
        """Quantities related to the different neutral species"""
        return List[CoreProfiles1D.Neutral](self["neutral"],  parent=self)

    @cached_property
    def t_i_average(self):
        """Ion temperature(averaged on charge states and ion species) {dynamic}[eV]"""
        return Function(self._axis, self["t_i_average"])

    @cached_property
    def t_i_average_fit(self):
        """Information on the fit used to obtain the t_i_average profile[eV]"""
        return Function(self._axis, self["t_i_average_fit"])

    @cached_property
    def n_i_total(self):
        """ total ion density(sum over species and charge states)   (thermal+non-thermal) {dynamic}[-]"""
        res = Function(self._axis, 0.0)
        for ion in self.ion:
            res += ion.z_ion*(ion.density_thermal+ion.density_fast)
        return res

    @cached_property
    def n_i_total_over_n_e(self):
        """Ratio of total ion density(sum over species and charge states) over electron density. (thermal+non-thermal) {dynamic}[-]"""
        return self.n_i_total/self.electrons.density

    @cached_property
    def n_i_thermal_total(self):
        """Total ion thermal density(sum over species and charge states) {dynamic}[m ^ -3]"""
        res = Function(self._axis, 0.0)
        for ion in self.ion:
            res += ion.z_ion * ion.density_thermal
        return res

    @cached_property
    def zeff(self):
        """Effective charge {dynamic}[-]"""
        res = Function(self._axis, 0.0)
        for ion in self.ion:
            res += ion.z_ion * ion.z_ion * ion.density
        return res/self.n_i_total

    @cached_property
    def zeff_fit(self):
        """Information on the fit used to obtain the zeff profile[-]  """
        return Function(self._axis, 0.0)

    @cached_property
    def momentum_tor(self):
        """Total plasma toroidal momentum, summed over ion species and electrons weighted by their density and major radius,
            i.e. sum_over_species(n*R*m*Vphi) {dynamic}[kg.m ^ -1.s ^ -1]"""
        return NotImplemented

    @cached_property
    def pressure_ion_total(self):
        """Total(sum over ion species) thermal ion pressure {dynamic}[Pa]"""
        return np.sum([ion.pressure for ion in self.ion])

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
        return self.current_parallel_inside.derivative * self._r0*TWOPI/self.grid.dvolume_drho_tor

    @cached_property
    def current_parallel_inside(self) -> Function:
        """Parallel current driven inside the flux surface. Cumulative surface integral of j_total {dynamic}[A]"""
        return self.grid.dpsi_drho_tor

    @cached_property
    def j_tor(self):
        """Total toroidal current density = average(J_Tor/R) / average(1/R) {dynamic}[A/m ^ 2]"""
        return Function(self._axis, ["j_tor"])

    @cached_property
    def j_ohmic(self):
        """Ohmic parallel current density = average(J_Ohmic.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        return Function(self._axis, ["j_ohmic"])

    @cached_property
    def j_non_inductive(self):
        """Non-inductive(includes bootstrap) parallel current density = average(jni.B) / B0,
        where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        return Function(self._axis, ["j_non_inductive"])

    @cached_property
    def j_bootstrap(self):
        """Bootstrap current density = average(J_Bootstrap.B) / B0,
            where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        return Function(self._axis, self["j_bootstrap"])

    @cached_property
    def conductivity_parallel(self):
        """Parallel conductivity {dynamic}[ohm ^ -1.m ^ -1]"""
        Te = self.electrons.temperature
        ne = self.electrons.density

        # Electron collisions: Coulomb logarithm
        clog = np.asarray([
            (24.0 - 1.15*np.log10(ne[idx]*1.0e-6) + 2.30*np.log10(Te[idx]))
            if Te[idx] >= 10 else (23.0 - 1.15*np.log10(ne[idx]*1.0e-6) + 3.45*np.log10(Te[idx]))
            for idx in range(len(ne))
        ])
        # electron collision time:
        # tau_e = (np.sqrt(2.*scipy.constants.electron_mass)*(Te**1.5)) / 1.8e-19 / (ne * 1.0e-6) / clog

        # Plasma electrical conductivity:
        return 1.96e0 * scipy.constants.elementary_charge**2   \
            * ((np.sqrt(2.*scipy.constants.electron_mass)*(Te**1.5)) / 1.8e-19 / clog) \
            / scipy.constants.m_e

    class EField(Profiles):
        def __init__(self,   *args, axis=None, parent=None, **kwargs):
            super().__init__(*args, axis=axis if axis is not None else parent._axis,  **kwargs)

        @cached_property
        def parallel(self):
            return Function(self._axis, self["parallel"])

    @cached_property
    def e_field(self):
        """Electric field, averaged on the magnetic surface. E.g for the parallel component, average(E.B) / B0,
            using core_profiles/vacuum_toroidal_field/b0[V.m ^ -1]  """
        return CoreProfiles1D.EField(self["e_field"], axis=self._axis, parent=self)

    @cached_property
    def phi_potential(self):
        """Electrostatic potential, averaged on the magnetic flux surface {dynamic}[V]"""
        return Function(self._axis, self["phi_potential"])

    @cached_property
    def rotation_frequency_tor_sonic(self):
        """Derivative of the flux surface averaged electrostatic potential with respect to the poloidal flux, multiplied by - 1.
        This quantity is the toroidal angular rotation frequency due to the ExB drift, introduced in formula(43) of Hinton and Wong,
        Physics of Fluids 3082 (1985), also referred to as sonic flow in regimes in which the toroidal velocity is dominant over the
        poloidal velocity Click here for further documentation. {dynamic}[s ^ -1]"""
        return Function(self._axis, self["rotation_frequency_tor_sonic"])

    @cached_property
    def q(self):
        """Safety factor(IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic}[-].
        This quantity is COCOS-dependent, with the following transformation: """
        return Function(self._axis, self["q"])

    @cached_property
    def magnetic_shear(self):
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic}[-]"""
        return Function(self._axis, self["magnetic_shear"])


class CoreProfilesGlobalQuantities(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)


class CoreProfilesTimeSlice(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def profiles_1d(self) -> CoreProfiles1D:
        return CoreProfiles1D(self["profiles_1d"], parent=self)

    @cached_property
    def global_quantities(self) -> CoreProfilesGlobalQuantities:
        return CoreProfilesGlobalQuantities(self["global_quantities"], parent=self)


class CoreProfiles(IDS):
    """CoreProfiles
    """
    _IDS = "core_profiles"
    TimeSlice = CoreProfilesTimeSlice

    def __init__(self,  *args,  grid: RadialGrid = None, ** kwargs):
        super().__init__(*args,  ** kwargs)
        self._grid = grid

    @cached_property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        vacuum_toroidal_field = self["vacuum_toroidal_field"]
        if vacuum_toroidal_field == None:
            vacuum_toroidal_field = self._grid.vacuum_toroidal_field
        else:
            vacuum_toroidal_field = VacuumToroidalField(* vacuum_toroidal_field ._as_dict())
        return vacuum_toroidal_field

    @cached_property
    def time_slice(self) -> TimeSeries[TimeSlice]:
        return TimeSeries[CoreProfilesTimeSlice](self["time_slice"], parent=self, grid=self._grid)

        # AoS({
        #     "profiles_1d": self["profiles_1d"],
        #     "global_quantities": self["global_quantities"]
        # })

    @cached_property
    def profiles_1d(self) -> List[CoreProfiles1D]:
        return self.time_slice.to_aos("profiles_1d")

    @cached_property
    def global_quantities(self) -> CoreProfilesGlobalQuantities:
        return self.time_slice.to_soa("global_quantities")
