import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.constants
from spdm.data.Function import Function
from spdm.data.Node import Dict, List
from spdm.data.Profiles import Profiles
from spdm.data.Node import sp_property
from spdm.util.logger import logger

from ..common.IDS import IDS
from ..common.Misc import VacuumToroidalField
from ..common.Species import Species, SpeciesElectron, SpeciesIon
from .MagneticCoordSystem import TWOPI, RadialGrid


class CoreProfilesElectrons(SpeciesElectron):
    def __init__(self,   *args,  **kwargs):
        super().__init__(*args,    **kwargs)

    @sp_property
    def temperature(self) -> Function:
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
    # @sp_property
    # def temperature_fit(self):
    #     """Information on the fit used to obtain the temperature profile [eV]  """
    #     return NotImplemented
    @sp_property
    def density(self) -> Function:
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
    # @sp_property
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

    @sp_property
    def pressure(self) -> Function:
        """Pressure(thermal+non-thermal) {dynamic}[Pa]"""
        return self.density*self.temperature*scipy.constants.electron_volt

        # if self.pressure_fast_perpendicular is not NotImplemented:
        #     return self.pressure_thermal+self.pressure_fast_perpendicular+self.pressure_fast_parallel
        # else:
        #     return self.pressure_thermal

    @sp_property
    def pressure_thermal(self) -> Function:
        """Pressure(thermal) associated with random motion ~average((v-average(v)) ^ 2) {dynamic}[Pa]"""
        return self.density*self.temperature*scipy.constants.electron_volt

    @sp_property
    def pressure_fast_perpendicular(self) -> Function:
        """Fast(non-thermal) perpendicular pressure {dynamic}[Pa]"""
        return NotImplemented

    @sp_property
    def pressure_fast_parallel(self) -> Function:
        """Fast(non-thermal) parallel pressure {dynamic}[Pa]"""
        return NotImplemented

    @sp_property
    def collisionality_norm(self) -> Function:
        """Collisionality normalised to the bounce frequency {dynamic}[-]"""
        return NotImplemented


class CoreProfilesIon(SpeciesIon):
    def __init__(self,   *args,   **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def z_ion_1d(self):
        d = self.get("z_ion_id", None)
        if isinstance(d, np.ndarray):
            return d
        else:
            return Function(self._axis, self.z_ion)

    @sp_property
    def z_ion_square_1d(self):
        d = self.get("z_ion_square_1d", None)
        if isinstance(d, np.ndarray):
            return d
        else:
            return Function(self._axis, self.z_ion*self.z_ion)

    @sp_property
    def temperature(self) -> Function:
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

    # @sp_property
    # def temperature_fit(self):
    #     """Information on the fit used to obtain the temperature profile [eV]    """
    #     return NotImplemented

    @sp_property
    def density(self) -> Function:
        """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        d = self["density"]
        if not isinstance(d, np.ndarray) or d != None:
            return d
        else:
            return self.density_fast+self.density_thermal

    # @property
    # def density_validity(self):
    #     """Indicator of the validity of the density profile.
    #      0: valid from automated processing,
    #      1: valid and certified by the RO;
    #      - 1 means problem identified in the data processing (request verification by the RO),
    #      -2: invalid data, should not be used {dynamic}    """
    #     return NotImplemented

    # @sp_property
    # def density_fit(self):
    #     """Information on the fit used to obtain the density profile [m^-3]    """
    #     return NotImplemented

    @property
    def density_thermal(self) -> Function:
        """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        return self["density_thermal"]

    @property
    def density_fast(self) -> Function:
        """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        return self["density_fast"]

    @sp_property
    def pressure(self) -> Function:
        """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        # if self.pressure_fast_perpendicular is not NotImplemented:
        #     return self.pressure_thermal+self.pressure_fast_perpendicular+self.pressure_fast_parallel
        # else:
        return self.density*self.temperature*scipy.constants.electron_volt

    @sp_property
    def pressure_thermal(self) -> Function:
        """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
        (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        return self.density_thermal*self.temperature*scipy.constants.electron_volt

    @sp_property
    def pressure_fast_perpendicular(self) -> Function:
        """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        return self["pressure_fast_perpendicular"]

    @sp_property
    def pressure_fast_parallel(self) -> Function:
        """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        return self["pressure_fast_parallel"]

    @sp_property
    def rotation_frequency_tor(self) -> Function:
        """Toroidal rotation frequency (i.e. toroidal velocity divided by the major radius at which the toroidal velocity is taken)
        (average over charge states when multiple charge states are considered) {dynamic} [rad.s^-1]  """
        return self["rotation_frequency_tor"]

    @sp_property
    def velocity(self) -> Function:
        """Velocity (average over charge states when multiple charge states are considered) at the position of maximum major
        radius on every flux surface [m.s^-1]    """
        return self["velocity"]


class CoreProfilesNeutral(Species):
    def __init__(self,   *args, axis=None, parent=None, **kwargs):
        super().__init__(*args, axis=axis if axis is not None else parent.grid.rho_tor_norm,  **kwargs)

    @property
    def ion_index(self):
        """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
        return self.__raw_get__("ion_index")

    @property
    def element(self):
        """List of elements forming the atom or molecule  struct_array [max_size=unbounded]  1- 1...N"""
        return NotImplemented

    @property
    def label(self):
        """String identifying the species (e.g. H, D, T, He, C, D2, DT, CD4, ...) {dynamic}  STR_0D  """
        return NotImplemented

    @property
    def ion_index(self):
        """Index of the corresponding ion species in the ../../ion array {dynamic}  INT_0D  """
        return NotImplemented

    @property
    def temperature(self):
        """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]  """
        return NotImplemented

    @property
    def density(self):
        """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        return NotImplemented

    @property
    def density_thermal(self):
        """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        return NotImplemented

    @property
    def density_fast(self):
        """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        return NotImplemented

    @property
    def pressure(self):
        """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        return NotImplemented

    @property
    def pressure_thermal(self):
        """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
        (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        return NotImplemented

    @property
    def pressure_fast_perpendicular(self):
        """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        return NotImplemented

    @property
    def pressure_fast_parallel(self):
        """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        return NotImplemented

    @property
    def multiple_states_flag(self):
        """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the
        state structure {dynamic}  INT_0D  """
        return NotImplemented

    @property
    def state(self):
        """Quantities related to the different states of the species (energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
        return NotImplemented


class CoreProfiles1D(Profiles):
    def __init__(self, *args, grid: RadialGrid = None, time=None, parent=None, **kwargs):
        grid = grid or getattr(parent, "_grid", None)
        assert(grid is not None)
        super().__init__(*args, axis=grid.rho_tor_norm, parent=parent, **kwargs)
        self._grid = grid
        self._r0 = self._grid.vacuum_toroidal_field.r0
        self._b0 = self._grid.vacuum_toroidal_field.b0

    @property
    def grid(self) -> RadialGrid:
        return self._grid

    @sp_property
    def electrons(self) -> CoreProfilesElectrons:
        """Quantities related to the electrons"""
        return CoreProfilesElectrons(self["electrons"], axis=self._axis, parent=self)

    @sp_property
    def ion(self) -> List[CoreProfilesIon]:
        """Quantities related to the different ion species"""
        return List[CoreProfilesIon](self["ion"],  parent=self)

    @sp_property
    def neutral(self) -> List[CoreProfilesNeutral]:
        """Quantities related to the different neutral species"""
        return List[CoreProfilesNeutral](self["neutral"],  parent=self)

    @sp_property
    def t_i_average(self):
        """Ion temperature(averaged on charge states and ion species) {dynamic}[eV]"""
        return Function(self._axis, np.sum([np.asarray(ion.temperature*ion.density) for ion in self.ion])/np.sum([np.asarray(ion.density) for ion in self.ion]))

    @sp_property
    def t_i_average_fit(self):
        """Information on the fit used to obtain the t_i_average profile[eV]"""
        return Function(self._axis, self["t_i_average_fit"])

    @sp_property
    def n_i_total(self):
        """ total ion density(sum over species and charge states)   (thermal+non-thermal) {dynamic}[-]"""
        return Function(self._axis, np.sum([np.asarray(ion.z_ion*ion.density) for ion in self.ion]))

    @sp_property
    def n_i_total_over_n_e(self):
        """Ratio of total ion density(sum over species and charge states) over electron density. (thermal+non-thermal) {dynamic}[-]"""
        return self.n_i_total/self.electrons.density

    @sp_property
    def n_i_thermal_total(self):
        """Total ion thermal density(sum over species and charge states) {dynamic}[m ^ -3]"""
        return Function(self._axis, np.sum([np.asarray(ion.z_ion*ion.density_thermal) for ion in self.ion]))

    @sp_property
    def zeff(self):
        """Effective charge {dynamic}[-]"""
        d = self.get("zeff")
        if isinstance(d, np.ndarray):
            return d
        else:
            # zeff = 0.0
            # for ion in self.ion:
            #     zeff = zeff + np.asarray(ion.z_ion*ion.z_ion*ion.density)
            return sum([np.asarray(ion.z_ion*ion.z_ion*ion.density) for ion in self.ion]) / self.electrons.density

    @sp_property
    def zeff_fit(self):
        """Information on the fit used to obtain the zeff profile[-]  """
        return NotImplemented

    @sp_property
    def momentum_tor(self):
        """Total plasma toroidal momentum, summed over ion species and electrons weighted by their density and major radius,
            i.e. sum_over_species(n*R*m*Vphi) {dynamic}[kg.m ^ -1.s ^ -1]"""
        return NotImplemented

    @sp_property
    def pressure_ion_total(self):
        """Total(sum over ion species) thermal ion pressure {dynamic}[Pa]"""
        return np.sum([ion.pressure for ion in self.ion])

    @sp_property
    def pressure_thermal(self):
        """Thermal pressure(electrons+ions) {dynamic}[Pa]"""
        return NotImplemented

    @sp_property
    def pressure_perpendicular(self):
        """Total perpendicular pressure(electrons+ions, thermal+non-thermal) {dynamic}[Pa]"""
        return NotImplemented

    @sp_property
    def pressure_parallel(self):
        """Total parallel pressure(electrons+ions, thermal+non-thermal) {dynamic}[Pa]"""
        return NotImplemented

    @sp_property
    def j_total(self):
        """Total parallel current density = average(jtot.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        d = self["j_total"]
        if isinstance(d, np.ndarray) or d != None:
            return Function(self._axis, d)
        else:
            return self.current_parallel_inside.derivative * self._r0*TWOPI/self.grid.dvolume_drho_tor

    @sp_property
    def current_parallel_inside(self) -> Function:
        """Parallel current driven inside the flux surface. Cumulative surface integral of j_total {dynamic}[A]"""
        return Function(self._axis, self["current_parallel_inside"])

    @sp_property
    def j_tor(self):
        """Total toroidal current density = average(J_Tor/R) / average(1/R) {dynamic}[A/m ^ 2]"""
        d = self["j_tor"]
        if isinstance(d, np.ndarray) or d != None:
            return Function(self._axis, d)
        else:
            return NotImplemented

    @sp_property
    def j_ohmic(self):
        """Ohmic parallel current density = average(J_Ohmic.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        return self.conductivity_parallel*self.e_field.parallel  # Function(self._axis, ["j_ohmic"])

    @sp_property
    def j_non_inductive(self):
        """Non-inductive(includes bootstrap) parallel current density = average(jni.B) / B0,
        where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        return self.j_total - self.j_ohmic

    @sp_property
    def j_bootstrap(self):
        """Bootstrap current density = average(J_Bootstrap.B) / B0,
            where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        return Function(self._axis, self["j_bootstrap"])

    @sp_property
    def conductivity_parallel(self):
        """Parallel conductivity {dynamic}[ohm ^ -1.m ^ -1]"""

        d = self["conductivity_parallel"]

        if isinstance(d, np.ndarray) or d != None:
            return Function(self._axis, d)
        else:
            Te = self.electrons.temperature
            ne = self.electrons.density

            # Electron collisions: Coulomb logarithm
            # clog = np.asarray([
            #     (24.0 - 1.15*np.log10(ne[idx]*1.0e-6) + 2.30*np.log10(Te[idx]))
            #     if Te[idx] >= 10 else (23.0 - 1.15*np.log10(ne[idx]*1.0e-6) + 3.45*np.log10(Te[idx]))
            #     for idx in range(len(ne))
            # ])
            clog = self.coulomb_logarithm
            # electron collision time:
            # tau_e = (np.sqrt(2.*scipy.constants.electron_mass)*(Te**1.5)) / 1.8e-19 / (ne * 1.0e-6) / clog

            # Plasma electrical conductivity:
            return 1.96e0 * scipy.constants.elementary_charge**2   \
                * ((np.sqrt(2.*scipy.constants.electron_mass)*(Te**1.5)) / 1.8e-19 / clog) \
                / scipy.constants.m_e

    @sp_property
    def coulomb_logarithm(self):
        """ Coulomb logarithm, Tokamaks   Ch.14.5 p727 ,2003
        """
        Te = np.asarray(self.electrons.temperature)
        Ne = np.asarray(self.electrons.density)

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003
        lnCoul = (14.9 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te < 10) +\
            (15.2 - 0.5*np.log(Ne/1e20) + np.log(Te/1000))*(Te >= 10)

        return lnCoul

    class EField(Profiles):
        def __init__(self,   *args, axis=None, parent=None, **kwargs):
            super().__init__(*args, axis=axis if axis is not None else parent._axis,  **kwargs)

        @sp_property
        def parallel(self):
            return Function(self._axis, self["parallel"])

    @sp_property
    def e_field(self):
        """Electric field, averaged on the magnetic surface. E.g for the parallel component, average(E.B) / B0,
            using core_profiles/vacuum_toroidal_field/b0[V.m ^ -1]  """
        return CoreProfiles1D.EField(self["e_field"], axis=self._axis, parent=self)

    @sp_property
    def phi_potential(self):
        """Electrostatic potential, averaged on the magnetic flux surface {dynamic}[V]"""
        return Function(self._axis, self["phi_potential"])

    @sp_property
    def rotation_frequency_tor_sonic(self):
        """Derivative of the flux surface averaged electrostatic potential with respect to the poloidal flux, multiplied by - 1.
        This quantity is the toroidal angular rotation frequency due to the ExB drift, introduced in formula(43) of Hinton and Wong,
        Physics of Fluids 3082 (1985), also referred to as sonic flow in regimes in which the toroidal velocity is dominant over the
        poloidal velocity Click here for further documentation. {dynamic}[s ^ -1]"""
        return Function(self._axis, self["rotation_frequency_tor_sonic"])

    @sp_property
    def q(self):
        """Safety factor(IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic}[-].
        This quantity is COCOS-dependent, with the following transformation: """
        return self["q"]

    @sp_property
    def magnetic_shear(self):
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic}[-]"""
        return self["magnetic_shear"]


class CoreProfilesGlobalQuantities(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)


class CoreProfiles(IDS):
    """CoreProfiles
    """
    _IDS = "core_profiles"
    Profiles1D = CoreProfiles1D
    GlobalQuantities = CoreProfilesGlobalQuantities

    @dataclass
    class State(IDS.State):
        profiles_1d: CoreProfiles1D

    def __init__(self,  *args, grid: Optional[RadialGrid] = None, ** kwargs):
        super().__init__(*args,  ** kwargs)
        self._grid = grid or self._parent.grid

    @sp_property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        d = self["vacuum_toroidal_field"]
        if d == None:
            d = self._grid.vacuum_toroidal_field
        elif isinstance(d, Dict):
            d = VacuumToroidalField(**d._as_dict())
        elif not isinstance(d, VacuumToroidalField):
            raise TypeError(type(d))
        return d

    @property
    def grid(self) -> RadialGrid:
        return self._grid

    @sp_property
    def profiles_1d(self) -> Profiles1D:
        return self["profiles_1d"]

    @sp_property
    def global_quantities(self) -> GlobalQuantities:
        return self["global_quantities"]
