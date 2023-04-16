
import numpy as np
from scipy import constants
from spdm.common.tags import _not_found_
from spdm.data.Dict import Dict
from spdm.data.Function import Function, function_like
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.util.logger import logger

from ..common.IDS import IDS
from ..transport.MagneticCoordSystem import TWOPI, RadialGrid
from .Species import Species, SpeciesElectron, SpeciesIon


class CoreProfilesElectrons(SpeciesElectron):

    @property
    def grid(self) -> RadialGrid:
        return self._parent.grid

    temperature: Function = sp_property(default_value=0.0)
    """Temperature {dynamic} [eV]"""

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
    def density(self, value) -> Function:
        """Density (thermal+non-thermal) {dynamic} [m^-3]"""
        if value is not None:
            return function_like(self._parent.grid.rho_tor_norm, value)
        else:
            return self.density_thermal+self.density_fast
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

    density_thermal: Function = sp_property(default_value=0.0)
    """Density of thermal particles {dynamic} [m^-3]"""

    density_fast: Function = sp_property(default_value=0.0)
    """Density of fast (non-thermal) particles {dynamic} [m^-3]"""

    @sp_property
    def pressure(self, value) -> Function:
        """Pressure(thermal+non-thermal) {dynamic}[Pa]"""
        if value is not None:
            return function_like(self._parent.grid.rho_tor_norm, value)
        else:
            return self.pressure_thermal+self.pressure_fast_parallel+self.pressure_fast_perpendicular

    @sp_property
    def pressure_thermal(self, value) -> Function:
        """Pressure(thermal) associated with random motion ~average((v-average(v)) ^ 2) {dynamic}[Pa]"""
        if value is not None:
            return function_like(self._parent.grid.rho_tor_norm, value)
        else:
            return self.density*self.temperature*constants.electron_volt

    pressure_fast_perpendicular: Function = sp_property(default_value=0.0)
    """Fast(non-thermal) perpendicular pressure {dynamic}[Pa]"""

    pressure_fast_parallel: Function = sp_property(default_value=0.0)
    """Fast(non-thermal) parallel pressure {dynamic}[Pa]"""

    collisionality_norm: Function = sp_property(default_value=0.0)
    """Collisionality normalised to the bounce frequency {dynamic}[-]"""

    @sp_property
    def tau(self) -> Function:
        """electron collision time"""
        return 1.09e16*((self.temperature/1000)**(3/2))/self.density/self._parent.coulomb_logarithm

    @sp_property
    def vT(self) -> Function:
        return np.sqrt(self.temperature*constants.electron_volt/constants.electron_mass)


class CoreProfilesIon(SpeciesIon):

    
    @sp_property
    def z_ion_1d(self, value) -> Function:
        if value is not None:
            return function_like(self._parent.grid.rho_tor_norm, value)
        else:
            return function_like(self._parent.grid.rho_tor_norm, self.z)

    @sp_property
    def z_ion_square_1d(self, value) -> Function:
        if value is not None:
            return function_like(self._parent.grid.rho_tor_norm, value)
        else:
            return function_like(self._parent.grid.rho_tor_norm, self.z_ion*self.z_ion)

    temperature: Function = sp_property(default_value=0.0)
    """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]  """

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
    def density(self, value) -> Function:
        """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        if self.has_fast_particle:
            return self.density_fast+self.density_thermal
        else:
            return function_like(self._parent.grid.rho_tor_norm, value)

    @sp_property
    def density_flux(self, value) -> Function:
        """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """
        if self.has_fast_particle:
            return function_like(self._parent.grid.rho_tor_norm, self.get("density_fast_flux", 0)) + \
                function_like(self._parent.grid.rho_tor_norm, self.get("density_thermal_flux", 0))
        else:
            return function_like(self._parent.grid.rho_tor_norm, value)

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

    density_thermal: Function = sp_property(default_value=0.0)
    """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """

    density_fast: Function = sp_property(default_value=0.0)
    """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]  """

    @sp_property
    def pressure(self, value) -> Function:
        """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """
        if value is not None:
            return function_like(self.grid.rho_tor_norm, value)
        else:
            return self.pressure_thermal+self.pressure_fast_parallel+self.pressure_fast_perpendicular

    @sp_property
    def pressure_thermal(self, value) -> Function:
        """Pressure (thermal) associated with random motion ~average((v-average(v))^2)
        (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """

        if value is not None:
            return function_like(self._parent.grid.rho_tor_norm, value)
        else:
            return self.density_thermal*self.temperature*constants.electron_volt

    pressure_fast_perpendicular: Function = sp_property(default_value=0.0)
    """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """

    pressure_fast_parallel: Function = sp_property(default_value=0.0)
    """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]  """

    rotation_frequency_tor: Function = sp_property(default_value=0.0)
    """Toroidal rotation frequency (i.e. toroidal velocity divided by the major radius at which the toroidal velocity is taken)
        (average over charge states when multiple charge states are considered) {dynamic} [rad.s^-1]  """

    velocity: Function = sp_property(default_value=0.0)
    """Velocity (average over charge states when multiple charge states are considered) at the position of maximum major
        radius on every flux surface [m.s^-1]    """


class CoreProfilesNeutral(Species):
    def __init__(self,   *args,  **kwargs):
        super().__init__(*args,  **kwargs)

    @property
    def grid(self) -> RadialGrid:
        return self._parent.grid

    ion_index: int = sp_property(default_value=0)
    """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """

    element: List[Species] = sp_property()
    """List of elements forming the atom or molecule  structure_array [max_size=unbounded]  1- 1...N"""

    label: str = sp_property()
    """String identifying the species (e.g. H, D, T, He, C, D2, DT, CD4, ...) {dynamic}  STR_0D  """

    ion_index: int = sp_property()
    """Index of the corresponding ion species in the ../../ion array {dynamic}  INT_0D  """

    temperature: float = sp_property()
    """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]  """

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


class CoreProfiles1D(Dict[Node]):
    Electrons = CoreProfilesElectrons
    Ion = CoreProfilesIon
    Neutral = CoreProfilesNeutral

    grid: RadialGrid = sp_property()

    electrons: Electrons = sp_property()
    """Quantities related to the electrons"""

    ion: List[Ion] = sp_property()
    """Quantities related to the different ion species"""

    neutral: List[Ion] = sp_property()
    """Quantities related to the different neutral species"""

    @sp_property
    def t_i_average(self) -> Function:
        """Ion temperature(averaged on charge states and ion species) {dynamic}[eV]"""
        return sum([ion.z_ion_1d*ion.temperature*ion.density for ion in self.ion]) / self.n_i_total

    @sp_property
    def t_i_average_fit(self) -> Function:
        """Information on the fit used to obtain the t_i_average profile[eV]"""
        return NotImplemented

    @sp_property
    def n_i_total(self) -> Function:
        """ total ion density(sum over species and charge states)   (thermal+non-thermal) {dynamic}[-]"""
        return sum([(ion.z_ion_1d * ion.density) for ion in self.ion])

    @sp_property
    def n_i_total_over_n_e(self) -> Function:
        """Ratio of total ion density(sum over species and charge states) over electron density. (thermal+non-thermal) {dynamic}[-]"""
        return self.n_i_total/self.electrons.density

    @sp_property
    def n_i_thermal_total(self) -> Function:
        """Total ion thermal density(sum over species and charge states) {dynamic}[m ^ -3]"""
        return sum([ion.z*ion.density_thermal for ion in self.ion])

    @sp_property
    def zeff(self, value) -> Function:
        """Effective charge {dynamic}[-]"""
        if value is not None:
            return value
        else:
            return sum([((ion.z_ion_1d**2)*ion.density) for ion in self.ion]) / self.n_i_total

    @sp_property
    def zeff_fit(self) -> Function:
        """Information on the fit used to obtain the zeff profile[-]  """
        return NotImplemented

    @sp_property
    def momentum_tor(self) -> Function:
        """Total plasma toroidal momentum, summed over ion species and electrons weighted by their density and major radius,
            i.e. sum_over_species(n*R*m*Vphi) {dynamic}[kg.m ^ -1.s ^ -1]"""
        return NotImplemented

    @sp_property
    def pressure(self) -> Function:
        """Total(sum over ion species) thermal ion pressure {dynamic}[Pa]"""
        return np.sum([ion.pressure for ion in self.ion])

    @sp_property
    def pressure_thermal(self) -> Function:
        """Thermal pressure(electrons+ions) {dynamic}[Pa]"""
        return sum([ion.pressure_thermal for ion in self.ion])+self.electrons.pressure_thermal

    @sp_property
    def pressure_perpendicular(self) -> Function:
        """Total perpendicular pressure(electrons+ions, thermal+non-thermal) {dynamic}[Pa]"""
        return NotImplemented

    @sp_property
    def pressure_parallel(self) -> Function:
        """Total parallel pressure(electrons+ions, thermal+non-thermal) {dynamic}[Pa]"""
        return NotImplemented

    @sp_property
    def j_total(self, value) -> Function:
        """Total parallel current density = average(jtot.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        if value is None:
            value = self.current_parallel_inside.derivative * \
                self._parent.grid.r0*TWOPI/self._parent.grid.dvolume_drho_tor
        return Function(self._parent.grid.rho_tor_norm, value)

    @sp_property
    def current_parallel_inside(self, value) -> Function:
        """Parallel current driven inside the flux surface. Cumulative surface integral of j_total {dynamic}[A]"""
        return function_like(self._parent.grid.rho_tor_norm, value)

    j_tor: Function = sp_property()
    """Total toroidal current density = average(J_Tor/R) / average(1/R) {dynamic}[A/m ^ 2]"""

    j_ohmic: Function = sp_property()
    """Ohmic parallel current density = average(J_Ohmic.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""

    @sp_property
    def j_non_inductive(self) -> Function:
        """Non-inductive(includes bootstrap) parallel current density = average(jni.B) / B0,
        where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        return self.j_total - self.j_ohmic

    @sp_property
    def j_bootstrap(self, value) -> Function:
        """Bootstrap current density = average(J_Bootstrap.B) / B0,
            where B0 = Core_Profiles/Vacuum_Toroidal_Field / B0 {dynamic}[A/m ^ 2]"""
        return Function(self._parent.grid.rho_tor_norm, value)

    @sp_property
    def conductivity_parallel(self, value) -> Function:
        """Parallel conductivity {dynamic}[ohm ^ -1.m ^ -1]"""
        if value is _not_found_:
            value = self.j_ohmic/self.e_field.parallel
        return Function(self._parent.grid.rho_tor_norm, value)

    beta_pol: Function = sp_property()
    """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""

    # if isinstance(d, np.ndarray) or (hasattr(d.__class__, 'empty') and not d.empty):
    #     return d

    # else:
    #     Te = self.electrons.temperature
    #     ne = self.electrons.density

    #     # Electron collisions: Coulomb logarithm
    #     # clog = np.asarray([
    #     #     (24.0 - 1.15*np.log10(ne[idx]*1.0e-6) + 2.30*np.log10(Te[idx]))
    #     #     if Te[idx] >= 10 else (23.0 - 1.15*np.log10(ne[idx]*1.0e-6) + 3.45*np.log10(Te[idx]))
    #     #     for idx in range(len(ne))
    #     # ])
    #     clog = self.coulomb_logarithm
    #     # electron collision time:
    #     # tau_e = (np.sqrt(2.*constants.electron_mass)*(Te**1.5)) / 1.8e-19 / (ne * 1.0e-6) / clog

    #     # Plasma electrical conductivity:
    #     return 1.96e0 * constants.elementary_charge**2   \
    #         * ((np.sqrt(2.*constants.electron_mass)*(Te**1.5)) / 1.8e-19 / clog) \
    #         / constants.m_e

    @sp_property
    def coulomb_logarithm(self) -> Function:
        """ Coulomb logarithm,
            @ref: Tokamaks 2003  Ch.14.5 p727 ,2003
        """
        Te = self.electrons.temperature(self.grid.rho_tor_norm)
        Ne = self.electrons.density(self.grid.rho_tor_norm)

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003

        return Function(self.grid.rho_tor_norm, ((14.9 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te < 10) +
                                                         (15.2 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te >= 10)))

    @sp_property
    def electron_collision_time(self) -> Function:
        """ electron collision time ,
            @ref: Tokamak 2003, eq 14.6.1
        """
        Te = self.electrons.temperature(self.grid.rho_tor_norm)
        Ne = self.electrons.density(self.grid.rho_tor_norm)
        lnCoul = self.coulomb_logarithm(self.grid.rho_tor_norm)
        return 1.09e16*((Te/1000.0)**(3/2))/Ne/lnCoul

    class EField(Dict[Node]):
        def __init__(self,   *args,     **kwargs):
            super().__init__(*args, **kwargs)

        @sp_property
        def parallel(self) -> Function:
            e_par = self.get("parallel", _not_found_)
            if e_par is _not_found_:
                vloop = self._parent.get("vloop", _not_found_)
                if vloop is _not_found_:
                    logger.error(f"Can not calculate E_parallel from vloop!")
                    e_par = 0.0
                else:
                    e_par = vloop/(TWOPI*self._parent.grid.r0)
            return Function(self._parent.grid.rho_tor_norm, e_par)

        @sp_property
        def radial(self) -> Function:
            return Function(self._parent.grid.rho_tor_norm, self.get("radial", 0))

        @sp_property
        def diamagnetic(self) -> Function:
            return Function(self._parent.grid.rho_tor_norm, self.get("diamagnetic", 0))

        @sp_property
        def poloidal(self) -> Function:
            return Function(self._parent.grid.rho_tor_norm, self.get("poloidal", 0))

        @sp_property
        def toroidal(self) -> Function:
            return Function(self._parent.grid.rho_tor_norm, self.get("toroidal", 0))

    @sp_property
    def e_field(self) -> EField:
        """Electric field, averaged on the magnetic surface. E.g for the parallel component, average(E.B) / B0,
            using core_profiles/vacuum_toroidal_field/b0[V.m ^ -1]  """
        return self.get("e_field", {})

    @sp_property
    def phi_potential(self) -> Function:
        """Electrostatic potential, averaged on the magnetic flux surface {dynamic}[V]"""
        return Function(self._parent.grid.rho_tor_norm, self.get("phi_potential"))

    @sp_property
    def rotation_frequency_tor_sonic(self) -> Function:
        """Derivative of the flux surface averaged electrostatic potential with respect to the poloidal flux, multiplied by - 1.
        This quantity is the toroidal angular rotation frequency due to the ExB drift, introduced in formula(43) of Hinton and Wong,
        Physics of Fluids 3082 (1985), also referred to as sonic flow in regimes in which the toroidal velocity is dominant over the
        poloidal velocity Click here for further documentation. {dynamic}[s ^ -1]"""
        return Function(self._parent.grid.rho_tor_norm, self.get("rotation_frequency_tor_sonic", 0))

    @sp_property
    def q(self) -> Function:
        """Safety factor(IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic}[-].
        This quantity is COCOS-dependent, with the following transformation: """
        return Function(self._parent.grid.rho_tor_norm, self.get("q"))

    @sp_property
    def magnetic_shear(self) -> Function:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic}[-]"""
        return Function(self._parent.grid.rho_tor_norm, self.q.derivative(self._parent.grid.rho_tor_norm)/self.q(self._parent.grid.rho_tor_norm)*self._parent.grid.rho_tor_norm)
        # return Function(self._parent.grid.rho_tor_norm, self.get("magnetic_shear"))


class CoreProfilesGlobalQuantities(Dict):
    def __init__(self,   *args,   **kwargs):
        super().__init__(*args,  **kwargs)

    @property
    def grid(self) -> RadialGrid:
        return self._parent.grid


class CoreProfiles(IDS):
    """CoreProfiles
    """
    _IDS = "core_profiles"

    Profiles1D = CoreProfiles1D

    GlobalQuantities = CoreProfilesGlobalQuantities

    profiles_1d: Profiles1D = sp_property()

    global_quantities: GlobalQuantities = sp_property()

    def refresh(self, *args,  **kwargs) -> None:
        super().refresh(*args, **kwargs)
