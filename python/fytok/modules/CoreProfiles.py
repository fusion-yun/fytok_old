
import numpy as np
from fytok._imas.core_profiles import (_T_core_profiles,
                                       _T_core_profiles_profiles_1d)
from fytok._imas.utilities import (_T_core_profile_ions,
                                   _T_core_profiles_profiles_1d_electrons,
                                   _T_core_profiles_vector_components_1)
from scipy import constants
from spdm.data.List import List
from spdm.data.Profile import Profile
from spdm.data.sp_property import sp_property, SpPropertyClass
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.utils.logger import logger

from .Utilities import CoreRadialGrid


class CoreProfilesElectrons(_T_core_profiles_profiles_1d_electrons):

    @sp_property
    def density(self) -> Profile[float]:
        return self.density_thermal+self.density_fast

    @sp_property
    def pressure(self) -> Profile[float]:

        return self.pressure_thermal+self.pressure_fast_parallel+self.pressure_fast_perpendicular

    @sp_property
    def pressure_thermal(self) -> Profile[float]:
        return self.density*self.temperature*constants.electron_volt

    @sp_property
    def tau(self) -> Profile[float]:
        return 1.09e16*((self.temperature/1000)**(3/2))/self.density/self._parent.coulomb_logarithm

    @sp_property
    def vT(self) -> Profile[float]:
        return np.sqrt(self.temperature*constants.electron_volt/constants.electron_mass)


class CoreProfilesIon(_T_core_profile_ions):

    @sp_property
    def z_ion_1d(self) -> Profile[float]:
        return super().z_ion_1d

    @sp_property
    def z_ion_square_1d(self) -> Profile[float]:

        return self.z_ion*self.z_ion

    @sp_property
    def density(self) -> Profile[float]:
        return self.density_fast+self.density_thermal

    @sp_property
    def density_flux(self) -> Profile[float]:
        return self.density_fast_flux + self.density_thermal_flux

    @sp_property
    def pressure(self) -> Profile[float]:
        return self.pressure_thermal+self.pressure_fast_parallel+self.pressure_fast_perpendicular

    @sp_property
    def pressure_thermal(self) -> Profile[float]:
        return self.density_thermal*self.temperature*constants.electron_volt


class CoreProfiles1D(_T_core_profiles_profiles_1d):

    grid: CoreRadialGrid = sp_property()

    electrons: CoreProfilesElectrons = sp_property()

    ion: List[CoreProfilesIon] = sp_property()

    @sp_property
    def t_i_average(self) -> Profile[float]:
        return sum([ion.z_ion_1d*ion.temperature*ion.density for ion in self.ion]) / self.n_i_total

    @sp_property
    def n_i_total(self) -> Profile[float]:
        return sum([(ion.z_ion_1d * ion.density) for ion in self.ion])

    @sp_property
    def n_i_total_over_n_e(self) -> Profile[float]:
        return self.n_i_total/self.electrons.density

    @sp_property
    def n_i_thermal_total(self) -> Profile[float]:
        return sum([ion.z*ion.density_thermal for ion in self.ion])

    @sp_property
    def zeff(self, value) -> Profile[float]:
        if value is not None:
            return value
        else:
            return sum([((ion.z_ion_1d**2)*ion.density) for ion in self.ion]) / self.n_i_total

    @sp_property
    def pressure(self) -> Profile[float]:
        return np.sum([ion.pressure for ion in self.ion])

    @sp_property
    def pressure_thermal(self) -> Profile[float]:
        return sum([ion.pressure_thermal for ion in self.ion])+self.electrons.pressure_thermal

    @sp_property
    def j_total(self) -> Profile[float]:
        return self.current_parallel_inside.derivative * \
            self.grid.r0*TWOPI/self.grid.dvolume_drho_tor

    @sp_property
    def current_parallel_inside(self, value) -> Profile[float]:
        return value

    @sp_property
    def j_non_inductive(self) -> Profile[float]:
        return self.j_total - self.j_ohmic

    @sp_property
    def conductivity_parallel(self) -> Profile[float]:
        return self.j_ohmic/self.e_field.parallel

    @sp_property
    def beta_pol(self) -> Profile[float]:
        return 4*self.pressure.antiderivative()/(self.grid.r0*constants.mu_0 * (self.j_total**2))

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
    def coulomb_logarithm(self) -> Profile[float]:
        """ Coulomb logarithm,
            @ref: Tokamaks 2003  Ch.14.5 p727 ,2003
        """
        Te = self.electrons.temperature(self.grid.rho_tor_norm)
        Ne = self.electrons.density(self.grid.rho_tor_norm)

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003

        return ((14.9 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te < 10) +
                (15.2 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te >= 10))

    @sp_property
    def electron_collision_time(self) -> Profile[float]:
        """ electron collision time ,
            @ref: Tokamak 2003, eq 14.6.1
        """
        Te = self.electrons.temperature(self.grid.rho_tor_norm)
        Ne = self.electrons.density(self.grid.rho_tor_norm)
        lnCoul = self.coulomb_logarithm(self.grid.rho_tor_norm)
        return 1.09e16*((Te/1000.0)**(3/2))/Ne/lnCoul

    class EField(_T_core_profiles_vector_components_1):

        @sp_property
        def parallel(self) -> Profile[float]:
            e_par = self.get("parallel", None)
            if e_par is None:
                vloop = self._parent.get("vloop", None)
                if vloop is None:
                    logger.error(f"Can not calculate E_parallel from vloop!")
                    e_par = 0.0
                else:
                    e_par = vloop/(TWOPI*self.grid.r0)
            return e_par

    @sp_property
    def e_field(self) -> EField:
        return self.get("e_field", {})

    @sp_property
    def magnetic_shear(self, value) -> Profile[float]:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic}[-]"""
        return self.q.derivative(self.grid.rho_tor_norm)/self.q(self.grid.rho_tor_norm)*self.grid.rho_tor_norm


class CoreProfiles(_T_core_profiles):

    Profiles1D = CoreProfiles1D

    profiles_1d: TimeSeriesAoS[Profiles1D] = sp_property()

    def update(self, *args,  **kwargs) -> None:
        super().update(*args, **kwargs)
