from __future__ import annotations


import numpy as np
import scipy.constants
from scipy import constants
from spdm.data.AoS import AoS
from spdm.data.Function import Function
from spdm.data.Expression import Expression
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.utils.tree_utils import merge_tree_recursive

from ..utils.atoms import atoms
from ..utils.logger import logger

from .Utilities import *
from ..ontology import core_profiles, utilities

PI = scipy.constants.pi
TWOPI = 2.0 * PI


@sp_tree(coordinate1="../grid/rho_tor_norm")
class CoreProfilesIon(utilities._T_core_profile_ions):

    _metadata = {"identifier": "label"}

    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, **kwargs)

        atom_desc = atoms.get(self.label.capitalize(), None)

        self._cache = merge_tree_recursive(self._cache, atom_desc)

    is_impurity: bool = sp_property(default_value=False)

    has_fast_particle: bool = sp_property(default_value=False)

    element: AoS[PlasmaCompositionNeutralElement]

    z_ion: float = sp_property(units="Elementary Charge Unit")

    label: str

    neutral_index: int

    z: float

    a: float

    z_ion_1d: Function = sp_property(units="-")

    z_ion_square_1d: Function = sp_property(units="-")

    temperature: Function = sp_property(units="eV", default_value=0.0)

    @sp_property(units="m^-3")
    def density(self) -> Function: return self.density_thermal+self.density_fast

    density_thermal: Function = sp_property(units="m^-3")
    density_fast: Function = sp_property(units="m^-3")

    @sp_property(units="Pa")
    def pressure(self) -> Function:
        # FIXME: coefficient on pressure fast
        return self.pressure_thermal + \
            self.pressure_fast_perpendicular + \
            self.pressure_fast_parallel

    @sp_property(units="Pa")
    def pressure_thermal(self) -> Function:
        return self.density_thermal*self.temperature*scipy.constants.electron_volt

    pressure_fast_perpendicular: Function = sp_property(units="Pa", default=0.0)
    pressure_fast_parallel: Function = sp_property(units="Pa", default=0.0)

    rotation_frequency_tor: Function = sp_property(units="rad.s^-1")

    # velocity: _T_core_profiles_vector_components_2 = sp_property(units="m.s^-1")

    multiple_states_flag: int

    @property
    def mass(self) -> float: return self.element[0].a

    @property
    def charge(self) -> float: return self.element[0].z

    @sp_property
    def z_ion_1d(self) -> Function:
        return super().z_ion_1d

    @sp_property
    def z_ion_square_1d(self) -> Function: return self.z_ion * self.z_ion

    @sp_property(units="m^-3")
    def density(self) -> Function: return self.density_thermal + self.density_fast

    density_thermal: Function = sp_property(default_value=0.0)

    density_fast: Function = sp_property(default_value=0.0)

    @sp_property
    def pressure_thermal(self) -> Function:
        return self.density_thermal * self.temperature * scipy.constants.electron_volt

    pressure_fast_perpendicular: Function = sp_property(default_value=0.0)

    pressure_fast_parallel: Function = sp_property(default_value=0.0)

    @sp_property
    def pressure(self) -> Function:
        return (
            self.pressure_thermal.__array__() +
            self.pressure_fast_parallel.__array__() +
            self.pressure_fast_perpendicular.__array__()
        )


@sp_tree(coordinate1="../grid/rho_tor_norm")
class CoreProfilesNeutral(utilities._T_core_profile_neutral):
    _metadata = {"identifier": "label"}

    label: str

    ion_index: int

    element: AoS[PlasmaCompositionNeutralElement]

    temperature: Function = sp_property(units="eV")

    density: Function = sp_property(units="m^-3")

    density_thermal: Function = sp_property(units="m^-3")

    density_fast: Function = sp_property(units="m^-3")

    pressure: Function = sp_property(units="Pa")

    pressure_thermal: Function = sp_property(units="Pa")

    pressure_fast_perpendicular: Function = sp_property(units="Pa")

    pressure_fast_parallel: Function = sp_property(units="Pa")

    multiple_states_flag: int

    # state: AoS[_T_core_profiles_neutral_state]


@sp_tree(coordinate1="../grid/rho_tor_norm")
class CoreProfilesElectrons(utilities._T_core_profiles_profiles_1d_electrons):

    @sp_property
    def tau(self):
        return (1.09e16 * ((self.temperature / 1000) ** (3 / 2)) / self.density / self._parent.coulomb_logarithm)

    @sp_property
    def vT(self):
        return np.sqrt(self.temperature * scipy.constants.electron_volt / scipy.constants.electron_mass)

    temperature: Function = sp_property(units="eV")

    @sp_property(units="m^-3")
    def density(self) -> Function: return self.density_thermal + self.density_fast

    density_thermal: Function = sp_property(units="m^-3", default_value=0.0)

    density_fast: Function = sp_property(units="m^-3", default_value=0.0)

    @sp_property(units="Pa")
    def pressure(self) -> Expression:
        return (
            self.pressure_thermal
            + self.pressure_fast_parallel
            + self.pressure_fast_perpendicular
        )

    @sp_property(units="Pa")
    def pressure_thermal(self) -> Expression:
        return self.density * self.temperature * scipy.constants.electron_volt

    pressure_fast_perpendicular: Function = sp_property(units="Pa", default_value=0.0)

    pressure_fast_parallel: Function = sp_property(units="Pa", default_value=0.0)

    collisionality_norm: Function = sp_property(units="-", default_value=0.0)


@sp_tree(coordinate1="grid/rho_tor_norm")
class CoreProfiles1D(core_profiles._T_core_profiles_profiles_1d):

    Ion = CoreProfilesIon

    Electrons = CoreProfilesElectrons

    Neutral = CoreProfilesNeutral

    grid: CoreRadialGrid

    electrons: CoreProfilesElectrons

    ion: AoS[CoreProfilesIon]

    neutral: AoS[CoreProfilesNeutral]

    @sp_property
    def t_i_average(self) -> Function:
        return (sum([ion.z_ion_1d * ion.temperature * ion.density for ion in self.ion]) / self.n_i_total)

    @sp_property
    def n_i_total(self) -> Function:
        return sum([(ion.z_ion_1d * ion.density) for ion in self.ion])

    @sp_property
    def n_i_total_over_n_e(self) -> Function:
        return self.n_i_total / self.electrons.density

    @sp_property
    def n_i_thermal_total(self) -> Function:
        return sum([ion.z * ion.density_thermal for ion in self.ion])

    @sp_property
    def zeff(self) -> Function:
        return (sum([((ion.z_ion_1d**2) * ion.density) for ion in self.ion]) / self.n_i_total)

    @sp_property
    def pressure(self) -> Function:
        p = [ion.pressure.__array__() for ion in self.ion]
        if len(p) == 1:
            return p[0]
        else:
            return np.sum(p, axis=0)

    @sp_property
    def pprime(self) -> Function: return self.pressure.d()

    @sp_property
    def pressure_thermal(self) -> Function:
        return (sum([ion.pressure_thermal for ion in self.ion]) + self.electrons.pressure_thermal)

    t_i_average: Function = sp_property(units="eV")
    # t_i_average_fit: _T_core_profiles_1D_fit = sp_property(units="eV")

    n_i_total_over_n_e: Function = sp_property(units="-")

    n_i_thermal_total: Function = sp_property(units="m^-3")

    momentum_tor: Function = sp_property(units="kg.m^-1.s^-1")

    zeff: Function = sp_property(units="-")
    # zeff_fit: _T_core_profiles_1D_fit = sp_property(units="-")

    pressure_ion_total: Function = sp_property(units="Pa")

    pressure_thermal: Function = sp_property(units="Pa")

    pressure_perpendicular: Function = sp_property(units="Pa")

    pressure_parallel: Function = sp_property(units="Pa")

    j_total: Function = sp_property(units="A/m^2")

    @sp_property(units="A")
    def current_parallel_inside(self) -> Function: return self.j_total.antiderivative()

    j_tor: Function = sp_property(units="A/m^2")

    j_ohmic: Function = sp_property(units="A/m^2")

    @sp_property(units="A/m^2")
    def j_non_inductive(self) -> Function: return self.j_total - self.j_ohmic

    j_bootstrap: Function = sp_property(units="A/m^2")

    @sp_property(units="ohm^-1.m^-1")
    def conductivity_parallel(self) -> Function: return self.j_ohmic / self.e_field.parallel

    @sp_tree
    class EFieldVectorComponents:

        radial: Function

        diamagnetic: Function

        # parallel: Function

        poloidal: Function

        toroidal: Function

        @sp_property
        def parallel(self) -> Function:
            vloop = self._parent.get("vloop", None)
            if vloop is None:
                logger.error(f"Can not calculate E_parallel from vloop!")
                e_par = 0.0
            else:
                e_par = vloop / (TWOPI * self._parent.grid.r0)
            return e_par

    e_field: EFieldVectorComponents = sp_property(units="V.m^-1")

    phi_potential: Function = sp_property(units="V")

    rotation_frequency_tor_sonic: Function

    q: Function = sp_property(units="-")

    @sp_property(units="-")
    def magnetic_shear(self) -> Function:
        return self.grid.rho_tor_norm * (self.q.derivative() / self.q())

    @sp_property
    def beta_pol(self) -> Function:
        return (4 * self.pressure.antiderivative() / (self._parent.vacuum_toroidal_field.r0 * constants.mu_0 * (self.j_total**2)))

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
        """Coulomb logarithm,
        @ref: Tokamaks 2003  Ch.14.5 p727 ,2003
        """
        Te = self.electrons.temperature
        Ne = self.electrons.density

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003

        return (14.9 - 0.5 * np.log(Ne / 1e20) + np.log(Te / 1000)) * (Te < 10) +\
            (15.2 - 0.5 * np.log(Ne / 1e20) + np.log(Te / 1000)) * (Te >= 10)

    @sp_property
    def electron_collision_time(self) -> Function:
        """electron collision time ,
        @ref: Tokamak 2003, eq 14.6.1
        """
        Te = self.electrons.temperature(self.grid.rho_tor_norm)
        Ne = self.electrons.density(self.grid.rho_tor_norm)
        lnCoul = self.coulomb_logarithm(self.grid.rho_tor_norm)
        return 1.09e16 * ((Te / 1000.0) ** (3 / 2)) / Ne / lnCoul

    ffprime: Function = sp_property(label="$ff^{\prime}$")

    pprime: Function = sp_property(label="$p^{\prime}$")


@sp_tree
class CoreGlobalQuantities(core_profiles._T_core_profiles_global_quantities):

    vacuum_toroidal_field: VacuumToroidalField

    ip: float = sp_property(units="A")

    current_non_inductive: float = sp_property(units="A")

    current_bootstrap: float = sp_property(units="A")

    v_loop: float = sp_property(units="V")

    li_3: float = sp_property(units="-")

    beta_tor: float = sp_property(units="-")

    beta_tor_norm: float = sp_property(units="-")

    beta_pol: float = sp_property(units="-")

    energy_diamagnetic: float = sp_property(units="J")

    z_eff_resistive: float = sp_property(units="-")

    t_e_peaking: float = sp_property(units="-")

    t_i_average_peaking: float = sp_property(units="-")

    resistive_psi_losses: float = sp_property(units="Wb")

    ejima: float = sp_property(units="-")

    t_e_volume_average: float = sp_property(units="eV")

    n_e_volume_average: float = sp_property(units="m^-3")

    @sp_tree
    class GlobalQuantitiesIon:
        t_i_volume_average: float = sp_property(units="eV")
        n_i_volume_average: float = sp_property(units="m^-3")

    ion: AoS[GlobalQuantitiesIon]

    ion_time_slice: float = sp_property(units="s")


@sp_tree
class CoreProfilesTimeSlice(TimeSlice):

    Profiles1D = CoreProfiles1D

    GlobalQuantities = CoreGlobalQuantities

    profiles_1d: CoreProfiles1D

    global_quantities: CoreGlobalQuantities

    vacuum_toroidal_field: VacuumToroidalField


@sp_tree
class CoreProfiles(TimeBasedActor):

    ids_properties: IDSProperties

    TimeSlice = CoreProfilesTimeSlice

    time_slice: TimeSeriesAoS[CoreProfilesTimeSlice]

    def refresh(self, *args,   core_transport, core_source, transport_solver, **kwargs):
        super().refresh(*args, **kwargs)

        prev_iter = self.time_slice.current

        next_iter = transport_solver.refresh(prev_iter, core_transport=core_transport, core_source=core_source)

        self.time_slice.current.update(next_iter)

    def advance(self, *args,  core_transport, core_source, transport_solver, **kwargs):
        prev_iter = self.time_slice.current

        next_iter = transport_solver.advance(prev_iter, core_transport=core_transport, core_source=core_source)

        super().advance(*args, **kwargs)

        self.time_slice.current.update(next_iter)
