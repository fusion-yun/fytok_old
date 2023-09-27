import typing

import numpy as np
import scipy.constants
from scipy import constants
from spdm.data.AoS import AoS
from spdm.data.Entry import Entry
from spdm.data.Function import Function
from spdm.data.HTree import HTree
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.utils.tags import _not_found_
from spdm.utils.tree_utils import merge_tree_recursive
from spdm.utils.typing import HTreeLike

from .._imas.lastest.core_profiles import (_T_core_profiles,
                                           _T_core_profiles_profiles_1d)
from .._imas.lastest.utilities import (_T_core_profile_ions,
                                       _T_core_profiles_profiles_1d_electrons,
                                       _T_core_profiles_vector_components_1)
from ..utils.atoms import atoms
from ..utils.logger import logger
from ..utils.utilities import CoreRadialGrid

PI = scipy.constants.pi
TWOPI = 2.0 * PI


class CoreProfilesElectrons(_T_core_profiles_profiles_1d_electrons):
    density_fast: Function = sp_property(default_value=0.0)

    @sp_property[Function]
    def density(self):
        return self.density_thermal + self.density_fast

    @sp_property[Function]
    def pressure_thermal(self):
        return self.density * self.temperature * scipy.constants.electron_volt

    pressure_fast_perpendicular: Function = sp_property(default_value=0.0)

    pressure_fast_parallel: Function = sp_property(default_value=0.0)

    @sp_property[Function]
    def pressure(self):
        return (
            self.pressure_thermal
            + self.pressure_fast_parallel
            + self.pressure_fast_perpendicular
        )

    @sp_property[Function]
    def tau(self):
        return (1.09e16 * ((self.temperature / 1000) ** (3 / 2)) / self.density / self._parent.coulomb_logarithm)

    @sp_property[Function]
    def vT(self):
        return np.sqrt(self.temperature * scipy.constants.electron_volt / scipy.constants.electron_mass)


class CoreProfilesIon(_T_core_profile_ions):
    def __init__(self, cache: typing.Any = None, /, entry: HTreeLike | Entry = None, **kwargs) -> None:
        if cache is None or cache is _not_found_:
            cache = {}

        label = (cache.get("label", None) or entry.get("label", "")).upper()

        desc = atoms.get(label, None)

        if desc is None:
            raise RuntimeError(f"Can not find ion {label}")

        cache = merge_tree_recursive(cache, desc)
        cache["label"] = label

        super().__init__(cache, entry=entry, **kwargs)

    is_impurity: bool = sp_property(default_value=False)

    has_fast_particle: bool = sp_property(default_value=False)

    @property
    def mass(self) -> float:
        return self.element[0].a

    @property
    def charge(self) -> float:
        return self.element[0].z

    @sp_property
    def z_ion_1d(self) -> Function:
        return super().z_ion_1d

    @sp_property
    def z_ion_square_1d(self) -> Function: return self.z_ion * self.z_ion

    @property
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
            self.pressure_thermal
            + self.pressure_fast_parallel
            + self.pressure_fast_perpendicular
        )


class CoreProfiles1D(_T_core_profiles_profiles_1d):
    Ion = CoreProfilesIon

    Electrons = CoreProfilesElectrons

    grid: CoreRadialGrid = sp_property()

    electrons: CoreProfilesElectrons = sp_property()

    ion: AoS[CoreProfilesIon] = sp_property(identifier="label")

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
        p = [ion.pressure for ion in self.ion]
        if len(p) == 1:
            return p[0]
        else:
            return np.sum(p, axis=0)

    @sp_property
    def pprime(self) -> Function: return self.pressure.d()

    @sp_property
    def pressure_thermal(self) -> Function:
        return (sum([ion.pressure_thermal for ion in self.ion]) + self.electrons.pressure_thermal)

    # @sp_property
    # def j_total(self) -> Function:
    #     return self.current_parallel_inside.derivative * \
    #         self.grid.r0*TWOPI/self.grid.dvolume_drho_tor

    j_total: Function = sp_property(coordinate1="../grid/rho_tor_norm")

    @sp_property
    def current_parallel_inside(self) -> Function: return self.j_total.antiderivative()

    # current_parallel_inside: Function = sp_property()

    @sp_property
    def j_non_inductive(self) -> Function: return self.j_total - self.j_ohmic

    @sp_property
    def conductivity_parallel(self) -> Function: return self.j_ohmic / self.e_field.parallel

    @sp_property
    def beta_pol(self) -> Function:
        return (4 * self.pressure.antiderivative() / (self.grid.r0 * constants.mu_0 * (self.j_total**2)))

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

    @sp_property(coordinate1="../grid/rho_tor_norm")
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

    @sp_property(coordinate1="../grid/rho_tor_norm")
    def electron_collision_time(self) -> Function:
        """electron collision time ,
        @ref: Tokamak 2003, eq 14.6.1
        """
        Te = self.electrons.temperature(self.grid.rho_tor_norm)
        Ne = self.electrons.density(self.grid.rho_tor_norm)
        lnCoul = self.coulomb_logarithm(self.grid.rho_tor_norm)
        return 1.09e16 * ((Te / 1000.0) ** (3 / 2)) / Ne / lnCoul

    class EField(_T_core_profiles_vector_components_1):
        @sp_property
        def parallel(self) -> Function:
            e_par = self.get("parallel", None)
            if e_par is None:
                vloop = self._parent.get("vloop", None)
                if vloop is None:
                    logger.error(f"Can not calculate E_parallel from vloop!")
                    e_par = 0.0
                else:
                    e_par = vloop / (TWOPI * self.grid.r0)
            return e_par

    @sp_property
    def e_field(self) -> EField: return self.get("e_field", {})

    @sp_property
    def magnetic_shear(self) -> Function:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic}[-]"""
        return self.grid.rho_tor_norm * (self.q.derivative() / self.q())

    ffprime: Function = sp_property(coordinate1="../grid/rho_tor_norm", label="$ff^{\prime}$")

    pprime: Function = sp_property(coordinate1="../grid/rho_tor_norm", label="$p^{\prime}$")


class CoreProfiles(_T_core_profiles):
    Profiles1D = CoreProfiles1D

    profiles_1d: TimeSeriesAoS[Profiles1D] = sp_property()

    def refresh(self, *args, **kwargs) -> Profiles1D:
        return self.profiles_1d.refresh(*args, **kwargs)

    def advance(self, *args, **kwargs) -> Profiles1D:
        return self.profiles_1d.advance(*args, **kwargs)
