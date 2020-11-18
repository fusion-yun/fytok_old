
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


class CoreProfiles(AttributeTree):
    """
        imas dd version 3.28

        ids = core_profiles.profiles_1d
    """

    def __init__(self, config, *args,  tokamak=None, ** kwargs):
        super().__init__(*args, ** kwargs)
        self.tokamak = tokamak
        self.load(config)

    # def load(self, entry=None, *args, dims=None, itime=0, **kwargs):
    #     if dims is None:
    #         dims = 129
    #     self.vacuum_toroidal_field.b0 = 1.0
    #     self.vacuum_toroidal_field.r0 = 1.0
    #     self.profiles_1d = Profiles(np.linspace(1.0/(dims+1), 1, dims))
    #     self.profiles_1d |= {"grid": {}, "electron": {}, "ion": [],
    #                          "neutral": [],
    #                          "efield": {}
    #                          }
    #     self.profiles_1d.conductivity_parallel = np.linspace(1.0/(dims+1), 1, dims)

    def load(self, config):
        self._cache = config

    @property
    def cache(self):
        if isinstance(self._cache, LazyProxy):
            self._cache = self._cache()
        elif isinstance(self._cache, AttributeTree):
            self._cache = self._cache
        else:
            self._cache = AttributeTree(self._cache)
        return self._cache

    def interpolate(self, key):
        return NotImplemented

    class Grid(AttributeTree):
        def __init__(self,  core_profiles,  *args,  **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_core_profiles'] = core_profiles

    class Ion(AttributeTree):
        def __init__(self,  core_profiles,  *args,  **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_core_profiles'] = core_profiles

    class Electons(AttributeTree):
        def __init__(self,  core_profiles,  *args,  **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_core_profiles'] = core_profiles

    class Neutral(AttributeTree):
        def __init__(self,  core_profiles,  *args,  **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_core_profiles'] = core_profiles

    @cached_property
    def grid(self):
        """Radial grid"""
        return CoreProfiles.Grid(self)

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

    @cached_property
    def t_i_average(self):
        """	Ion temperature (averaged on charge states and ion species) {dynamic} [eV]	"""
        return NotImplemented

    @cached_property
    def t_i_average_fit(self):
        """Information on the fit used to obtain the t_i_average profile [eV]	structure"""
        return NotImplemented

    @cached_property
    def n_i_total_over_n_e(self):
        """	Ratio of total ion density (sum over species and charge states) over electron density. (thermal+non-thermal) {dynamic} [-]	"""
        return NotImplemented

    @cached_property
    def n_i_thermal_total(self):
        """	Total ion thermal density (sum over species and charge states) {dynamic} [m^-3]	"""
        return NotImplemented

    @cached_property
    def momentum_tor(self):
        """	Total plasma toroidal momentum, summed over ion species and electrons weighted by their density and major radius, i.e. sum_over_species(n*R*m*Vphi) {dynamic} [kg.m^-1.s^-1]	"""
        return NotImplemented

    @cached_property
    def zeff(self):
        """	Effective charge {dynamic} [-]	"""
        return NotImplemented

    @cached_property
    def zeff_fit(self):
        """Information on the fit used to obtain the zeff profile [-]	structure"""
        return NotImplemented

    @cached_property
    def pressure_ion_total(self):
        """	Total (sum over ion species) thermal ion pressure {dynamic} [Pa]	"""
        return NotImplemented

    @cached_property
    def pressure_thermal(self):
        """	Thermal pressure (electrons+ions) {dynamic} [Pa]	"""
        return NotImplemented

    @cached_property
    def pressure_perpendicular(self):
        """	Total perpendicular pressure (electrons+ions, thermal+non-thermal) {dynamic} [Pa]	"""
        return NotImplemented

    @cached_property
    def pressure_parallel(self):
        """	Total parallel pressure (electrons+ions, thermal+non-thermal) {dynamic} [Pa]	"""
        return NotImplemented

    @cached_property
    def j_total(self):
        """	Total parallel current density = average(jtot.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field/ B0 {dynamic} [A/m^2]	"""
        return NotImplemented

    @cached_property
    def current_parallel_inside(self):
        """	Parallel current driven inside the flux surface. Cumulative surface integral of j_total {dynamic} [A]	"""
        return NotImplemented

    @cached_property
    def j_tor(self):
        """	Total toroidal current density = average(J_Tor/R) / average(1/R) {dynamic} [A/m^2]	"""
        return NotImplemented

    @cached_property
    def j_ohmic(self):
        """	Ohmic parallel current density = average(J_Ohmic.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field/ B0 {dynamic} [A/m^2]	"""
        return NotImplemented

    @cached_property
    def j_non_inductive(self):
        """	Non-inductive (includes bootstrap) parallel current density = average(jni.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field/ B0 {dynamic} [A/m^2]	"""
        return NotImplemented

    @cached_property
    def j_bootstrap(self):
        """	Bootstrap current density = average(J_Bootstrap.B) / B0, where B0 = Core_Profiles/Vacuum_Toroidal_Field/ B0 {dynamic} [A/m^2]	"""
        return NotImplemented

    @cached_property
    def conductivity_parallel(self):
        """	Parallel conductivity {dynamic} [ohm^-1.m^-1]	"""
        return NotImplemented

    @cached_property
    def e_field(self):
        """Electric field, averaged on the magnetic surface. E.g for the parallel component, average(E.B) / B0, using core_profiles/vacuum_toroidal_field/b0 [V.m^-1]	structure"""
        return NotImplemented

    @cached_property
    def phi_potential(self):
        """	Electrostatic potential, averaged on the magnetic flux surface {dynamic} [V]	"""
        return NotImplemented

    @cached_property
    def rotation_frequency_tor_sonic(self):
        """  	Derivative of the flux surface averaged electrostatic potential with respect to the poloidal flux, multiplied by -1. This quantity is the toroidal angular rotation frequency due to the ExB drift, introduced in formula (43) of Hinton and Wong, Physics of Fluids 3082 (1985), also referred to as sonic flow in regimes in which the toroidal velocity is dominant over the poloidal velocity Click here for further documentation. {dynamic} [s^-1]	"""
        return NotImplemented

    @cached_property
    def q(self):
        """	Safety factor (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic} [-]. This quantity is COCOS-dependent, with the following transformation :"""
        return NotImplemented

    @cached_property
    def magnetic_shear(self):
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic}[-]"""
        return NotImplemented
