import collections
import math
from functools import cached_property

import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from matplotlib.pyplot import sci
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger

# class NeoClassicalProfiles1D(CoreTransport.Model.TimeSlice):
#     def __init__(self, *args, grid: RadialGrid,
#                  equilibrium: Equilibrium.TimeSlice = None,
#                  core_profile: CoreProfiles.TimeSlice = None,
#                  **kwargs):
#         super().__init__(*args, grid=grid, **kwargs)


class NeoClassical(CoreTransport.Model):
    """
        Neoclassiical Transport Model
        ===============================

        References:
        =============
        - Tokamaks, Third Edition, Chapter 4 Confinement,p149,  J.A.Wesson 2003
    """

    def __init__(self, d, *args, **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {
                "name": f"neoclassical",
                "index": 5,
                "description": f"{self.__class__.__name__}"
            }}, d or {}), *args, **kwargs)

    def update(self,
               equilibrium: Equilibrium.TimeSlice,
               core_profiles: CoreProfiles.TimeSlice,
               impurity_list=[],
               **kwargs):

        super().update(equilibrium=equilibrium, core_profiles=core_profiles)

        # Coulomb logarithm
        lnCoul = 14

        B0 = equilibrium.vacuum_toroidal_field.b0
        R0 = equilibrium.vacuum_toroidal_field.r0

        eq_profile = equilibrium.profiles_1d
        core_profile = core_profiles.profiles_1d
        prof = self.profiles_1d[-1]

        rho_tor_norm = core_profile.grid.rho_tor_norm
        rho_tor = core_profile.grid.rho_tor

        epsilon = np.sqrt(rho_tor/R0)

        Tavg = np.sum([ion.density*ion.temperature for ion in core_profile.ion]) / \
            np.sum([ion.density for ion in core_profile.ion])
        species = [core_profile.electrons, *[ion for ion in core_profile.ion]]
        for sp in species:
            T = sp.temperature
            N = sp.density
            dlnT = T.derivative/T
            dlnN = N.derivative/N
            m_s = sp.a
            z_s = sp.z

            if sp.label in impurity_list or sp.neutral_index in impurity_list:
                raise NotImplementedError("TODO: impurity")
            elif sp.label == "electron":

                # electron collision time , Tokamaks 3ed, eq 14.6.1 p730
                tau_s = 1.09e16*((T/1000)**(3/2))/N/lnCoul
                # Larmor radius, Tokamaks 3ed, eq 14.7.2
                rho_s = 1.07e-4*((T/1000)**(1/2))/B0

                dlnTe = dlnT
                dlnNe = dlnN
                Te = T
                Ne = N
                tau_e = tau_s
            else:
                # ion collision time Tokamaks 3ed, eq 14.6.2 p730
                tau_s = 6.6e17*np.sqrt(m_s)*((T/1000)**(3/2))/N/(1.1*lnCoul)
                # Larmor radius, Tokamaks 3ed, eq 14.7.2
                rho_s = 4.57e-4 * np.sqrt(m_s*T/1000) / B0

            # thermal velocity
            v_s = np.sqrt(T*(scipy.constants.electron_volt/scipy.constants.m_p/m_s))

            if sp.label == 'electron':
                prof.conductivity_parallel = (
                    (1-epsilon)**2)*N*(scipy.constants.elementary_charge**2)*tau_s/0.51/scipy.constants.electron_mass
                prof.j_bootstrap = N*scipy.constants.electron_volt*T*(2.44*dlnN+0.69*dlnT)
            else:
                prof.j_bootstrap += N*scipy.constants.electron_volt*T*(2.44*dlnNe-0.44*dlnT)

            logger.debug(sp)


__SP_EXPORT__ = NeoClassical
