import collections
from  functools import cached_property

from spdm.numlib import np
from spdm.numlib import constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport, CoreTransportProfiles1D
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger


class Spitzer(CoreTransport.Model):
    """
        Spitzer Resistivity
        ===============================

        References:
        =============
        - Tokamaks, Third Edition, Chapter 14  ,p727,  J.A.Wesson 2003
    """

    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)

    def advance(self, *args, time=None, dt=None,   **kwargs) -> float:
        return super().advance(*args, time=time, dt=dt,   **kwargs)

    def update(self, *args, core_profiles: CoreProfiles.TimeSlice = None, **kwargs):
        super().update(*args, grid=core_profiles.profiles_1d.grid, **kwargs)
        prof = self.profiles_1d[-1]

        Te = np.asarray(core_profiles.profiles_1d.electrons.temperature)
        ne = np.asarray(core_profiles.profiles_1d.electrons.density)

        # Coulomb logarithm  Ch.14.5 p727 Tokamaks 2003
        clog = (31.0 - 0.5*np.log(ne) + np.log(Te))*(Te < 10) +\
            (31.3 - 0.5*np.log(ne) + np.log(Te))*(Te >= 10)

        # (29.96 - 0.5*np.log(ne) + 1.5 * np.log(Te))*(Te >= 10000)

        # Collision times Eq. 14.6.1 p729 Tokamaks 2003
        # tau_e = 1.09e16*((Te*1.0e-3)**(3/2))/ne/clog

        # Spitzer resistivity, Eq. 14.10.1 p735 Tokamaks 2003
        sigma = Function(core_profiles.profiles_1d.grid.rho_tor_norm, 1.65e-9 * clog*((Te*1.0e-3)**(-3/2)))

        prof.conductivity_parallel = Function(prof.grid_d.rho_tor_norm, sigma)

        return 0.0


__SP_EXPORT__ = Spitzer
