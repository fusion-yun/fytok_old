import collections
from functools import cached_property

import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport, CoreTransportProfiles1D
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger


class GyroBohm(CoreTransport.Model):
    """
        Heat conductivity Anomalous gyroBohm
        ===============================

        References:
        =============
        - Tokamaks, Third Edition, Chapter  4.16  ,p197,  J.A.Wesson 2003
    """

    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args,
               core_profiles: CoreProfiles.TimeSlice = None,
               equilibrium: Equilibrium.TimeSlice = None,
               **kwargs):
        super().update(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)
        prof = self.profiles_1d[-1]
        rho_tor_norm = core_profiles.profiles_1d.grid.rho_tor_norm
        psi_norm = core_profiles.profiles_1d.grid.psi_norm

        Te = np.asarray(core_profiles.profiles_1d.electrons.temperature)/1.e3
        ne = np.asarray(core_profiles.profiles_1d.electrons.density)/1.e19
        mu = 1.0/np.asarray(equilibrium.profiles_1d.q(psi_norm))

        for ion in prof.ion:

            # ion.particles.d = 0
            # ion.particles.v = 0
            ion.energy.d = Chi_i
            ion.energy.v = 0

        # prof.electrons.particles.d = 0
        # prof.electrons.particles.v = 0
        # prof.electrons.energy.d = Chi_e
        # prof.electrons.energy.v = 0


__SP_EXPORT__ = GyroBohm
