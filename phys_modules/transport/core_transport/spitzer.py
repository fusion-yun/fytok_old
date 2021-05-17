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


class Spitzer(CoreTransport.Model):
    """
        Spitzer Resistivity
        ===============================

        References:
        =============
        - Tokamaks, Third Edition, Chapter 2,16 Confinement,p149,  J.A.Wesson 2003
    """

    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, core_profiles: CoreProfiles.TimeSlice = None, **kwargs):
        prof = self.profiles_1d[-1]
        prof.update(core_profiles=core_profiles, **kwargs)
        prof.conductivity_parallel = prof.grid_d.rho_tor_norm
        return 0.0


__SP_EXPORT__ = Spitzer
