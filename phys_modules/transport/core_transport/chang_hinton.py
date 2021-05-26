import collections
from  functools import cached_property

from spdm.util.numlib import np
from spdm.util.numlib import constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport, CoreTransportProfiles1D
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger


class ChangHiton(CoreTransport.Model):
    """
        Chang-Hiton formula for \Chi_{i}
        ===============================

        References:
        =============
        - Tokamaks, Third Edition, Chapter 14.11  ,p737,  J.A.Wesson 2003
    """

    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args,
               core_profiles: CoreProfiles.TimeSlice = None,
               equilibrium: Equilibrium.TimeSlice = None,
               **kwargs):
        super().update(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)

        raise NotImplementedError()


__SP_EXPORT__ = ChangHiton
