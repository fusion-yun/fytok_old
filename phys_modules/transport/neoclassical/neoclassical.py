from functools import cached_property

import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger


class NeoClassicalProfiles1D(CoreTransport.Profiles1D):
    def __init__(self, *args, grid: RadialGrid,
                 equilibrium: Equilibrium.TimeSlice = None,
                 core_profile: CoreProfiles.TimeSlice = None,
                 **kwargs):
        super().__init__(*args, grid=grid, **kwargs)

    def update(self):
        equilibrium_prof = equilibrium.profiles_1d
        core_profiles_prof = core_profiles.profiles_1d
        core_transport_prof = core_transport.profiles_1d
        species = [core_profiles_prof.electrons, *[ion for ion in core_profiles_prof.ion]]


class NeoClassical(CoreTransport.Model):
    Profiles1D = NeoClassicalProfiles1D

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def profiles_1d(self) -> TimeSeries[Profiles1D]:
        return TimeSeries[NeoClassical.Profiles1D](self["profiles_1d"], parent=self)


__SP_EXPORT__ = NeoClassical
