import collections
from functools import cached_property

import numpy as np
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreTransport import CoreTransport, CoreTransportProfiles1D
from fytok.modules.Equilibrium import Equilibrium
from scipy import constants
from spdm.data import Function
from fytok.utils.logger import logger


class ChangHiton(CoreTransport.Model):
    """
        Chang-Hiton formula for \Chi_{i}
        ===============================

        References:
        =============
        - Tokamaks, Third Edition, Chapter 14.11  ,p737,  J.A.Wesson 2003
    """

    def refresh(self, *args,
                core_profiles_1d: CoreProfiles.Profiles1d = None,
                equilibrium: Equilibrium.TimeSlice = None,
                **kwargs):
        return super().refresh(*args, **kwargs)


__SP_EXPORT__ = ChangHiton
