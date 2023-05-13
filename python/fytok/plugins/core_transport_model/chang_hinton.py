import collections
from functools import cached_property

import numpy as np
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreTransport import (CoreTransport,
                                           CoreTransportProfiles1D)
from fytok.transport.Equilibrium import Equilibrium
from fytok.transport.MagneticCoordSystem import RadialMesh
from scipy import constants
from spdm.data import Function
from spdm.utils.logger import logger
from spdm.utils.tags import _next_


class ChangHiton(CoreTransport.Model):
    """
        Chang-Hiton formula for \Chi_{i}
        ===============================

        References:
        =============
        - Tokamaks, Third Edition, Chapter 14.11  ,p737,  J.A.Wesson 2003
    """

    def update(self, *args,
                core_profiles: CoreProfiles = None,
                equilibrium: Equilibrium = None,
                **kwargs) -> float:
        residual = super().refresh(*args, **kwargs)

        logger.warning(f"Not IMPLEMENTED!")

        return residual


__SP_EXPORT__ = ChangHiton
