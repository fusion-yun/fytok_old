
import collections
import collections.abc

import scipy
from fytok.transport.CoreProfiles import CoreProfiles, CoreProfiles1D
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport, CoreTransportModel
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.numlib import constants, np
from spdm.util.logger import logger

from .glf23_mod import glf2d, glf
logger.debug(glf2d.__doc__)
# logger.debug(glf.__doc__)


class GLF23(CoreTransport.Model):
    r"""
        GLF23
        ===============================
            - 2D GLF equations with massless isothermal passing electrons from

        References:
        =============
            - Waltz et al, Phys. of Plasmas 6(1995)2408
    """

    def __init__(self, d=None, *args,  **kwargs):
        super().__init__(collections.ChainMap(
            {"identifier": {"name": "anomalous", "index": 6,
                            "description": f"anomalous {self.__class__.__name__}"},
             "code": {"name": "glf23"}}, d or {}),
            *args, **kwargs)

    def refresh(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles,  **kwargs):
        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)
        return


__SP_EXPORT__ = GLF23
