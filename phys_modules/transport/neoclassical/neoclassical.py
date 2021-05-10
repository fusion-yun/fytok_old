import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.data.Node import _next_


def neoclassical(equilibrium: Equilibrium.TimeSlice,
                 core_profiles: CoreProfiles.Profiles1D,
                 core_transport: CoreTransport.Profiles1D,
                 grid: np.ndarray = None) -> CoreTransport.Profiles1D:

    species = [core_profiles.electrons, *[ion for ion in core_profiles.ion]]
    return core_transport
