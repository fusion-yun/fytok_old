import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.data.Node import _next_


def neoclassical(equilibrium: Equilibrium.TimeSlice,
                 core_profiles: CoreProfiles.TimeSlice,
                 core_transport: CoreTransport.TimeSlice,
                 grid: np.ndarray = None):
    equilibrium_prof = equilibrium.profiles_1d
    core_profiles_prof = core_profiles.profiles_1d
    core_transport_prof = core_transport.profiles_1d

    species = [core_profiles_prof.electrons, *[ion for ion in core_profiles_prof.ion]]
