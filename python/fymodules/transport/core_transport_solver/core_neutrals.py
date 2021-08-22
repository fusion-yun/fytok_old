
import collections
import enum
from itertools import chain
from math import isclose, log
from typing import (Any, Callable, Iterator, Mapping, Optional, Sequence,
                    Tuple, Type, Union)

from fytok.common.Atoms import atoms
from fytok.common.IDS import IDS
from fytok.common.Misc import Identifier, VacuumToroidalField
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport, TransportCoeff
from fytok.transport.CoreTransportSolver import CoreTransportSolver
from fytok.transport.Equilibrium import Equilibrium
from fytok.transport.MagneticCoordSystem import RadialGrid
from matplotlib.pyplot import loglog
from numpy.core.fromnumeric import var
from spdm.data.Function import Function, function_like
from spdm.data.Node import Dict, List, _not_found_
from spdm.numlib import constants, np
from spdm.numlib.misc import array_like
from spdm.util.logger import logger

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class CoreNeutrals(CoreTransportSolver):
    """
        Calculate densities, fluxes and  temperatures of neutrals.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, /,
              core_profiles_next: CoreProfiles,
              core_profiles_prev: CoreProfiles,
              core_transport: CoreTransport.Model,
              core_sources: CoreSources.Source,
              equilibrium_next: Equilibrium,
              equilibrium_prev: Equilibrium = None,
              dt: float = None,
              **kwargs) -> float:
        parameters = collections.ChainMap(kwargs, self.get("code.parameters", {}))

        residual = 0.0

        logger.warning("TODO: Calculate densities, fluxes and  temperatures of neutrals.")

        return residual


__SP_EXPORT__ = CoreNeutrals
