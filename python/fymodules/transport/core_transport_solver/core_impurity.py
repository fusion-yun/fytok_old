
import collections
import enum
from itertools import chain
from math import isclose, log
from typing import (Any, Callable, Iterator, Mapping, Optional, Sequence,
                    Tuple, Type, Union)

from fytok.common.Misc import Identifier, VacuumToroidalField
from fytok.constants.Atoms import atoms
from fytok.IDS import IDS
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport, TransportCoeff
from fytok.transport.CoreTransportSolver import CoreTransportSolver
from fytok.transport.Equilibrium import Equilibrium
from fytok.transport.MagneticCoordSystem import RadialGrid
from spdm.numlib.bvp import BVPResult, solve_bvp
from spdm.numlib.misc import array_like
from fytok.plasma.Radiation import Radiation
from scipy import constants
from spdm.data import Dict, Function, List, function_like, sp_property
from spdm.util.logger import logger
from spdm.common.tags import _not_found_
from spdm.util.misc import convert_to_named_tuple

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class Impurities(CoreTransportSolver):
    """
        Calculate impurity  density
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve_impurity(self, /,
                       ion_next: CoreProfiles.Profiles1D.Ion,
                       ion_prev: CoreProfiles.Profiles1D.Ion,
                       radiation: Radiation,
                       ion_transport: CoreTransport.Model.Profiles1D.Ion,
                       ion_sources: CoreSources.Source.Profiles1D.Ion,
                       ):
        return 0.0

    def solve(self, /,
              core_profiles_next: CoreProfiles,
              core_profiles_prev: CoreProfiles,
              radiation: Radiation,
              core_transport: CoreTransport,
              core_sources: CoreSources,
              equilibrium_next: Equilibrium,
              equilibrium_prev: Equilibrium = None,
              dt: float = None,
              **kwargs) -> float:
        parameters = collections.ChainMap(kwargs, self.get("code.parameters", {}))

        residual = 0.0

        logger.warning("TODO:Calculate densities, fluxes and  temperatures of impurities.")

        for ion_prev in core_profiles_prev.profiles_1d.ion:

            ion_next = core_profiles_next.profiles_1d.ion[{"label": ion_prev.label}]

            residual += self.solve_impurity(
                ion_next=ion_next,
                ion_prev=ion_prev,
                core_transport=core_transport.profiles_1d.ion[{"label": ion_prev.label}],
                radiation=radiation
            )

        return residual


__SP_EXPORT__ = Impurities
