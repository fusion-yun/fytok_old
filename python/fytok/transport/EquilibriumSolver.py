from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional, Tuple

from fytok.common.Species import SpeciesElectron, SpeciesIon
from fytok.transport.EdgeProfiles import EdgeProfiles
from fytok.transport.EdgeSources import EdgeSources
from fytok.transport.EdgeTransport import EdgeTransport
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Node import Dict, List, _not_found_, sp_property
from spdm.numlib import constants, np
from spdm.util.logger import logger

from ..common.IDS import IDS
from ..common.Misc import Identifier
from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .Equilibrium import Equilibrium
from .MagneticCoordSystem import RadialGrid


class EquilibriumSolver(IDS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, /,

              equilibrium_prev: Equilibrium,
              core_profiles: CoreProfiles,
              dt: float = None,
              ** kwargs) -> Tuple[float, Equilibrium]:

        return 0.0, equilibrium_prev
