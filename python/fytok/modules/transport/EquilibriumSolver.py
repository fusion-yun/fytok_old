from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional, Tuple

from scipy import constants
from spdm.logger import logger
from spdm.data import Dict, File, Link, List, Node, Path, Query, sp_property,Function

from ...IDS import IDS
from ..common.Misc import Identifier
from ..common.Species import SpeciesElectron, SpeciesIon
from ..transport.CoreProfiles import CoreProfiles
from ..transport.CoreSources import CoreSources
from ..transport.CoreTransport import CoreTransport
from ..transport.EdgeProfiles import EdgeProfiles
from ..transport.EdgeSources import EdgeSources
from ..transport.EdgeTransport import EdgeTransport
from ..transport.Equilibrium import Equilibrium
from ..transport.MagneticCoordSystem import RadialGrid


class EquilibriumSolver(IDS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, /,
              equilibrium_prev: Equilibrium,
              core_profiles: CoreProfiles,
              dt: float = None,
              ** kwargs) -> float:

        return equilibrium_prev
