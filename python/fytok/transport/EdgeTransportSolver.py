from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional, Tuple

from scipy import constants
from spdm.data import (Dict, File, Function, Link, List, Node, Path, Query,
                       sp_property)
from spdm.util.logger import logger

from ..common.IDS import IDS
from ..common.Misc import Identifier
from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .EdgeProfiles import EdgeProfiles
from .EdgeSources import EdgeSources
from .EdgeTransport import EdgeTransport
from .Equilibrium import Equilibrium
from .MagneticCoordSystem import RadialGrid
from .Species import SpeciesElectron, SpeciesIon


class EdgeTransportSolver(IDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, /,
              edge_profiles_prev: EdgeProfiles,
              edge_transport: EdgeTransport.Model,
              edge_sources: EdgeSources.Source,
              equilibrium_prev: Equilibrium,
              equilibrium_next: Equilibrium,
              tolerance=1.0e-3,
              ** kwargs) -> EdgeProfiles:
        """
            solve transport equation until residual < tolerance
            return residual , core_profiles, edge_profiles
        """

        return edge_profiles_prev
