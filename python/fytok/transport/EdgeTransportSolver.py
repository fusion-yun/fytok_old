from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional, Tuple

from fytok.common.Species import SpeciesElectron, SpeciesIon
from fytok.transport.EdgeProfiles import EdgeProfiles
from fytok.transport.EdgeSources import EdgeSources
from fytok.transport.EdgeTransport import EdgeTransport
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Node import Dict, List, _not_found_, sp_property
from scipy import constants
from spdm.util.logger import logger

from ..common.IDS import IDS
from ..common.Misc import Identifier
from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .Equilibrium import Equilibrium
from .MagneticCoordSystem import RadialGrid


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
