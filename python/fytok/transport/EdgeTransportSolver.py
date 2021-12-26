from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional, Tuple

from scipy import constants
from spdm.common.logger import logger
from spdm.data import Dict, File, Link, List, Node, Path, Query, sp_property,Function

from ..common.IDS import IDS
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
