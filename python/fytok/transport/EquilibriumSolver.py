from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional, Tuple

from scipy import constants
from spdm.common.logger import logger
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Function import Function, function_like
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property

from ..common.IDS import IDS
from ..common.Misc import Identifier
from ..common.Species import SpeciesElectron, SpeciesIon
from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .EdgeProfiles import EdgeProfiles
from .EdgeSources import EdgeSources
from .EdgeTransport import EdgeTransport
from .Equilibrium import Equilibrium
from .MagneticCoordSystem import RadialGrid


class EquilibriumSolver(IDS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, /,
              equilibrium_prev: Equilibrium,
              core_profiles: CoreProfiles,
              dt: float = None,
              ** kwargs) -> float:

        return equilibrium_prev
