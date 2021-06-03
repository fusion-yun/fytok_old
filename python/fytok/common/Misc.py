
import collections
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Any

from spdm.numlib import np
from spdm.data.Node import Dict
from spdm.data.Node import sp_property

# VacuumToroidalField = collections.namedtuple("VacuumToroidalField", "r0 b0", defaults=(0.0, 0.0))
# Identifier = collections.namedtuple("Identifier", " ", defaults=("unamed", 0, ""))


@dataclass
class Identifier:
    name: str = "unnamed"
    index: int = 0
    description: str = ""


@dataclass
class VacuumToroidalField:
    r0: float = 0.0
    b0: float = 0.0


@dataclass
class RZTuple:
    r: Any
    z: Any


@dataclass
class Signal:
    data: np.ndarray
    time: np.ndarray
