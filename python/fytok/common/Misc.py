
import collections
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, TypeVar, Union

from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
import numpy as np

# VacuumToroidalField = collections.namedtuple("VacuumToroidalField", "r0 b0", defaults=(0.0, 0.0))
# Identifier = collections.namedtuple("Identifier", " ", defaults=("unamed", 0, ""))


@dataclass
class Identifier:
    name: str = "unnamed"
    index: int = 0
    description: str = ""


_TData = TypeVar("_TData")


@dataclass
class Decomposition(Generic[_TData]):
    implicit_part: _TData  # Implicit part
    explicit_part: _TData  # Explicit part


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
