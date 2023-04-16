
import collections
from dataclasses import dataclass
from datetime import datetime
import typing

import numpy as np
# VacuumToroidalField = collections.namedtuple("VacuumToroidalField", "r0 b0", defaults=(0.0, 0.0))
# Identifier = collections.namedtuple("Identifier", " ", defaults=("unamed", 0, ""))
from spdm.data.Dict import Dict
from spdm.data.Node import Node


@dataclass
class Identifier:
    name: str = "unnamed"
    index: int = 0
    description: str = ""


_TData = typing.TypeVar("_TData")


@dataclass
class Decomposition(typing.Generic[_TData]):
    implicit_part: _TData  # Implicit part
    explicit_part: _TData  # Explicit part


@dataclass
class VacuumToroidalField:
    r0: float = 0.0
    b0: float = 0.0


@dataclass
class RZTuple(typing.Generic[_TData]):
    r:   _TData
    z:   _TData


class Profiles1D(Dict[Node]):
    def _as_child(self, *args, **kwargs):
        super()._as_child(*args, **kwargs)
