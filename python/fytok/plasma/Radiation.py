
import collections
import collections.abc
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, TypeVar, Union

import scipy
from scipy import constants
from spdm.common.logger import logger
from spdm.common.tags import _not_found_, _undefined_
from spdm.data import Dict, Function, List, Node, function_like, sp_property
from spdm.data.Field import Field
from spdm.util.utilities import try_get

from ..common.GGD import GGD
from ..common.IDS import IDS


class RadiationProcess(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Radiation(IDS):
    r"""Radiation emitted by the plasma and neutrals

        Note: Radiation is an ids
    """
    _IDS = "radiation"
    Process = RadiationProcess

    default_process_list = [
        {"identifier":  {"name": "nuclear_decay", "index": 6	, "description": "Emission from nuclear decay"}, },
        {"identifier":  {"name": "bremsstrahlung	", "index": 8,	"description": "Emission from bremsstrahlung"}, },
        {"identifier":  {"name": "synchrotron_radiation	", "index": 9,	"description": "Emission from synchrotron radiation"}, },
        {"identifier":  {"name": "line_radiation	", "index": 10	, "description": "Emission from line radiation"}, },
        {"identifier":  {"name": "recombination	", "index": 11,	"description": "Emission from recombination"}, },
        {"identifier":  {"name": "runaways	", "index": 501,
                         "description": "Emission from run-away processes; includes both electron and ion run-away"}, },
    ]

    def __init__(self, d=None, **kwargs):
        super().__init__(d if d is not None else {"process": Radiation.default_process_list}, **kwargs)

    @sp_property
    def grid_ggd(self) -> GGD:
        return self.get("grid_ggd", {})

    @sp_property
    def process(self) -> List[Process]:
        return List[Radiation.Process](self.get("process", []), parent=self)
