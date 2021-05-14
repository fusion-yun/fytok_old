import collections
from functools import cached_property

import numpy as np
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Combiner import Combiner
from spdm.data.Function import Function
from spdm.data.Node import Dict, List
from spdm.data.Profiles import Profiles
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module

from .IDS import IDS, IDSCode
from .Misc import Identifier


class Actor(Dict):

    _module_prefix = ""

    def __new__(cls, d, *args, **kwargs):
        prefix = getattr(cls, "_module_prefix", "")
        name = d["code"]["name"]
        n_cls = sp_find_module(f"{prefix}.{name}")
        return object.__new__(n_cls)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def update(self,  *args,  **kwargs):
        return NotImplemented
