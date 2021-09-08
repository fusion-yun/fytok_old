
from ..common.IDS import IDS
import collections
import collections.abc

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, TypeVar, Union

import scipy
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.Node import Dict, List, Node, sp_property
from scipy import constants
from spdm.util.logger import logger
from spdm.util.utilities import _not_found_, _undefined_, try_get


class AMNSData(IDS):
    r"""
        Atomic, molecular, nuclear and surface physics data. Each occurrence contains the data for a given element (nuclear charge),
        describing various physical processes. For each process, data tables are organized by charge states. The coordinate system used
        by the data tables is described under the coordinate_system node.

        Note: AMNSData is an ids
    """
    _IDS = "amns_data"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def z_n(self) -> float:
        """Nuclear charge {static} [Elementary Charge Unit]"""
        return NotImplemented

    def a(self) -> float:
        """	Mass of atom {static} [Atomic Mass Unit]"""
        return NotImplemented

    def process(self) -> List:
        """Description and data for a set of physical processes."""
        return NotImplemented

    def coordinate_system(self) -> List:
        """	Array of possible coordinate systems for process tables"""
        return NotImplemented
