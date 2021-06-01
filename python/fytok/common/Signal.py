
import collections
from dataclasses import dataclass
from typing import Any, Union

from spdm.numlib import np
from spdm.data.Node import Dict
from spdm.data.Node import sp_property


@dataclass
class Signal:
    data: np.ndarray
    time: np.ndarray
