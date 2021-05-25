
import collections
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from spdm.data.Node import Dict
from spdm.data.sp_property import sp_property


@dataclass
class Signal:
    data: np.ndarray
    time: np.ndarray
