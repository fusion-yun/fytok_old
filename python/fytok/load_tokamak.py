import collections
import pathlib
import sys
from typing import Union

import numpy as np
import pandas as pd
from scipy import constants
from spdm.data import (Dict, Entry, File, Function, Link, List, Node, Path,
                       Query, function_like, sp_property)
from spdm.data.Function import PiecewiseFunction, function_like
from spdm.util.logger import logger
from .device.Wall import Wall
from .device.PFActive import PFActive
from .device.TF import TF
from .device.Magnetics import Magnetics
from .transport.Equilibrium import Equilibrium


from spdm import open_entry


def load_tokamak(entry: Union[str, Entry], *args, **kwargs):
    if not isinstance(entry, Entry):
        entry = open_entry(entry, *args, **kwargs)

    magnetics = Magnetics(entry.get(["magnetics"]))
    pf_active = PFActive(entry.get(["pf_active"]))
    wall = Wall(entry.get(["wall"]))
    equilibrium = Equilibrium(entry.get(["equilibrium"]))
