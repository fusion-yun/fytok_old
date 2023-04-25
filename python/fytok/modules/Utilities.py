from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from dataclasses import dataclass
from _imas.utilities import _T_rz1d_dynamic_aos

import typing

_T = typing.TypeVar("_T")


class RZTuple(_T_rz1d_dynamic_aos):
    r = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
    z = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
# @dataclass
# class RZTuple:
#     r: typing.Any
#     z: typing.Any
