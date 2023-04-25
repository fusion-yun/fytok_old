from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property

import typing
_T = typing.TypeVar("_T")


class RZTuple(Dict[Node], typing.Generic[_T]):
    r: _T = sp_property(type="dynamic", units="m", ndims=1, data_type=float)

    z: _T = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
