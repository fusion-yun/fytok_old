import collections.abc

import numpy as np
from spdm.utils.Plugin import Pluggable
from spdm.data.Dict import Dict
from spdm.data.List import List

from spdm.data.Node import Node
from spdm.data.sp_property import sp_property

from .utilities import _T_code, _T_library, _T_ids_properties


class Library(_T_library):

    parameters: Dict[Node] = sp_property(type="constant")
    """List of the code specific parameters in XML format"""


class Code(_T_code):
    parameters: Dict[Node] = sp_property(type="constant")
    """List of the code specific parameters in XML format"""

    library: List[Library] = sp_property()
    """List of external libraries used by the code that has produced this IDS"""


class TimeSlice(Dict[Node]):
    
    time: float = sp_property(type="dynamic", units="s")
    """Time"""


class Module(Dict[Node], Pluggable):

    def __new__(cls, *args, **kwargs):
        return Pluggable.__new__(cls, *args, **kwargs)

    @classmethod
    def _guess_plugin_name(cls,  *args, code=None, **kwargs):
        if len(args) == 0 and code is None:
            return []

        module_name = None
        if isinstance(code, collections.abc.Mapping):
            module_name = code.get("name", None)

        if module_name is None and isinstance(args[0], collections.abc.Mapping):
            module_name = args[0].get("code", {}).get("name", None)
        elif hasattr(args[0], "__as_entry__"):
            module_name = args[0].__as_entry__().get("code/name", None)

        if module_name is not None:
            prefix: str = getattr(cls, "_plugin_prefix", cls.__name__.lower())
            if prefix.startswith('_t_'):
                prefix = prefix[3:]
            return [f"fytok/plugins/{prefix}/{module_name}"]
        else:
            return []

    code: Code = sp_property()
    """Generic decription of the code-specific parameters for the code that has produced this IDS"""

    def update(self,  *args,  ** kwargs):
        self._cache.clear()


class IDS(Module):
    """ Base class of IDS """

    ids_properties: _T_ids_properties = sp_property()
    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    time: np.ndarray = sp_property(type="dynamic", units="s", ndims=1, data_type=float)
    """Generic time"""
