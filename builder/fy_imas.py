

import collections.abc

from spdm.utils.Pluggable import Pluggable
import numpy as np


class _T_Library(_T_library):
    parameters: Dict[Node] = sp_property(type="constant")
    """List of the code specific parameters in XML format"""


class _T_Code(_T_code):
    parameters: Dict[Node] = sp_property(type="constant")
    """List of the code specific parameters in XML format"""

    library: List[_T_Library] = sp_property()
    """List of external libraries used by the code that has produced this IDS"""


class _T_Module(Dict[Node], Pluggable):
    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, name_list, self, code, *args, **kwargs) -> None:
        if name_list is None:
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
                name_list = [f"fytok/plugins/{prefix}/{module_name}"]

        super().__dispatch__init__(name_list, self, code, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if self.__class__ is _T_Module or "_plugin_registry" in vars(self.__class__):
            _T_Module.__dispatch__init__(None, self, *args, **kwargs)
            return
        super().__init__(*args, **kwargs)

    code: _T_Code = sp_property()
    """Generic decription of the code-specific parameters for the code that has produced this IDS"""


class _T_IDS(_T_Module):
    """ Base class of IDS """

    _plugin_registry = {}

    ids_properties: _T_ids_properties = sp_property()
    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    time: np.ndarray = sp_property(type="dynamic", units="s", ndims=1, data_type=float)
    """Generic time"""

    def update(self,  *args, time=None, ** kwargs):
        if time is not None:
            self.time.append(time)
