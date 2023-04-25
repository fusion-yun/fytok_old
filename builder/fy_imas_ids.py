import collections.abc

import numpy as np
from spdm.common.Plugin import Pluggable
from spdm.data.Dict import Dict
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property

from .utilities import _T_code, _T_ids_properties


class _T_module(Dict[Node], Pluggable):

    def __new__(cls, *args, **kwargs):
        return Pluggable.__new__(cls, *args, **kwargs)

    @classmethod
    def _guess_plugin_name(cls,  *args, **kwargs):
        if len(args) == 0:
            return []

        module_name = None

        if isinstance(args[0], collections.abc.Mapping):
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

    def update(self,  *args,  ** kwargs):
        """Refresh the data from the source"""
        pass


class _T_ids(_T_module):
    """ Base class of IDS """

    ids_properties: _T_ids_properties = sp_property()
    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    code: _T_code = sp_property()
    """Generic decription of the code-specific parameters for the code that has produced this IDS"""

    time: np.ndarray = sp_property(type="dynamic", units="s", ndims=1, data_type=float)
    """Generic time"""
