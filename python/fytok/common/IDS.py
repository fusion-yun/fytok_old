import typing

from spdm.data.sp_property import sp_property

# from _imas.utilities import _T_code, _T_ids_properties
from .Module import Module


class IDS(Module):
    """ Base class of all IDS classes.  """
    _IDS = None

    # code:  _T_code = sp_property()

    # ids_properties: _T_ids_properties = sp_property(default_value={})

    def __serialize__(self, properties: typing.Optional[typing.Sequence] = None):
        res = super().__serialize__(properties=properties)
        res["@ids"] = self._IDS
        return res

    @classmethod
    def __deserialize__(cls, desc: typing.Mapping):
        ids = desc.get("@ids", None)
        if ids is None:
            raise ValueError(desc)
        else:
            raise NotImplementedError(ids)
