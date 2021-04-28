
import collections
from datetime import datetime
from functools import cached_property

import numpy as np
from spdm.data.Node import Dict

VacuumToroidalField = collections.namedtuple("VacuumToroidalField", "r0 b0")

class Identifier(Dict):
    def __init__(self, *args, name=None, index=0, description=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name or ""
        self.index = index
        self.description = ""


class Signal(Dict):
    def __init__(self,  *args, time=None, data=None, **kwargs):
        super().__init__(*args, **kwargs)
        if type(time) is int:
            time = np.linspace(0, 1, time)
        elif type(time) is float:
            time = np.linspace(0, time, 128)
        elif isinstance(np.ndarray):
            pass
        elif time is None:
            time = np.linspace(*args, **kwargs)
        else:
            raise TypeError(type(time))
        self.__dict__["_time"] = time

        if isinstance(data, LazyProxy):
            data = data()
        self.__dict__["_data"] = data or np.full(self._time.shape, np.nan)

        assert(self._data.shape == self._time.shape)

    @property
    def time(self):
        return self._time

    @property
    def data(self):
        return self._data

    @cached_property
    def data_error_upper(self):
        """Upper error for "data" {dynamic} [as_parent]   """
        return NotImplemented

    @cached_property
    def data_error_lower(self):
        """Lower error for "data" {dynamic} [as_parent]   """
        return NotImplemented

    @cached_property
    def data_error_index(self):
        """Index in the error_description list for "data" {constant}"""
        return NotImplemented


class IDSProperties(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    @cached_property
    def comment(self):
        """Any comment describing the content of this IDS {constant}     STR_0D     """
        return super().__getitem__("comment")

    @cached_property
    def homogeneous_time(self):
        """This node must be filled (with 0, 1, or 2) for the IDS to be valid. If 1, the time of this IDS is homogeneous, i.e. the time values for this IDS are stored in the time node just below the root of this IDS. If 0, the time values are stored in the various time fields at lower levels in the tree. In the case only constant or static nodes are filled within the IDS, homogeneous_time must be set to 2 {constant}     INT_0D     """
        return super().__getitem__("homogeneous_time") or 1

    @cached_property
    def source(self):
        """Source of the data (any comment describing the origin of the data : code, path to diagnostic signals, processing method, ...) {constant}     STR_0D     """
        return super().__getitem__("source") or "non-documented"

    @cached_property
    def provider(self):
        """Name of the person in charge of producing this data {constant}     STR_0D     """
        return super().__getitem__("provider") or "non-documented"

    @cached_property
    def creation_date(self):
        """Date at which this data has been produced {constant}     STR_0D     """
        return super().__getitem__("creation_date") or datetime.now().strftime('%Y-%m-%d %H:%M:%S  %Z%z')

    @cached_property
    def version_put(self):
        """Version of the access layer package used to PUT this IDS"""
        return NotImplemented
