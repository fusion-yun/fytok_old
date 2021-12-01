
import collections
import collections.abc

from spdm.common.logger import logger
from spdm.common.SpObject import SpObject
from spdm.common.tags import _undefined_
from spdm.data.Entry import Entry
from spdm.data.Node import Dict, Node, sp_property

from .Misc import Identifier


class Module(SpObject, Dict[Node]):

    def __new__(cls, desc=None, *args, **kwargs):

        prefix = getattr(cls, "_fy_module_prefix", None)

        cls_name = None

        if cls is not Module and prefix is None:
            pass
        elif isinstance(desc, collections.abc.Mapping):
            cls_name = desc.get("code", {}).get("name", None)
        elif isinstance(desc, Entry):
            cls_name = desc.get("code.name", "")

        if isinstance(cls_name, str):
            cls_name = f"{prefix}{cls_name}"

        if isinstance(cls_name, str):
            return SpObject.new_object(cls_name)
        else:
            return object.__new__(cls)

    def __init__(self, d=None,  /, time=None,  **kwargs):
        Dict.__init__(self, d,  **kwargs)
        self._job_id = 0  # Session.current().job_id(self.__class__.__name__)
        self._time = time if time is not None else 0.0

    def __del__(self):
        # logger.debug(f"Delete Module {guess_class_name(self.__class__)}")
        pass

    @property
    def time(self):
        return self._time

    def job_id(self):
        return self._job_id

    @sp_property
    def code(self):
        return self.get("code", {})

    @sp_property
    def identifier(self) -> Identifier:
        return self.get("identifier", {})

    @sp_property
    def comment(self) -> str:
        return self.get("comment", "")

    def refresh(self,  *args, time=_undefined_, ** kwargs) -> float:
        """
            Function: update the current state of the Module without advancing the time.
            Return  : return the residual between the updated state and the previous state
        """
        if time is not _undefined_:
            self._time = time
        return 0.0
