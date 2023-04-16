
import collections
import collections.abc
from dataclasses import dataclass

from spdm.common.Factory import Factory
from spdm.common.tags import _undefined_
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.util.logger import logger

from ..common.Misc import Identifier


class LibraryDesc(Dict[Node]):
    name: str = sp_property(default_value="")
    """ Name of software {constant}	STR_0D"""
    commit: str = sp_property(default_value="")      #
    """Unique commit reference of software {constant}	STR_0D"""
    version: str = sp_property(default_value="")      #
    """Unique version (tag) of software {constant}	STR_0D"""
    repository: str = sp_property(default_value="")      #
    """URL of software repository {constant}	STR_0D"""
    parameters: list = sp_property(default_value=None)     #
    """List of the code specific parameters in XML format {constant}"""


class ModuleCode(Dict[Node]):

    @sp_property
    def name(self) -> str:
        """Name of software generating IDS {constant}	STR_0D"""
        return self.get("name", None) or f"{self._parent.__class__.__module__}.{self._parent.__class__.__name__}"

    commit: str = sp_property()
    """	Unique commit reference of software {constant}	STR_0D"""

    version: str = sp_property()
    """Unique version (tag) of software {constant}	STR_0D"""

    repository: str = sp_property()
    """URL of software repository {constant}	STR_0D"""

    parameters: Dict[str] = sp_property()
    """List of the code specific parameters  {constant}	dict"""

    output_flag: List[int] = sp_property()
    """Output flag : 0 means the run is successful, other values mean some difficulty has been encountered, 
           the exact meaning is then code specific. Negative values mean the result shall not be used. {dynamic}	INT_1D	1- time"""

    library: List[LibraryDesc] = sp_property()
    "List of external libraries used by the code that has produced this IDS	struct_array [max_size=10]	1- 1...N"


class Module(Dict[Node], Factory):
    Code = ModuleCode

    _registry = {}

    def __new__(cls, *args, **kwargs):
        if not issubclass(cls, Module) or getattr(cls, "_IDS", None) is None:
            return object.__new__(cls)
        else:
            return Factory.__new__(cls, *args, **kwargs)

    @classmethod
    def _guess_class_name(cls,  *args, **kwargs):
        # pkg_prefix = getattr(cls, "_fy_module_prefix", None)

        # if cls is not Module or pkg_prefix is None:
        #     return super().__new__(cls)

        if len(args) == 0:
            return []
            # raise TypeError("Module() missing 1 required positional argument: 'desc'")

        module_name = None

        if isinstance(args[0], collections.abc.Mapping):
            module_name = args[0].get("code", {}).get("name", None)
        elif isinstance(args[0], Entry):
            module_name = args[0].get("code/name", None)

        if module_name is None:
            return []
        else:
            ids_name = getattr(cls, "_IDS", cls.__class__.__name__.lower())
            return ["/".join(["fymodules", ids_name, module_name])]

        # if isinstance(cls_name, str):
        #     return super().create(cls_name)
        # else:
        #     return object.__new__(cls)

    def __init__(self, *args, time=None, **kwargs):
        super().__init__(*args, **kwargs)
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

    code: Code = sp_property()

    identifier: Identifier = sp_property()

    comment: str = sp_property()

    def refresh(self,  *args, time=_undefined_, ** kwargs) -> float:
        """
            Function: update the current state of the Module without advancing the time.
            Return  : return the residual between the updated state and the previous state
        """
        if time is not _undefined_:
            self._time = time
        return 0.0
