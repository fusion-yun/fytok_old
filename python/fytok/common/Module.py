
import collections
import collections.abc
from spdm.util.logger import logger
from spdm.common.tags import _undefined_
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.common.Factory import Factory
from ..common.Identifier import Identifier


class Module(Dict[Node], Factory):

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

    @ property
    def time(self):
        return self._time

    def job_id(self):
        return self._job_id

    code: Dict = sp_property()

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
