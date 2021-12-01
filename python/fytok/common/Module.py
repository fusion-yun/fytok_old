
import collections
import collections.abc

from spdm.common.logger import logger
from spdm.common.SpObject import SpObject
from spdm.common.tags import _undefined_
from spdm.data.Entry import Entry
from spdm.data.Node import sp_property
from spdm.flow.Actor import Actor

from .Misc import Identifier


class Module(Actor):

    def __new__(cls, desc=None, *args, **kwargs):

        prefix = getattr(cls, "_fy_module_prefix", None)

        cls_name = None

        if cls is not Actor and prefix is None:
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def code(self):
        return self.get("code", {})

    @sp_property
    def identifier(self) -> Identifier:
        return self.get("identifier", {})

    @sp_property
    def comment(self) -> str:
        return self.get("comment", "")

    def refresh(self, *args, **kwargs) -> float:
        return super().refresh(*args, **kwargs)
