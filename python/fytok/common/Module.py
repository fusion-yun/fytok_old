
from spdm.data.Node import Dict, List, Node, sp_property, TypeVar, Sequence
from spdm.flow.Actor import Actor
from spdm.util.logger import logger
from spdm.util.utilities import _undefined_, guess_class_name
from typing import Mapping, Union
from .Misc import Identifier
from .IDS import IDSCode


_TState = TypeVar("_TState")


class Module(Actor[_TState]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def code(self) -> IDSCode:
        return self.get("code", {})

    @sp_property
    def identifier(self) -> Identifier:
        return self.get("code", {})

    @sp_property
    def comment(self) -> str:
        return self.get("comment", "")

    def refresh(self, *args, **kwargs) -> float:
        return super().refresh(*args, **kwargs)
