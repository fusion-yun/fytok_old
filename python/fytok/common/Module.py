
from spdm.data.Node import Dict, List, Node, sp_property, TypeVar, Sequence
from spdm.flow.Actor import Actor
from spdm.util.logger import logger
from spdm.util.utilities import _undefined_, guess_class_name
from typing import Mapping, Union
from .Misc import Identifier
from .IDS import IDSCode


_TState = TypeVar("_TState")


class Module(Actor[_TState]):
    _actor_module_prefix = _undefined_

    def __init__(self,   d, /,
                 identifier: Union[Mapping, Identifier] = _undefined_,
                 code: Union[Mapping, IDSCode] = _undefined_,
                 comment: str = _undefined_,
                 **kwargs):
        super().__init__(d, **kwargs)
        self.update({"identifier": identifier, "code": code, "comment": comment})
        self._inputs = kwargs
        logger.debug(f"Inititalize Module {guess_class_name(self.__class__)}")

    def __del__(self):
        logger.debug(f"Delete Module {guess_class_name(self.__class__)}")

    @sp_property
    def code(self) -> IDSCode:
        return self.get("code", {})

    @sp_property
    def identifier(self) -> Identifier:
        return self.get("code", {})

    @sp_property
    def comment(self) -> str:
        return self.get("comment", "")

    def refresh(self, d=None, /, **inputs) -> None:
        logger.debug(f"Refresh SubModule {guess_class_name(self.__class__)}")
        pass
