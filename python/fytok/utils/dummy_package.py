
from spdm.data.sp_property import PropertyTree
import typing


class DummyModule:
    def __init__(self, name):
        self._module = name

    def __str__(self) -> str:
        return f"<dummy_module '{__package__}.dummy.{self._module}'>"

    def __getattr__(self, __name: str) -> typing.Type[PropertyTree]:
        cls = type(__name, (PropertyTree,), {})
        cls.__module__ = f"{__package__}.dummy.{self._module}"
        return cls


def __getattr__(key: str) -> DummyModule:
    return DummyModule(key)
