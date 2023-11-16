
from fytok.modules.CoreTransport import CoreTransport
from spdm.data.Path import update_tree


class TransportModelDummy(CoreTransport.Model):
    """
       Dummy CoreTransport.Model
       ===============================

    """

    def __init__(self, d, *args,   **kwargs):
        super().__init__(update_tree({
            "identifier": "unspecified",
            "code": {"name": "dummy",
                     "description": f"{self.__class__.__name__} Dummy CoreTransport.Model "}}, d),
            *args, **kwargs)

    def refresh(self, *args, **kwargs) -> float:
        return super().refresh(*args, **kwargs)

    def advance(self, *args, **kwargs):
        return super().advance(*args, **kwargs)


__SP_EXPORT__ = TransportModelDummy
