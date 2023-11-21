from fytok.modules.CoreTransport import CoreTransport
from spdm.data.Path import update_tree
from spdm.data.sp_property import sp_tree


@sp_tree
class TransportModelDummy(CoreTransport.Model):
    """
    Dummy CoreTransport.Model
    ===============================

    """

    code = {"name": "dummy", "description": f" Dummy CoreTransport.Model "}

    identifier = "unspecified"

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)

    def advance(self, *args, **kwargs):
        super().advance(*args, **kwargs)


__SP_EXPORT__ = TransportModelDummy
