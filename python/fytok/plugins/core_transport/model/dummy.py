from fytok.modules.CoreTransport import CoreTransport
from spdm.data.sp_property import sp_tree


@sp_tree
class TransportModelDummy(CoreTransport.Model):
    """
    Dummy CoreTransport.Model
    """

    code = {"name": "dummy", "description": f" Dummy CoreTransport.Model "}

    identifier = "unspecified"
