from functools import cached_property
from spdm.data.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from spdm.util.utilities import convert_to_named_tuple
import collections
VersionPut = collections.namedtuple("VersionPut", "data_dictionary  access_layer access_layer_language ")
IDSProperties = collections.namedtuple("IDSProperties", [
    # Any comment describing the content of this IDS {constant}     STR_0D
    "comment",
    # This node must be filled (with 0, 1, or 2) for the IDS to be valid. If 1, the time of this IDS is homogeneous, i.e.
    #  the time values for this IDS are stored in the time node just below the root of this IDS. If 0, the time values are stored
    #  in the various time fields at lower levels in the tree. In the case only constant or static nodes are filled within the IDS,
    #  homogeneous_time must be set to 2 {constant}     INT_0D
    "homogeneous_time",
    # Source of the data (any comment describing the origin of the data : code, path to diagnostic signals, processing method, ...) {constant}     STR_0D
    "source",
    # Name of the person in charge of producing this data {constant}     STR_0D
    "provider",
    # Date at which this data has been produced {constant}     STR_0D
    "creation_date",
    # Version of the access layer package used to PUT this IDS
    "version_put",
],
    defaults=["", 0, "", "", 0, VersionPut(0, 0, 0)]
)


class IDS(AttributeTree):
    """%%%DESCRIPTION%%%.      

        .. todo:: '___NAME___' IS NOT IMPLEMENTED
    """
    @staticmethod
    def __new__(cls,     *args, **kwargs):
        if cls is not IDS:
            return super(IDS, cls).__new__(cls)
        return NotImplemented
        # backend = str(config.engine) or ""
        # n_cls = cls

        # if not backend:
        #     pass
        # else:
        #     try:
        #         plugin_name = f"{__package__}.plugins.{cls.__name__.lower()}.Plugin{backend}"
        #         n_cls = sp_find_module(plugin_name, fragment=f"{cls.__name__}{backend}")
        #     except ModuleNotFoundError as error:
        #         logger.debug(error)
        #         n_cls = cls

        # return Dict.__new__(n_cls)

    def __init__(self, *args, ** kwargs):
        super().__init__(*args, ** kwargs)

    @cached_property
    def ids_properties(self):
        return IDSProperties(**self["ids_properties"])
