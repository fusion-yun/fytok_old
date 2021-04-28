from spdm.data.AttributeTree import AttributeTree
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module


class IDS(AttributeTree):
    """%%%DESCRIPTION%%%.      

        .. todo:: '___NAME___' IS NOT IMPLEMENTED
    """
    @staticmethod
    def __new__(cls,   config,  *args, **kwargs):
        if cls is not IDS:
            return super(IDS, cls).__new__(cls)

        backend = str(config.engine) or ""
        n_cls = cls

        if not backend:
            pass
        else:
            try:
                plugin_name = f"{__package__}.plugins.{cls.__name__.lower()}.Plugin{backend}"
                n_cls = sp_find_module(plugin_name, fragment=f"{cls.__name__}{backend}")
            except ModuleNotFoundError as error:
                logger.debug(error)
                n_cls = cls

        return AttributeTree.__new__(n_cls)

    def __init__(self, *args, ** kwargs):
        super().__init__(*args, ** kwargs)



