from scipy.optimize import root_scalar
from spdm.data.Entry import open_entry
from spdm.data.Profile import Profiles
from spdm.util.AttributeTree import AttributeTree, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from spdm.util.SpObject import SpObject

class FyModule(AttributeTree, SpObject):

    @staticmethod
    def __new__(cls, *args, _metadata=None, **kwargs):
        if cls is not FyModule and _metadata is None:
            return object.__new__(cls)
      
         
        if config is None:
            config = {}
        backend = config.get("engine", "FreeGS")
        n_cls = cls

        if backend != "":
            try:
                path = __package__.split(".")
                plugin_name = ".".join([path[0], "plugins", *path[1:], "equilibrium", f"Plugin{backend}"])
                n_cls = sp_find_module(plugin_name)

            except ModuleNotFoundError as error:
                logger.debug(error)
                n_cls = cls
            else:
                logger.info(f"Load '{cls.__name__}' module {backend}!")

        return SpObject.__new__(n_cls)
