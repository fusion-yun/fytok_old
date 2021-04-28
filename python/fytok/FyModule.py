from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.SpObject import SpObject

import collections


class FyModule(SpObject):

    @staticmethod
    def __new__(cls, metadata=None, *args,  **kwargs):
        if cls is not FyModule and metadata is None:
            return object.__new__(cls)

        if isinstance(metadata, collections.abc.Mapping):
            plugin_name = metadata.get("$plugin", None) or kwargs.get("_plugin", None)
        # elif isinstance(metadata, LazyProxy) and "$plugin" in metadata:
        #     plugin_name = metadata["$plugin"].__fetch__()
        else:
            plugin_name = None

        plugin_name = plugin_name or getattr(cls, "DEFAULT_PLUGIN", None)

        if not plugin_name:
            return SpObject.__new__(cls)
        else:
            path = __package__.split(".")
            ids = (getattr(cls, "IDS", None) or cls.__name__.lower()).split('.')
            n_cls = ".".join([path[0], "plugins", "modules", *ids, f"Plugin{plugin_name}"])
            logger.debug(f"Load FyModule {n_cls}")
            return SpObject.__new__(cls, n_cls)

    # @staticmethod
    # def __new__(cls,    *args, config=None,  **kwargs):
    #     if cls is not Equilibrium:
    #         return super(Equilibrium, cls).__new__(cls)
    #     if config is None:
    #         config = {}
    #     backend = config.get("engine", "FreeGS")
    #     n_cls = cls

    #     if backend != "":
    #         try:
    #             path = __package__.split(".")
    #             plugin_name = ".".join([path[0], "plugins", *path[1:], "equilibrium", f"Plugin{backend}"])
    #             n_cls = sp_find_module(plugin_name, fragment=f"Equilibrium{backend}")

    #         except ModuleNotFoundError as error:
    #             logger.debug(error)
    #             n_cls = cls
    #         else:
    #             logger.info(f"Load '{cls.__name__}' module {backend}!")

    #     return AttributeTree.__new__(n_cls)
