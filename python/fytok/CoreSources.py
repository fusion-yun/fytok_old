
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from functools import cached_property, lru_cache
from spdm.util.Profiles import Profiles


class CoreSources(AttributeTree):
    """CoreSources
    """
    IDS = "core_sources"

    def __init__(self, cache, *args, tokamak=None, **kwargs):
        super().__init__(*args, **kwargs)

    # class Profiles1D(Profiles):
    #     def __init__(self, cache=None,  *args, parent=None, **kwargs):
    #         pass

    #     @property
    #     def grid(self):

    # @cached_property
    # def profiles_1d(self):
    #     return Profiles1D(self._cache.profiles_1d, parent=self)
