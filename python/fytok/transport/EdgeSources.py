from functools import cached_property

from ..common.IDS import IDS
from spdm.common.tags import _not_found_, _undefined_
from spdm.data import (Dict, File, Function, Link, List, Node, Path, Query,
                       sp_property)


class EdgeSourcesSource(Dict):
    pass


class EdgeSources(IDS):
    """Edge plasma sources. 

        Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)

        .. todo:: 'EdgeSource' IS NOT IMPLEMENTED
    """
    _IDS = "edge_sources"
    Source = EdgeSourcesSource

    source: List[Source] = sp_property()

    @cached_property
    def source_combiner(self) -> Source:
        return self.source.combine({
            "identifier": {"name": "total", "index": 1,
                           "description": "Total source; combines all sources"},
            "code": {"name": _undefined_}
        })

    def refresh(self, *args,   **kwargs) -> float:
        if "source_combiner" in self.__dict__:
            del self.__dict__["source_combiner"]
        return sum([src.refresh(*args,   **kwargs) for src in self.source])
