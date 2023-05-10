from functools import cached_property

from  fytok._imas.lastest.edge_sources import _T_edge_sources, _T_edge_sources_source
from spdm.utils.tags import _undefined_
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.sp_property import sp_property, SpPropertyClass


class EdgeSources(_T_edge_sources):

    Source = _T_edge_sources_source

    @cached_property
    def source_combiner(self) -> Source:
        return self.source.combine({
            "identifier": {"name": "total", "index": 1,
                           "description": "Total source; combines all sources"},
            "code": {"name": _undefined_}
        })

    def update(self, *args,   **kwargs) -> float:
        if "source_combiner" in self.__dict__:
            del self.__dict__["source_combiner"]
        return sum([src.refresh(*args,   **kwargs) for src in self.source])
