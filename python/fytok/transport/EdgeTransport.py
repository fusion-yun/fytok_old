from functools import cached_property

from spdm.common.tags import _undefined_
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.sp_property import sp_property
from ..common.IDS import IDS


class EdgeTransportModel(Dict):
    pass


class EdgeTransport(IDS):
    """Edge plasma transport. 

        Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)

        Todo:
            'EdgeTransport' IS NOT IMPLEMENTED
    """
    _IDS = "edge_transport"
    Model = EdgeTransportModel

    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def model(self) -> List[Model]:
        return List[EdgeTransport.Model](self.get("model", []),  parent=self)

    @cached_property
    def model_combiner(self) -> Model:
        return self.model.combine({
            "identifier": {"name": "combined", "index": 1,
                           "description": """Combination of data from all available transport models"""},
            "code": {"name": _undefined_}
        })

    def refresh(self, *args, **kwargs) -> float:
        if "model_combiner" in self.__dict__:
            del self.__dict__["model_combiner"]
        return sum([model.refresh(*args,   **kwargs) for model in self.model])
