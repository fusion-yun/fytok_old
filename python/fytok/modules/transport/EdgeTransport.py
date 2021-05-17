from spdm.data.Node import Dict
from ..common.IDS import IDS


class EdgeTransport(IDS):
    """Edge plasma transport. 

        Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)

        Todo:
            'EdgeTransport' IS NOT IMPLEMENTED
    """
    _IDS = "edge_transport"

    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)
