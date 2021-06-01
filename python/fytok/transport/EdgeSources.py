from spdm.data.Node import Dict
from ..common.IDS import IDS


class EdgeSources(IDS):
    """Edge plasma sources. 

        Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)

        .. todo:: 'EdgeSource' IS NOT IMPLEMENTED
    """
    _IDS = "edge_sources"

    def __init__(self,   *args,   ** kwargs):
        super().__init__(*args, ** kwargs)
