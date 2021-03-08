from spdm.data.PhysicalGraph import PhysicalGraph, _next_

class EdgeSources(PhysicalGraph):
    """Edge plasma sources. 
    
        Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)

        .. todo:: 'EdgeSource' IS NOT IMPLEMENTED
    """
    IDS="edge_sources"
    def __init__(self, cache=None, *args, equilibrium=None, rho_tor_norm=None, ** kwargs):
        super().__init__(*args, ** kwargs)
        self.__dict__['_cache'] = cache or PhysicalGraph()

