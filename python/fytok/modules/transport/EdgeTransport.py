from spdm.data.PhysicalGraph import PhysicalGraph, _next_

class EdgeTransport(PhysicalGraph):
    """Edge plasma transport. 
      
        Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)

        Todo:
            'EdgeTransport' IS NOT IMPLEMENTED
    """
    IDS="edge_transport"
    def __init__(self, cache=None, *args, equilibrium=None, rho_tor_norm=None, ** kwargs):
        pass
