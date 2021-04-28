from spdm.data.AttributeTree import AttributeTree


class EdgeTransport(AttributeTree):
    """Edge plasma transport. 

        Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)

        Todo:
            'EdgeTransport' IS NOT IMPLEMENTED
    """
    IDS = "edge_transport"

    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)
