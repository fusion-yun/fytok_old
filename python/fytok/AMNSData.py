
# This is file is generated from template
from fytok.IDS import IDS

class AMNSData(IDS):
    r"""Atomic, molecular, nuclear and surface physics data. Each occurrence contains the data for a given element (nuclear charge), describing various physical processes. For each process, data tables are organized by charge states. The coordinate system used by the data tables is described under the coordinate_system node.
        
        Note: AMNSData is an ids
    """
    IDS="amns_data"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
