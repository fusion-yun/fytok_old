from ..ontology import amns_data
from spdm.data.HTree import List, Dict, Node
from spdm.data.AoS import AoS


class AMNSData(amns_data._T_amns_data):
    pass


class AMNS(Dict[str, AMNSData]):
    pass



