from spdm.data.HTree import Dict

from ..ontology import amns_data

class AMNSData(amns_data._T_amns_data):
    pass


class AMNS(Dict[AMNSData]):
    pass
