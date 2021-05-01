from spdm.data.Node import Dict


class EdgeProfiles(Dict):
    """Edge plasma profiles
         (includes the scrape-off layer and possibly part of the confined plasma)

       .. todo:: 'EdgeProfiles' IS NOT IMPLEMENTED
    """
    _IDS = "edge_profiles"

    def __init__(self, cache=None, *args, equilibrium=None, rho_tor_norm=None, ** kwargs):
        super().__init__(*args, ** kwargs)
        self.__dict__['_cache'] = cache or Dict()
