from spdm.data.Node import Dict
from ..common.IDS import IDS


class EdgeProfiles(Dict):
    """Edge plasma profiles
         (includes the scrape-off layer and possibly part of the confined plasma)

       .. todo:: 'EdgeProfiles' IS NOT IMPLEMENTED
    """
    _IDS = "edge_profiles"

    def __init__(self, *args,   ** kwargs):
        super().__init__(*args, ** kwargs)
