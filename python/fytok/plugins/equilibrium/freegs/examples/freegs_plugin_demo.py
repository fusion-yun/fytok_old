import sys
from spdm.utils.logger import logger
import pprint

# sys.path
sys.path.append("/gpfs/fuyun/projects/fuyun/fytok_ext/python")
# sys.path.append("/scratch/liuxj/workspace_plugins/freegs_plugin/python")


from fytok.Tokamak import Tokamak

# /scratch/liuxj/workspace_plugins/fytok_tutorial/tutorial/data/g070754.05000
tokamak = Tokamak(
    # f"file+geqdsk:///scratch/liuxj/workspace_plugins/fytok_tutorial/tutorial/data/g070754.05000",
    f"file+geqdsk:///home/salmon/workspace/fytok_data/data/g070754.05000",
    device="east",
    # shot=70754,
    equilibrium={
        "code": {
            "name": "freegs_plugin",
            "parameters": {"boundary": "fixed", "trim": 1},
        }
    },
)


tokamak.equilibrium.refresh()
# pprint.pprint(tokamak.equilibrium.time_slice.current.profiles_2d.psi._cache)
# pprint.pprint(tokamak.equilibrium.time_slice.current.profiles_2d.psi._cache[0])
# pprint.pprint(tokamak.equilibrium.time_slice.current.profiles_2d.psi._cache[-1])
