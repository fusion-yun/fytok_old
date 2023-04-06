import pandas as pd
import numpy as np
import pathlib
from scipy import constants
from pprint import pprint
from fytok.device.Wall import Wall

if True:
    import sys
    sys.path.append("/home/salmon/workspace/fytok/python")
    sys.path.append("/home/salmon/workspace/SpDB/python")

    from fytok.load_profiles import (load_core_profiles, load_core_source,
                                     load_core_transport, load_equilibrium)
    from fytok.transport.Equilibrium import Equilibrium
    from spdm.data import File, Function, Query
    from spdm.util.logger import logger


eqdsk_file = File("/home/salmon/workspace/atec/ATEC_craft/g070754.003740", format="geqdsk").read()
    #"/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="geqdsk").read()
desc = load_equilibrium(eqdsk_file,
                        coordinate_system={
                            "psi_norm": np.linspace(0, 0.995, 32),
                            "theta": 64},
                        code={"name": "dummy"},
                        boundary={"psi_norm": 0.995}
                        )
device_desc = File("/home/salmon/workspace/fytok_data/mapping/EAST/imas/3/static/config.xml", format="XML").read()

eq = Equilibrium(desc)

wall = Wall(device_desc.get("wall"))

wall.description_2d[0].limiter.unit[0].outline.r

psi_norm = np.linspace(0.0, 0.995, 32)


eqdsk_file = File("test.geqdsk", mode="w", format="geqdsk").write({
    "equilibrium": eq, "wall": wall
})


# pprint(p()[:10])
# pprint(p(psi_norm)[:10])
# pprint(eq.profiles_1d.q())
