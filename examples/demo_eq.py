import pandas as pd
import numpy as np
import pathlib
from scipy import constants
from pprint import pprint

if True:
    import sys
    sys.path.append("/home/salmon/workspace/fytok/python")
    sys.path.append("/home/salmon/workspace/SpDB/python")

    from fytok.load_profiles import (load_core_profiles, load_core_source,
                                     load_core_transport, load_equilibrium)
    from fytok.modules.transport.Equilibrium import Equilibrium
    from fytok.numlib.smooth import rms_residual
    from spdm.data import File, Function, Query
    from spdm.logger import logger


###########################


# from spdm.view.plot_profiles import plot_profiles, sp_figure


if __name__ == "__main__":
    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="geqdsk").read()
    desc = load_equilibrium(eqdsk_file,
                            coordinate_system={"theta": 64},
                            code={"name": "dummy"},
                            boundary={"psi_norm": 0.995}
                            )
    print(type(desc))

    pprint(desc["coordinate_system"])

    eq = Equilibrium(desc)

    print(np.asarray(eq.profiles_1d.rho_tor))
    print(np.asarray(eq.profiles_1d.q))
    print(np.asarray(eq.profiles_1d.dpressure_dpsi))
