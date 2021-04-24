import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants
from fytok.Tokamak import Tokamak
from spdm.data.Collection import Collection
from spdm.data.File import File
from spdm.numerical.Function import Function
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles
import sys


if __name__ == "__main__":
    sys.path.append("/home/salmon/workspace/fytok/phys_modules/")
    import nclass

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml").entry
    equilibrium = File(
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/15MA inductive - burn/Increased domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16VVHR.txt",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        format="geqdsk").entry

    profile = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')

    tok = Tokamak({
        "radial_grid": {"axis": 64, "primary": "rho_tor_norm"},
        "wall":  device.wall,
        "pf_active": device.pf_active,
        "equilibrium": {
            "vacuum_toroidal_field": equilibrium.vacuum_toroidal_field,
            "global_quantities": equilibrium.global_quantities,
            "profiles_1d": equilibrium.profiles_1d,
            "profiles_2d": equilibrium.profiles_2d,
            "coordinate_system": {"grid": {"dim1": 64, "dim2": 512}}
        },
        "core_profiles": {
            "electrons": {
                "label": "electrons",
                "density":     1e19,
                "temperature": lambda x: (1-x**2)**2
            },
            "ion": [
                {
                    "label": "H+",
                    "z_ion": 1,
                    "neutral_index": 1
                },
                {
                    "label": "D+",
                    "z_ion": 1,
                    "neutral_index": 2
                }

            ],
            "conductivity_parallel": 1.0,
            "psi":   1.0,
        }
    })

    for ion in tok.core_profiles.ion:
        logger.debug((ion.label,ion.z_ion))

    core_transport = nclass.transport_nclass(tok.equilibrium, tok.core_profiles, tok.core_transport)
