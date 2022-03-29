import matplotlib.pyplot as plt
import numpy as np
from fytok.Tokamak import Tokamak
from spdm.logger import logger
from spdm.data import Dict, Function, Collection
from spdm.data.File import File
from spdm.util.plot_profiles import plot_profiles

if __name__ == "__main__":

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml")

    equilibrium = File(
        # "/home/salmon/workspace/fytok/examples/data/g063982.04800",
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        format="geqdsk")

    tok = Tokamak({
        "wall":  device.entry.wall,
        "pf_active": device.entry.pf_active,
        "equilibrium": equilibrium.entry.equilibrium
    })

    logger.debug(equilibrium.entry.equilibrium)

    fig = plt.figure()

    fig.gca().contour(
        tok.equilibrium.profiles_2d.r,  # .reshape([81, 56]),
        tok.equilibrium.profiles_2d.z,  # .reshape([81, 56]),
        tok.equilibrium.profiles_2d.psi,
        levels=32,
        linewidths=0.1
    )

    tok.plot(fig.gca(),
             wall={"limiter": {"edgecolor": "green"}, "vessel": {"edgecolor": "blue"}},
             pf_active={"facecolor": 'red'},
             equilibrium={"boundary": True, "mesh": True}
             )

    # bdry = np.asarray([[r, z] for r, z in tok.equilibrium.magnetic_flux_coordinates.find_by_psinorm(1.0)]).T

    # fig.gca().plot(bdry[ 0], bdry[1])

    plt.savefig("/home/salmon/workspace/output/iter_contour.svg")
