import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Collection import Collection
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.data.File import File
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles
from fytok.Tokamak import Tokamak

if __name__ == "__main__":

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml")

    equilibrium = File(
        # "/home/salmon/workspace/fytok/examples/data/g063982.04800",
        "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        format="geqdsk")

    tok = Tokamak({
        "wall":  device.entry.wall,
        "pf_active": device.entry.pf_active,
        "equilibrium": {"time_slice": equilibrium.entry.equilibrium}
    })

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
             equilibrium={"boundary": False, "mesh": False}
             )

    plt.savefig("/home/salmon/workspace/output/iter_contour.svg")
