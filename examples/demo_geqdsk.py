import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Collection import Collection
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.data.File import File
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles

if __name__ == "__main__":

    doc = File("/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1", format="geqdsk")

    logger.debug(doc.entry.equilibrium.profiles_2d.psi)

