import matplotlib.pyplot as plt
import pprint
import sys

sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


##################################################################################################

if __name__ == "__main__":

    from fytok.Tokamak import Tokamak
    from fytok.CoreProfiles import CoreProfiles

    from fytok.PFActive import PFActive
    from fytok.Wall import Wall
    from spdm.data.Entry import open_entry
    from spdm.util.logger import logger

    ids = open_entry("east+mdsplus:///home/salmon/public_data/~t/?default_tree_name=efit_east#shot=55555")
    itime = 10

    tok = Tokamak.load_imas(ids, itime)

    lfcs_r = ids.equilibrium.time_slice[itime].boundary.outline.r()[:, 0]
    lfcs_z = ids.equilibrium.time_slice[itime].boundary.outline.z()[:, 0]
    psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    tok.equilibrium.solve(constraints={"psivals": psivals})
    tok.equilibrium.update_boundary()
    # fig = tok.equilibrium.plot_full()

    # tok.update(constraints={"psivals": psivals})

    fig = plt.figure()

    axis = fig.add_subplot(111)

    tok.equilibrium.plot(axis=axis)

    # tok.plot(axis=axis)

    fig.savefig("../output/tokamak.svg")

    logger.info("Done")
