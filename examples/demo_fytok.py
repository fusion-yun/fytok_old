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

    device = open_entry("east+mdsplus:///home/salmon/public_data/~t/?default_tree_name=efit_east#shot=55555")
 
    tok = Tokamak()

    tok.load(device)

    lfcs_r = device.equilibrium.time_slice[10].boundary.outline.r()[:, 0]
    lfcs_z = device.equilibrium.time_slice[10].boundary.outline.z()[:, 0]
    psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    tok.equilibrium.solve(None, constraints={"psivals": psivals})

    # tok.solve(0.1, max_iters=1, constraints={"psivals": psivals})
    # fig = plt.figure()
    # tok.plot(axis=fig.add_subplot(111))
    # fig.savefig("tokamak.svg")

    tok.equilibrium.plot_full().savefig("../output/equilibrium_full.svg")


    logger.info("Done")
