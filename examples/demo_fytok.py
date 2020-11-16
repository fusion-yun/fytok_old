import matplotlib.pyplot as plt
import pprint
import sys
import numpy as np
sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


##################################################################################################

if __name__ == "__main__":

    from fytok.Tokamak import Tokamak
    from spdm.util.logger import logger

    tok = Tokamak.load_from("east+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east",
                            shot=55555, time_slice=100)

    # lfcs_r = ids.equilibrium.time_slice[itime].boundary.outline.r()[:, 0]
    # lfcs_z = ids.equilibrium.time_slice[itime].boundary.outline.z()[:, 0]
    # psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    # tok.equilibrium.update(constraints={"psivals": psivals})

    # bdr = np.array([p for p in tok.equilibrium.find_surface(0.6)])

    # tok.update(constraints={"psivals": psivals})

    fig = plt.figure()

    axis = fig.add_subplot(111)

    tok.equilibrium.plot(axis=axis)

    # axis.plot(bdr[:, 0], bdr[:, 1], "b--")

    tok.wall.plot(axis)

    # tok.plot(axis=axis)

    axis.axis("scaled")

    # fig = tok.equilibrium.plot_full(
    #     # x_axis=("rho_tor_norm", r"$\rho_{tor}/\rho_b$", r"$[-]$"),
    #     # profiles=[
    #     #     ("rho_tor", r"$\rho_{tor}$", r"$[m]$"),
    #     #     ("phi", r"$\Phi_{tor}$", r"$[Wb]$"),
    #     #     ("dpsi_drho_tor", r"$d\psi/d\rho_{tor}$", r"$[Wb/m]$"),
    #     #     ("vprime", r"$V^{\prime}$", r"$[m^3/Wb]$")
    #     # ]
    # )

    fig.savefig("../output/tokamak.svg", transparent=True)

    logger.info("Done")
