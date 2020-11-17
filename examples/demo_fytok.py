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

    # lfcs_r = tok.equilibrium.boundary.outline.r
    # lfcs_z = tok.equilibrium.boundary.outline.z
    # psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    # tok.equilibrium.update(constraints={"psivals": psivals})

    # bdr = np.array([p for p in tok.equilibrium.find_surface(0.6)])

    # tok.update(constraints={"psivals": psivals})

    # fig = plt.figure()

    # axis = fig.add_subplot(111)

    # tok.equilibrium.plot(axis=axis)

    # # axis.plot(bdr[:, 0], bdr[:, 1], "b--")

    # tok.wall.plot(axis)

    # # tok.plot(axis=axis)

    # axis.axis("scaled")

    fig = tok.equilibrium.plot_full(
        profiles=[
            ([{"name": "q", "opts": {"label": r"$q_{exp}$"}},
              {"name": "q1", "opts": {"label": r"$q_{average}$"}}],   r"$[-]$"),
            ({"name": "pprime", "opts": {"label": r"$p^{\prime}$"}}, r"$p^{\prime}[Pa/Wb]$"),
            ({"name": "ffprime", "opts": {"label": r"$f f^{\prime}$"}}, r"$f f^{\prime}[T^2 \cdot m^2/Wb]$"),
            ({"name": "fpol", "opts": {"label":  r"$f_{pol}$"}}, r"$f_{pol}[T \cdot m]$"),
            ({"name": "rho_tor", "opts": {"label": r"$\rho_{tor}$"}}, r"$\rho_{tor}[m]$"),
            ({"name": "phi", "opts": {"label": r"$\Phi_{tor}$"}}, r"$\Phi_{tor}[Wb]$"),
            ({"name": "dpsi_drho_tor", "opts": {"label": r"$d\psi/d\rho_{tor}$"}}, r"$d\psi/d\rho_{tor}[Wb/m]$"),
            ({"name": "vprime", "opts": {"label": r"$V^{\prime}$"}}, r"$V^{\prime}[m^3/Wb]$")
            # ("pressure", r"pressure", r"$[Pa]$")
        ]
    )
    fig.savefig("../output/tokamak.svg", transparent=True)

    logger.info("Done")
