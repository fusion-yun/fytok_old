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
        x_axis=("psi_norm",   r'$(\psi-\psi_{axis})/(\psi_{boundary}-\psi_{axis}) [-]$'),
        # x_axis=("rho_tor_norm",  r"$\rho_{tor}/\rho_{bndry}$"),
        profiles=[
            ({"name": "psi", "opts": {"label": r"$\psi$"}}, r"$[Wb]$"),
            ({"name": "rho_tor", "opts": {"label": r"$\rho_{tor}$"}}, r"$[m]$"),
            # ({"name": "rho_tor_norm", "opts": {"label": r"$\rho_{tor}/\rho_{bndry}$"}}, r"[m]"),
            # ({"name": "phi", "opts": {"label": r"$\Phi_{tor}$"}}, r"$[Wb]$"),
            # ([{"name": "q", "opts": {"label": r"$q_{exp}$"}},
            #   {"name": "flux_surface_average.q", "opts": {"label": r"$q_{average}$"}}],   r"$[-]$"),
            ({"name": "pprime", "opts": {"label": r"$p^{\prime}$"}}, r"$[Pa/Wb]$"),
            ({"name": "ffprime", "opts": {"label": r"$f f^{\prime}$"}}, r"$[T^2 \cdot m^2/Wb]$"),
            ({"name": "fpol", "opts": {"label":  r"$f_{pol}$"}}, r"$[T \cdot m]$"),
            ([{"name": "drho_tor_dpsi", "opts": {"label": r"$d\rho_{tor}/d\psi$"}}], r"$[m\cdot Wb]$"),
            ({"name": "dpsi_drho_tor", "opts": {"label": r"$d\psi/d\rho_{tor}$"}}, r"$[Wb/m]$"),
            ({"name": "vprime", "opts": {"label": r"$V^{\prime}$"}}, r"$[m^3/Wb]$"),
            ({"name": "gm1", "opts": {"label": r"$gm1=<\frac{1}{R^2}>$"}}, r"$[m^{-2}]$")
            ({"name": "gm4", "opts": {"label": r"$\left\langle \frac{1}{B^{2}}\right\rangle$"}}, r"$\left[T^{-2}\right]$")

            # ("pressure", r"pressure", r"$[Pa]$")
        ]
    )
    fig.savefig("../output/tokamak.svg", transparent=True)

    logger.info("Done")
