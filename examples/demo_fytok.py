import matplotlib.pyplot as plt
import pprint
import sys
import numpy as np
import scipy.constants as constants
sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


##################################################################################################

def draw(tok):

    return tok.equilibrium.plot(
        # x_axis=("psi_norm",   r'$(\psi-\psi_{axis})/(\psi_{boundary}-\psi_{axis}) [-]$'),
        x_axis=("rho_tor_norm",  r"$\rho_{tor}/\rho_{bndry}$"),
        profiles=[
            ({"name": "pprime", "opts": {"label": r"$p^{\prime}$"}}, r"$[Pa/Wb]$"),
            ({"name": "pressure", "opts": {"label": r"$p$"}}, r"$[Pa]$"),
            ({"name": "ffprime", "opts": {"label": r"$f f^{\prime}$"}}, r"$[T^2 \cdot m^2/Wb]$"),
            ({"name": "fpol", "opts": {"label": r"$f_{pol}$"}}, r"$\left[T\cdot m\right]$"),
            ({"name": "phi", "opts": {"label": r"$\Phi_{tor}$"}}, r"$[Wb]$"),
            ({"name": "rho_tor", "opts": {"label": r"$\rho_{tor}$"}}, r"$[m]$"),
            ({"name": "drho_tor_dpsi", "opts": {"label": r"$d\rho / d\psi$"}}, r"$\left[m/Wb\right]$"),
            # ({"name": "dpsi_drho_tor", "opts": {"label": r"$d\psi/d\rho_{tor}$"}}, r"$[Wb/m]$"),
            # ({"name": "rho_tor_norm", "opts": {"label": r"$\rho_{tor}/\rho_{bndry}$"}}, r"[m]"),
            ([{"name": "q", "opts": {"label": r"$q$", "marker": "o", "markersize": 0.5}},
              #   {"name": "cache.q", "opts": {"label": r"$q_{exp}$", "marker": "o", "markersize": 0.5}}
              ], r"$[-]$"),
            ({"name": "vprime", "opts": {"label": r"$V^{\prime}$"}}, r"$[m^3/Wb]$"),
            ({"name": "gm1", "opts": {"label": r"$gm1=<R^{-2}>$"}}, r"$[m^{-2}]$"),
            ({"name": "gm2", "opts": {"label": r"$gm2=<|\nabla \rho / R|^2>$"}}, r"$[m^{-2}]$"),
            ({"name": "gm3", "opts": {"label": r"$gm3: \left\langle  |\nabla\rho |^{2}\right\rangle $"}},
             r"$\left[-\right]$"),
            # ({"name": "gm4", "opts": {"label": r"$gm4: \left\langle 1/B^{2} \right\rangle$"}},   r"$\left[T^{-2}\right]$"),
            ({"name": "gm5", "opts": {"label": r"$gm5: \left\langle B^{2}\right\rangle$"}},  r"$\left[T^2\right]$"),
            ({"name": "psi"}, r"$[Wb]$"),
            ({"name": "dpsi_drho_tor"}, r"$[Wb]$"),
            ({"name": "psi_norm", "opts": {"label": r"$\psi_{norm}$"}}, r"$[Wb]$")
        ],
        # profiles_2d=[("phi", {})],
        vec_field=[("b_field_r", "b_field_z", {"linewidth": 0.5, "arrowsize": 0.2})],
        # surface_mesh=True,
        machine={"coils": False}
    )


if __name__ == "__main__":

    from fytok.Tokamak import Tokamak
    from spdm.util.logger import logger
    from spdm.data.Entry import open_entry
    from fytok.Plot import plot_profiles
    from spdm.util.AttributeTree import _next_

    tok = Tokamak(open_entry("east+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east", shot=55555, time_slice=20))

    # draw(tok).savefig("../output/tokamak0.png", transparent=True)

    lfcs_r = tok.equilibrium.boundary.outline.r
    lfcs_z = tok.equilibrium.boundary.outline.z
    # psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    xpoints = [[p.r, p.z] for p in tok.equilibrium.boundary.x_point]

    ir_min = np.argmin(lfcs_r)
    ir_max = np.argmax(lfcs_r)
    iz_min = np.argmin(lfcs_z)
    iz_max = np.argmax(lfcs_z)

    isoflux = [(lfcs_r[ir_min], lfcs_z[ir_min], lfcs_r[ir_max], lfcs_z[ir_max]),
               (lfcs_r[iz_min], lfcs_z[iz_min], lfcs_r[iz_max], lfcs_z[iz_max])]  # (R1,Z1, R2,Z2) pair of locations

    # xpoints[0][0] += 0.1
    # xpoints[0][1] -= 0.1
    # xpoints[1][0] += 0.1
    # xpoints[1][1] += 0.1

    # tok.equilibrium.update(
    #     constraints={
    #         # "psivals": psivals,
    #         "xpoints": xpoints,
    #         "isoflux": isoflux
    #     })

    fun = tok.equilibrium.profiles_1d.gm2*tok.equilibrium.profiles_1d.vprime * tok.equilibrium.profiles_1d.dpsi_drho_tor / \
        tok.equilibrium.profiles_1d.fpol / (4.0*(constants.pi**2)*tok.equilibrium.profiles_1d.rho_tor[-1])
    dfun = tok.equilibrium.profiles_1d.derivative(fun)

    j_total = -dfun*(tok.equilibrium.profiles_1d.fpol**2)/tok.equilibrium.profiles_1d.vprime/tok.equilibrium.profiles_1d.rho_tor / \
        (constants.mu_0*tok.vacuum_toroidal_field.b0*constants.pi*2.0)

    tok.constraints = {"xpoints": xpoints,
                       "isoflux": isoflux,
                       # "psivals": psivals,
                       }

    tok.core_transport[_next_] = {"identifier": {"name": "unspecified", "index": 0},
                                  "profiles_1d": {"conductivity_parallel": 1.0e-8}}

    tok.core_sources[_next_] = {"identifier": {"name": "unspecified", "index": 0},
                                "profiles_1d": {"j_parallel": j_total}}

    plot_profiles(tok.equilibrium.profiles_1d,
                  profiles=["psi_norm",["q","q1"] ,"vprime", "fpol", "gm2"],
                  x_axis="psi_norm", grid=True)[1] .savefig("../output/eq_profiles_1d.svg")

    tok.update()

    draw(tok).savefig("../output/tokamak1.svg", transparent=True)

    plot_profiles(tok.core_profiles.profiles_1d,
                  profiles=[
                      [
                          {"name": "psi0", "opts": {"marker": "+", "label": r"$\psi^{-1}$"}},
                          #   {"name": "psi", "x_axis": "rho_tor_norm2", "opts": {"marker": "+", "label": r"$\psi$"}}
                      ],
                      ([
                          {"name": "dpsi0", "opts": {"marker": "+", "label": r"$d\psi^{-1}/d\rho_{tor,norm}$"}},
                          #   {"name": "dpsi_drho_tor", "x_axis": "rho_tor_norm2",
                          #       "opts": {"marker": "+", "label": r"$d\psi/d\rho_{tor,norm}$"}}
                      ], r"$[Wb/m]$"),
                      "j_total", "fcoeff", "A", "B", "C", "volume", "dc", "c",
                      ["ddpsi0", "C_A"]

                  ],
                  x_axis="rho_tor_norm", grid=True)[1] .savefig("../output/core_profiles.svg")
    #
    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0)
    # fig.align_ylabels()
    # fig.savefig("../output/core_profiles.svg", transparent=True)

    # tok.equilibrium.plot().savefig("../output/tokamak1.svg", transparent=True)
    # bdr = np.array([p for p in tok.equilibrium.find_surface(0.6)])

    # tok.update(constraints={"psivals": psivals})

    # fig = plt.figure()

    # axis = fig.add_subplot(111)

    # tok.equilibrium.plot(axis=axis)

    # # axis.plot(bdr[:, 0], bdr[:, 1], "b--")

    # tok.wall.plot(axis)

    # # tok.plot(axis=axis)

    # axis.axis("scaled")

    logger.info("Done")
