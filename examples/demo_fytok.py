import matplotlib.pyplot as plt
import pprint
import sys
import numpy as np
import scipy.stats
import scipy.constants as constants
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

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
    from spdm.util.Profiles import Profile
    from spdm.data.Entry import open_entry
    from fytok.Plot import plot_profiles
    from spdm.util.AttributeTree import _next_

    tok = Tokamak(open_entry("east+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east", shot=55555, time_slice=20))

    rho_b = 0.96
    rho = np.linspace(0, 1.0, 129)

    def D(x): return 0.5 + (x**3) if x < rho_b else 0.1

    def v(x): return -(x**2)*0.4 if x < rho_b else 0.1

    tok.core_transport[_next_] = {"identifier": {"name": "unspecified", "index": 0}}

    trans = tok.core_transport[-1].profiles_1d

    trans.conductivity_parallel = 1.0e-8

    trans.electrons.particles.d = D
    trans.electrons.particles.v = v

    plot_profiles(trans, profiles=[
        ["electrons.particles.d", "electrons.particles.v"], "conductivity_parallel"
    ], x_axis="grid_d.rho_tor_norm", grid=True).savefig("../output/core_tranport.svg")

    tok.core_sources[_next_] = {"identifier": {"name": "unspecified", "index": 0}}

    src = tok.core_sources[-1].profiles_1d

    def S_pel(rho, pos=0.7, w=0.1, S0=1.0e19): return scipy.stats.norm.pdf((rho-0.7)/w) * \
        np.sqrt(scipy.constants.pi*2.0)*S0

    def S_edge(rho, pos=0.7, w=0.03, S0=8.0e16): return np.piecewise(
        rho, [rho < pos, rho >= pos], [0, lambda x: (np.exp((x-pos)/w-10)*S0-1.0)])

    gamma = tok.equilibrium.profiles_1d.dvolume_drho_tor / (4.0*(constants.pi**2)) \
        * tok.equilibrium.profiles_1d.gm2    \
        / tok.equilibrium.profiles_1d.fpol \
        * tok.equilibrium.profiles_1d.dpsi_drho_tor

    dgamma = Profile(tok.equilibrium.profiles_1d.rho_tor, gamma).derivative

    j_total = dgamma  \
        * (tok.equilibrium.profiles_1d.fpol**2) \
        / (constants.mu_0*tok.vacuum_toroidal_field.b0) \
        * (constants.pi*2.0)
    j_total[1:] /= tok.equilibrium.profiles_1d.dvolume_drho_tor[1:]

    j_total[0] = 2*j_total[1]-j_total[2]

    src.j_parallel = tok.equilibrium.profiles_1d.mapping("rho_tor_norm", j_total)

    src.electrons.particles = lambda rho: S_edge(rho)

    plot_profiles(src, profiles=[
        "electrons.particles", "grid.rho_tor_norm"
    ], x_axis="grid.rho_tor_norm", grid=True).savefig("../output/core_sources.svg")

    def ne(rho, rho_b=rho_b, w=2.0, n_0=0.95e19): return np.piecewise(rho, [rho < rho_b, rho > rho_b], [
        lambda x:n_0*((1-(x/w)**2)**2), lambda x:n_0*((1-(rho_b/w)**2)**2)*np.exp(-((x-rho_b)*20)**2)])

    tok.core_profiles.profiles_1d.electrons.density = ne

    # draw(tok).savefig("../output/tokamak0.png", transparent=True)

    # lfcs_r = tok.equilibrium.boundary.outline.r
    # lfcs_z = tok.equilibrium.boundary.outline.z
    # # psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    # xpoints = [[p.r, p.z] for p in tok.equilibrium.boundary.x_point]

    # ir_min = np.argmin(lfcs_r)
    # ir_max = np.argmax(lfcs_r)
    # iz_min = np.argmin(lfcs_z)
    # iz_max = np.argmax(lfcs_z)

    # isoflux = [(lfcs_r[ir_min], lfcs_z[ir_min], lfcs_r[ir_max], lfcs_z[ir_max]),
    #            (lfcs_r[iz_min], lfcs_z[iz_min], lfcs_r[iz_max], lfcs_z[iz_max])]  # (R1,Z1, R2,Z2) pair of locations

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

    # tok.constraints = {"xpoints": xpoints,
    #                    "isoflux": isoflux,
    #                    # "psivals": psivals,
    #                    }

    plot_profiles(tok.equilibrium.profiles_1d,
                  profiles=[["q", "q1"],
                            "rho_tor",
                            "fpol",
                            "phi",
                            "psi",
                            "vprime",
                            ["dpsi_drho_tor",
                             "dpsi_drho_tor1"]
                            ],
                  x_axis="rho_tor_norm", grid=True) .savefig("../output/eq_profiles_1d.svg")

    tok.update()

    # draw(tok).savefig("../output/tokamak1.svg", transparent=True)

    plot_profiles(tok.core_profiles.profiles_1d,
                  profiles=[
                      [{"name": "psi0", "opts": {"marker": "+", "label": r"$\psi_{0}$"}},
                       {"name": "psi", "opts": {"marker": "+", "label": r"$\psi$"}}],
                      ["psi0_prime", "psi0_prime1", "psi_prime", "psi_prime1"],
                      ["q0",   "q1"],
                      ["dgamma_current", "f_current"],
                      ["j_total0", "j_ni_exp"],
                      ["electrons.density0", "electrons.density"],
                      #   ["electrons.density_prime", "electrons.density0_prime"],
                      ["electrons.dgamma", "electrons.se_exp0"],
                      "j_tor", "j_parallel",
                      #   "e_field.parallel",

                  ],
                  x_axis="grid.rho_tor_norm", grid=True).savefig("../output/core_profiles.svg")
    # #
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
