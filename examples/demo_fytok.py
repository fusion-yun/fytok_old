import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import scipy.stats
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


##################################################################################################

def draw(tok):

    return tok.equilibrium.plot(
        # axis=("psi_norm",   r'$(\psi-\psi_{axis})/(\psi_{boundary}-\psi_{axis}) [-]$'),
        axis=("rho_tor_norm",  r"$\rho_{tor}/\rho_{bndry}$"),
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
            # ({"name": "gm5", "opts": {"label": r"$gm5: \left\langle B^{2}\right\rangle$"}},  r"$\left[T^2\right]$"),
            # ({"name": "psi"}, r"$[Wb]$"),
            # ({"name": "dpsi_drho_tor"}, r"$[Wb]$"),
            # ({"name": "psi_norm", "opts": {"label": r"$\psi_{norm}$"}}, r"$[Wb]$")
        ],
        # profiles_2d=[("phi", {})],
        vec_field=[("b_field_r", "b_field_z", {"linewidth": 0.5, "arrowsize": 0.2})],
        # surface_mesh=True,
        machine={"coils": False}
    )


if __name__ == "__main__":

    from fytok.util.Plot import plot_profiles
    from fytok.Tokamak import Tokamak
    from spdm.data.Entry import open_entry
    from spdm.util.AttributeTree import _next_
    from spdm.util.logger import logger
    from spdm.util.Profiles import Profile

    tok = Tokamak(open_entry("east+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east", shot=55555, time_slice=20))

    tok.create_dummy_profile()

    # tok = Tokamak(open_entry("cfetr+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east", shot=55555, time_slice=20))
    # rho_b = 0.96
    # rho_norm = np.linspace(0, 1.0, 129)
    # def D(r): return np.piecewise(r, [r < rho_b, r >= rho_b], [lambda x: (0.2 + (x**3)), 0.1])
    # def v(r): return np.piecewise(r, [r < rho_b, r >= rho_b], [lambda x: -(x**3)*0.3,  0])
    # tok.core_transport[_next_] = {"identifier": {"name": "unspecified", "index": 0}}

    # trans = tok.core_transport[-1].profiles_1d

    # trans.conductivity_parallel = 1.0e-8

    # trans.electrons.particles.d = D
    # trans.electrons.particles.v = v

    # tok.core_sources[_next_] = {"identifier": {"name": "unspecified", "index": 0}}

    # src = tok.core_sources[-1].profiles_1d

    # def S_pel(rho, pos=0.7, w=0.1, S0=1.0e19): return scipy.stats.norm.pdf((rho-0.7)/w) * \
    #     np.sqrt(scipy.constants.pi*2.0)*S0

    # def S_edge(rho, pos=0.7, w=0.03, S0=5.0e15): return np.piecewise(
    #     rho, [rho < pos, rho >= pos], [0, lambda x: - (np.exp((x-pos)/w-10)*S0-1.0)])

    # src.electrons.particles = lambda rho: S_edge(rho)

    # def n_core(x, n0, w): return n0*((1-(x/w)**2)**2)
    # def n_ped(x, n0, rho_b): return n0*(1 - ((x-rho_b)*16)**2)

    # def ne(rho, rho_b=rho_b, w=2.0, n0=1e20): return np.piecewise(
    #     rho, [rho < rho_b, rho > rho_b],
    #     [lambda x: n_core(x, n0, w),
    #      lambda x: n_ped(x, n_core(rho_b, n0, w), rho_b)])

    # tok.core_profiles.profiles_1d.electrons.density = ne

    # lfcs_r = tok.equilibrium.boundary.outline.r
    # lfcs_z = tok.equilibrium.boundary.outline.z
    # psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]
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
    # plot_profiles(tok.equilibrium.profiles_1d,
    #               profiles=[["q", "q1"],
    #                         "rho_tor",
    #                         "fpol",
    #                         "phi",
    #                         "psi",
    #                         "dvolume_dpsi",
    #                         "gm1", "gm2", "gm3"
    #                         ],
    #               axis="rho_tor_norm", grid=True) .savefig("../output/eq_profiles_1d.svg")
    # plot_profiles(trans, profiles=[
    #     ["electrons.particles.d", "electrons.particles.v"], "conductivity_parallel"
    # ], axis="grid_d.rho_tor_norm", grid=True).savefig("../output/core_tranport.svg")
    # plot_profiles(src, profiles=[
    #     "electrons.particles", "grid.rho_tor_norm"
    # ], axis="grid.rho_tor_norm", grid=True).savefig("../output/core_sources.svg")
    # draw(tok).savefig("../output/tokamak0.png", transparent=True)

    tok.update()

    # draw(tok).savefig("../output/tokamak1.svg", transparent=True)
    plot_profiles(tok.core_profiles.profiles_1d,
                  profiles=[
                      [{"name": "psi0_eq", "opts": {"marker": ".", "label": r"$\psi_{0}$"}},
                       {"name": "psi", "opts":  {"marker": "+", "label": r"$\psi$"}}],
                      [{"name": "q0", "opts": {"marker": ".", "label": r"$q_{0}$"}},
                       {"name": "q", "opts":  {"marker": "+", "label": r"$q$"}}],
                      [
                          {"name": "rho_star", "opts": {"marker": ".", "label": r"$\rho^{\dagger}_{tor}$"}},
                          {"name": "rho_tor", "opts": {"marker": ".", "label": r"$\rho_{tor}$"}},
                      ],
                      [
                          {"name": "electrons.density0", "opts": {"marker": ".", "label": r"$n_{e0}$"}},
                          {"name": "electrons.density", "opts":  {"marker": "+", "label": r"$n_{e}$"}},
                      ],
                      #   [
                      #       {"name": "electrons.density0_residual_left", "opts":  {"label": r"$n_{e,residual,left}$"}},
                      #       #   {"name": "electrons.density0_residual_left1", "opts":  {"label": r"$n_{e,residual,left}$"}},
                      #       {"name": "electrons.density0_residual_right", "opts":  {"label": r"$n_{e,residual,right}$"}},
                      #   ],
                      #   "electrons.se_exp0",
                      [
                          {"name": "electrons.density0_prime", "opts":  {"marker": "+", "label": r"$n^{\prime}_{e0}$"}},
                          {"name": "electrons.density_prime", "opts":  {"marker": "+", "label": r"$n^{\prime}_{e}$"}},
                      ],
                      [
                          {"name": "electrons.diff", "opts": {"marker": ".", "label": r"$D$"}},
                          {"name": "electrons.vconv", "opts": {"marker": ".", "label": r"$v$"}},
                      ],
                      #   {"name": "vpr", "opts": {"marker": "*"}},
                        # "gm3", "vpr",
                      #   {"name": "dpsi_drho_tor", "opts":{"marker": "*"}},
                      #   "a", "b",  # "c",
                    #   "d", "e", "f", "g",
                      [
                          "electrons.diff_flux",
                          "electrons.vconv_flux",
                          "electrons.s_exp_flux",
                          "electrons.density_residual"
                      ],
                      #   [
                      #       #       #   {"name": "electrons.density_flux0", "opts": {"label": r"$\Gamma_{e0}$"}},
                      #       {"name": "electrons.density_flux", "opts": {"marker": "o", "label": r"$\Gamma_{e}$"}},
                      #       #   {"name": "electrons.density_flux1", "opts": {"marker": "+", "label": r"$\Gamma_{e2}$"}},

                      #   ],
                      #   {"name": "electrons.density_flux_error", "opts": {"marker": "+", "label": r"$\Gamma_{e,error}$"}},

                      #   #   #   {"name": "electrons.density_flux0_prime", "opts": {"label": r"$\Gamma_{e0}^{\prime}$"}},
                      #   [
                      #       {"name": "electrons.density_flux_prime", "opts": {
                      #           "marker": "o", "label": r"$\Gamma_{e}^{\prime}$"}},
                      #       {"name": "electrons.density_flux1_prime", "opts": {
                      #           "marker": "+", "label": r"$\Gamma_{e1}^{\prime}$"}},
                      #       "electrons.se_exp0",
                      #   ],

                      #   {"name": "electrons.density0_prime", "opts": {"marker": ".", "label": r"$n^{\prime}_{e0}$"}},

                      #   ["psi0_prime", "psi0_prime1",  "psi1_prime", "psi1_prime1"],
                      #   {"name": "dpsi_drho_tor", "opts": {"marker": "+"}},
                      #   ["dgamma_current", "f_current"],
                      #   ["j_total0", "j_ni_exp"],
                      #   ["electrons.density0",
                      #    "electrons.density"],

                      #   "electrons.density0_prime", "electrons.density_prime",
                      #   ["electrons.gamma0_prime", "electrons.se_exp0","f"],
                      #   ["electrons.gamma0"],

                      #   "j_tor", "j_parallel",
                      #   "e_field.parallel",

                  ],
                  axis={"name": "grid.rho_tor_norm", "opts": {"label": r"$\rho_{tor}/\rho_{tor,bdry}$"}}, grid=True)\
        .savefig("../output/core_profiles.svg")
    # #
    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0)
    # fig.align_ylabels()
    # fig.savefig("../output/core_profiles.svg", transparent=True)

    # tok.plot()
    # plt.savefig("../output/east.svg", transparent=True)
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
