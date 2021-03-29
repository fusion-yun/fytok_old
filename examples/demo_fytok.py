from math import log
import matplotlib.pyplot as plt
import numpy as np
from fytok.Scenario import Scenario
from spdm.data.Collection import Collection
from spdm.data.File import File
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles

if __name__ == "__main__":
    # db = Collection(schema="mapping",
    #                 source="mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east",
    #                 mapping={"schema": "EAST", "version": "imas/3",
    #                          "path": "/home/salmon/workspace/fytok/data/mapping"})

    # doc = db.open(shot=55555, time_slice=40)

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml").entry
    equilibrium = File(
        "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1", format="geqdsk").entry

    # device = File("/home/salmon/workspace/fytok/data/mapping/EAST/imas/3/static/config.xml").entry
    # equilibrium = File("/home/salmon/workspace/fytok/examples/data/g063982.04800",  format="geqdsk").entry

    scenario = Scenario({"tokamak": {
        "wall":  device.wall,
        "pf_active": device.pf_active,
        "equilibrium": {
            "vacuum_toroidal_field": equilibrium.vacuum_toroidal_field,
            "profiles_1d": equilibrium.profiles_1d,
            "profiles_2d": equilibrium.profiles_2d,
            "coordinate_system": {"grid": {"dim1": 64, "dim2": 128}}

        }
    }})

    fig = plt.figure()

    scenario.tokamak.plot(fig.gca(),
                          wall={"limiter": {"edgecolor": "green"},  "vessel": {"edgecolor": "blue"}},
                          pf_active={"facecolor": 'red'},
                          equilibrium={"mesh": True, "boundary": True}
                          )
    r, z = np.meshgrid(equilibrium.profiles_2d.grid.dim1, equilibrium.profiles_2d.grid.dim2, indexing="ij")
    fig.gca().contour(r, z,  equilibrium.profiles_2d.psi, levels=32,  linewidths=0.2)

    plt.savefig("/home/salmon/workspace/output/contour.svg")

    psi_axis = scenario.tokamak.equilibrium.global_quantities.psi_axis
    psi_boundary = scenario.tokamak.equilibrium.global_quantities.psi_boundary

    ffprime = scenario.tokamak.equilibrium.profiles_1d.f_df_dpsi
    fpol = scenario.tokamak.equilibrium.profiles_1d.f

    psi_norm = np.linspace(0, 1, len(ffprime))

    fvac = fpol[0]

    plot_profiles(
        [
            [
                # (tok.equilibrium.profiles_1d.ffprime, r"$ff^{\prime}$"),
                (Function(psi_norm, ffprime), r"$ff^{\prime}_0$"),
                # (Function(psi_norm, (fpol**2)/(psi_boundary-psi_axis)*0.5).derivative, r"$d(f^{2}_0)$"),
                (scenario.tokamak.equilibrium.profiles_1d.ffprime, r"$ff^{\prime}$"),
            ],

            [(Function(psi_norm, fpol),  r"$f_{pol} $"),
             (Function(psi_norm, np.sqrt(2.0*Function(psi_norm, ffprime).antiderivative * \
                                         (psi_boundary-psi_axis)+fpol[0]**2)), r"$f_{pol}$"),
             (scenario.tokamak.equilibrium.profiles_1d.fpol, r"$f_{pol}$"), ],

            # (scenario.tokamak.equilibrium.profiles_1d.ffprime, r"$ff^{\prime}$"),

            (scenario.tokamak.equilibrium.profiles_1d.vprime, r"$V^{\prime}$"),
            (scenario.tokamak.equilibrium.profiles_1d.volume, r"$V$"),
            (scenario.tokamak.equilibrium.profiles_1d.q,      r"$q$"),
            (scenario.tokamak.equilibrium.profiles_1d.phi,    r"$\phi$"),

            (scenario.tokamak.equilibrium.profiles_1d.gm1, r"$gm1$"),
            (scenario.tokamak.equilibrium.profiles_1d.gm2, r"$gm2$"),
            (scenario.tokamak.equilibrium.profiles_1d.gm3, r"$gm3$"),
            (scenario.tokamak.equilibrium.profiles_1d.gm4, r"$gm4$"),
            (scenario.tokamak.equilibrium.profiles_1d.gm5, r"$gm5$"),
            (scenario.tokamak.equilibrium.profiles_1d.gm6, r"$gm6$"),
            (scenario.tokamak.equilibrium.profiles_1d.gm7, r"$gm7$"),
            (scenario.tokamak.equilibrium.profiles_1d.gm8, r"$gm8$"),
            (scenario.tokamak.equilibrium.profiles_1d.gm9, r"$gm9$"),

            # (tok.equilibrium.profiles_1d.vprime, "vprime"),
            # {"name": "volume"},
            # [{"name": "q"},
            #  {"name": "safety_factor"}]
        ],
        x_axis=(scenario.tokamak.equilibrium.profiles_1d.psi_norm,
                {"label": r"$\bar{\psi}$"}), grid=True)\
        .savefig("/home/salmon/workspace/output/profiles_1d.svg")

    # scenario.initialize({
    #     "particle": {
    #         "source": {"S0": 7.5e20, "profile": lambda x: np.exp(15.0*(x-1.0))},
    #         "diffusivity": {"D0": 0.5, "D1": 1.0, "D2": 1.1},
    #         "pinch_number": {"V0": 1.385}
    #     }
    # })

    # plot_profiles(tok.core_profiles.profiles_1d,
    #               profiles=[
    #                   [{"name": "psi0_eq", "opts": {"marker": ".", "label": r"$\psi_{0}$"}},
    #                    {"name": "psi", "opts":  {"marker": "+", "label": r"$\psi$"}}],
    #                   [{"name": "q0", "opts": {"marker": ".", "label": r"$q_{0}$"}},
    #                    {"name": "q", "opts":  {"marker": "+", "label": r"$q$"}}],
    #                   [
    #                       {"name": "rho_star", "opts": {"marker": ".", "label": r"$\rho^{\dagger}_{tor}$"}},
    #                       {"name": "rho_tor", "opts": {"marker": ".", "label": r"$\rho_{tor}$"}},
    #                   ],
    #                   [
    #                       {"name": "electrons.density0", "opts": {"marker": ".", "label": r"$n_{e0}$"}},
    #                       {"name": "electrons.density", "opts":  {"marker": "+", "label": r"$n_{e}$"}},
    #                   ],
    #                   #   [
    #                   #       {"name": "electrons.density0_residual_left", "opts":  {"label": r"$n_{e,residual,left}$"}},
    #                   #       #   {"name": "electrons.density0_residual_left1", "opts":  {"label": r"$n_{e,residual,left}$"}},
    #                   #       {"name": "electrons.density0_residual_right", "opts":  {"label": r"$n_{e,residual,right}$"}},
    #                   #   ],
    #                   #   "electrons.se_exp0",
    #                   [
    #                       {"name": "electrons.density0_prime", "opts":  {"marker": "+", "label": r"$n^{\prime}_{e0}$"}},
    #                       {"name": "electrons.density_prime", "opts":  {"marker": "+", "label": r"$n^{\prime}_{e}$"}},
    #                   ],
    #                   [
    #                       {"name": "electrons.diff", "opts": {"label": r"$D$"}},
    #                       {"name": "electrons.vconv", "opts": {"label": r"$v$"}},
    #                   ],
    #                   #   {"name": "vpr", "opts": {"marker": "*"}},
    #                   # "gm3", "vpr",
    #                   #   {"name": "dpsi_drho_tor", "opts":{"marker": "*"}},
    #                   #   "a", "b",  # "c",
    #                   #   "d", "e", "f", "g",
    #                   [
    #                       "electrons.diff_flux",
    #                       "electrons.vconv_flux",
    #                       "electrons.s_exp_flux",
    #                       "electrons.density_residual",
    #                   ],
    #                   #   [
    #                   #       #       #   {"name": "electrons.density_flux0", "opts": {"label": r"$\Gamma_{e0}$"}},
    #                   #       {"name": "electrons.density_flux", "opts": {"marker": "o", "label": r"$\Gamma_{e}$"}},
    #                   #       #   {"name": "electrons.density_flux1", "opts": {"marker": "+", "label": r"$\Gamma_{e2}$"}},
    #                   #   ],
    #                   #   {"name": "electrons.density_flux_error", "opts": {"marker": "+", "label": r"$\Gamma_{e,error}$"}},
    #                   #   #   #   {"name": "electrons.density_flux0_prime", "opts": {"label": r"$\Gamma_{e0}^{\prime}$"}},
    #                   #   [
    #                   #       {"name": "electrons.density_flux_prime", "opts": {
    #                   #           "marker": "o", "label": r"$\Gamma_{e}^{\prime}$"}},
    #                   #       {"name": "electrons.density_flux1_prime", "opts": {
    #                   #           "marker": "+", "label": r"$\Gamma_{e1}^{\prime}$"}},
    #                   #       "electrons.se_exp0",
    #                   #   ],
    #                   #   {"name": "electrons.density0_prime", "opts": {"marker": ".", "label": r"$n^{\prime}_{e0}$"}},
    #                   #   ["psi0_prime", "psi0_prime1",  "psi1_prime", "psi1_prime1"],
    #                   #   {"name": "dpsi_drho_tor", "opts": {"marker": "+"}},
    #                   #   ["dgamma_current", "f_current"],
    #                   #   ["j_total0", "j_ni_exp"],
    #                   #   ["electrons.density0",
    #                   #    "electrons.density"],
    #                   #   "electrons.density0_prime", "electrons.density_prime",
    #                   #   ["electrons.gamma0_prime", "electrons.se_exp0","f"],
    #                   #   ["electrons.gamma0"],
    #                   #   "j_tor", "j_parallel",
    #                   #   "e_field.parallel",
    #               ],
    #               axis={"name": "grid.rho_tor_norm", "opts": {"label": r"$\rho_{tor}/\rho_{tor,bdry}$"}}, grid=True)\
    #     .savefig("/home/salmon/workspace/output/core_profiles.svg")
    # fig.savefig("/home/salmon/workspace/output/core_profiles.svg", transparent=True)
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
