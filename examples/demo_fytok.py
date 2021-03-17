import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt
from fytok.Tokamak import Tokamak
from fytok.util.Plot import plot_profiles
from spdm.util.logger import logger
from spdm.data.Collection import Collection
if __name__ == "__main__":
    db = Collection(schema="mapping",
                    source="mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east",
                    mapping={"schema": "EAST", "version": "imas/3",
                             "path": "/home/salmon/workspace/fytok/data/mapping"})

    doc = db.open(shot=55555, time_slice=10)

    tok = Tokamak(doc.entry)

    fig = plt.figure()
    tok.plot(fig.gca(),
             wall={"limiter": {"edgecolor": "green"},
                   "vessel": {"edgecolor": "blue"}},
             pf_active={"facecolor": 'red'})
    # fig.savefig("/home/salmon/workspace/output/tokamak.svg")

    # # plt.plot(tok.equilibrium.flux_surface.psi_norm, tok.equilibrium.flux_surface.ffprime)
    # plt.contourf(tok.equilibrium.flux_surface.R, tok.equilibrium.flux_surface.Z, tok.equilibrium.flux_surface.dl)
    # ax0 = tok.equilibrium.flux_surface.surface_mesh.axis(10, axis=0)
    # ax1 = tok.equilibrium.flux_surface.surface_mesh.axis(10, axis=1)
    u = np.linspace(0, 1.0, 128)

    for idx in range(0, 128, 8):
        ax1 = tok.equilibrium.flux_surface.rz_mesh.axis(idx, axis=0)
        plt.plot(*ax1(u), "b")

    for idx in range(0, 128, 8):
        ax1 = tok.equilibrium.flux_surface.rz_mesh.axis(idx, axis=1)
        plt.plot(* ax1(u), "r")

        # plt.plot(* ax1(np.linspace(0, 1.0, 128)))

    # plt.plot(*tok.equilibrium.flux_surface.surface_mesh.xy, "+")

    # plt.contourf(tok.equilibrium.flux_surface.R, tok.equilibrium.flux_surface.Z, tok.equilibrium.flux_surface.Z)
    plt.savefig("/home/salmon/workspace/output/contour.svg")

    psi_norm = tok.equilibrium.flux_surface.psi_norm
    fig = plt.figure()
    # plt.plot(psi_norm, tok.equilibrium.flux_surface.rz_mesh.axis(10, axis=0).apply(lambda r, z: 1.0/r, psi_norm))
    r, z = tok.equilibrium.flux_surface.rz_mesh.points
    # br, bz = tok.equilibrium.flux_surface.grad_psi(r, z)
    jacob = tok.equilibrium.flux_surface.jacobian(r, z)
    # plt.contourf(r, z, r/np.sqrt(br**2+bz**2))
    plt.contourf(r, z, jacob)
    plt.savefig("/home/salmon/workspace/output/contour1.svg")

    plt.figure()
    plt.contourf(r, z, jacob)
    plt.savefig("/home/salmon/workspace/output/contour2.svg")

    # tok.initialize_profile()

    plot_profiles(
        tok.equilibrium.flux_surface, [
            {"name": "ffprime"},
            {"name": "fpol"},
            {"name": "q"},
            {"name": "volume"},
        ],
        axis={"name": "psi_norm", "opts": {"label": r"$\bar{\psi}$"}},
        grid=True)\
        .savefig("/home/salmon/workspace/output/profiles_1d.svg")

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
