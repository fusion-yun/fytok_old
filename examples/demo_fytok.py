

import matplotlib.pyplot as plt
import numpy as np
from fytok.Tokamak import Tokamak
from spdm.data.Collection import Collection
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles

if __name__ == "__main__":
    db = Collection(schema="mapping",
                    source="mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east",
                    mapping={"schema": "EAST", "version": "imas/3",
                             "path": "/home/salmon/workspace/fytok/data/mapping"})

    doc = db.open(shot=55555, time_slice=40)

    tok = Tokamak(doc.entry)

    fig = plt.figure()

    tok.plot(fig.gca(),
             wall={"limiter": {"edgecolor": "green"},
                   "vessel": {"edgecolor": "blue"}},
             pf_active={"facecolor": 'red'},
             equilibrium={"mesh": False}
             )

    fig.gca().contourf(tok.equilibrium.flux_surface.R, tok.equilibrium.flux_surface.Z, tok.equilibrium.flux_surface.dl)

    # fig.savefig("/home/salmon/workspace/output/tokamak.svg")

    # for idx in range(tok.equilibrium.flux_surface.mesh.shape[0]):
    #     ax1 = tok.equilibrium.flux_surface.mesh.axis(idx, axis=0)
    #     fig.gca().add_patch(plt.Polygon(ax1.points, fill=False, closed=True, color="b", linewidth=0.2))

    # for idx in range(0, tok.equilibrium.flux_surface.mesh.shape[1], 8):
    #     ax1 = tok.equilibrium.flux_surface.mesh.axis(idx, axis=1)
    #     plt.plot(*ax1.points.T,  "r", linewidth=0.2)

    # r = tok.equilibrium.flux_surface.rz_mesh.points[0]
    # z = tok.equilibrium.flux_surface.rz_mesh.points[1]
    # grad_psi = np.linalg.norm(tok.equilibrium.flux_surface.grad_psi(r, z), axis=0)
    # plt.contourf(r[:, :], z[:, :],  grad_psi[:, :])

    plt.savefig("/home/salmon/workspace/output/contour.svg")

    # psi_norm = tok.equilibrium.flux_surface.psi_norm
    # fig = plt.figure()
    # # plt.plot(psi_norm, tok.equilibrium.flux_surface.rz_mesh.axis(10, axis=0).apply(lambda r, z: 1.0/r, psi_norm))
    # r, z = tok.equilibrium.flux_surface.rz_mesh.points
    # # br, bz = tok.equilibrium.flux_surface.grad_psi(r, z)
    # jacob = tok.equilibrium.flux_surface.jacobian(r, z)
    # # plt.contourf(r, z, r/np.sqrt(br**2+bz**2))
    # plt.contourf(r, z, jacob)
    # plt.savefig("/home/salmon/workspace/output/contour1.svg")

    # plt.figure()
    # plt.contourf(r, z, jacob)
    # plt.savefig("/home/salmon/workspace/output/contour2.svg")

    # tok.initialize_profile()
    # fig = plt.figure()
    # u = np.linspace(0, 1.0, 128, endpoint=False)
    # idx = 124

    # # plt.plot(tok.equilibrium.flux_surface.Jdl[120](u))
    # plt.plot(u, tok.equilibrium.flux_surface.Jdl[idx](u))
    # logger.debug(scipy.integrate.romberg(tok.equilibrium.flux_surface.Jdl[idx], 0, 1.0))

    # plt.savefig("/home/salmon/workspace/output/profiles_1d.svg")
    # logger.debug(tok.equilibrium.profiles_1d["f"])
    # logger.debug(tok.equilibrium.profiles_1d.fpol)
    # logger.debug(tok.equilibrium.profiles_1d["f_df_dpsi"])

    psi_axis = tok.equilibrium.global_quantities.psi_axis
    psi_boundary = tok.equilibrium.global_quantities.psi_boundary

    ffprime = tok.equilibrium["profiles_1d.f_df_dpsi"].__fetch__()
    fpol = tok.equilibrium["profiles_1d.f"].__fetch__()

    psi_norm = np.linspace(0, 1, len(ffprime))

    fvac = fpol[0]

    plot_profiles([
        [
            # (tok.equilibrium.flux_surface.ffprime, r"$ff^{\prime}$"),
            (Function(psi_norm, ffprime), r"$ff^{\prime}_0$"),
            (Function(psi_norm, (fpol**2)/(psi_boundary-psi_axis)*0.5).derivative, r"$d(f^{2}_0)$"),
        ],

        # [(Function(psi_norm, fpol**2),  r"$f_{pol}^2$"),
        #  (Function(psi_norm, 2.0*Function(psi_norm, ffprime).antiderivative*(psi_boundary-psi_axis)+fpol[0]**2), r"$\int ff^{\prime}$")],

        (tok.equilibrium.flux_surface.ffprime, r"$ff^{\prime}$"),
        (tok.equilibrium.flux_surface.fpol, r"$f_{pol}$"),
        (tok.equilibrium.flux_surface.vprime, r"$V^{\prime}$"),
        (tok.equilibrium.flux_surface.volume, r"$V$"),
        (tok.equilibrium.flux_surface.q,        r"$q$"),
        (tok.equilibrium.flux_surface.phi, r"$\phi$"),

        # (tok.equilibrium.flux_surface.vprime, "vprime"),
        # {"name": "volume"},
        # [{"name": "q"},
        #  {"name": "safety_factor"}]
    ],
        x_axis=(tok.equilibrium.flux_surface.psi_norm,   {"label": r"$\bar{\psi}$"}),
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
