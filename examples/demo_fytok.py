from math import log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants
from fytok.Tokamak import Tokamak
from spdm.data.Collection import Collection
from spdm.data.File import File
from spdm.numerical.Function import Function
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles

if __name__ == "__main__":
    # db = Collection(schema="mapping",
    #                 source="mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east",
    #                 mapping={"schema": "EAST", "version": "imas/3",
    #                          "path": "/home/salmon/workspace/fytok/data/mapping"})

    # doc = db.open(shot=55555, time_slice=40)

    # device = File("/home/salmon/workspace/fytok/data/mapping/EAST/imas/3/static/config.xml").entry
    # equilibrium = File("/home/salmon/workspace/fytok/examples/data/g063982.04800",  format="geqdsk").entry

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml").entry
    equilibrium = File(
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/15MA inductive - burn/Increased domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16VVHR.txt",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        format="geqdsk").entry

    profile = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')

    tok = Tokamak({
        "radial_grid": {
            "axis": 128,
            "label": "rho_tor_norm"
        },
        "wall":  device.wall,
        "pf_active": device.pf_active,
        "equilibrium": {
            "vacuum_toroidal_field": equilibrium.vacuum_toroidal_field,
            "global_quantities": equilibrium.global_quantities,
            "profiles_1d": equilibrium.profiles_1d,
            "profiles_2d": equilibrium.profiles_2d,
            "coordinate_system": {"grid": {"dim1": 64, "dim2": 256}}
        },
        # "core_profiles":{ion": [{}]}
    })
    rho_tor_norm = np.linspace(0, 1.0, 128)
    r_ped = np.sqrt(0.9)
    tok.initialize({
        "r_ped": r_ped,  # \frac{\Phi}{\Phi_a}=0.88
        "electron": {
            "density": {
                "n0": 1e20,
                "source": {"S0": 7.5e20},  # S0 7.5e20
                "diffusivity":  Function(rho_tor_norm, [lambda r:r < r_ped, lambda r:r >= r_ped], [lambda x:0.2 + 0.6*(x**2), lambda x: 0.2]),
                # {"D0": 0.2, "D1": 3.0, "D2": 0.11},
                # "diffusivity": Function(profile["x"].values, profile["He"].values),
                "pinch_number": {"V0": 1.385},
                "boundary_condition": {"value": 4.6e18}
            },
            "temperature": {
                "T0": 0.95e19,
                "profile": lambda r: (1-r**2)**2,
            }}
    })

    fig = plt.figure()

    tok.plot(fig.gca(),
             wall={"limiter": {"edgecolor": "green"},  "vessel": {"edgecolor": "blue"}},
             pf_active={"facecolor": 'red'},
             equilibrium={"mesh": True, "boundary": True}
             )

    plt.savefig("/home/salmon/workspace/output/contour.svg", transparent=True)

    #
    # logger.debug((
    #     tok.equilibrium.profiles_1d.phi,
    #     tok.equilibrium.profiles_1d.rho_tor
    #     #     (equilibrium.vacuum_toroidal_field.r0*equilibrium.vacuum_toroidal_field.b0,
    #     #      tok.equilibrium.vacuum_toroidal_field.r0 * tok.equilibrium.vacuum_toroidal_field.b0),
    #     #     (equilibrium.global_quantities.psi_boundary-equilibrium.global_quantities.psi_axis,
    #     #      tok.equilibrium.global_quantities.psi_boundary-tok.equilibrium.global_quantities.psi_axis),
    #     #     (tok.equilibrium.profiles_1d.fpol/Function(equilibrium.profiles_1d.psi_norm,
    #     #                                                equilibrium.profiles_1d.f)(tok.equilibrium.profiles_1d.psi_norm)),
    #     #     (tok.equilibrium.profiles_1d.q/Function(equilibrium.profiles_1d.psi_norm, equilibrium.profiles_1d.q))
    # ))

    plot_profiles(
        [
            # (tok.core_profiles.grid.psi_norm, r"$\psi_{N}$"),
            (tok.equilibrium.profiles_1d.phi, r"$\phi$"),
            (tok.equilibrium.profiles_1d.rho_tor, r"$\rho_{tor}$"),
            (tok.equilibrium.profiles_1d.rho_tor_norm, r"$\rho_{tor,N}$"),
            # [
            #     (tok.core_transport[0].electrons.particles.d, r"$d_{e}$"),
            #     (np.abs(tok.core_transport[0].electrons.particles.v), r"$v_{e}$"),
            # ],
            # (tok.equilibrium.profiles_1d.f_df_dpsi, r"$ff^{\prime}$"),

            [
                (tok.equilibrium.profiles_1d.fpol, r"$fpol$"),
                (Function(equilibrium.profiles_1d.psi_norm, np.abs(
                    equilibrium.profiles_1d.f)), r"$\left|f_{pol0}\right|$"),
            ],

            [
                (Function(equilibrium.profiles_1d.psi_norm, equilibrium.profiles_1d.q), r"$q_0$"),
                (tok.equilibrium.profiles_1d.q, r"$q$"),
                (tok.equilibrium.profiles_1d.dphi_dpsi, r"$\frac{d\phi}{d\psi}$"),
            ],
            (tok.equilibrium.profiles_1d.volume, r"$V$"),
            (tok.equilibrium.profiles_1d.vprime, r"$V^{\prime}$"),
            (tok.equilibrium.profiles_1d.dvolume_drho_tor * \
             tok.equilibrium.profiles_1d.rho_tor[-1], r"$\frac{dV}{d\rho_{N}}$"),

            # (tok.equilibrium.profiles_1d.dphi_dpsi, r"$\frac{d\phi}{d\psi}$"),


            # (tok.equilibrium.profiles_1d.drho_tor_dpsi.pullback(
            #     psi_norm, rho_tor_norm), r"$\frac{d\rho_{tor}}{d\psi}$"),
            # (tok.equilibrium.profiles_1d.dpsi_drho_tor.pullback(
            #     psi_norm, rho_tor_norm), r"$\frac{d\psi}{d\rho_{tor}}$"),
            # (tok.equilibrium.profiles_1d.dvolume_drho_tor_norm.pullback(
            #     psi_norm, rho_tor_norm), r"$\frac{dV}{d\rho_{N}}$"),
            # (tok.equilibrium.profiles_1d.drho_tor_dpsi, r"$\frac{d\rho_{tor}}{d\psi}$"),
            # (tok.equilibrium.profiles_1d.gm3, r"$gm3$")

            # (tok.core_profiles.electrons.temperature, r"$T_{e}$"),

        ],
        # x_axis=(tok.equilibrium.profiles_1d.rho_tor_norm,   {"label": r"$\rho_{N}$"}),  # asd
        # x_axis=(tok.equilibrium.profiles_1d.phi,   {"label": r"$\Phi$"}),  # asd
        x_axis=(equilibrium.profiles_1d.psi_norm,  {"label": r"$\psi_{N}$"}),  # asd
        grid=True) .savefig("/home/salmon/workspace/output/profiles_1d.svg", transparent=True)

    tok.update(transport_solver={})

    # rho_tor_bdry = tok.core_profiles.grid.rho_tor[-1]

    # vpr = Function(tok.equilibrium.profiles_1d.rho_tor_norm,
    #                tok.equilibrium.profiles_1d.vprime)(tok.core_profiles.grid.rho_tor)
    # logger.debug(tok.core_profiles.electrons.density.x)
    # x = tok.core_profiles.electrons.density.x
    # dx = (np.roll(x, -1)-x)
    # dx[-1] = dx[-2]
    # logger.debug(tok.core_profiles.electrons.density.x)
    psi_norm = tok.equilibrium.profiles_1d.psi_norm
    rho_tor_norm = tok.equilibrium.profiles_1d.rho_tor_norm
    plot_profiles(
        [
            # (1.0/dx,                                          {"marker": ".", "label": r"$1/dx$"}),
            (tok.core_profiles.electrons.density.x,           r"$rho_{tor,N}$"),
            [
                (Function(profile["x"].values, profile["NE"].values*1.0e19),             r"$n_{e0}$"),
                (tok.core_profiles.electrons.density,             r"$n_{e}$"),
            ],
            [(tok.core_profiles.electrons.density.derivative, {"color": "green", "label":  r"$n_{e}^{\prime}$"}),
             (tok.core_profiles.electrons.density_prime,      {"color": "black", "label":  r"$n_{e}^{\prime}$"})],
            (tok.core_profiles.electrons.density.derivative - \
             tok.core_profiles.electrons.density_prime,        {"marker": ".", "label": r"$\Delta n_{e}^{\prime}$"}),

            (tok.core_profiles.electrons.n_gamma,             r"$\Gamma_{e}$"),
            (tok.core_profiles.electrons.n_gamma_prime,       r"$\Gamma_{e}^{\prime}$"),
            # (tok.core_profiles.electrons.n_rms_residuals,     {"marker": ".", "label":  r"residuals"}),
            [
                (tok.core_profiles.electrons.n_diff,          {"color": "green", "label": r"D"}),
                (np.abs(tok.core_profiles.electrons.n_conv),  {"color": "black",  "label": r"v"}),
                (Function(profile["x"].values, profile["He"].values), r"$D_e$")
            ],

            # (tok.core_profiles.electrons.n_diff.derivative,   {"color": "green", "label": r"$D^{\prime}$"}),

            [
                (tok.core_profiles.electrons.n_s_exp_flux,    {"color": "green", "label": r"Source"}),
                (tok.core_profiles.electrons.n_diff_flux,     {"color": "black", "label": r"Diffusive flux"}),
                (tok.core_profiles.electrons.n_conv_flux,     {"color": "blue",  "label": r"Convective flux"}),
                (tok.core_profiles.electrons.n_residual,      {"color": "red",   "label": r"Residual"})
            ],
            # (tok.core_profiles.electrons.residuals_y,         r"$dn_{residuals}$"),
            # (tok.core_profiles.electrons.residuals_gamma,     r"$dgamma_{residuals}$"),
            [
                (tok.equilibrium.profiles_1d.q.pullback(psi_norm, rho_tor_norm),                   r"$q$"),
                (tok.equilibrium.profiles_1d.dphi_dpsi.pullback(psi_norm, rho_tor_norm),
                 r"$\frac{d\phi}{d\psi}$"),
                (Function(profile["x"].values, profile["q"].values),             r"$q_{0}$"),

            ],

            (tok.equilibrium.profiles_1d.volume.pullback(psi_norm, rho_tor_norm),        r"$V$"),

            (tok.equilibrium.profiles_1d.vprime.pullback(psi_norm, rho_tor_norm),        r"$V^{\prime}$"),

            (tok.equilibrium.profiles_1d.dvolume_drho_tor.pullback(psi_norm, rho_tor_norm),
             r"$\frac{dV}{d\rho_{tor}}$"),

            [
                (tok.equilibrium.profiles_1d.rho_tor.pullback(psi_norm, rho_tor_norm),             r"$\rho_{tor}$"),
                (tok.equilibrium.profiles_1d.dvolume_drho_tor.pullback(psi_norm, rho_tor_norm) / ((scipy.constants.pi**3) * 8.0 * tok.equilibrium.vacuum_toroidal_field.r0),
                    r"$\frac{dV/d\rho_{tor}}{4\pi^2 R_0}$"),

            ],

            [
                (tok.equilibrium.profiles_1d.dvolume_drho_tor_norm.pullback(psi_norm, rho_tor_norm),
                    r"$\frac{dV}{d\rho_{tor,N}}$"),

                (tok.core_profiles.electrons.vpr,                 r"$vpr$"),
            ]
            # (tok.core_profiles.electrons.gm3,                 r"$gm3$"),
            # (tok.core_profiles.electrons.n_a,                 r"$a$"),
            # (tok.core_profiles.electrons.n_b,                 r"$b$"),
            # (tok.core_profiles.electrons.n_c,       r"$c$"),
            # (tok.core_profiles.electrons.n_d,                 r"$d$"),
            # (tok.core_profiles.electrons.n_e,                 r"$e$"),
            # (tok.core_profiles.electrons.n_f,                 r"$f$"),
            # (tok.core_profiles.electrons.n_g,                 r"$g$"),

        ],
        x_axis=(tok.core_profiles.electrons.density.x,   {"label": r"$\rho_{N}$"}),  # x axis,
        # index_slice=slice(-100,None, 1),
        grid=True) .savefig("/home/salmon/workspace/output/electron_1d.svg", transparent=True)

    logger.info("Done")

    # psi_axis = tok.equilibrium.global_quantities.psi_axis
    # psi_boundary = tok.equilibrium.global_quantities.psi_boundary

    # ffprime = tok.equilibrium.profiles_1d.f_df_dpsi
    # fpol = tok.equilibrium.profiles_1d.f

    # psi_norm = np.linspace(0, 1, len(ffprime))

    # fvac = fpol[0]

    # plot_profiles(
    #     [
    #         # [
    #         #     # (tok.equilibrium.profiles_1d.ffprime, r"$ff^{\prime}$"),
    #         #     # (Function(psi_norm, ffprime), r"$ff^{\prime}_0$"),
    #         #     # (Function(psi_norm, (fpol**2)/(psi_boundary-psi_axis)*0.5).derivative, r"$d(f^{2}_0)$"),
    #         #     (tok.equilibrium.profiles_1d.ffprime, r"$ff^{\prime}$"),
    #         # ],

    #         # [
    #         #     # (Function(psi_norm, fpol),  r"$f_{pol} $"),
    #         #     #  (Function(psi_norm, np.sqrt(2.0*Function(psi_norm, ffprime).antiderivative * \
    #         #     #                              (psi_boundary-psi_axis)+fpol[0]**2)), r"$f_{pol}$"),
    #         #     (tok.equilibrium.profiles_1d.fpol, r"$f_{pol}$"), ],

    #         # # (tok.equilibrium.profiles_1d.ffprime, r"$ff^{\prime}$"),

    #         (tok.equilibrium.profiles_1d.vprime, r"$V^{\prime}$"),
    #         (tok.equilibrium.profiles_1d.volume, r"$V$"),
    #         (tok.equilibrium.profiles_1d.q,      r"$q$"),
    #         (tok.equilibrium.profiles_1d.fpol,   r"$fpol$"),
    #         (tok.equilibrium.profiles_1d.phi,    r"$\phi$"),
    #         (tok.equilibrium.profiles_1d.rho_tor_norm, r"$\rho_{N}$"),

    #         (tok.equilibrium.profiles_1d.gm1, r"$gm1$"),
    #         (tok.equilibrium.profiles_1d.gm2, r"$gm2$"),
    #         (tok.equilibrium.profiles_1d.gm3, r"$gm3$"),
    #         (tok.equilibrium.profiles_1d.gm4, r"$gm4$"),
    #         (tok.equilibrium.profiles_1d.gm5, r"$gm5$"),
    #         (tok.equilibrium.profiles_1d.gm6, r"$gm6$"),
    #         (tok.equilibrium.profiles_1d.gm7, r"$gm7$"),
    #         (tok.equilibrium.profiles_1d.gm8, r"$gm8$"),
    #         (tok.equilibrium.profiles_1d.gm9, r"$gm9$"),

    #         # (tok.equilibrium.profiles_1d.vprime, "vprime"),
    #         # {"name": "volume"},
    #         # [{"name": "q"},
    #         #  {"name": "safety_factor"}]
    #     ],
    #     x_axis=(tok.equilibrium.profiles_1d.psi_norm, {"label": r"$\bar{\psi}$"}), \
    #     # x_axis=(tok.equilibrium.profiles_1d.rho_tor_norm, {"label": r"$\rho_{N}$"}) , # asd
    #     grid=True) .savefig("/home/salmon/workspace/output/profiles_1d.svg")
