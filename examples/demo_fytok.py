from math import log
import matplotlib.pyplot as plt
import numpy as np
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

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml").entry
    equilibrium = File(
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/15MA inductive - burn/Increased domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16VVHR.txt",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_inbord.EQDSK",
        format="geqdsk").entry

    # device = File("/home/salmon/workspace/fytok/data/mapping/EAST/imas/3/static/config.xml").entry
    # equilibrium = File("/home/salmon/workspace/fytok/examples/data/g063982.04800",  format="geqdsk").entry
   
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
            "coordinate_system": {"grid": {"dim1": 64, "dim2": 128}}
        },
        # "core_profiles":{ion": [{}]}
    })

    tok.initialize({
        "pedestal_top": 0.88,  # \frac{\Phi}{\Phi_a}=0.88
        "electron": {
            "density": {
                "n0": 1e20,
                "source": {"S0": 7.5e20},  # S0 7.5e20
                "diffusivity": {"D0": 0.5, "D1": 1.0, "D2": 0.11},
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

    profile = np.asarray([
        [0.00000E+00, -1.42094E+00,  1.01371E+00, 1.01930E+00],
        [3.85482E-04, -1.42045E+00,  1.01306E+00, 1.02463E+00],
        [1.54133E-03, -1.41975E+00,  1.01108E+00, 1.02649E+00],
        [3.46577E-03, -1.42388E+00,  1.00764E+00, 1.02488E+00],
        [6.15583E-03, -1.42416E+00,  1.00258E+00, 1.02479E+00],
        [9.60736E-03, -1.42664E+00,  9.95737E-01, 1.02502E+00],
        [1.38150E-02, -1.42849E+00,  9.86993E-01, 1.02503E+00],
        [1.87724E-02, -1.42960E+00,  9.76327E-01, 1.02543E+00],
        [2.44717E-02, -1.42888E+00,  9.63850E-01, 1.02635E+00],
        [3.09043E-02, -1.42938E+00,  9.49755E-01, 1.02773E+00],
        [3.80602E-02, -1.43126E+00,  9.34264E-01, 1.02919E+00],
        [4.59284E-02, -1.43544E+00,  9.17591E-01, 1.03053E+00],
        [5.44967E-02, -1.44149E+00,  8.99912E-01, 1.03154E+00],
        [6.37520E-02, -1.45060E+00,  8.81353E-01, 1.03210E+00],
        [7.36799E-02, -1.46292E+00,  8.61989E-01, 1.03202E+00],
        [8.42652E-02, -1.47597E+00,  8.41858E-01, 1.03127E+00],
        [9.54915E-02, -1.49051E+00,  8.20987E-01, 1.02998E+00],
        [1.07342E-01, -1.50547E+00,  7.99414E-01, 1.02822E+00],
        [1.19797E-01, -1.51845E+00,  7.77209E-01, 1.02615E+00],
        [1.32839E-01, -1.52910E+00,  7.54473E-01, 1.02405E+00],
        [1.46447E-01, -1.53808E+00,  7.31335E-01, 1.02207E+00],
        [1.60600E-01, -1.54359E+00,  7.07952E-01, 1.02034E+00],
        [1.75276E-01, -1.54627E+00,  6.84480E-01, 1.01904E+00],
        [1.90453E-01, -1.54401E+00,  6.61044E-01, 1.01829E+00],
        [2.06107E-01, -1.53595E+00,  6.37745E-01, 1.01829E+00],
        [2.22215E-01, -1.52153E+00,  6.14669E-01, 1.01926E+00],
        [2.38751E-01, -1.50083E+00,  5.91895E-01, 1.02136E+00],
        [2.55689E-01, -1.47049E+00,  5.69490E-01, 1.02483E+00],
        [2.73005E-01, -1.43445E+00,  5.47525E-01, 1.02990E+00],
        [2.90670E-01, -1.39301E+00,  5.26066E-01, 1.03664E+00],
        [3.08658E-01, -1.34534E+00,  5.05144E-01, 1.04516E+00],
        [3.26941E-01, -1.29333E+00,  4.84770E-01, 1.05560E+00],
        [3.45492E-01, -1.23784E+00,  4.64958E-01, 1.06804E+00],
        [3.64280E-01, -1.17976E+00,  4.45742E-01, 1.08255E+00],
        [3.83277E-01, -1.12160E+00,  4.27173E-01, 1.09916E+00],
        [4.02455E-01, -1.06355E+00,  4.09293E-01, 1.11787E+00],
        [4.21783E-01, -1.00859E+00,  3.92112E-01, 1.13866E+00],
        [4.41231E-01, -9.54598E-01,  3.75605E-01, 1.16148E+00],
        [4.60770E-01, -9.02409E-01,  3.59738E-01, 1.18635E+00],
        [4.80370E-01, -8.52793E-01,  3.44500E-01, 1.21328E+00],
        [5.00000E-01, -8.05028E-01,  3.29899E-01, 1.24229E+00],
        [5.19630E-01, -7.59672E-01,  3.15952E-01, 1.27341E+00],
        [5.39230E-01, -7.17197E-01,  3.02658E-01, 1.30666E+00],
        [5.58769E-01, -6.78723E-01,  2.89994E-01, 1.34200E+00],
        [5.78217E-01, -6.41601E-01,  2.77900E-01, 1.37945E+00],
        [5.97545E-01, -6.08153E-01,  2.66279E-01, 1.41906E+00],
        [6.16723E-01, -5.82301E-01,  2.55018E-01, 1.46068E+00],
        [6.35720E-01, -5.55849E-01,  2.44050E-01, 1.50417E+00],
        [6.54508E-01, -5.23961E-01,  2.33408E-01, 1.54991E+00],
        [6.73059E-01, -4.94919E-01,  2.23192E-01, 1.59824E+00],
        [6.91342E-01, -4.67560E-01,  2.13479E-01, 1.64916E+00],
        [7.09330E-01, -4.43497E-01,  2.04271E-01, 1.70272E+00],
        [7.26995E-01, -4.21878E-01,  1.95524E-01, 1.75889E+00],
        [7.44311E-01, -4.00622E-01,  1.87184E-01, 1.81778E+00],
        [7.61249E-01, -3.83130E-01,  1.79202E-01, 1.87950E+00],
        [7.77785E-01, -3.69073E-01,  1.71537E-01, 1.94390E+00],
        [7.93893E-01, -3.53939E-01,  1.64169E-01, 2.01105E+00],
        [8.09547E-01, -3.37756E-01,  1.57100E-01, 2.08131E+00],
        [8.24724E-01, -3.24291E-01,  1.50347E-01, 2.15488E+00],
        [8.39400E-01, -3.10721E-01,  1.43927E-01, 2.23182E+00],
        [8.53553E-01, -2.99308E-01,  1.37837E-01, 2.31238E+00],
        [8.67161E-01, -2.88626E-01,  1.32056E-01, 2.39667E+00],
        [8.80203E-01, -2.80194E-01,  1.26558E-01, 2.48490E+00],
        [8.92658E-01, -2.71828E-01,  1.21329E-01, 2.57733E+00],
        [9.04508E-01, -2.63717E-01,  1.16358E-01, 2.67450E+00],
        [9.15735E-01, -2.57950E-01,  1.11636E-01, 2.77690E+00],
        [9.26320E-01, -2.52451E-01,  1.07162E-01, 2.88519E+00],
        [9.36248E-01, -2.43968E-01,  1.02928E-01, 3.00096E+00],
        [9.45503E-01, -2.28732E-01,  9.87739E-02, 3.12807E+00],
        [9.54072E-01, -2.41495E-01,  9.39894E-02, 3.27092E+00],
        [9.61940E-01, -3.34413E-01,  8.70067E-02, 3.42457E+00],
        [9.69096E-01, -8.78493E-01,  7.63438E-02, 3.52050E+00],
        [9.75528E-01, -8.18307E-01,  6.24286E-02, 3.54247E+00],
        [9.81228E-01, -8.14591E-01,  4.76031E-02, 3.59528E+00],
        [9.86185E-01, -7.52130E-01,  3.41075E-02, 3.65826E+00],
        [9.90393E-01, -6.51636E-01,  2.29752E-02, 3.73986E+00],
        [9.93844E-01, -5.34893E-01,  1.44718E-02, 3.84587E+00],
        [9.96534E-01, -3.58018E-01,  8.50578E-03, 3.99276E+00],
        [9.98459E-01, -1.77563E-01,  4.71737E-03, 4.22766E+00],
        [9.99615E-01, -4.66826E-02,  2.66421E-03, 4.74240E+00],
        [1.00000E+00,  7.83676E-17,  2.02545E-03, 4.97643E+00]
    ])

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
                (-tok.equilibrium.profiles_1d.fpol, r"$fpol$"),
                (Function(equilibrium.profiles_1d.psi_norm, equilibrium.profiles_1d.f), r"$f_{pol0}$"),
            ],

            [
                (Function(equilibrium.profiles_1d.psi_norm, equilibrium.profiles_1d.q), r"$q_0$"),
                (Function(profile[:, 0], profile[:, 3]), r"$q_1$"),
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
        # x_axis=(tok.equilibrium.profiles_1d.rho_tor_norm,   {"label": r"$\rho$"}),  # asd
        # x_axis=(tok.equilibrium.profiles_1d.phi,   {"label": r"$\Phi$"}),  # asd
        x_axis=(tok.equilibrium.profiles_1d.psi_norm,   {"label": r"$\rho_{N}$"}),  # asd
        # x_axis=(tok.equilibrium.profiles_1d.psi_norm,   {"label": r"$\psi_{N}$"}),  # asd
        grid=True) .savefig("/home/salmon/workspace/output/profiles_1d.svg", transparent=True)

    tok.update(transport_solver={})

    # rho_tor_bdry = tok.core_profiles.grid.rho_tor[-1]

    # vpr = Function(tok.equilibrium.profiles_1d.rho_tor_norm,
    #                tok.equilibrium.profiles_1d.vprime)(tok.core_profiles.grid.rho_tor)
    # logger.debug(tok.core_profiles.electrons.density.x)
    # x = tok.core_profiles.electrons.density.x
    # dx = (np.roll(x, -1)-x)
    # dx[-1] = dx[-2]

    plot_profiles(
        [
            # (1.0/dx,                                          {"marker": ".", "label": r"$1/dx$"}),

            (tok.core_profiles.electrons.density,             r"$n_{e}$"),
            [(tok.core_profiles.electrons.density.derivative, {"color": "green", "label":  r"$n_{e}^{\prime}$"}),
             (tok.core_profiles.electrons.density_prime,      {"color": "black", "label":  r"$n_{e}^{\prime}$"})],
            (tok.core_profiles.electrons.density.derivative - \
             tok.core_profiles.electrons.density_prime,        {"marker": ".", "label": r"$\Delta n_{e}^{\prime}$"}),

            (tok.core_profiles.electrons.n_gamma,             r"$\Gamma_{e}$"),
            (tok.core_profiles.electrons.n_gamma_prime,       r"$\Gamma_{e}^{\prime}$"),
            # (tok.core_profiles.electrons.n_rms_residuals,     {"marker": ".", "label":  r"residuals"}),
            # [
            #     (tok.core_profiles.electrons.n_diff,          {"color": "green", "label": r"D"}),
            #     (np.abs(tok.core_profiles.electrons.n_conv),  {"color": "black",  "label": r"v"}),
            # ],
            (tok.core_profiles.electrons.n_diff.derivative,   {"color": "green", "label": r"$D^{\prime}$"}),

            [
                (tok.core_profiles.electrons.n_s_exp_flux,    {"color": "green", "label": r"Source"}),
                (tok.core_profiles.electrons.n_diff_flux,     {"color": "black", "label": r"Diffusive flux"}),
                (tok.core_profiles.electrons.n_conv_flux,     {"color": "blue",  "label": r"Convective flux"}),
                (tok.core_profiles.electrons.n_residual,      {"color": "red",   "label": r"Residual"})
            ],
            # (tok.core_profiles.electrons.residuals_y,         r"$dn_{residuals}$"),
            # (tok.core_profiles.electrons.residuals_gamma,     r"$dgamma_{residuals}$"),
            [(tok.equilibrium.profiles_1d.q,                   r"$q$"),
             (tok.equilibrium.profiles_1d.dphi_dpsi,           r"$\frac{d\phi}{d\psi}$")],

            (tok.equilibrium.profiles_1d.vprime,              r"$V^{\prime}_{\psi_N}$"),
            (tok.core_profiles.electrons.vpr,                 r"$vpr$"),
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

    # # plot_profiles(tok.core_profiles.profiles_1d,
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
    #                       "electrons.conv_flux",
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
