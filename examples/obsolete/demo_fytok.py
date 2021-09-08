
import numpy as np
import pandas as pd
from scipy import constants
from fytok.Tokamak import Tokamak
from spdm.data.Collection import Collection
from spdm.data.File import File
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles, sp_figure

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
        "radial_grid": {"axis": 64},
        "wall":  device.wall,
        "pf_active": device.pf_active,
        "equilibrium": equilibrium  # {
        #     "vacuum_toroidal_field": equilibrium.vacuum_toroidal_field,
        #     "time_slice": [{
        #         "global_quantities": equilibrium.global_quantities,
        #         "profiles_1d": equilibrium.profiles_1d,
        #         "profiles_2d": equilibrium.profiles_2d,
        #         "coordinate_system": {"grid": {"dim1": 64, "dim2": 512}}
        #     }]
        # }
        ,
        "core_profiles": {
            "profiles_1d": {
                "electrons": {
                    "label": "electrons",
                    "density":     1e19,
                    "temperature": lambda x: (1-x**2)**2
                },
                "conductivity_parallel": 1.0,
                "psi":   1.0, }
        }
    })

    sp_figure(tok,
              wall={"limiter": {"edgecolor": "green"},  "vessel": {"edgecolor": "blue"}},
              pf_active={"facecolor": 'red'},
              equilibrium={"mesh": True, "boundary": True,
                           "scalar_field": [
                               #   ("coordinate_system.norm_grad_psi", {"levels": 32, "linewidths": 0.1}),
                               ("psirz", {"levels": 32, "linewidths": 0.1}),
                           ],
                           }
              ) .savefig("/home/salmon/workspace/output/contour.svg", transparent=True)

    eq_f = equilibrium.time_slice[-1]
    eq = tok.equilibrium.time_slice[-1]
    plot_profiles(
        [
            [
                (eq.profiles_1d.ffprime,            r"$ff^{\prime}$"),
                (Function(eq_f.profiles_1d.psi_norm,
                          eq_f.profiles_1d.f_df_dpsi),   r"$ff^{\prime}_{0}$"),
            ],
            [
                (eq.profiles_1d.fpol,              r"$fpol$"),
                (Function(eq_f.profiles_1d.psi_norm,
                          np.abs(eq_f.profiles_1d.f)),   r"$\left|f_{pol0}\right|$"),
            ],
            [
                (eq.profiles_1d.q,                    r"$q$"),
                # (eq.profiles_1d.dphi_dpsi,                    r"$\frac{d\phi}{d\psi}$"),
                # (Function(eq_f.profiles_1d.psi_norm, eq_f.profiles_1d.q), r"$q_0$"),
                (Function(profile["Fp"].values, profile["q"].values),             r"$q^{\star}$"),
            ],
            [
                (eq.profiles_1d.rho_tor,           r"$\rho_{tor}$"),
                (Function(profile["Fp"].values, profile["rho"].values),             r"$\rho_{tor}^{\star}$"),
                #     # (eq.profiles_1d.dvolume_drho_tor / ((constants.pi**2) * 4.0 * eq.vacuum_toroidal_field.r0),
                #     #     r"$\frac{dV/d\rho_{tor}}{4\pi^2 R_0}$"),
            ],
            (eq.profiles_1d.rho_tor_norm,           r"$\rho_{tor}/\rho_{tor,0}$"),

            # [
            #     # (eq.profiles_1d.j_tor, r"$j_{tor}$"),
            #     (eq.profiles_1d.j_parallel,                          r"$j_{\parallel}$"),
            #     (Function(profile["Fp"].values, profile["Jtot"].values*1e6),      r"$j_{\parallel}^{\star}$"),
            # ],
            # [
            #     (eq.profiles_1d.geometric_axis.r,                   r"$geometric_{axis.r}$"),
            #     (eq.profiles_1d.r_inboard,                          r"$r_{inboard}$"),
            #     (eq.profiles_1d.r_outboard,                         r"$r_{outboard}$"),

            # ],
            # [
            # (Function(profile["Fp"].values, (profile["Jtot"].values-profile["Jbs"].values-- \
            #                                  profile["Jext"].values)**2),                   r"$j_{total}^2$"),
            # (Function(profile["Fp"].values, profile["Poh"].values),                       r"$P_{oh}^{\star}$"),

            # ],
            # (eq.profiles_1d.phi,                   r"$\Phi$"),
            # (eq.profiles_1d.dpsi_drho_tor,         r"$\frac{d\psi}{d\rho_{tor}}$"),
            # [
            #     (Function(eq_f.profiles_1d.psi_norm, eq_f.profiles_1d.q), r"$q_0$"),
            #     (eq.profiles_1d.q,                 r"$q$"),
            #     (eq.profiles_1d.dphi_dpsi,         r"$\frac{d\phi}{d\psi}$"),
            # ],
            # (eq.profiles_1d.rho_tor,                r"$\rho_{tor}$"),

            # [
            #     (eq.profiles_1d.volume,                r"$V$"),
            #     (Function(eq.profiles_1d.rho_tor, eq.profiles_1d.dvolume_drho_tor.view(np.ndarray)).antiderivative,
            #      r"$\int \frac{dV}{d\rho_{tor}}  d\rho_{tor}$"),
            #     (eq.profiles_1d.dvolume_dpsi.antiderivative * \
            #      (eq.global_quantities.psi_boundary - eq.global_quantities.psi_axis),\
            #      r"$\int \frac{dV}{d\psi}  d\psi$"),
            # ],
            # (eq.profiles_1d.dvolume_drho_tor,      r"$\frac{dV}{d\rho}$"),
            # (eq.profiles_1d.dpsi_drho_tor,         r"$\frac{d\psi}{d\rho_{tor}}$"),
            # (eq.profiles_1d.drho_tor_dpsi,         r"$\frac{d\rho_{tor}}{d\psi}$"),
            # (eq.profiles_1d.gm1,                   r"$\left<\frac{1}{R^2}\right>$"),
            # (eq.profiles_1d.gm2,       r"$\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
            # (eq.profiles_1d.gm3,                   r"$\left<\left|\nabla \rho\right|^2\right>$"),
            # (eq.profiles_1d.gm7,                   r"$\left<\left|\nabla \rho\right|\right>$"),
            # (eq.profiles_1d.dphi_dpsi, r"$\frac{d\phi}{d\psi}$"),
            # (eq.profiles_1d.drho_tor_dpsi, r"$\frac{d\rho_{tor}}{d\psi}$"),
            # (tok.core_profiles.profiles_1d.electrons.temperature, r"$T_{e}$"),
            # [
            #     (eq.coordinate_system.surface_integrate2(lambda r, z:1.0/r**2), \
            #      r"$\left<\frac{1}{R^2}\right>$"),
            #     (eq.coordinate_system.surface_integrate(1/eq.coordinate_system.r**2), \
            #      r"$\left<\frac{1}{R^2}\right>$"),
            # ]

        ],
        # x_axis=(eq.profiles_1d.rho_tor_norm,   {"label": r"$\rho_{N}$"}),  # asd
        # x_axis=(eq.profiles_1d.phi,   {"label": r"$\Phi$"}),  # asd
        x_axis=(eq.profiles_1d.psi_norm,    r"$\psi_{N}$"),  # asd
        grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/profiles_1d.svg", transparent=True)

    psi_norm = eq.profiles_1d.psi_norm
    rho_tor_norm = eq.profiles_1d.rho_tor_norm

    r_ped = 0.96  # np.sqrt(0.88)

    S0 = 7.5e20

    n_src = Function(rho_tor_norm, lambda x: S0 * np.exp(15.0*(x**2-1.0)))

    diff = Function(rho_tor_norm,
                    [lambda r:r < r_ped, lambda r:r >= r_ped],
                    [lambda x:0.4*(1.0 + 3*(x**2)), lambda x: 0.1785])

    v_pinch = diff * rho_tor_norm * 1.5 / equilibrium.vacuum_toroidal_field.r0

    j_parallel = eq.profiles_1d.j_parallel.pullback(psi_norm, rho_tor_norm)
    Qoh = Function(profile['x'].values, profile['Poh'].values*1e6/constants.elementary_charge)
    conductivity_parallel = 2.0e-9

    tok.update(
        core_transport={
            "profiles_1d": {
                "conductivity_parallel": conductivity_parallel,
                "electrons": {
                    "particles": {"d": diff,  "v": - v_pinch},
                    "energy": {"d":  diff,  "v": - v_pinch},
                }}
        },

        core_sources={
            "profiles_1d": {
                "electrons": {
                    "particles": n_src,
                    "energy": j_parallel**2/conductivity_parallel,
                },
                "j_parallel": j_parallel,
                "conductivity_parallel": conductivity_parallel
            }
        },

        boundary_conditions={
            "current": {
                "identifier": {"index": 1},
                "value": eq.global_quantities.psi_boundary
            },
            "electrons": {
                "particles": {
                    "identifier": {"index": 1},
                    "value": 3e19
                },
                "energy": {
                    "identifier": {"index": 1},
                    "value": 0.2  # 　keV
                }
            },
            "ion": [
                {
                    "label": "H",
                    "a": 1.0,
                    "z_ion": 1,
                    "particles": {
                        "identifier": {"index": 1},
                        "value": (1, 0, 4.6e19)
                    },
                    "energy": {
                        "identifier": {"index": 1},
                        "value": (1, 0, 4.6e19)
                    }
                }
            ]
        }
    )

    psi_norm = eq.profiles_1d.psi_norm
    rho_tor_norm = eq.profiles_1d.rho_tor_norm
    psi_prev = eq.profiles_1d.psi.pullback(psi_norm, rho_tor_norm)
    psi_next = tok.core_profiles.profiles_1d.psi

    plot_profiles(
        [
            [
                (tok.core_profiles.profiles_1d.electrons.diff,       r"D",    {"color": "green"}),
                (Function(profile["x"].values, profile["Dn"].values),   r"$D^{\star}$"),
                (np.abs(tok.core_profiles.profiles_1d.electrons.conv), r"$\left|v\right|$",  {"color": "black"}),
            ],

            [
                (tok.core_profiles.profiles_1d.electrons.s_exp_flux,   r"Source",  {"color": "green", }),
                (tok.core_profiles.profiles_1d.electrons.diff_flux,    r"Diffusive flux",  {"color": "black", }),
                (tok.core_profiles.profiles_1d.electrons.conv_flux,    r"Convective flux",  {"color": "red", }),
                (tok.core_profiles.profiles_1d.electrons.residual,     r"Residual",  {"color": "blue", }),
            ],
            (n_src,   r"$S_{edge}$",           {"color": "green"}),
            # [
            #     ((psi_prev-psi_prev[0])/(psi_prev[-1]-psi_prev[0]),
            #      r"$(\psi^{-}-\psi^{-}_{axis})/(\psi^{-}_{bdry}\psi^{-}_{axis})$"),
            #     ((psi_next-psi_next[0])/(psi_next[-1]-psi_next[0]),
            #      r"$(\psi^{+}-\psi^{+}_{axis})/(\psi^{+}_{bdry}\psi^{+}_{axis})$"),
            # ],

            [
                (tok.core_profiles.profiles_1d.electrons.density,                           r"$n_{e}$"),
                (Function(profile["x"].values, profile["NE"].values*1e19),      r"$n_{e}^{\star}$"),
            ],
            [
                (tok.core_profiles.profiles_1d.electrons.temperature,                       r"$T_{e}$"),
                (Function(profile["x"].values, profile["TE"].values),           r"$T_{e}^{\star}$"),
            ],
            # (eq.profiles_1d.q.pullback(psi_norm, rho_tor_norm),                    r"$q$"),
            # [
            #     (eq.profiles_1d.j_parallel.pullback(psi_norm, rho_tor_norm),       r"$j_{\parallel}$"),
            #     (Function(profile["x"].values, profile["Jtot"].values*1e6),     r"$j_{\parallel}^{\star}$"),

            # ],
            # (tok.core_profiles.profiles_1d.electrons.density_flux,                              r"$\Gamma_{e}$"),
            # [
            #     (tok.core_profiles.profiles_1d.electrons.density.derivative, {"color": "green", "label":  r"$n_{e}^{\prime}$"}),
            #     (tok.core_profiles.profiles_1d.electrons.density_prime,      {"color": "black", "label":  r"$n_{e}^{\prime}$"}),
            # ],


            # (tok.core_profiles.profiles_1d.electrons.temperature_prime,                     r"$T_{e}^{\prime}$"),
            # [
            #     (-tok.core_profiles.profiles_1d.electrons.heat_flux + \
            #      tok.core_profiles.profiles_1d.electrons.T_conv_flux,                             r"$T_e V_{e}^{pinch}$"),
            # ],
            # [
            #     (tok.core_profiles.profiles_1d.electrons.T_d,                                   r"$\chi_{e}$"),
            #     (tok.core_profiles.profiles_1d.electrons.T_e,                                   r"$V_{e}^{pinch}$"),
            # ],
            # (tok.core_profiles.profiles_1d.electrons.T_s_exp_flux,    {"color": "green", "label": r"Source"}),
            # [
            #     (tok.core_profiles.profiles_1d.electrons.T_s_exp_flux,    {"color": "green", "label": r"Source"}),
            #     (tok.core_profiles.profiles_1d.electrons.T_diff_flux,     {"color": "black", "label": r"Diffusive flux"}),
            #     (tok.core_profiles.profiles_1d.electrons.T_conv_flux,     {"color": "red",   "label": r"Convective flux"}),
            #     (tok.core_profiles.profiles_1d.electrons.T_residual,      {"color": "blue",  "label": r"Residual"}),
            # ],

        ],
        x_axis=(tok.core_profiles.profiles_1d.electrons.density.x,   r"$\rho_{N}$"),  # x axis,
        # index_slice=slice(-100,None, 1),
        grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/electron_1d.svg", transparent=True)

    logger.info("Done")
