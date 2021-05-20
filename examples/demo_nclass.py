
from logging import log
import numpy as np
import pandas as pd
import scipy.constants

from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.Tokamak import Tokamak
from spdm.data.Collection import Collection
from spdm.data.File import File
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles, sp_figure


if __name__ == "__main__":
    logger.info("====== START ========")

    baseline = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml").entry

    tok = Tokamak(
        wall=device.wall,
        pf_active=device.pf_active,
        tf=device.tf,
        magnetics=device.magnetics)

    eq_conf = File(
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        format="geqdsk").entry

    eq_conf.coordinate_system = {"grid": {"dim1": 100, "dim2": 256}}

    eq_slice = tok.equilibrium.time_slice.insert(eq_conf, time=0.0)

    ###################################################################################################
    if True:
        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},  "vessel": {"edgecolor": "blue"}},
                  pf_active={"facecolor": 'red'},
                  equilibrium={
                      "mesh": True,
                      "boundary": True,
                      "scalar_field": [("psirz", {"levels": 32, "linewidths": 0.1}), ],
                  }
                  ) .savefig("/home/salmon/workspace/output/tokamak.svg", transparent=True)
    if True:
        eq_profile = tok.equilibrium.current_state.profiles_1d

        plot_profiles(
            [
                (eq_profile.dpressure_dpsi,                                                       r"$dP/d\psi$"),
                [
                    (eq_profile.ffprime,                                                       r"$ff^{\prime}$"),
                    (Function(eq_profile.psi_norm,  eq_profile.f_df_dpsi),                 r"$ff^{\prime}_{0}$"),
                ],
                [
                    (eq_profile.fpol,                                                                 r"$fpol$"),
                    (Function(eq_profile.psi_norm, np.abs(eq_profile.f)),            r"$\left|f_{pol0}\right|$"),
                ],
                [
                    (Function(baseline["Fp"].values, baseline["q"].values),                      r"$q^{\star}$"),
                    (eq_profile.q,                                                                       r"$q$"),
                    (eq_profile.dphi_dpsi,                                             r"$\frac{d\phi}{d\psi}$"),
                ],
                [
                    (Function(baseline["Fp"].values, baseline["rho"].values),           r"$\rho_{tor}^{\star}$"),
                    (eq_profile.rho_tor,                                                        r"$\rho_{tor}$"),
                ],
                [
                    (Function(baseline["Fp"].values, baseline["x"].values),           r"$\rho_{tor,0}^{\star}$"),
                    (eq_profile.rho_tor_norm,                                                 r"$\rho_{tor,0}$"),
                ],
                [
                    (Function(baseline["Fp"].values, baseline["Jtot"].values*1e6),   r"$j_{\parallel}^{\star}$"),
                    (eq_profile.j_parallel,                                                  r"$j_{\parallel}$"),
                ],
                # [
                #     (eq_profile.geometric_axis.r,                                     r"$geometric_{axis.r}$"),
                #     (eq_profile.r_inboard,                                                   r"$r_{inboard}$"),
                #     (eq_profile.r_outboard,                                                 r"$r_{outboard}$"),
                # ],

                # [
                #     (eq_profile.volume,                r"$V$"),
                #     # (Function(eq_profile.rho_tor, eq_profile.dvolume_drho_tor).antiderivative,
                #     #  r"$\int \frac{dV}{d\rho_{tor}}  d\rho_{tor}$"),
                #     (eq_profile.dvolume_dpsi.antiderivative * \
                #      (eq.global_quantities.psi_boundary - eq.global_quantities.psi_axis),\
                #      r"$\int \frac{dV}{d\psi}  d\psi$"),
                # ],

                (eq_profile.gm1,                                             r"$gm1=\left<\frac{1}{R^2}\right>$"),
                (eq_profile.gm2,                    r"$gm2=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                (eq_profile.gm3,                                r"$gm3=\left<\left|\nabla \rho\right|^2\right>$"),
                (eq_profile.gm7,                                  r"$gm7=\left<\left|\nabla \rho\right|\right>$"),
                (eq_profile.dphi_dpsi,                                                  r"$\frac{d\phi}{d\psi}$"),
                (eq_profile.drho_tor_dpsi,                                        r"$\frac{d\rho_{tor}}{d\psi}$"),
                (eq_profile.dvolume_drho_tor,                                              r"$\frac{dV}{d\rho}$"),
                (eq_profile.dpsi_drho_tor,                                        r"$\frac{d\psi}{d\rho_{tor}}$"),
                # [
                #     (eq.coordinate_system.surface_integrate2(lambda r, z:1.0/r**2), \
                #      r"$\left<\frac{1}{R^2}\right>$"),
                #     (eq.coordinate_system.surface_integrate(1/eq.coordinate_system.r**2), \
                #      r"$\left<\frac{1}{R^2}\right>$"),
                # ]
            ],
            x_axis=(eq_profile.psi_norm,                                                              r"$\psi_{N}$"),
            grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/equilibrium.svg", transparent=True)

    ###################################################################################################

    ne = Function(baseline["x"].values, baseline["NE"].values*1.0e19)
    Te = Function(baseline["x"].values, baseline["TE"].values*1000)
    nDT = Function(baseline["x"].values, baseline["Nd+t"].values*1.0e19)
    Ti = Function(baseline["x"].values, baseline["TI"].values*1000)
    n_alpha = Function(baseline["x"].values, baseline["Nalf"].values*1000)
    Zeff = Function(baseline["x"].values, baseline["Zeff"].values)

    core_profiles_conf = {
        "profiles_1d": {
            "electrons": {
                "label": "electrons",
                "density":     ne,
                "temperature": Te,
            },
            "ion": [
                {
                    "label": "H^+",
                    "z_ion": 1,
                    "neutral_index": 1,
                    "element": [{"a": 1, "z_n": 1, "atoms_n": 1}],
                    "density":  nDT/2.0,
                    "temperature": Ti,
                },
                {
                    "label": "D^+",
                    "z_ion": 1,
                    "element": [{"a": 2, "z_n": 1, "atoms_n": 1}],
                    "neutral_index": 2,
                    "density":  nDT/2.0,
                    "temperature": Ti,
                },
                {
                    "label": r"\alpha^{2+}",
                    "z_ion": 2,
                    "element": [{"a": 4, "z_n": 1, "atoms_n": 1}],
                    "neutral_index": 2,
                    "density": n_alpha,
                    "temperature": Ti,
                },
                {
                    "label": "Be^{2+}",
                    "z_ion": 2,
                    "element": [{"a": 9, "z_n": 1, "atoms_n":   1}],
                    "neutral_index": 4,
                    "density":    0.02*ne,
                    "temperature": Ti,
                },
                {
                    "label": "Ar^+",
                    "z_ion": 1,
                    "element": [{"a": 40, "z_n": 1, "atoms_n":   1}],
                    "neutral_index": 18,
                    "density":    0.0012*ne,
                    "temperature": Ti,
                }
            ],
            "zeff": Zeff
        }}

    core_profile_slice = tok.core_profiles.time_slice.push_back(core_profiles_conf,
                                                                grid=eq_slice.radial_grid(),
                                                                vacuum_toroidal_field=eq_slice.vacuum_toroidal_field)
    if True:
        core_profile = core_profile_slice.profiles_1d

        plot_profiles(
            [
                [
                    # (Function(baseline["x"].values, baseline["NE"].values*1.0e19),              r"$n_{e}^{\star}$"),
                    (core_profile.electrons.density,                                                      r"$n_e$"),
                    *[(ion.density,                            f"$n_{{{ion.label}}}$") for ion in core_profile.ion],

                ],
                [
                    # (Function(baseline["x"].values, baseline["TE"].values),                     r"$T_{e}^{\star}$"),
                    (core_profile.electrons.temperature,                                                  r"$T_e$"),
                    *[(ion.temperature,                        f"$T_{{{ion.label}}}$") for ion in core_profile.ion],
                ],

                # [
                (Function(baseline["x"].values, baseline["Zeff"].values),                     r"$Z_{eff}^{\star}$"),
                (core_profile.zeff,                                                                   r"$z_{eff}$"),
                # ],
                # [
                (core_profile.j_ohmic,                                                              r"$j_{ohmic}$"),
                (Function(baseline["x"].values, baseline["Joh"].values),                       r"$j_{oh}^{\star}$"),
                # ],
                (core_profile.grid.psi,                                                                  r"$\psi$"),
                (core_profile.electrons.pressure,                                                        r"$p_e $"),
                (core_profile.electrons.density,                                                         r"$n_e $"),
                (core_profile.electrons.temperature,                                                     r"$T_e $"),
                (core_profile.electrons.pressure.derivative,                                     r"$p_e^{\prime}$"),
                (core_profile.electrons.density.derivative,                                      r"$n_e^{\prime}$"),
                (core_profile.electrons.temperature.derivative,                                  r"$T_e^{\prime}$"),

            ],
            x_axis=(core_profile.grid.rho_tor_norm,                                   r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_profile.svg", transparent=True)

    ###################################################################################################
    if True:
        core_transport = CoreTransport({"model": [
            {"code": {"name": "neoclassical"}},
            # {"code": {"name": "nclass"}},
            # {"code": {"name": "spitzer"}},
            # {"code": {"name": "gyroBhom"}},
        ]})

        core_transport.advance(dt=0.1,
                               equilibrium=tok.equilibrium.current_state,
                               core_profiles=tok.core_profiles.current_state)

        # tok.equilibrium.advance(dt=0.1)

        # tok.core_profiles.advance(dt=0.1)

        # core_transport.advance(dt=0.1,
        #                        equilibrium=tok.equilibrium.current_state,
        #                        core_profiles=tok.core_profiles.current_state)

        # core_transport1d = core_transport.current_state.profiles_1d
        core_transport1d = core_transport.model[0].profiles_1d[-1]
        logger.debug(core_profile_slice.vacuum_toroidal_field.r0)
    # if False:
        plot_profiles(
            [
                # [
                #     (core_transport1d.electrons.particles.flux,                                 r"$\Gamma_e$"),
                #     *[(ion.particles.flux,       f"$\Gamma_{{{ion.label}}}$") for ion in core_transport1d.ion],
                # ],
                # [
                #     (core_transport1d.electrons.particles.d,                                         r"$D_e$"),
                #     *[(ion.particles.d,               f"$D_{{{ion.label}}}$") for ion in core_transport1d.ion],
                # ],
                # [
                #     (core_transport1d.electrons.particles.v,                                         r"$v_e$"),
                #     *[(ion.particles.v,               f"$v_{{{ion.label}}}$") for ion in core_transport1d.ion],
                # ],
                # [
                #     (core_transport1d.electrons.energy.flux,                                         r"$q_e$"),
                #     *[(ion.energy.flux,               f"$q_{{{ion.label}}}$") for ion in core_transport1d.ion],
                # ],

                # (Function(baseline["x"].values, baseline["Xi"].values),          r"$\chi_{i}^{\star}$"),
                [
                    (Function(baseline["x"].values, baseline["XiNC"].values*0.2),     r"$0.2 \chi_{i,nc}^{\star}$"),
                    # (core_transport1d.electrons.energy.d,                                         r"$\chi_e$"),
                    # *[(ion.energy.d,               f"$\chi_{{{ion.label}}},nclass$")
                    #   for ion in core_transport.model[1].profiles_1d[-1].ion],
                    *[(ion.energy.d,               f"$\chi_{{{ion.label},wesson}}$")
                      for ion in core_transport1d.ion],
                ],
                # [
                #     (core_transport1d.electrons.energy.v,                                         r"$v_{Te}$"),
                #     *[(ion.energy.v,                f"$v_{{T,{ion.label}}}$") for ion in core_transport1d.ion],
                # ],
                [
                    (Function(baseline["x"].values, baseline["Zeff"].values),                 r"$Z_{eff}^{\star}$"),
                    (core_profile.zeff,                                                               r"$z_{eff}$"),
                ],
                [
                    (Function(baseline["x"].values, baseline["Joh"].values*1.0e6 / baseline["U"].values * \
                              (2.0*scipy.constants.pi * core_profile_slice.vacuum_toroidal_field.r0)),     r"$\sigma_{\parallel}^{\star}$"),
                    (core_transport1d.conductivity_parallel*11/14,
                     r"$\sigma_{\parallel}^{wesson}$"),
                ],
                [
                    (Function(baseline["x"].values, baseline["Jbs"].values*1.0e6),     r"$j_{bootstrap}^{\star}$"),
                    (core_transport1d.j_bootstrap,                                    r"$j_{bootstrap}^{wesson}$"),
                ],
                # (core_transport1d.e_field_radial,                                             r"$E_{radial}$"),
                # (tok.equilibrium.time_slice[-1].profiles_1d.trapped_fraction(
                # core_transport.model[0].profiles_1d[-1].grid_v.psi_norm),      r"trapped"),

                # (core_profile.electrons.pressure,                                                  r"$p_{e}$"),

            ],
            x_axis=(core_transport.model[0].profiles_1d[-1].grid_v.rho_tor_norm, r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            # annotation=core_transport.model[0].identifier.name,
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_transport.svg", transparent=True)

    logger.info("====== DONE ========")
