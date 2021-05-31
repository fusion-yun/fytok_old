import pandas as pd
from fytok.modules.transport.CoreSources import CoreSources
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.Tokamak import Tokamak
from spdm.data.File import File
from spdm.data.Function import Function
from spdm.numlib import constants, np
from spdm.numlib.smooth import smooth
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles, sp_figure

if __name__ == "__main__":
    logger.info("====== START ========")

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml")

    baseline = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')
    bs_psi = baseline["Fp"].values
    bs_r_nrom = baseline["x"].values

    tok = Tokamak(
        wall=device.entry.find("wall"),
        pf_active=device.entry.find("pf_active"),
        tf=device.entry.find("tf"),
        magnetics=device.entry.find("magnetics"))

    ###################################################################################################
    if True:
        eqdsk = File(
            # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
            "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt",
            # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
            format="geqdsk")

        tok.equilibrium.update({"time": 0.0,
                                "time_slice": {
                                    "profiles_1d": eqdsk.entry.find("profiles_1d"),
                                    "profiles_2d": eqdsk.entry.find("profiles_2d"),
                                    "coordinate_system": {"grid": {"dim1": 128, "dim2": 256}}
                                },
                                "vacuum_toroidal_field":  eqdsk.entry.find("vacuum_toroidal_field"),
                                })

    if True:
        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},  "vessel": {"edgecolor": "blue"}},
                  pf_active={"facecolor": 'red'},
                  equilibrium={
                      "mesh": True,
                      "boundary": True,
                      "scalar_field": [("psirz", {"levels": 16, "linewidths": 0.1}), ],
                  }
                  ) .savefig("/home/salmon/workspace/output/tokamak.svg", transparent=True)

    if True:
        eq_profile = tok.equilibrium.time_slice.profiles_1d

        plot_profiles(
            [
                # (eq_profile.dpressure_dpsi,                                                       r"$dP/d\psi$"),
                # [
                #     (Function(eq_profile.psi_norm,  eq_profile.f_df_dpsi),  r"$ff^{\prime}_{0}$" ),
                #     (eq_profile.ffprime,                                                       r"$ff^{\prime}$"),
                # ],
                # [
                #     (Function(eq_profile.psi_norm, np.abs(eq_profile.f)),  r"$\left|f_{pol0}\right|$" ),
                #     (eq_profile.fpol,                                                                 r"$fpol$"),
                # ],
                [
                    (Function(bs_psi, baseline["q"].values),                     r"$q^{astra}$", {"marker": "+"}),
                    (eq_profile.q,                                                                        r"$q$"),
                    # (eq_profile.dphi_dpsi,                                             r"$\frac{d\phi}{d\psi}$"),
                ],
                [
                    (Function(bs_psi, baseline["rho"].values),           r"$\rho_{tor}^{astra}$", {"marker": "+"}),
                    (eq_profile.rho_tor,                                                        r"$\rho_{tor}$"),
                ],
                [
                    (Function(bs_psi, baseline["x"].values),           r"$\rho_{tor,0}^{astra}$", {"marker": "+"}),
                    (eq_profile.rho_tor_norm,                                                 r"$\rho_{tor,0}$"),
                ],
                [
                    (Function(bs_psi, baseline["Jtot"].values*1e6),   r"$j_{\parallel}^{astra}$", {"marker": "+"}),
                    (eq_profile.j_parallel,                                                  r"$j_{\parallel}$"),
                ],
                [
                    (Function(bs_psi, baseline["shif"].values), r"$\Delta^{astra}$ shafranov shift", {"marker": "+"}),
                    (eq_profile.geometric_axis.r-tok.equilibrium.vacuum_toroidal_field.r0,          r"$\Delta$ "),
                ],
                [
                    (Function(bs_psi, baseline["k"].values),         r"$k^{astra}$ elongation", {"marker": "+"}),
                    (eq_profile.elongation,                                                 r"$k$ elongation"),
                ],

                # (eq_profile.gm1,                                             r"$gm1=\left<\frac{1}{R^2}\right>$"),
                # (eq_profile.gm2,                    r"$gm2=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                # (eq_profile.gm3,                                r"$gm3=\left<\left|\nabla \rho\right|^2\right>$"),
                # (eq_profile.gm7,                                  r"$gm7=\left<\left|\nabla \rho\right|\right>$"),
                # (eq_profile.dphi_dpsi,                                                  r"$\frac{d\phi}{d\psi}$"),
                # (eq_profile.drho_tor_dpsi,                                        r"$\frac{d\rho_{tor}}{d\psi}$"),
                # (eq_profile.dvolume_drho_tor,                                              r"$\frac{dV}{d\rho}$"),
                # (eq_profile.dpsi_drho_tor,                                        r"$\frac{d\psi}{d\rho_{tor}}$"),
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
    if True:
        s_range = -1  # slice(0, 140, 1)
        Te = Function(bs_r_nrom, smooth(baseline["TE"].values*1000, s_range))
        Ti = Function(bs_r_nrom, smooth(baseline["TI"].values*1000, s_range))

        ne = Function(bs_r_nrom, smooth(baseline["NE"].values*1.0e19, s_range))
        nHe = Function(bs_r_nrom, baseline["Nalf"].values*1.0e19)
        # nDT = Function(bs_r_nrom, baseline["Nd+t"].values*1.0e19)
        nDT = ne * (1.0 - 0.02*4 - 0.0012*18) - nHe*2.0

        # Zeff = Function(bs_r_nrom, baseline["Zeff"].values)

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
                        "element": [{"a": 1, "z_n": 1, "atoms_n": 1}],
                        "density":  nDT/2.0,
                        "temperature": Ti,
                    },
                    {
                        "label": "D^+",
                        "z_ion": 1,
                        "element": [{"a": 2, "z_n": 1, "atoms_n": 1}],
                        "density":  nDT/2.0,
                        "temperature": Ti,
                    },
                    {
                        "label": r"He^{2}",
                        "z_ion": 2,
                        "element": [{"a": 4, "z_n": 1, "atoms_n": 1}],
                        "density": nHe,
                        "temperature": Ti,
                    },
                    {
                        "label": "Be^{4}",
                        "z_ion": 4,
                        "element": [{"a": 9, "z_n": 1, "atoms_n":   1}],
                        "density":    0.02*ne,
                        "temperature": Ti,
                    },
                    {
                        "label": "Ar^{18}",
                        "z_ion": 18,
                        "element": [{"a": 40, "z_n": 1, "atoms_n":   1}],
                        "density":    0.0012*ne,
                        "temperature": Ti,
                    }
                ],
                # "zeff": Zeff
            }}

        tok.core_profiles.update(core_profiles_conf,
                                 grid=tok.equilibrium.time_slice.radial_grid(),
                                 vacuum_toroidal_field=tok.equilibrium.vacuum_toroidal_field)

        core_profile = tok.core_profiles.profiles_1d

        plot_profiles(
            [
                [
                    # (Function(bs_r_nrom, baseline["NE"].values*1.0e19),              r"$n_{e}^{astra}$"),
                    (core_profile.electrons.density,                                                      r"$n_e$"),
                    *[(ion.density,                            f"$n_{{{ion.label}}}$") for ion in core_profile.ion],

                ],
                [
                    # (Function(bs_r_nrom, baseline["TE"].values),                     r"$T_{e}^{astra}$"),
                    (core_profile.electrons.temperature,                                                  r"$T_e$"),
                    *[(ion.temperature,                        f"$T_{{{ion.label}}}$") for ion in core_profile.ion],
                ],

                [
                    (Function(bs_r_nrom, baseline["Zeff"].values),                            r"$Z_{eff}^{astra}$"),
                    (core_profile.zeff,                                                               r"$z_{eff}$"),
                ],
                (core_profile.e_field.parallel,                                                 r"$E_{\parallel}$"),

                [
                    (core_profile.j_ohmic,                                                          r"$j_{ohmic}$"),
                    (Function(bs_r_nrom, baseline["Joh"].values),                              r"$j_{oh}^{astra}$"),
                ],
                (core_profile.grid.psi,                                                                  r"$\psi$"),
                (core_profile.electrons.pressure,                                                        r"$p_e $"),
                (core_profile.electrons.density,                                                         r"$n_e $"),
                (core_profile.electrons.temperature,                                                     r"$T_e $"),
                (core_profile.electrons.pressure.derivative(),                                   r"$p_e^{\prime}$"),
                (core_profile.electrons.density.derivative(),                                    r"$n_e^{\prime}$"),
                (core_profile.electrons.temperature.derivative(),                                r"$T_e^{\prime}$"),

            ],
            x_axis=(core_profile.grid.rho_tor_norm,                                   r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_profile.svg", transparent=True)

    ###################################################################################################
    if True:
        core_transport = CoreTransport({"model": [
            # {"code": {"name": "neoclassical"}},
            # {"code": {"name": "nclass"}},
            {"code": {"name": "spitzer"}},
            # {"code": {"name": "gyroBhom"}},
        ]},
            grid=tok.equilibrium.time_slice.coordinate_system.radial_grid()
        )

        core_transport.advance(dt=0.1,  equilibrium=tok.equilibrium,     core_profiles=tok.core_profiles)

        core_transport1d = core_transport.model[0].profiles_1d

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

                # (Function(bs_r_nrom, baseline["Xi"].values),          r"$\chi_{i}^{astra}$"),
                # [
                #     (core_transport1d.electrons.energy.v,                                         r"$v_{Te}$"),
                #     *[(ion.energy.v,                f"$v_{{T,{ion.label}}}$") for ion in core_transport1d.ion],
                # ],
                [
                    (Function(bs_r_nrom, np.log(baseline["XiNC"].values)),
                     r"$ln \chi_{i,nc}^{astra}$", {"marker": "+"}),
                    * [(np.log(ion.energy.d),   f"$ln \chi_{{{ion.label},wesson}}$") for ion in core_transport1d.ion],
                ],
                [
                    (Function(bs_r_nrom, baseline["Zeff"].values),          r"$Z_{eff}^{astra}$", {"marker": "+"}),
                    (core_profile.zeff,                                                              r"$z_{eff}$"),
                ],
                [
                    (Function(bs_r_nrom, baseline["Joh"].values*1.0e6 / baseline["U"].values * \
                              (2.0*constants.pi * tok.equilibrium.vacuum_toroidal_field.r0)),     r"$\sigma_{\parallel}^{astra}$", {"marker": "+"}),
                    (core_transport1d.conductivity_parallel,                 r"$\sigma_{\parallel}^{wesson}$"),
                    (core_transport.model.combine.profiles_1d.conductivity_parallel,  r"$\sigma_{\parallel}$"),

                ],

                # (core_transport1d.e_field_radial,                                             r"$E_{radial}$"),
                # (tok.equilibrium.time_slice[-1].profiles_1d.trapped_fraction(
                # core_transport.model[0].profiles_1d[-1].grid_v.psi_norm),      r"trapped"),
                # (core_profile.electrons.pressure,                                                  r"$p_{e}$"),

            ],
            x_axis=(core_transport.model[0].profiles_1d.grid_v.rho_tor_norm, r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            # annotation=core_transport.model[0].identifier.name,
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_transport.svg", transparent=True)

    if True:
        core_sources = CoreSources({"source": [
            {"code": {"name": "bootstrap_current"}},
            {"code": {"name": "dummy"}},
        ]},
            grid=tok.equilibrium.time_slice.coordinate_system.radial_grid()
        )

        core_sources.advance(dt=0.1, equilibrium=tok.equilibrium, core_profiles=tok.core_profiles)

        core_source_1d = core_sources.source[0].profiles_1d

        plot_profiles(
            [
                # [
                #     (Function(bs_r_nrom, baseline["Zeff"].values),          r"$Z_{eff}^{astra}$", {"marker": "+"}),
                #     (core_profile.zeff,                                                              r"$z_{eff}$"),
                # ],
                # [
                #     (Function(bs_r_nrom, baseline["Joh"].values*1.0e6 / baseline["U"].values *
                #               (2.0*constants.pi * tok.equilibrium.vacuum_toroidal_field.r0)),     r"$\sigma_{\parallel}^{astra}$", {"marker": "+"}),
                #     (core_source_1d.conductivity_parallel*11/14,
                #      r"$\sigma_{\parallel}^{wesson}$"),
                # ],
                [
                    (Function(bs_r_nrom, baseline["Jbs"].values*1.0e6), r"$j_{bootstrap}^{astra}$", {"marker": "+"}),
                    (core_source_1d.j_parallel,                                         r"$j_{bootstrap}^{wesson}$"),
                ],
                (core_sources.source.combine.profiles_1d.j_parallel,             r"$j_{bootstrap}^{wesson}$"),

                (tok.core_profiles.profiles_1d.electrons.density,                                       r"$ n_e $"),
                (tok.core_profiles.profiles_1d.electrons.temperature,                                   r"$ T_e $"),
                (tok.core_profiles.profiles_1d.electrons.temperature.dln(),          r"$\frac{T_e^{\prime}}{T_e}$"),
                (tok.core_profiles.profiles_1d.electrons.density.dln(),              r"$\frac{n_e^{\prime}}{n_e}$"),
                (tok.core_profiles.profiles_1d.electrons.pressure.dln(),             r"$\frac{p_e^{\prime}}{p_e}$"),
                (tok.core_profiles.profiles_1d.electrons.tau,                                          r"$\tau_e$"),
                (tok.core_profiles.profiles_1d.electrons.vT,                                          r"$v_{T,e}$"),

                # (core_source_1d.e_field_radial,                                             r"$E_{radial}$"),
                # (tok.equilibrium.time_slice[-1].profiles_1d.trapped_fraction(
                # core_transport.model[0].profiles_1d[-1].grid_v.psi_norm),      r"trapped"),
                # (core_profile.electrons.pressure,                                                  r"$p_{e}$"),

            ],
            x_axis=(core_source_1d.grid.rho_tor_norm, r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            # annotation=core_transport.model[0].identifier.name,
            # index_slice=slice(1, 110, 1),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_sources.svg", transparent=True)

    logger.info("====== DONE ========")
