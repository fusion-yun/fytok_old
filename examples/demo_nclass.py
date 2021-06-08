from operator import eq
import pandas as pd

from fytok.Tokamak import Tokamak
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.File import File
from spdm.data.Function import Function
from spdm.numlib import constants, np
from spdm.numlib.smooth import smooth
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles, sp_figure
from fytok.common.Atoms import atoms

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
        magnetics=device.entry.find("magnetics"),
        equilibrium={"code": {"name": "dummy"}},
        core_profiles={},
        transport_solver={"code": {"name": "bvp_solver"}},
    )

    ###################################################################################################
    if True:  # Equlibrium
        eqdsk = File(
            # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
            "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt",
            # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
            format="geqdsk")

        tok.equilibrium["vacuum_toroidal_field"] = eqdsk.entry.find("vacuum_toroidal_field", {})

        tok.equilibrium["time_slice"] = {
            "profiles_1d": eqdsk.entry.find("profiles_1d"),
            "profiles_2d": eqdsk.entry.find("profiles_2d"),
            "coordinate_system": {"grid": {"dim1": 128, "dim2": 128}}
        }

        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},  "vessel": {"edgecolor": "blue"}},
                  pf_active={"facecolor": 'red'},
                  equilibrium={
                      "mesh": True,
                      "boundary": True,
                      "scalar_field": [("psirz", {"levels": 16, "linewidths": 0.1}), ],
                  }
                  ) .savefig("/home/salmon/workspace/output/tokamak.svg", transparent=True)

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
                    (Function(bs_psi, baseline["q"].values),            r"astra",  r"$q [-]$", {"marker": "+"}),
                    (eq_profile.q,                                      r"fytok",  r"$q [-]$"),
                    # (eq_profile.dphi_dpsi,                                             r"$\frac{d\phi}{d\psi}$"),
                ],
                [
                    (Function(bs_psi, baseline["rho"].values),       r"astra", r"$\rho_{tor}[m]$",  {"marker": "+"}),
                    (eq_profile.rho_tor,                                      r"fytok",    r"$\rho_{tor}[m]$"),
                ],
                [
                    (Function(bs_psi, baseline["x"].values),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", {"marker": "+"}),
                    (eq_profile.rho_tor_norm,                        r"fytok"),
                ],

                [
                    (Function(bs_psi, baseline["shif"].values),
                     r"astra", "$\Delta$ shafranov \n shift $[m]$ ", {"marker": "+"}),
                    (eq_profile.geometric_axis.r-tok.equilibrium.vacuum_toroidal_field.r0,
                     r"fytok", "shafranov \n shift $\Delta [m]$ "),
                ],
                [
                    (Function(bs_psi, baseline["k"].values),         r"astra", r"$elongation[-]$", {"marker": "+"}),
                    (eq_profile.elongation,                                 r"fytok", r"$elongation[-]$"),
                ],
                [
                    (Function(bs_psi, baseline["Jtot"].values*1e6),   r"astra",
                     r"$j_{\parallel} [A\cdot m^{-2}]$", {"marker": "+"}),
                    (eq_profile.j_parallel,                                         r"fytok",     r"$j_{\parallel}$"),
                ],

                # [
                #     (eq_profile.geometric_axis.r,                                     r"$geometric_{axis.r}$"),
                #     (eq_profile.r_inboard,                                                   r"$r_{inboard}$"),
                #     (eq_profile.r_outboard,                                                 r"$r_{outboard}$"),
                # ],
                [
                    (4*(constants.pi**2)*tok.equilibrium.time_slice.vacuum_toroidal_field.r0*tok.equilibrium.time_slice.profiles_1d.rho_tor,
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho , dV/d\rho$"),
                    (tok.equilibrium.time_slice.profiles_1d.dvolume_drho_tor,   r"$V^{\prime}$", r"$dV/d\rho$"),
                ]
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
            x_axis=(eq_profile._coord.psi_norm,                                                r"$\psi/\psi_{bdry}$"),
            title="Equlibrium",
            grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/equilibrium.svg", transparent=True)

        rgrid = eq_profile._coord
        plot_profiles(
            [
                (rgrid.psi_norm,  r"$\bar{\psi}$", r"$[-]$"),
                (rgrid.rho_tor_norm,  r"$\bar{\rho}$", r"$[-]$"),
                (rgrid.psi,  r"$\psi$", r"$[Wb]$"),
                (rgrid.rho_tor,  r"$\rho$", r"$[m]$"),
                (eq_profile.dvolume_dpsi(rgrid.psi_norm),  r"$dV/d\psi$", r"$[m]$"),
                [
                    (4*(constants.pi**2)*rgrid.vacuum_toroidal_field.r0*rgrid.rho_tor,
                        r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho$"),

                    (rgrid.dvolume_drho_tor, r"$dV/d\rho_{tor}$", r"$[m^2]$")
                ],
                (rgrid.gm1,                                             r"$gm1=\left<\frac{1}{R^2}\right>$"),
                (rgrid.gm2,                    r"$gm2=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                (rgrid.gm3,                                r"$gm3=\left<\left|\nabla \rho\right|^2\right>$"),
                (rgrid.gm7,                                  r"$gm7=\left<\left|\nabla \rho\right|\right>$"),
                (rgrid.gm8,                                                         r"$gm8=\left<R\right>$"),

                (rgrid.dphi_dpsi,                                                  r"$\frac{d\phi}{d\psi}$"),
                (rgrid.drho_tor_dpsi,                                        r"$\frac{d\rho_{tor}}{d\psi}$"),
                # (rgrid.dvolume_drho_tor,                                              r"$\frac{dV}{d\rho}$"),
                (rgrid.dpsi_drho_tor,                                        r"$\frac{d\psi}{d\rho_{tor}}$"),
            ],
            x_axis=(rgrid.rho_tor_norm,      r"$\bar{\rho_{tor}}$"),
            title="Equlibrium",
            grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/equilibrium_coord.svg", transparent=True)

    ###################################################################################################
    if True:  # CoreProfile
        s_range = -1  # slice(0, 140, 1)
        b_Te = Function(bs_r_nrom, smooth(baseline["TE"].values*1000, s_range))
        b_Ti = Function(bs_r_nrom, smooth(baseline["TI"].values*1000, s_range))

        b_ne = Function(bs_r_nrom, baseline["NE"].values*1.0e19)
        b_nHe = Function(bs_r_nrom, baseline["Nalf"].values*1.0e19)
        # nDT = Function(bs_r_nrom, baseline["Nd+t"].values*1.0e19)
        b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

        # Zeff = Function(bs_r_nrom, baseline["Zeff"].values)

        tok.core_profiles["profiles_1d"] = {
            "electrons": {**atoms["e"], "density": 0.1*b_ne,  "temperature": 0.1*b_Te, },
            "ion": [
                {**atoms["D"],  "density":   0.5*b_nDT, "temperature": b_Ti, },
                {**atoms["T"],  "density":   0.5*b_nDT, "temperature": b_Ti, },
                {**atoms["He"], "density":       b_nHe, "temperature": b_Ti, },
                {**atoms["Be"], "density":   0.02*b_ne, "temperature": b_Ti, },
                {**atoms["Ar"], "density": 0.0012*b_ne, "temperature": b_Ti, },
            ]}

        tok.core_profiles.update()

        core_profile = tok.core_profiles.profiles_1d

        plot_profiles(
            [
                [
                    (b_ne, "astra"    r"$n_{e} [m \cdot s^{-3}]$"),
                    (core_profile.electrons.density,             r"$e$", r"n $[  m \cdot s^{-3}]$"),
                    *[(ion.density,                            f"${ion.label}$") for ion in core_profile.ion],

                ],
                [
                    (b_Te,    r"$T_e$astra"              r"$T_{e} [eV]$"),
                    (core_profile.electrons.temperature,       r"$e$", r"T $[eV]$"),
                    *[(ion.temperature,                      f"${ion.label}$") for ion in core_profile.ion],
                ],

                [
                    (Function(bs_r_nrom, baseline["Zeff"].values),                r"$^{astra}$", r"$Z_{eff}  [-]$"),
                    (core_profile.zeff,                                                                 r"$fytok$"),
                ],
                # (core_profile.e_field.parallel,                    r"fytok",   r"$E_{\parallel} [V\cdot m^{-1}]$ "),


                # (core_profile.grid.psi,                                                                  r"$\psi$"),
                # (core_profile.electrons.pressure,                                                        r"$p_e $"),
                # (core_profile.electrons.density,                                                         r"$n_e $"),
                # (core_profile.electrons.temperature,                                                     r"$T_e $"),
                # (core_profile.electrons.pressure.derivative(),                                   r"$p_e^{\prime}$"),
                # (core_profile.electrons.density.derivative(),                                    r"$n_e^{\prime}$"),
                # (core_profile.electrons.temperature.derivative(),                                r"$T_e^{\prime}$"),

            ],
            x_axis=(core_profile.grid.rho_tor_norm,                                   r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_profile.svg", transparent=True)

    ###################################################################################################
    if True:  # CoreTransport
        rho_tor_norm = tok.equilibrium.time_slice.profiles_1d.rho_tor_norm
        R0 = tok.equilibrium.vacuum_toroidal_field.r0

        r_ped = 0.96  # np.sqrt(0.88)
        Cped = 0.2
        Ccore = 0.4
        chi = Function(
            [lambda r:r < r_ped, lambda r:r >= r_ped],
            [lambda x: Ccore*(1.0 + 3*(x**2)), lambda x: Cped])
        chi_e = Function(
            [lambda r:r < r_ped, lambda r:r >= r_ped],
            [lambda x:0.5*chi(x), lambda x: chi(x)])

        conductivity_parallel = Function(bs_r_nrom, baseline["Joh"].values*1.0e6 / baseline["U"].values *
                                         (2.0*constants.pi * R0))

        # D = Function(
        #     [lambda r:r < r_ped, lambda r:r >= r_ped],
        #     [lambda x:  0.5+(x**4), lambda x: 0.11])
        D = 0.1*chi_e
        v_pinch = Function(None, lambda x: 0.4 * D(x) * x / R0)  # FIXME: The coefficient 0.4 is a guess.

        tok.core_transport["model"] = [
            {"code": {"name": "dummy"},
                "profiles_1d": {
                    "conductivity_parallel": conductivity_parallel,
                    "electrons": {
                        **atoms["e"],
                        "particles":   {"d": D, "v": -v_pinch},
                        "energy":      {"d": chi_e, "v": -v_pinch},
                    },
                    "ion": [
                        {
                            **atoms["D"],
                            "particles":{"d":  D, "v": v_pinch},
                            "energy": {"d": chi, "v": 0},
                        },
                        {
                            **atoms["T"],
                            "particles":{"d":  D, "v": v_pinch},
                            "energy": {"d": chi, "v": 0},
                        },
                        {
                            **atoms["He"],
                            "particles":{"d": 0.1 * (chi+chi_e), "v": 0},
                            "energy": {"d": 0.5*chi, "v": 0}, }
                    ]}
             },

            # {"code": {"name": "neoclassical"}},
            # {"code": {"name": "spitzer"}},
        ]

        # logger.debug(tok.core_transport.model[0].profiles_1d.electrons.particles.v(rho_tor_norm))

        tok.core_transport.update()
    if False:
        core_transport1d_nc = tok.core_transport.model[{"code.name": "neoclassical"}].profiles_1d
        core_transport1d_dummy = tok.core_transport.model[{"code.name": "dummy"}].profiles_1d
        core_transport1d = tok.core_transport.model.combine.profiles_1d

        plot_profiles(
            [
                [
                    (core_transport1d.electrons.particles.flux,                            "e",  r"$\Gamma$"),
                    *[(core_transport1d.ion[{"label": ion.label}].particles.flux,
                       f"{ion.label}", f"$\Gamma$") for ion in core_profile.ion],
                ],
                [
                    (core_transport1d.electrons.particles.d,                                  "e",     r"$D$"),
                    *[(core_transport1d.ion[{"label": ion.label}].particles.d,     f"{ion.label}", r"$D$")
                      for ion in core_profile.ion],
                ],
                [
                    (core_transport1d.electrons.particles.v,                        "e",              r"$v$"),
                    *[(core_transport1d.ion[{"label": ion.label}].particles.v, f"{ion.label}",  f"$v$")
                      for ion in core_profile.ion],
                ],


                [
                    (Function(bs_r_nrom, baseline["Xi"].values),          r"astra", r"$\chi_{i}$", {"marker": "+"}),
                    *[(core_transport1d_dummy.ion[{"label": ion.label}].energy.d,
                       f"{ion.label}", r"$\chi_{i}$") for ion in core_profile.ion],
                ],

                [
                    (Function(bs_r_nrom,  np.log(baseline["XiNC"].values)),
                     "astra", r"$ln \chi_{i,nc}$", {"marker": "+"}),
                    * [(np.log(core_transport1d_nc.ion[{"label": label}].energy.d),   f"${label}$", r"$ln \chi_{i,nc}$")
                        for label in ("H", "D", "He")],
                ],

                [
                    (core_transport1d.electrons.energy.flux,                             "e",         r"$q$"),
                    *[(core_transport1d.ion[{"label": ion.label}].energy.flux,
                       f"{ion.label}") for ion in core_profile.ion],
                ],
                [
                    (core_transport1d.electrons.energy.v,           r"$electron$", r"V pinch $[m\cdot s^{-1}]$"),
                    *[(core_transport1d.ion[{"label": ion.label}].energy.v,  f"${ion.label}$",)
                      for ion in core_profile.ion],
                ],
                [
                    (Function(bs_r_nrom, baseline["Zeff"].values),     r"astra", r"$Z_{eff}[-]$", {"marker": "+"}),
                    (core_profile.zeff,                                r"fytok", r"$Z_{eff}[-]$"),
                ],
                [
                    (Function(bs_r_nrom, baseline["Joh"].values*1.0e6 / baseline["U"].values *
                              (2.0*constants.pi * tok.equilibrium.vacuum_toroidal_field.r0)),     r"astra", r"$\sigma_{\parallel}$", {"marker": "+"}),
                    (tok.core_transport.model[{"code.name": "spitzer"}].profiles_1d.conductivity_parallel,
                     "spitzer", r"$\sigma_{\parallel}$"),
                    (core_transport1d.conductivity_parallel,  r"fytok"),

                ],

                # (core_transport1d.e_field_radial,                                             r"$E_{radial}$"),
                # (tok.equilibrium.time_slice[-1].profiles_1d.trapped_fraction(
                # core_transport.model[0].profiles_1d[-1].grid_v.psi_norm),      r"trapped"),
                # (core_profile.electrons.pressure,                                                  r"$p_{e}$"),

            ],
            x_axis=(rho_tor_norm, r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            # index_slice=slice(10, 110, 1),

            title=tok.core_transport.model[0].identifier.name,
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_transport.svg", transparent=True)

    if True:  # CoreSources

        S0 = 5e20

        ne_src = Function(rho_tor_norm, lambda x, _S0=S0: _S0 * np.exp(15.0*(x**2-1.0)))

        tok.core_sources["source"] = [
            {"code": {"name": "dummy"},
                "profiles_1d": {
                    "j_parallel": Function(bs_r_nrom, baseline["Jtot"].values*1e6),
                    "electrons":{
                        "particles": ne_src,
                        "energy": Function(bs_r_nrom,
                                           (baseline["Poh"].values
                                            + baseline["Paux"].values
                                            - baseline["Peic"].values
                                            - baseline["Prad"].values
                                            )*1000
                                           )},
                    "ion":[]
            }
            },
            # {"code": {"name": "q_ei"}, }
            # {"code": {"name": "bootstrap_current"}},
        ]
        # tok.core_sources.source[{"code.name": "dummy"}].profiles_1d["j_parallel"] = \
        #     Function(bs_r_nrom, baseline["Jtot"].values)
        tok.core_sources.update()
    if False:
        core_source_1d = tok.core_sources.source.combine.profiles_1d

        plot_profiles(
            [
                [
                    (Function(bs_r_nrom, baseline["Jtot"].values),  "astra", r"$J_{\parallel} [MA\cdot m^{-2}]$"),
                    (tok.core_sources.source[{"code.name": "dummy"}].profiles_1d.j_parallel,
                     "dummy", r"$J_{\parallel} [MA\cdot m^{-2}]$"),
                    (core_source_1d.j_parallel,                     "fytok", r"$J_{\parallel} [MA\cdot m^{-2}]$"),
                ],
                # [
                #     (Function(bs_r_nrom, baseline["Zeff"].values),          r"$Z_{eff}^{astra}$", {"marker": "+"}),
                #     (core_profile.zeff,                                                              r"$z_{eff}$"),
                # ],
                # [
                #     (Function(bs_r_nrom, baseline["Joh"].values*1.0e6 / baseline["U"].values *
                #               (2.0*constants.pi * tok.equilibrium.vacuum_toroidal_field.r0)),     r"$\sigma_{\parallel}^{astra}$", {"marker": "+"}),
                #     (core_source_1d.conductivity_parallel,
                #      r"$\sigma_{\parallel}^{wesson}$"),
                # ],
                # [
                #     (Function(bs_r_nrom, baseline["Joh"].values), "astra",    r"$j_{ohmic} [MA\cdot m^{-2}]$"),
                #     # (core_profile.j_ohmic,                        "fytok",    r"$j_{ohmic} [MA\cdot m^{-2}]$"),
                # ],
                # [
                #     (Function(bs_r_nrom, baseline["Jbs"].values),
                #      r"astra", r"$j_{bootstrap} [MA\cdot m^{-2}]$", {"marker": "+"}),
                #     (core_source_1d.j_parallel*1.0e-6,                                          r"wesson"),

                #     (tok.core_sources.source.combine.profiles_1d.j_parallel*1.0e-6,             r"fytok"),
                # ],
                # (tok.core_profiles.profiles_1d.electrons.density,                                       r"$ n_e $"),
                # (tok.core_profiles.profiles_1d.electrons.temperature,                                   r"$ T_e $"),
                # (tok.core_profiles.profiles_1d.electrons.temperature.dln(),          r"$\frac{T_e^{\prime}}{T_e}$"),
                # (tok.core_profiles.profiles_1d.electrons.density.dln(),              r"$\frac{n_e^{\prime}}{n_e}$"),
                # (tok.core_profiles.profiles_1d.electrons.pressure.dln(),             r"$\frac{p_e^{\prime}}{p_e}$"),
                # (tok.core_profiles.profiles_1d.electrons.tau,                                          r"$\tau_e$"),
                # (tok.core_profiles.profiles_1d.electrons.vT,                                          r"$v_{T,e}$"),

                # (core_source_1d.e_field_radial,                                             r"$E_{radial}$"),
                # (tok.equilibrium.time_slice[-1].profiles_1d.trapped_fraction(
                # core_transport.model[0].profiles_1d[-1].grid_v.psi_norm),      r"trapped"),
                # (core_profile.electrons.pressure,                                                  r"$p_{e}$"),

            ],
            x_axis=(core_source_1d.grid.rho_tor_norm, r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            # annotation=core_transport.model[0].identifier.name,
            # index_slice=slice(1, 110, 1),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_sources.svg", transparent=True)
    ###################################################################################################

    if True:  # TransportSolver
        tok.transport_solver["boundary_conditions_1d"] = {
            "current": {"identifier": {"index": 1}, "value": [tok.equilibrium.time_slice.global_quantities.psi_boundary]},
            "electrons": {"particles": {"identifier": {"index": 1}, "value": [b_ne[-1]]},
                          "energy": {"identifier": {"index": 1}, "value": [b_Te[-1]]}},
            "ion": [
                {**atoms["D"],
                 "particles": {"identifier": {"index": 1}, "value": [b_nDT[-1]]},
                 "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}},
                {**atoms["T"],
                    "particles": {"identifier": {"index": 1}, "value": [b_nDT[-1]]},
                    "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}},
                {**atoms["He"],
                    "particles": {"identifier": {"index": 1}, "value": [b_nHe[-1]]},
                    "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}}
            ]
        }

        tok.transport_solver.update(equilibrium=tok.equilibrium,
                                    core_profiles=tok.core_profiles,
                                    core_transport=tok.core_transport,
                                    core_sources=tok.core_sources)

        # psi0 = tok.core_profiles.profiles_1d['psi']
        # tok.core_profiles.profiles_1d["psi"] = np.zeros(tok.core_profiles.grid.rho_tor_norm.shape)
        # tok.core_profiles.profiles_1d["dpsi_drho_tor_norm"] = np.zeros(tok.core_profiles.grid.rho_tor_norm.shape)

        tok.transport_solver.solve(impurities=["D", "T", "He", "Ar", "Be"])

        core_profile = tok.core_profiles.profiles_1d

        b_psi = Function(bs_r_nrom, (baseline["Fp"].values * (tok.equilibrium.time_slice.global_quantities.psi_boundary-tok.equilibrium.time_slice.global_quantities.psi_axis)
                                     + tok.equilibrium.time_slice.global_quantities.psi_axis))

        plot_profiles(
            [

                ######################################################################
                # psi ,current
                [
                    (b_psi, r"astra", r"$\psi [Wb]$", {"marker": "+"}),
                    (core_profile["psi"],  r"fytok", r"$\psi  [Wb]$"),
                    (b_psi-core_profile["psi"], r"residual",
                     r"$n_e [10^{19} m^{-3}]$",  {"color": "red", "linestyle": "dashed"}),
                ],
                # [
                #     (Function(bs_r_nrom, baseline["Fp"].values), r"astra", r"$\psi/\psi_{bdry}  [-]$", {"marker": "+"}),
                #     ((core_profile["psi"]-core_profile["psi"][0])/(core_profile["psi"][-1]-core_profile["psi"][0]), r"fytok", r"$\psi  [-]$"),
                # ],
                # [
                #     (Function(bs_r_nrom, baseline["Jtot"].values), "astra",   r"$J_{\parallel} [MA]$", {"marker": "+"}),
                #     (tok.core_sources.source.combine.profiles_1d.j_parallel/1e6,    "fytok",   r"$J_{\parallel} [MA]$"),
                # ],
                # [
                #     (conductivity_parallel, "astra", r"$\sigma_{\parallel}$", {"marker": "+"}),
                #     (tok.core_transport.model.combine.profiles_1d.conductivity_parallel,   "fytok"),
                # ],
                # (core_profile["sol.current.fpol"],  r"sol.current.fpol", r"$fpol$"),
                # (core_profile["sol.current.gm2"],  r"sol.current.gm2", r"$gm2$"),
                ######################################################################
                # electron particles
                [
                    (b_ne, r"astra", r"$n_e [m^{-3}]$",  {"marker": "+"}),
                    (core_profile.electrons.density, r"fytok", r"$n_e [ m^{-3}]$"),
                    (b_ne-core_profile.electrons.density, r"residual",
                     r"$n_e [10^{19} m^{-3}]$",  {"color": "red", "linestyle": "dashed"}),
                    # *[(ion.density*1e-19,                            f"${ion.label}$") for ion in core_profile.ion],
                ],
                # [
                #     (core_profile.electrons["diff"],  r"D"),
                #     (np.abs(core_profile.electrons["conv"]),  r"$\left|V\right|$"),
                # ],

                # [
                #     (core_profile.electrons["s_exp_flux"],   r"Source", r"[10^{23} s^{-1}]", {"color": "green", }),
                #     (core_profile.electrons["diff_flux"],    r"Diffusive flux",  "", {"color": "black", }),
                #     (core_profile.electrons["conv_flux"],    r"Convective flux", "",  {"color": "red", }),
                #     (core_profile.electrons["residual"],     r"Residual",  "", {"color": "blue", }),
                # ],
                ######################################################################
                # electron energy
                # [
                (b_Te, r"astra", r"$T_e [eV]$",  {"marker": "+"}),
                (core_profile.electrons.temperature, r"fytok", r"$T_e [eV]$"),
                (b_Te-core_profile.electrons.temperature, r"residual",
                 r"$[eV]$",  {"color": "red", "linestyle": "dashed"}),
                # ],

                ######################################################################

                # [
                #     (4*(constants.pi**2)*R0 * core_profile.grid.rho_tor, r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho$"),
                #     (core_profile.electrons["vpr"],  r"vpr"),
                # ],
                # (core_profile.electrons["a"],  r"electron.a", r"$a$"),
                # (core_profile.electrons["b"],  r"electron.b", r"$b$"),
                # (core_profile.electrons["c"],  r"electron.c", r"$c$"),
                # (core_profile.electrons["d"],  r"electron.d", r"$d$"),
                # (core_profile.electrons["e"],  r"electron.e", r"$e$"),
                # (core_profile.electrons["f"],  r"electron.f", r"$f$"),
                # (core_profile.e_field.parallel,                    r"fytok",   r"$E_{\parallel} [V\cdot m^{-1}]$ "),
            ],
            x_axis=(core_profile.grid.rho_tor_norm,                                   r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title="Result of TransportSolver",
            # index_slice=slice(0, 100, 1),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_profile_result.svg", transparent=True)

    logger.info("====== DONE ========")
