
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import constants
# from spdm.numlib.smooth import rms_residual
from spdm.data.File import File
from spdm.data.Function import function_like
from spdm.utils.logger import logger
from fytok.utils.plot_profiles import plot_profiles, sp_figure

from fytok.utils.load_profiles import (load_core_profiles, load_core_source,
                                       load_core_transport, load_equilibrium)
from fytok.Tokamak import Tokamak

###################


if __name__ == "__main__":
    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    ###################################################################################################
    # baseline
    device_desc = File("/home/salmon/workspace/fytok_data/mapping/ITER/imas/3/static/config.xml", format="XML").read()

    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="GEQdsk").read()
    # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
    # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
    # profiles = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')

    R0 = eqdsk_file.get("vacuum_toroidal_field/r0")
    B0 = eqdsk_file.get("vacuum_toroidal_field/b0")

    psi_axis = eqdsk_file.get("time_slice/0/global_quantities/psi_axis")
    psi_boundary = eqdsk_file.get("time_slice/0/global_quantities/psi_boundary")

    bs_eq_psi = eqdsk_file.get("time_slice/0/profiles_1d/psi")
    bs_eq_psi_norm = (bs_eq_psi-psi_axis)/(psi_boundary-psi_axis)
    bs_eq_fpol = function_like(bs_eq_psi_norm, eqdsk_file.get("time_slice/0/profiles_1d/f"))

    profiles = pd.read_excel('/home/salmon/workspace/data/15MA inductive - burn/15MA Inductive at burn-ASTRA.xls',
                             sheet_name='15MA plasma', header=10, usecols="B:BN")

    bs_r_norm = profiles["x"].values

    b_Te = function_like(profiles["TE"].values*1000, bs_r_norm)
    b_Ti = function_like(profiles["TI"].values*1000, bs_r_norm)
    b_ne = function_like(profiles["NE"].values*1.0e19, bs_r_norm)

    b_nHe = function_like(profiles["Nalf"].values * 1.0e19, bs_r_norm)
    b_ni = function_like(profiles["Nd+t"].values * 1.0e19*0.5, bs_r_norm)
    b_nImp = function_like(profiles["Nz"].values * 1.0e19, bs_r_norm)
    b_zeff = function_like(profiles["Zeff"].values, bs_r_norm)
    bs_psi_norm = profiles["Fp"].values
    bs_psi = bs_psi_norm*(psi_boundary-psi_axis)+psi_axis

    bs_r_norm = profiles["x"].values
    bs_line_style = {"marker": '.', "linestyle": ''}
    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    time_slice = -1
    ###################################################################################################
    # Initialize Tokamak

    tok = Tokamak(device_desc[{"wall", "pf_active", "tf", "magnetics"}])

    # Equilibrium

    tok["equilibrium"] = {**load_equilibrium(eqdsk_file),
                          "code": {"name": "eq_analyze",
                                   "parameters": {
                                       "boundary": {"psi_norm": 0.995},
                                       "coordinate_system": {"psi_norm": np.linspace(0.001, 0.995, 64), "theta": 64}}
                                   }
                          }

    if True:
        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},
                        "vessel": {"edgecolor": "blue"}},
                  pf_active={"facecolor": 'red'},
                  equilibrium={
                      "contours": 16,
                      "boundary": True,
                      "separatrix": True,
                  }
                  ) .savefig(output_path/"tokamak.svg", transparent=True)

    if True:  # plot tokamak
        eq_profiles_1d = tok.equilibrium.time_slice[time_slice].profiles_1d

        plot_profiles(
            [
                [
                    (bs_eq_fpol, "astra", r"$F_{pol} [Wb\cdot m]$", bs_line_style),
                    (eq_profiles_1d.f,  r"fytok", r"$[Wb]$"),
                ],

                [
                    (function_like(profiles["q"].values, bs_psi_norm), r"astra", r"$q [-]$", bs_line_style),
                    # (function_like(eqdsk.get('profiles_1d.psi_norm'), eqdsk.get('profiles_1d.q')), "eqdsk"),
                    (eq_profiles_1d.q,  r"$fytok$", r"$[Wb]$"),
                    # (magnetic_surface.dphi_dpsi,  r"$\frac{d\phi}{d\psi}$", r"$[Wb]$"),
                ],
                [
                    (function_like(profiles["rho"].values, bs_psi_norm), r"astra", r"$\rho_{tor}[m]$",  bs_line_style),
                    (eq_profiles_1d.rho_tor,  r"$\rho$", r"$[m]$"),
                ],
                [
                    (function_like(profiles["x"].values, bs_psi_norm),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", bs_line_style),
                    (eq_profiles_1d.rho_tor_norm,
                     r"$\bar{\rho}$", r"$[-]$"),
                ],

                [
                    (function_like(4*(constants.pi**2) * R0 * profiles["rho"].values, bs_psi_norm),
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho$",  bs_line_style),
                    (eq_profiles_1d.dvolume_drho_tor,
                     r"$dV/d\rho_{tor}$", r"$[m^2]$"),
                ],

                # (magnetic_surface.dvolume_dpsi, r"$\frac{dV}{d\psi}$"),

                # [
                #     (magnetic_surface.volume, r"$V$  from $\psi$"),
                #     # (magnetic_surface.volume1, r"$V$ from $\rho_{tor}$"),
                # ],



                # (magnetic_surface.psi,  r"$\psi$", r"$[Wb]$"),
                # (magnetic_surface.phi,  r"$\phi$", r"$[Wb]$"),
                # (magnetic_surface.psi_norm,  r"$\bar{\psi}$", r"$[-]$"),


                # [
                #     (magnetic_surface.dpsi_drho_tor,
                #      r"$\frac{d\psi}{d\rho_{tor}}$", "", {"marker": '.'}),
                #     (magnetic_surface.dvolume_drho_tor/magnetic_surface.dvolume_dpsi,
                #      r"$\frac{dV}{d\rho_{tor}} / \frac{dV}{d\psi}$")
                # ],
                # (magnetic_surface.drho_tor_dpsi*magnetic_surface.dpsi_drho_tor,
                #  r"$\frac{d\rho_{tor}}{d\psi} \cdot \frac{d\psi}{d\rho_{tor}}$"),
                # (magnetic_surface.gm2_,
                #  r"$gm2_=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                # (magnetic_surface.dpsi_drho_tor, r"$\frac{d\rho_{tor}}{d\psi}$", "", {"marker": "."}),


                (eq_profiles_1d.gm1, r"$gm1=\left<\frac{1}{R^2}\right>$"),
                (eq_profiles_1d.gm2, r"$gm2=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                (eq_profiles_1d.gm3, r"$gm3=\left<\left|\nabla \rho\right|^2\right>$"),
                (eq_profiles_1d.gm7, r"$gm7=\left<\left|\nabla \rho\right|\right>$"),
                (eq_profiles_1d.gm8, r"$gm8=\left<R\right>$"),

                # (magnetic_surface.dphi_dpsi,                                                  r"$\frac{d\phi}{d\psi}$"),
                # (magnetic_surface.dpsi_drho_tor,                                        r"$\frac{d\psi}{d\rho_{tor}}$"),
            ],
            # x_axis=(magnetic_surface.rho_tor_norm,      r"$\bar{\rho}_{tor}$"),
            x_axis=(eq_profiles_1d.psi_norm,      r"$\bar{\psi}$"),
            title="Equilibrium",
            grid=True, fontsize=16) .savefig(output_path/"equilibrium_coord.svg", transparent=True)

        plot_profiles(
            [

                [
                    (function_like(profiles["q"].values, bs_psi_norm), r"astra",  r"$q [-]$", bs_line_style),
                    (eq_profiles_1d.q, r"fytok",  r"$q [-]$"),
                    (eq_profiles_1d.dphi_dpsi*np.sign(B0)/constants.pi/2.0,
                     r"$\frac{\sigma_{B_{p}}}{\left(2\pi\right)^{1-e_{B_{p}}}}\frac{d\Phi_{tor}}{d\psi_{ref}}$"),
                ],
                [
                    (function_like(profiles["rho"].values, bs_psi_norm), r"astra",
                     r"$\rho_{tor}[m]$",  bs_line_style),
                    (eq_profiles_1d.rho_tor,
                     r"fytok",    r"$\rho_{tor}[m]$"),
                ],
                [
                    (function_like(profiles["x"].values, bs_psi_norm),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", bs_line_style),
                    (eq_profiles_1d.rho_tor_norm,                        r"fytok"),
                ],

                [
                    (function_like(profiles["shif"].values, bs_psi_norm),
                     r"astra", "$\Delta$ shafranov \n shift $[m]$ ", bs_line_style),
                    (eq_profiles_1d.geometric_axis.r - R0,
                     r"fytok", "shafranov \n shift $\Delta [m]$ "),
                ],
                [
                    (function_like(profiles["k"].values, bs_psi_norm), r"astra", r"$elongation[-]$", bs_line_style),
                    (eq_profiles_1d.elongation,  r"fytok", r"$elongation[-]$"),
                ],
                [
                    (4*(constants.pi**2) * R0*eq_profiles_1d.rho_tor,
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho , dV/d\rho$"),
                    (eq_profiles_1d.dvolume_drho_tor, r"$V^{\prime}$", r"$dV/d\rho$"),
                ],
                # [
                #     (function_like(profiles["Jtot"].values*1e6),   r"astra",
                #      r"$j_{\parallel} [A\cdot m^{-2}]$", bs_line_style),
                #     (eq_profile.j_parallel,                                 r"fytok",     r"$j_{\parallel}$"),
                # ],

                # [
                #     (eq_profile.geometric_axis.r,                                     r"$geometric_{axis.r}$"),
                #     (eq_profile.r_inboard,                                                   r"$r_{inboard}$"),
                #     (eq_profile.r_outboard,                                                 r"$r_{outboard}$"),
                # ],

                # [
                #     (eq_profile.volume,                r"$V$"),
                #     # (function_like(eq_profile.rho_tor, eq_profile.dvolume_drho_tor).antiderivative,
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
            x_axis=(eq_profiles_1d.psi_norm,      r"$\psi/\psi_{bdry}$"),
            # x_axis=([0, 1.0],                                                r"$\psi/\psi_{bdry}$"),

            title="Equilibrium",
            grid=True, fontsize=16) .savefig(output_path/"equilibrium_profiles.svg", transparent=True)

        logger.info("Initialize Equilibrium ")

    if True:  # CoreProfile initialize value

        tok.core_profiles["profiles_1d"] = load_core_profiles(profiles, grid=tok.equilibrium.radial_grid)

        core_profile_1d = tok.core_profiles.profiles_1d[time_slice]

        plot_profiles(
            [
                [
                    (b_ne, "electron astra", r"density $n [m \cdot s^{-3}]$", bs_line_style),
                    (b_ni, "D astra", r"density $n [m \cdot s^{-3}]$", bs_line_style),
                    (b_nHe, "He astra", r"density $n [m \cdot s^{-3}]$", bs_line_style),
                    (core_profile_1d.electrons.density,    r"$electron$", ),
                    *[(core_profile_1d.ion[{"label": label}].density,  f"${label}$") for label in ['H', 'D', 'He']],

                ],
                [
                    (b_Te,    r"astra $T_e$",       r"$T [eV]$", bs_line_style),
                    (b_Ti,    r"astra $T_i$",       r"$T [eV]$", bs_line_style),
                    (core_profile_1d.electrons.temperature,  r"$e$", r"T $[eV]$"),
                    *[(core_profile_1d.ion[{"label": label}].temperature,   f"${label}$")
                      for label in ['H', 'D', 'He']],
                ],

                # [
                #     (function_like( profiles["Zeff"].values, bs_r_norm),       r"astra",
                #      r"$Z_{eff}  [-]$", bs_line_style),
                #     (core_profile_1d.zeff, r"$fytok$"),
                # ],
                [
                    (function_like(bs_eq_psi, bs_eq_psi_norm)(core_profile_1d.grid.psi_norm), "astra",      r"$\psi$"),
                    (core_profile_1d.grid.psi,              r"fytok"),
                ]
            ],
            x_axis=([0, 1.0], r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig(output_path/"core_profiles_initialize.svg", transparent=True)

        logger.info("Initialize Core Profiles ")

    if True:  # CoreTransport
        tok.core_transport["model"] = [
            {"code": {"name": "dummy"},
             "profiles_1d": load_core_transport(profiles, tok.core_profiles.profiles_1d.grid)},
            {"code": {"name": "fast_alpha"}},
            {"code": {"name": "spitzer"}},
            # {"code": {"name": "neoclassical"}},
            # {"code": {"name": "glf23"}},
            # {"code": {"name": "nclass"}},

        ]

        # logger.debug(tok.core_transport["model"].dump())

        tok.core_transport.refresh(equilibrium=tok.equilibrium, core_profiles=tok.core_profiles)

        core_transport_model = tok.core_transport.model_combiner

        core_transport_profiles_1d = core_transport_model.profiles_1d[time_slice]

        # logger.debug([[sp.energy.d for sp in model.profiles_1d.ion] for model in tok.core_transport.model])
        # logger.debug(energy.d)
        # logger.debug(core_transport_profiles_1d.electrons.energy.d(np.linspace(0, 1.0, 128)))
        # nc_profiles_1d = tok.core_transport.model[{"code.name": "neoclassical"}].profiles_1d
        # fast_alpha_profiles_1d = tok.core_transport.model[{"code.name": "fast_alpha"}].profiles_1d

        plot_profiles(
            [
                [
                    (function_like(profiles["Xi"].values, bs_r_norm),
                     r"astra", r"$\chi_{i}$", bs_line_style),
                    *[(core_transport_profiles_1d.ion[{"label": label}].energy.d, f"{label}", r"$\chi_{i}$")
                      for label in ['D', 'T', 'He']],
                ],
                [
                    (function_like(profiles["He"].values, bs_r_norm),
                     "astra", r"$\chi_{e}$", bs_line_style),
                    (core_transport_profiles_1d.electrons.energy.d,   "fytok", r"$\chi_{e}$"),
                ],
                [
                    (function_like(profiles["Joh"].values*1.0e6 / profiles["U"].values * (2.0*constants.pi * R0)),
                     r"astra", r"$\sigma_{\parallel}$", bs_line_style),

                    (core_transport_profiles_1d.conductivity_parallel,  r"fytok", r"$\sigma_{\parallel}$"),
                ],

                # [(ion.particles.d_fast_factor, f"{ion.label}", r"$D_{\alpha}/D_{He}$")
                #  for ion in fast_alpha_profiles_1d.ion],


                # [
                #     (function_like( np.log(profiles["XiNC"].values, bs_r_norm)),
                #      "astra", r"$ln \chi_{i,nc}$", bs_line_style),
                #     # * [(np.log(core_transport1d_nc.ion[{"label": label}].energy.d),   f"${label}$", r"$ln \chi_{i,nc}$")
                #     #     for label in ("H", "D", "He")],
                # ],
                # [
                #     (function_like(profiles["XiNC"].values, bs_r_norm), "astra",
                #      "neoclassical  $\\chi_{NC}$ \n ion heat conductivity", bs_line_style),

                #     # *[(ion.energy.d,  f"{ion.label}", r"Neoclassical $\chi_{NC}$")
                #     #   for ion in nc_profiles_1d.ion if not ion.is_impurity],
                # ],
                # [(ion.particles.d,  f"{ion.label}", r"Neoclassical $D_{NC}$")
                #  for ion in nc_profiles_1d.ion if not ion.is_impurity],

            ],
            x_axis=([0, 1.0],   r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title="combine",
            grid=True, fontsize=10) .savefig(output_path/"core_transport.svg", transparent=True)

        logger.info("Initialize Core Transport ")

    if True:  # CoreSources
        tok.core_sources["source"] = [
            {"code": {"name": "dummy"},
             "profiles_1d": load_core_source(profiles, tok.core_profiles.profiles_1d.grid)},
            {"code": {"name": "bootstrap_current"}},
            {"code": {"name": "fusion_reaction"}},
        ]

        tok.core_sources.refresh(equilibrium=tok.equilibrium,
                                 core_profiles=tok.core_profiles)

        core_source_profiles_1d = tok.core_sources.source_combiner.profiles_1d[time_slice]

        plot_profiles(
            [
                [
                    (function_like(profiles["Jtot"].values, bs_r_norm),  "astra",
                     "$J_{total}=j_{bootstrap}+j_{\\Omega}$ \n $[MA\\cdot m^{-2}]$", bs_line_style),
                    (core_source_profiles_1d.j_parallel*1e-6,     "fytok", ""),
                ],

                [
                    (function_like(profiles["Joh"].values, bs_r_norm), "astra",
                     r"$j_{ohmic} [MA\cdot m^{-2}]$", bs_line_style),
                    (core_profile_1d.j_ohmic*1e-6, "fytok", r"$j_{\Omega} [MA\cdot m^{-2}]$"),
                ],
                [
                    (core_source_profiles_1d.electrons.particles,             "e",   r"$S[m ^ {-3} s ^ {-1}]$"),
                    (core_source_profiles_1d.ion[{"label": "D"}].particles,   "D",   r"$S[m ^ {-3} s ^ {-1}]$"),
                    (core_source_profiles_1d.ion[{"label": "T"}].particles,   "T",   r"$S[m ^ {-3} s ^ {-1}]$"),
                    (core_source_profiles_1d.ion[{"label": "He"}].particles,  "He",  r"$S[m ^ {-3} s ^ {-1}]$"),
                ],
                [
                    (core_source_profiles_1d.electrons.energy,             "e",   r"$Q$"),
                    (core_source_profiles_1d.ion[{"label": "D"}].energy,   "D",   r"$Q$"),
                    (core_source_profiles_1d.ion[{"label": "T"}].energy,   "T",   r"$Q$"),
                    (core_source_profiles_1d.ion[{"label": "He"}].energy,  "He",  r"$Q$"),
                ],
                # [
                # (function_like(profiles["Jbs"].values, bs_r_norm),
                #  r"astra", "bootstrap current \n $[MA\\cdot m^{-2}]$", bs_line_style),
                # # (tok.core_sources.source[{"code/name": "bootstrap_current"}].profiles_1d.j_parallel*1e-6,
                #  r"fytok",                     ),
                # ],
                # [
                #     (rms_residual(function_like(profiles["Jbs"].values*1e6,bs_r_norm),
                #                   tok.core_sources.source[{"code.name": "bootstrap_current"}].profiles_1d.j_parallel),
                #      r"bootstrap current", r"  rms residual $[\%]$"),
                #     (rms_residual(function_like(profiles["Jtot"].values, bs_r_norm), core_source.j_parallel*1e-6),
                #      r"total current", r"  rms residual $[\%]$"),
                # ],
                # [
                #     (core_source_profiles_1d.ion[{"label": "D"}].particles,    r"D",  r"$S_{DT} [m^3 s^{-1}]$",),
                #     (core_source_profiles_1d.ion[{"label": "T"}].particles,    r"T",  r"$S_{DT} [m^3 s^{-1}]$",),
                #     (core_source_profiles_1d.ion[{"label": "He"}].particles_fast,
                #      r"fast", r"$S_{\alpha} [m^3 s^{-1}]$",),
                # ],
                # [
                #     (core_source_profiles_1d.ion[{"label": "He"}].particles,
                #      r"thermal", r"$S_{\alpha} [m^3 s^{-1}]$",),
                # ],
                # (core_source_profiles_1d.ion[{"label": "D"}].particles,           r"$D_{total}$", r"$S_{DT} [m^3 s^{-1}]$",),
                # [
                #     (core_source_profiles_1d.electrons.energy,  "electron",      r"$Q [eV\cdot m^{-3} s^{-1}]$"),
                #     # *[(ion.energy*1e-6,             f"{ion.label}",  r"$Q [eV\cdot m^{-3} s^{-1}]$")
                #     #   for ion in core_source_profiles_1d.ion if not ion.is_impurity],
                # ],
            ],
            x_axis=([0, 1.0], r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig(output_path/"core_sources.svg", transparent=True)

        logger.info("Initialize Core Source  ")

    ###################################################################################################
    # TransportSolver
    if True:

        tok["core_transport_solver"] = {
            "code": {
                "name": "bvp_solver_nonlinear",
                "parameters": {
                        "tolerance": 1.0e-4,
                        "particle_solver": "ion",
                        "max_nodes": 500,
                        "verbose": 2,
                        "bvp_rms_mask": [r_ped],
                }
            },
            "fusion_reaction": [r"D(t,n)\alpha"],
            "boundary_conditions_1d": {
                "current": {"identifier": {"index": 1}, "value": [psi_boundary]},
                "electrons": {"particles": {"identifier": {"index": 1}, "value": [b_ne[-1]]},
                              "energy": {"identifier": {"index": 1}, "value": [b_Te[-1]]}},

                "ion": [
                    {"label": "D",
                     "particles": {"identifier": {"index": 1}, "value": [b_ni[-1]]},
                     "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}},
                    {"label": "T",
                     "particles": {"identifier": {"index": 1}, "value": [b_ni[-1]]},
                     "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}},
                    {"label": "He",
                     "particles": {"identifier": {"index": 1}, "value": [b_nHe[-1]]},
                     "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]},
                     "particles_thermal": {"identifier": {"index": 1}, "value": [b_nHe[-1]]},
                     "particles_fast": {"identifier": {"index": 1}, "value": [0]},
                     }
                ]
            }}

        particle_solver = tok.core_transport_solver.code.parameters.get('particle_solver', 'ion')

        logger.info("Transport solver Start")

        tok.refresh()

        logger.info("Transport solver End")

        core_profile_1d = tok.core_profiles.profiles_1d

        b_nHe_fast = function_like(profiles["Naff"].values * 1.0e19, bs_r_norm)
        b_nHe_thermal = function_like(profiles["Nath"].values * 1.0e19, bs_r_norm)

        ionHe = core_profile_1d.ion[{"label": "He"}]

        plot_profiles(
            [
                # psi ,current
                [
                    (function_like(bs_psi, bs_r_norm),             r"astra", r"$\psi [Wb]$", bs_line_style),
                    (core_profile_1d["psi"],  r"fytok", r"$\psi  [Wb]$", {"marker": '+', "linestyle": '-'}),
                ],

                # (core_profile_1d["psi_flux"],  r"fytok", r"$\Gamma_{\psi}$", {"marker": '+', "linestyle": '-'}),

                # electron
                # [
                #     (b_ne/1.0e19, r"astra",  r"$n_e [10^{19} m^{-3}]$",  bs_line_style),
                #     (core_profile_1d.electrons.density/1.0e19,  r"fytok", r"$n_e [10^{19} m^{-3}]$"),
                # ],

                [
                    (b_Te/1000.0, r"astra",  r"$T_e [keV]$",  bs_line_style),
                    (core_profile_1d.electrons.temperature / 1000.0, r"fytok", r"$T_e [keV]$"),
                ],

                # ion
                [
                    (b_ni*1.0e-19,    r"$D_{astra}$",  r"$n_i  \, [10^{19} m^-3]$", bs_line_style),
                    (b_nHe*1.0e-19,   r"$He_{astra}$", r"$n_i  \,  [10^{19} m^-3]$", bs_line_style),
                    *[(core_profile_1d.ion[{"label": label}].density*1.0e-19,   f"${label}$", r"$n_i  \, [10^{19} m^-3]$")
                      for label in ['D', 'T', 'He']],
                ],


                [
                    (ionHe.density_thermal * 1.0e-19,  r"$n_{thermal}$", r"$n_{He}  \, [10^{19} m^-3]$"),
                    (ionHe.density_fast*1.0e-19,     r"$n_{\alpha}$", r"$n_{\alpha}  \, [10^{19} m^-3]$"),
                    (ionHe.density*1.0e-19,             r"$n_{He}$", r"$n_{He}  \, [10^{19} m^-3]$"),

                    (b_nHe*1.0e-19,   r"${astra}$", r"$n_{He}  \, [10^{19} m^-3]$", bs_line_style),
                ],

                [
                    (b_Ti/1000.0,    r"astra", r"$T_{i} \, [keV]$", bs_line_style),
                    * [(core_profile_1d.ion[{"label": label}].temperature/1000.0,  f"fytok ${label}$", r"$T_{i} [keV]$")
                        for label in ['D', 'T', 'He']],
                ],

                # ---------------------------------------------------------------------------------------------------

                # (core_profile_1d["rms_residuals"] * 100, r"bvp", r"residual $[\%]$"),

                # [
                #     # (rms_residual(function_like(bs_psi,bs_r_norm),
                #     #  core_profile_1d["psi"]), r"$\psi$", " rms residual [%]"),

                #     # (rms_residual(b_ne, core_profile_1d.electrons.density), r"$n_e$"),

                #     (rms_residual(b_Te, core_profile_1d.electrons.temperature), r"$T_e$", " rms residual [%]"),

                #     *[(rms_residual(b_ni, ion.density), f"$n_{ion.label}$")
                #       for ion in core_profile_1d.ion if not ion.is_impurity],
                #     # *[(rms_residual(b_Ti, ion.temperature), f"$T_{ion.label}$")
                #     #   for ion in core_profile_1d.ion if not ion.is_impurity],

                # ],
            ],
            x_axis=([0, 1.0],  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title=f" Particle solver '{particle_solver}'",
            grid=True, fontsize=10).savefig(output_path/f"core_profiles_result_{particle_solver}.svg", transparent=True)

        plot_profiles(
            [
                # psi ,current
                [
                    (function_like(bs_psi, bs_r_norm),            r"astra", r"$\psi [Wb]$", bs_line_style),
                    (core_profile_1d.grid.psi,  r"fytok", r"$\psi  [Wb]$", {"marker": '+', "linestyle": '-'}),
                ],

                # (core_profile_1d.grid.psi_flux,  r"fytok", r"$\Gamma_{\psi}$", {"marker": '+', "linestyle": '-'}),

                # electron
                [
                    (b_ne/1.0e19, r"astra",  r"$n_e [10^{19} m^{-3}]$",  bs_line_style),
                    (core_profile_1d.electrons.density/1.0e19,  r"fytok", r"$n_e [10^{19} m^{-3}]$"),
                ],
                [
                    (b_Te/1000.0, r"astra",  r"$T_e [keV]$",  bs_line_style),
                    (core_profile_1d.electrons.temperature / 1000.0, r"fytok", r"$T_e [keV]$"),
                ],

                # ion
                [
                    (b_ni*1.0e-19,    r"$D_{astra}$",  r"$n_i  \, [10^{19} m^-3]$", bs_line_style),
                    (b_nHe*1.0e-19,   r"$He_{astra}$", r"$n_i  \,  [10^{19} m^-3]$", bs_line_style),
                    *[(core_profile_1d.ion[{"label": label}].density*1.0e-19,   f"${label}$", r"$n_i  \, [10^{19} m^-3]$")
                        for label in ["D", "T", "He"]],
                ],

                # (fast_alpha_profiles_1d.ion[{"label": "He"}].particles.d_fast_factor,
                #  f""r"$D_{\alpha}/D_{He}$", r"$D_{\alpha}/D_{He}$"),

                # [
                #     (core_source_fusion.ion[{"label": "He"}].particles,
                #      r"$[ n_{\alpha}/\tau^{*}_{SD}]$", r"$S_{He} [m^3 s^{-1}]$",),

                #     (core_source_fusion.ion[{"label": "He"}].particles_fast,
                #         r"$n_{D} n_{T} \left<\sigma_{DT}\right>- n_{\alpha}/\tau^{*}_{SD}$", r"$S_{\alpha} [m^3 s^{-1}]$",),
                # ],


                [
                    (b_nHe_fast*1.0e-19,   r"$astra$",  r"$n_{He}  \, [10^{19} m^-3]$", bs_line_style),
                    (ionHe.density_fast*1.0e-19, r"fytok", r"$n_{\alpha} [10^{19} m^-3]$"),
                ],
                [
                    (b_nHe_thermal*1.0e-19,   r"astra",  r"$n_{He}  \, [10^{19} m^-3]$", bs_line_style),
                    (ionHe.density_thermal * 1.0e-19,   r"fytok", r"$n_{He}  \, [10^{19} m^-3]$"),
                ],
                [
                    (b_nHe*1.0e-19,   r"astra", r"$n_{total}  \, [10^{19} m^-3]$", bs_line_style),
                    (ionHe.density*1.0e-19, r"fytok$", r"$n_{total}  \, [10^{19} m^-3]$"),
                ]


                # ---------------------------------------------------------------------------------------------------


            ],
            x_axis=([0, 1.0],  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title=f" Particle solver '{particle_solver}'",
            grid=True, fontsize=10)\
            .savefig(output_path/f"core_profiles_result_{particle_solver}_alpha.svg", transparent=True)

    logger.info("====== DONE ========")
