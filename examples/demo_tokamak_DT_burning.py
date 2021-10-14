from copy import deepcopy
import pathlib
from fytok.transport.CoreProfiles import CoreProfiles

import numpy as np
import pandas as pd
from fytok.common.Atoms import atoms
from fytok.common.load_profiles import (load_core_profiles, load_core_source,
                                        load_core_transport, load_equilibrium)
from fytok.numlib.smooth import rms_residual, smooth_1d
from fytok.Tokamak import Tokamak
from scipy import constants
from spdm.data.File import File
from spdm.data.Function import Function, PiecewiseFunction
from spdm.util.logger import logger
from spdm.view.plot_profiles import plot_profiles, sp_figure

if __name__ == "__main__":
    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    ###################################################################################################
    # baseline
    device_desc = File(
        "/home/salmon/workspace/fytokdata/mapping/ITER/imas/3/static/config.xml", format="XML").read()

    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="geqdsk").read()
    # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
    # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
    # profiles = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')

    R0 = eqdsk_file.get("vacuum_toroidal_field.r0")
    B0 = eqdsk_file.get("vacuum_toroidal_field.b0")
    psi_axis = eqdsk_file.get("global_quantities.psi_axis")
    psi_boundary = eqdsk_file.get("global_quantities.psi_boundary")
    bs_eq_psi = eqdsk_file.get("profiles_1d.psi")
    bs_eq_psi_norm = (bs_eq_psi-psi_axis)/(psi_boundary-psi_axis)
    bs_eq_fpol = Function(bs_eq_psi_norm, eqdsk_file.get("profiles_1d.f"))

    profiles = pd.read_excel('/home/salmon/workspace/data/15MA inductive - burn/15MA Inductive at burn-ASTRA.xls',
                             sheet_name='15MA plasma', header=10, usecols="B:BN")

    bs_r_norm = profiles["x"].values

    b_Te = Function(bs_r_norm,  profiles["TE"].values*1000)
    b_Ti = Function(bs_r_norm,  profiles["TI"].values*1000)
    b_ne = Function(bs_r_norm,  profiles["NE"].values*1.0e19)

    b_nHe = Function(bs_r_norm,  profiles["Nalf"].values * 1.0e19)
    b_ni = Function(bs_r_norm,  profiles["Nd+t"].values * 1.0e19*0.5)
    b_nImp = Function(bs_r_norm, profiles["Nz"].values * 1.0e19)
    b_zeff = Function(bs_r_norm,   profiles["Zeff"].values)
    bs_psi_norm = profiles["Fp"].values
    bs_psi = bs_psi_norm*(psi_boundary-psi_axis)+psi_axis

    bs_r_norm = profiles["x"].values

    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    ###################################################################################################
    # Initialize Tokamak
    d = device_desc.get_many(["wall", "pf_active", "tf", "magnetics"])

    tok = Tokamak(**d)

    # Equilibrium
    eqdsk = load_equilibrium(eqdsk_file)

    tok["equilibrium"] = {**eqdsk,
                          "code": {"name": "dummy"},
                          "boundary": {"psi_norm": 0.995},
                          "coordinate_system": {"psi_norm": np.linspace(0.001, 0.995, 64), "theta": 128}}

    sp_figure(tok,
              wall={"limiter": {"edgecolor": "green"},
                    "vessel": {"edgecolor": "blue"}},
              pf_active={"facecolor": 'red'},
              equilibrium={
                  "contour": [0, 2],
                  "boundary": True,
                  "separatrix": True,
              }
              ) .savefig(output_path/"tokamak.svg", transparent=True)

    if True:  # plot tokamak

        magnetic_surface = tok.equilibrium.coordinate_system

        plot_profiles(
            [
                [
                    (bs_eq_fpol, "eqdsk", r"$F_{pol} [Wb\cdot m]$"),
                    (magnetic_surface.fpol,  r"fytok", r"$[Wb]$"),
                ],

                [
                    (Function(bs_psi_norm, profiles["q"].values), r"astra",
                     r"$q [-]$", {"marker": '.', "linestyle": ''}),
                    # (Function(eqdsk.get('profiles_1d.psi_norm'), eqdsk.get('profiles_1d.q')), "eqdsk"),
                    (magnetic_surface.q,  r"$fytok$", r"$[Wb]$"),
                    # (magnetic_surface.dphi_dpsi,  r"$\frac{d\phi}{d\psi}$", r"$[Wb]$"),
                ],
                [
                    (Function(bs_psi_norm, profiles["rho"].values), r"astra",
                     r"$\rho_{tor}[m]$",  {"marker": '.', "linestyle": ''}),
                    (magnetic_surface.rho_tor,  r"$\rho$", r"$[m]$"),
                ],
                [
                    (Function(bs_psi_norm, profiles["x"].values),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", {"marker": '.', "linestyle": ''}),
                    (magnetic_surface.rho_tor_norm,
                     r"$\bar{\rho}$", r"$[-]$"),
                ],

                [
                    (Function(bs_psi_norm, 4*(constants.pi**2) * R0 * profiles["rho"].values),
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho$",  {"marker": '.', "linestyle": ''}),
                    (magnetic_surface.dvolume_drho_tor,
                     r"$dV/d\rho_{tor}$", r"$[m^2]$"),
                ],

                (magnetic_surface.dvolume_dpsi, r"$\frac{dV}{d\psi}$"),

                [
                    (magnetic_surface.volume, r"$V$  from $\psi$"),
                    # (magnetic_surface.volume1, r"$V$ from $\rho_{tor}$"),
                ],



                (magnetic_surface.psi,  r"$\psi$", r"$[Wb]$"),
                (magnetic_surface.phi,  r"$\phi$", r"$[Wb]$"),
                (magnetic_surface.psi_norm,  r"$\bar{\psi}$", r"$[-]$"),


                [
                    (magnetic_surface.dpsi_drho_tor,
                     r"$\frac{d\psi}{d\rho_{tor}}$", "", {"marker": '.'}),
                    (magnetic_surface.dvolume_drho_tor/magnetic_surface.dvolume_dpsi,
                     r"$\frac{dV}{d\rho_{tor}} / \frac{dV}{d\psi}$")
                ],
                # (magnetic_surface.drho_tor_dpsi*magnetic_surface.dpsi_drho_tor,
                #  r"$\frac{d\rho_{tor}}{d\psi} \cdot \frac{d\psi}{d\rho_{tor}}$"),
                # (magnetic_surface.gm2_,
                #  r"$gm2_=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                # (magnetic_surface.dpsi_drho_tor, r"$\frac{d\rho_{tor}}{d\psi}$", "", {"marker": "."}),


                (magnetic_surface.gm1,
                 r"$gm1=\left<\frac{1}{R^2}\right>$"),
                (magnetic_surface.gm2,
                 r"$gm2=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                (magnetic_surface.gm3,
                 r"$gm3=\left<\left|\nabla \rho\right|^2\right>$"),
                (magnetic_surface.gm7,
                 r"$gm7=\left<\left|\nabla \rho\right|\right>$"),
                (magnetic_surface.gm8,
                 r"$gm8=\left<R\right>$"),

                # (magnetic_surface.dphi_dpsi,                                                  r"$\frac{d\phi}{d\psi}$"),
                # (magnetic_surface.dpsi_drho_tor,                                        r"$\frac{d\psi}{d\rho_{tor}}$"),
            ],
            # x_axis=(magnetic_surface.rho_tor_norm,      r"$\bar{\rho}_{tor}$"),
            x_axis=(magnetic_surface.psi_norm,      r"$\bar{\psi}$"),
            title="Equilibrium",
            grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/equilibrium_coord.svg", transparent=True)

        eq_profile = tok.equilibrium.profiles_1d

        plot_profiles(
            [

                [
                    (Function(bs_psi_norm, profiles["q"].values),
                     r"astra",  r"$q [-]$", {"marker": '.', "linestyle": ''}),
                    (eq_profile.q,
                     r"fytok",  r"$q [-]$"),
                    (eq_profile.dphi_dpsi*np.sign(B0)/constants.pi/2.0,
                     r"$\frac{\sigma_{B_{p}}}{\left(2\pi\right)^{1-e_{B_{p}}}}\frac{d\Phi_{tor}}{d\psi_{ref}}$"),
                ],
                [
                    (Function(bs_psi_norm, profiles["rho"].values), r"astra",
                     r"$\rho_{tor}[m]$",  {"marker": '.', "linestyle": ''}),
                    (eq_profile.rho_tor,
                     r"fytok",    r"$\rho_{tor}[m]$"),
                ],
                [
                    (Function(bs_psi_norm, profiles["x"].values),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", {"marker": '.', "linestyle": ''}),
                    (eq_profile.rho_tor_norm,                        r"fytok"),
                ],

                [
                    (Function(bs_psi_norm, profiles["shif"].values),
                     r"astra", "$\Delta$ shafranov \n shift $[m]$ ", {"marker": '.', "linestyle": ''}),
                    (eq_profile.geometric_axis.r - R0,
                     r"fytok", "shafranov \n shift $\Delta [m]$ "),
                ],
                [
                    (Function(bs_psi_norm, profiles["k"].values),
                     r"astra", r"$elongation[-]$", {"marker": '.', "linestyle": ''}),
                    (eq_profile.elongation,
                     r"fytok", r"$elongation[-]$"),
                ],
                [
                    (4*(constants.pi**2) * R0*tok.equilibrium.profiles_1d.rho_tor,
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho , dV/d\rho$"),
                    (tok.equilibrium.profiles_1d.dvolume_drho_tor,
                     r"$V^{\prime}$", r"$dV/d\rho$"),
                ],
                # [
                #     (Function(bs_psi_norm, profiles["Jtot"].values*1e6),   r"astra",
                #      r"$j_{\parallel} [A\cdot m^{-2}]$", {"marker": '.', "linestyle": ''}),
                #     (eq_profile.j_parallel,                                 r"fytok",     r"$j_{\parallel}$"),
                # ],

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
            x_axis=(eq_profile._coord.psi_norm,      r"$\psi/\psi_{bdry}$"),
            # x_axis=([0, 1.0],                                                r"$\psi/\psi_{bdry}$"),

            title="Equilibrium",
            grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/equilibrium.svg", transparent=True)

    if True:  # CoreProfile initialize value

        tok["core_profiles.profiles_1d"] = load_core_profiles(profiles, grid=tok.equilibrium.radial_grid)

        core_profile_1d = tok.core_profiles.profiles_1d

        plot_profiles(
            [
                [
                    (b_ne,      "electron astra",
                     r"density $n [m \cdot s^{-3}]$", {"marker": '.', "linestyle": ''}),
                    (b_ni, "D astra",
                     r"density $n [m \cdot s^{-3}]$", {"marker": '.', "linestyle": ''}),
                    (b_nHe,     "He astra",
                     r"density $n [m \cdot s^{-3}]$", {"marker": '.', "linestyle": ''}),

                    (core_profile_1d.electrons.density,             r"$electron$", ),
                    *[(ion.density,          f"${ion.label}$")
                      for ion in core_profile_1d.ion],

                ],
                [
                    (b_Te,    r"astra $T_e$",       r"$T [eV]$", {
                     "marker": '.', "linestyle": ''}),
                    (b_Ti,    r"astra $T_i$",       r"$T [eV]$", {
                     "marker": '.', "linestyle": ''}),
                    (core_profile_1d.electrons.temperature,
                     r"$e$", r"T $[eV]$"),
                    *[(ion.temperature,      f"${ion.label}$")
                      for ion in core_profile_1d.ion],
                ],

                [
                    (Function(bs_r_norm, profiles["Zeff"].values),       r"astra",
                     r"$Z_{eff}  [-]$", {"marker": '.', "linestyle": ''}),
                    (core_profile_1d.zeff,
                     r"$fytok$"),
                ],
                [
                    (Function(bs_eq_psi_norm, bs_eq_psi)
                     (core_profile_1d.grid.psi_norm), "astra",      r"$\psi$"),
                    (core_profile_1d.grid.psi,              r"fytok"),
                ]
            ],
            x_axis=([0, 1.0],
                    r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig(output_path/"core_profiles_initialize.svg", transparent=True)

    if True:  # CoreTransport
        tok["core_transport.model"] = [
            {"code": {"name": "dummy"}, "profiles_1d": load_core_transport(
                profiles, tok.core_profiles.profiles_1d.grid)},
            # {"code": {"name": "spitzer"}},
            # {"code": {"name": "neoclassical"}},
            # {"code": {"name": "glf23"}},
            # {"code": {"name": "nclass"}},
        ]

        # tok.core_transport.refresh(equilibrium=tok.equilibrium, core_profiles=tok.core_profiles)

        core_transport_model = tok.core_transport.model_combiner

        core_transport = core_transport_model.profiles_1d

        # nc_profiles_1d = tok.core_transport.model[{"code.name": "neoclassical"}].profiles_1d

        plot_profiles(
            [
                [
                    (Function(bs_r_norm, profiles["Xi"].values),          r"astra",
                     r"$\chi_{i}$", {"marker": '.', "linestyle": ''}),
                    *[(core_transport.ion[{"label": ion.label}].energy.d,
                       f"{ion.label}", r"$\chi_{i}$") for ion in core_profile_1d.ion if not ion.is_impurity],
                ],
                [
                    (Function(bs_r_norm, profiles["He"].values), "astra",
                     r"$\chi_{e}$", {"marker": '.', "linestyle": ''}),
                    (core_transport.electrons.energy.d,
                     "fytok", r"$\chi_{e}$"),
                ],
                [
                    (Function(bs_r_norm, profiles["Joh"].values*1.0e6 / profiles["U"].values * (2.0*constants.pi * R0)),
                     r"astra", r"$\sigma_{\parallel}$", {"marker": '.', "linestyle": ''}),

                    (core_transport.conductivity_parallel,
                     r"fytok", r"$\sigma_{\parallel}$"),
                ],


                # [
                #     (Function(bs_r_norm,  np.log(profiles["XiNC"].values)),
                #      "astra", r"$ln \chi_{i,nc}$", {"marker": '.', "linestyle": ''}),
                #     # * [(np.log(core_transport1d_nc.ion[{"label": label}].energy.d),   f"${label}$", r"$ln \chi_{i,nc}$")
                #     #     for label in ("H", "D", "He")],
                # ],
                # [
                #     (Function(bs_r_norm, profiles["XiNC"].values), "astra",
                #      "neoclassical  $\\chi_{NC}$ \n ion heat conductivity", {"marker": '.', "linestyle": ''}),

                #     # *[(ion.energy.d,  f"{ion.label}", r"Neoclassical $\chi_{NC}$")
                #     #   for ion in nc_profiles_1d.ion if not ion.is_impurity],
                # ],
                # [(ion.particles.d,  f"{ion.label}", r"Neoclassical $D_{NC}$")
                #  for ion in nc_profiles_1d.ion if not ion.is_impurity],

                # (core_transport1d.e_field_radial,                                             r"$E_{radial}$"),

            ],
            x_axis=([0, 1.0],   r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title="combine",
            grid=True, fontsize=10) .savefig(output_path/"core_transport.svg", transparent=True)

    if True:  # CoreSources
        tok["core_sources.source"] = [
            {"code": {"name": "dummy"}, "profiles_1d": load_core_source(profiles, tok.core_profiles.profiles_1d.grid)},
            # {"code": {"name": "bootstrap_current"}},
            {"code": {"name": "fusion_reaction"}},
        ]

        tok.core_sources.refresh(equilibrium=tok.equilibrium, core_profiles=tok.core_profiles)

        core_source = tok.core_sources.source_combiner.profiles_1d
        core_source_fusion = tok.core_sources.source[{"code.name": "fusion_reaction"}].profiles_1d
        plot_profiles(
            [
                [
                    (Function(bs_r_norm, profiles["Jtot"].values),  "astra",
                     r"$J_{\parallel} [MA\cdot m^{-2}]$", {"marker": '.', "linestyle": ''}),
                    (core_source.j_parallel*1e-6,     "fytok",
                     r"$J_{\parallel} [A\cdot m^{-2}]$"),
                ],

                [
                    (Function(bs_r_norm, profiles["Joh"].values), "astra",
                     r"$j_{ohmic} [MA\cdot m^{-2}]$", {"marker": '.', "linestyle": ''}),
                    (core_profile_1d.j_ohmic*1e-6,
                     "fytok",    r"$j_{ohmic} [MA\cdot m^{-2}]$"),
                ],

                # [
                #     (Function(bs_r_norm, profiles["Jbs"].values),
                #      r"astra", "bootstrap current \n $[MA\\cdot m^{-2}]$", {"marker": '.', "linestyle": ''}),
                #     (tok.core_sources.source[{"code.name": "bootstrap_current"}].profiles_1d.j_parallel*1e-6,
                #      r"fytok",),
                # ],
                # [
                #     (rms_residual(Function(bs_r_norm, profiles["Jbs"].values*1e6),
                #                   tok.core_sources.source[{"code.name": "bootstrap_current"}].profiles_1d.j_parallel),
                #      r"bootstrap current", r"  rms residual $[\%]$"),
                #     (rms_residual(Function(bs_r_norm, profiles["Jtot"].values), core_source.j_parallel*1e-6),
                #      r"total current", r"  rms residual $[\%]$"),
                # ],
                [
                    (core_source_fusion.ion[{"label": "D"}].particles,    r"D",  r"$S_{DT} [m^3 s^{-1}]$",),
                    (core_source_fusion.ion[{"label": "T"}].particles,    r"T",  r"$S_{DT} [m^3 s^{-1}]$",),
                    (core_source_fusion.ion[{"label": "He"}].particles,   r"He", r"$S_{DT} [m^3 s^{-1}]$",),
                    (core_source.ion[{"label": "D"}].particles,           r"$D_{total}$", r"$S_{DT} [m^3 s^{-1}]$",),

                ],

                [
                    (core_source.electrons.energy,  "electron",
                     r"$Q [eV\cdot m^{-3} s^{-1}]$"),
                    *[(ion.energy*1e-6,             f"{ion.label}",  r"$Q [eV\cdot m^{-3} s^{-1}]$")
                      for ion in core_source.ion if not ion.is_impurity],
                ],
            ],
            x_axis=([0, 1.0], r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig(output_path/"core_sources.svg", transparent=True)

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
                     "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}}
                ]
            }}

        particle_solver = tok.core_transport_solver.get('code.parameters.particle_solver', 'ion')

        tok.refresh()

        core_profile_1d = tok.core_profiles.profiles_1d

        plot_profiles(
            [
                # psi ,current
                [
                    (Function(bs_r_norm, bs_psi),
                     r"astra", r"$\psi [Wb]$", {"marker": '.', "linestyle": ''}),
                    (core_profile_1d["psi"],  r"fytok", r"$\psi  [Wb]$", {
                     "marker": '+', "linestyle": '-'}),
                ],

                # (core_profile_1d["psi_flux"],  r"fytok", r"$\Gamma_{\psi}$", {"marker": '+', "linestyle": '-'}),

                # electron
                [
                    (b_ne/1.0e19, r"astra",
                     r"$n_e [10^{19} m^{-3}]$",  {"marker": '.', "linestyle": ''}),
                    (core_profile_1d.electrons.density/1.0e19,
                     r"fytok", r"$n_e [10^{19} m^{-3}]$"),
                ],

                [
                    (b_Te/1000.0, r"astra",
                     r"$T_e [keV]$",  {"marker": '.', "linestyle": ''}),
                    (core_profile_1d.electrons.temperature / \
                     1000.0, r"fytok", r"$T_e [keV]$"),
                ],

                # ion
                [
                    (b_ni/1.0e19,    r"astra",
                     r"$n_i  \, [10^{19} m^-3]$", {"marker": '.', "linestyle": ''}),
                    * [(ion.density/1.0e19,   f"${ion.label}$", r"$n_i  \, [10^{19} m^-3]$")
                        for ion in core_profile_1d.ion if not ion.is_impurity and ion.label != "He"],
                ],

                [
                    (b_Ti/1000.0,    r"astra",
                     r"$T_{i} \, [keV]$", {"marker": '.', "linestyle": ''}),
                    * [(ion.temperature/1000.0,  f"fytok {ion.label}$", r"$T_{i} [keV]$")
                        for ion in core_profile_1d.ion if not ion.is_impurity and ion.label != "He"],
                ],

                # ---------------------------------------------------------------------------------------------------

                (core_profile_1d["rms_residuals"] * 100, r"bvp", r"residual $[\%]$"),


                [
                    (rms_residual(Function(bs_r_norm, bs_psi),
                     core_profile_1d["psi"]), r"$\psi$", " rms residual [%]"),

                    (rms_residual(b_ne, core_profile_1d.electrons.density), r"$n_e$"),

                    (rms_residual(b_Te, core_profile_1d.electrons.temperature), r"$T_e$"),

                    *[(rms_residual(b_ni, ion.density), f"$n_{ion.label}$")
                      for ion in core_profile_1d.ion if not ion.is_impurity],
                    *[(rms_residual(b_Ti, ion.temperature), f"$T_{ion.label}$")
                      for ion in core_profile_1d.ion if not ion.is_impurity],

                ],
            ],
            x_axis=([0, 1.0],  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title=f" Particle solver '{particle_solver}'",
            grid=True, fontsize=10)\
            .savefig(output_path/f"core_profiles_result_{particle_solver}.svg", transparent=True)

        ion_He: CoreProfiles.Profiles1D.Ion = core_profile_1d.ion[{"label": "He"}]

        plot_profiles(
            [

                # ion
                [
                    (b_nHe/1.0e19,    r"astra",
                     r"$n_i  \, [10^{19} m^-3]$", {"marker": '.', "linestyle": ''}),
                    (ion_He.density/1.0e19,
                     f"${ion_He.label}$", r"$n_i  \, [10^{19} m^-3]$"),
                ],

                # [
                #     (b_Ti/1000.0,    r"astra",
                #      r"$T_{i} \, [keV]$", {"marker": '.', "linestyle": ''}),
                #     * [(ion.temperature/1000.0,  f"fytok ${ion.label}$", r"$T_{i} [keV]$")
                #         for ion in core_profile_1d.ion if not ion.is_impurity],
                # ],

                # # ---------------------------------------------------------------------------------------------------

                # (core_profile_1d["rms_residuals"]
                #  * 100, r"bvp", r"residual $[\%]$"),


                # [
                #     (rms_residual(Function(bs_r_norm, bs_psi),
                #      core_profile_1d["psi"]), r"$\psi$", " rms residual [%]"),

                #     (rms_residual(b_ne, core_profile_1d.electrons.density), r"$n_e$"),

                #     (rms_residual(b_Te, core_profile_1d.electrons.temperature), r"$T_e$"),

                #     *[(rms_residual(b_ni, ion.density), f"$n_{ion.label}$")
                #       for ion in core_profile_1d.ion if not ion.is_impurity],
                #     *[(rms_residual(b_Ti, ion.temperature), f"$T_{ion.label}$")
                #       for ion in core_profile_1d.ion if not ion.is_impurity],

                # ],
            ],
            x_axis=([0, 1.0],  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title=f" Particle solver '{particle_solver}'",
            grid=True, fontsize=10).savefig(output_path/f"core_profiles_result_{particle_solver}_alpha.svg", transparent=True)

    logger.info("====== DONE ========")
