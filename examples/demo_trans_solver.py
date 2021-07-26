from math import log
from operator import eq

import pandas as pd
from fytok.common.Atoms import atoms
from fytok.Tokamak import TWOPI, Tokamak
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.File import File
from spdm.data.Function import Function, PiecewiseFunction
from spdm.numlib import constants, np
from spdm.numlib.smooth import rms_residual, smooth_1d
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles, sp_figure

if __name__ == "__main__":
    logger.info("====== START ========")

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml")

    baseline = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')
    bs_psi_norm = baseline["Fp"].values
    bs_r_norm = baseline["x"].values

    eqdsk = File(
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        format="geqdsk").entry

    ###################################################################################################
    # Configure

    c_tokamak = {
        "wall": device.entry.get("wall"),
        "pf_active": device.entry.get("pf_active"),
        "tf": device.entry.get("tf"),
        "magnetics": device.entry.get("magnetics"),
        "radial_grid": {"rho_tor_norm": np.linspace(0, 0.9995, 100)},
    }

    # Equilibrium

    R0 = eqdsk.get("vacuum_toroidal_field.r0")
    B0 = eqdsk.get("vacuum_toroidal_field.b0")
    psi_axis = eqdsk.get("global_quantities.psi_axis")
    psi_boundary = eqdsk.get("global_quantities.psi_boundary")
    noise = 1  # np.random.random(bs_r_norm.shape)*0.1

    c_tokamak["equilibrium"] = {
        "code": {"name": "dummy"},
        "vacuum_toroidal_field": {
            "b0": eqdsk.get("vacuum_toroidal_field.b0"),
            "r0": eqdsk.get("vacuum_toroidal_field.r0"),
        },
        "time_slice": {
            "vacuum_toroidal_field": {
                "b0": eqdsk.get("vacuum_toroidal_field.b0"),
                "r0": eqdsk.get("vacuum_toroidal_field.r0"),
            },
            "global_quantities": eqdsk.get("global_quantities"),
            "profiles_1d":   eqdsk.get("profiles_1d"),
            "profiles_2d": {
                "psi": eqdsk.get("profiles_2d.psi")*TWOPI,
                "grid_type": "rectangular",
                "grid_index": 1,
                "grid": {
                    "dim1": eqdsk.get("profiles_2d.grid.dim1"),
                    "dim2": eqdsk.get("profiles_2d.grid.dim2"),
                }
            },
            "boundary_separatrix": eqdsk.get("boundary"),
            "coordinate_system": {"psi_norm": {"axis": 0.0, "boundary": 0.995, "npoints": 128}}
            # "coordinate_system": {"psi_norm": baseline["Fp"].values[:-1]}
        }}

    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    b_Te = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["TE"].values, i_end=i_ped-10, window_len=21)*1000)
    b_Ti = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["TI"].values, i_end=i_ped-10, window_len=21)*1000)
    b_ne = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["NE"].values, i_end=i_ped-10, window_len=21)*1.0e19)
    b_nDT = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["Nd+t"].values, i_end=i_ped-10, window_len=21)*1.0e19)
    b_nHe = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["Nalf"].values, i_end=i_ped-10, window_len=21)*1.0e19)

    # b_nHe = Function(bs_r_norm, baseline["Nalf"].values*1.0e19)
    # nDT = Function(bs_r_norm, baseline["Nd+t"].values*1.0e19)
    # b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_norm, baseline["Zeff"].values)

    c_tokamak["core_profiles"] = {
        "profiles_1d": {
            "electrons": {**atoms["e"], "density":              b_ne,   "temperature":        b_Te, },
            "ion": [
                {**atoms["D"],          "density":         0.5*b_nDT,   "temperature":        b_Ti, },
                {**atoms["T"],          "density":         0.5*b_nDT,   "temperature":        b_Ti, },
                {**atoms["He"],         "density":             b_nHe,   "temperature":        b_Ti, "is_impurity":True},
                {**atoms["Be"],         "density":         0.02*b_ne,   "temperature":        b_Ti, "is_impurity":True},
                {**atoms["Ar"],         "density":       0.0012*b_ne,   "temperature":        b_Ti, "is_impurity":True},
            ]}}

    # Core Transport
    conductivity_parallel = Function(bs_r_norm, baseline["Joh"].values*1.0e6 / baseline["U"].values *
                                     (2.0*constants.pi * R0))

    Cped = 0.17
    Ccore = 0.4
    # Function(bs_r_norm, baseline["Xi"].values)  Cped = 0.2
    chi = PiecewiseFunction([0, r_ped, 1.0],  [lambda x: Ccore*(1.0 + 3*(x**2)), lambda x: Cped])
    chi_e = PiecewiseFunction([0, r_ped, 1.0],  [lambda x: 0.5 * Ccore*(1.0 + 3*(x**2)), lambda x: Cped])

    D = 0.1*(chi+chi_e)

    v_pinch_ne = Function([0, r_ped, 1.0], lambda x: -0.6 * D(x) * x / R0)
    v_pinch_Te = Function([0, r_ped, 1.0], lambda x:  2.5 * chi_e(x) * x / R0)

    v_pinch_ni = Function([0, r_ped, 1.0], lambda x:  D(x) * x / R0)
    v_pinch_Ti = Function([0, r_ped, 1.0], lambda x:  chi(x) * x / R0)

    c_tokamak["core_transport"] = {
        "model": [
            {
                "code": {"name": "dummy"},
                "profiles_1d": {
                    # "conductivity_parallel": conductivity_parallel,
                    "electrons": {
                        **atoms["e"],
                        "particles":   {"d": D,     "v": v_pinch_ne},
                        "energy":      {"d": chi_e, "v": v_pinch_Te},
                    },
                    "ion": [
                        {
                            **atoms["D"],
                            "particles":{"d":  D, "v": v_pinch_ni},
                            "energy": {"d":  chi, "v": v_pinch_Ti},
                        },
                        {
                            **atoms["T"],
                            "particles":{"d":  D, "v": v_pinch_ni},
                            "energy": {"d":  chi, "v": v_pinch_Ti},
                        },
                        {
                            **atoms["He"],
                            "particles":{"d": D, "v": v_pinch_ni},
                            "energy": {"d": chi, "v": v_pinch_Ti}, }
                    ]}
            },
            # {"code": {"name": "neoclassical"}},
            {"code": {"name": "spitzer"}},
            # {"code": {"name": "glf23"}},
            # {"code": {"name": "nclass"}},
        ]}

    S = Function(lambda x: 9e20 * np.exp(15.0*(x**2-1.0)))

    Q_e = Function(bs_r_norm,
                   (baseline["Poh"].values
                    + baseline["Pdte"].values
                    + baseline["Paux"].values
                    - baseline["Peic"].values
                    - baseline["Prad"].values
                    # - baseline["Pneu"].values
                    )*1e6/constants.electron_volt)

    Q_DT = Function(bs_r_norm,
                    (baseline["Peic"].values
                     + baseline["Pdti"].values
                     + baseline["Pibm"].values
                     )*1e6/constants.electron_volt)

    Q_He = Function(bs_r_norm,
                    (- baseline["Pdti"].values
                     - baseline["Pdte"].values
                     )*1e6/constants.electron_volt)

    # Core Source
    c_tokamak["core_sources"] = {
        "source": [
            {"code": {"name": "dummy"},
             "profiles_1d": {
                "j_parallel": Function(
                    bs_r_norm,
                    (
                        # baseline["Jtot"].values
                        baseline["Joh"].values
                        # + baseline["Jbs"].values
                        + baseline["Jnb"].values
                        + baseline["Jrf"].values
                    ) * 1e6),
                "electrons":{**atoms["e"],  "particles": S, "energy": Q_e},
                "ion": [
                    {**atoms["D"],          "particles":S*0.5,      "energy":Q_DT*0.5},
                    {**atoms["T"],          "particles":S*0.5,      "energy":Q_DT*0.5},
                    {**atoms["He"],         "particles":0,          "energy":Q_He}
                ]}},
            # {"code": {"name": "collisional_equipartition"}, },
            {"code": {"name": "bootstrap_current"}},
        ]}

    #  TransportSolver
    c_tokamak["transport_solver"] = {
        "code": {"name": "bvp_solver2"},
        "boundary_conditions_1d": {
            "current": {"identifier": {"index": 1}, "value": [(psi_boundary-psi_axis)+psi_axis]},
            "electrons": {"particles": {"identifier": {"index": 1}, "value": [b_ne[-1]]},
                          "energy": {"identifier": {"index": 1}, "value": [b_Te[-1]]}},
            "ion": [
                {**atoms["D"],
                 "particles": {"identifier": {"index": 1}, "value": [0.5*b_nDT[-1]]},
                 "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}},
                {**atoms["T"],
                 "particles": {"identifier": {"index": 1}, "value": [0.5*b_nDT[-1]]},
                 "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}},
                {**atoms["He"],
                 "particles": {"identifier": {"index": 1}, "value": [b_nHe[-1]]},
                 "energy": {"identifier": {"index": 1}, "value": [b_Ti[-1]]}}
            ]
        }}

    ###################################################################################################
    # Initialize
    tok = Tokamak(c_tokamak)

    ###################################################################################################
    # Plot profiles

    e_psi = eqdsk.get('profiles_1d.psi')

    if True:  # Equilibrium
        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},  "vessel": {"edgecolor": "blue"}},
                  pf_active={"facecolor": 'red'},
                  equilibrium={
                      "contour": [0, 2],
                      "boundary": True,
                      "separatrix": True,
                      #   "scalar_field": [("psirz", {"levels": 16, "linewidths": 0.1}), ],
                  }
                  ) .savefig("/home/salmon/workspace/output/tokamak.svg", transparent=True)

        # _, spearatrix_surf = next(magnetic_surface.find_surface_by_psi_norm([1.0]))
        # bpol = np.asarray([magnetic_surface.Bpol(p[0], p[1]) for p in spearatrix_surf.points()])
        # logger.debug(bpol.min())
        # plot_profiles(
        #     [(bpol, r"$B_{pol}$"),
        #      (spearatrix_surf.points()[:, 0], r"$r$"),
        #      (spearatrix_surf.points()[:, 1], r"$z$"),
        #      (spearatrix_surf._mesh[0], r"$\theta$"),

        #      ],
        #     x_axis=([0, 1], "u"),
        #     grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/equilibrium_surf.svg", transparent=True)

    if True:

        magnetic_surface = tok.equilibrium.time_slice.coordinate_system

        plot_profiles(
            [
                [
                    (Function(bs_psi_norm, baseline["q"].values), r"astra",
                     r"$q [-]$", {"marker": '.', "linestyle": ''}),
                    # (Function(eqdsk.get('profiles_1d.psi_norm'), eqdsk.get('profiles_1d.q')), "eqdsk"),
                    (magnetic_surface.q,  r"$fytok$", r"$[Wb]$"),
                    # (magnetic_surface.dphi_dpsi,  r"$\frac{d\phi}{d\psi}$", r"$[Wb]$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["rho"].values), r"astra",
                     r"$\rho_{tor}[m]$",  {"marker": '.', "linestyle": ''}),
                    (magnetic_surface.rho_tor,  r"$\rho$", r"$[m]$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["x"].values),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", {"marker": '.', "linestyle": ''}),
                    (magnetic_surface.rho_tor_norm,  r"$\bar{\rho}$", r"$[-]$"),
                ],

                [
                    (Function(bs_psi_norm, 4*(constants.pi**2)*magnetic_surface.vacuum_toroidal_field.r0 * baseline["rho"].values),
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho$",  {"marker": '.', "linestyle": ''}),
                    (magnetic_surface.dvolume_drho_tor, r"$dV/d\rho_{tor}$", r"$[m^2]$"),
                ],

                (magnetic_surface.dvolume_dpsi, r"$\frac{dV}{d\psi}$"),

                [
                    (magnetic_surface.volume, r"$V$  from $\psi$"),
                    # (magnetic_surface.volume1, r"$V$ from $\rho_{tor}$"),
                ],

                [
                    (Function([0, 1], eqdsk.get('profiles_1d.f')), "eqdsk", r"$F_{pol} [Wb\cdot m]$"),
                    (magnetic_surface.fpol,  r"fytok", r"$[Wb]$"),
                ],


                (magnetic_surface.psi,  r"$\psi$", r"$[Wb]$"),
                (magnetic_surface.phi,  r"$\phi$", r"$[Wb]$"),
                (magnetic_surface.psi_norm,  r"$\bar{\psi}$", r"$[-]$"),


                [
                    (magnetic_surface.dpsi_drho_tor, r"$\frac{d\psi}{d\rho_{tor}}$", "", {"marker": '.'}),
                    (magnetic_surface.dvolume_drho_tor/magnetic_surface.dvolume_dpsi,
                     r"$\frac{dV}{d\rho_{tor}} / \frac{dV}{d\psi}$")
                ],
                # (magnetic_surface.drho_tor_dpsi*magnetic_surface.dpsi_drho_tor,
                #  r"$\frac{d\rho_{tor}}{d\psi} \cdot \frac{d\psi}{d\rho_{tor}}$"),
                # (magnetic_surface.gm2_,
                #  r"$gm2_=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                # (magnetic_surface.dpsi_drho_tor, r"$\frac{d\rho_{tor}}{d\psi}$", "", {"marker": "."}),


                (magnetic_surface.gm1,                                             r"$gm1=\left<\frac{1}{R^2}\right>$"),
                (magnetic_surface.gm2,                    r"$gm2=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                (magnetic_surface.gm3,                                r"$gm3=\left<\left|\nabla \rho\right|^2\right>$"),
                (magnetic_surface.gm7,                                  r"$gm7=\left<\left|\nabla \rho\right|\right>$"),
                (magnetic_surface.gm8,                                                         r"$gm8=\left<R\right>$"),

                # (magnetic_surface.dphi_dpsi,                                                  r"$\frac{d\phi}{d\psi}$"),
                # (magnetic_surface.dpsi_drho_tor,                                        r"$\frac{d\psi}{d\rho_{tor}}$"),
            ],
            # x_axis=(magnetic_surface.rho_tor_norm,      r"$\bar{\rho}_{tor}$"),
            x_axis=(magnetic_surface.psi_norm,      r"$\bar{\psi}$"),

            title="Equlibrium",
            grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/equilibrium_coord.svg", transparent=True)

    if True:
        eq_profile = tok.equilibrium.profiles_1d

        plot_profiles(
            [

                [
                    (Function(bs_psi_norm, baseline["q"].values),
                     r"astra",  r"$q [-]$", {"marker": '.', "linestyle": ''}),
                    (eq_profile.q,                                      r"fytok",  r"$q [-]$"),
                    (eq_profile.dphi_dpsi*np.sign(B0),
                     f"${np.sign(B0)}\\frac{{d\\phi}}{{d\\psi}}$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["rho"].values), r"astra",
                     r"$\rho_{tor}[m]$",  {"marker": '.', "linestyle": ''}),
                    (eq_profile.rho_tor,                                      r"fytok",    r"$\rho_{tor}[m]$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["x"].values),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", {"marker": '.', "linestyle": ''}),
                    (eq_profile.rho_tor_norm,                        r"fytok"),
                ],

                [
                    (Function(bs_psi_norm, baseline["shif"].values),
                     r"astra", "$\Delta$ shafranov \n shift $[m]$ ", {"marker": '.', "linestyle": ''}),
                    (eq_profile.geometric_axis.r-tok.equilibrium.time_slice.vacuum_toroidal_field.r0,
                     r"fytok", "shafranov \n shift $\Delta [m]$ "),
                ],
                [
                    (Function(bs_psi_norm, baseline["k"].values),
                     r"astra", r"$elongation[-]$", {"marker": '.', "linestyle": ''}),
                    (eq_profile.elongation,                                 r"fytok", r"$elongation[-]$"),
                ],
                [
                    (4*(constants.pi**2)*R0*tok.equilibrium.profiles_1d.rho_tor,
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho , dV/d\rho$"),
                    (tok.equilibrium.profiles_1d.dvolume_drho_tor,   r"$V^{\prime}$", r"$dV/d\rho$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["Jtot"].values*1e6),   r"astra",
                     r"$j_{\parallel} [A\cdot m^{-2}]$", {"marker": '.', "linestyle": ''}),
                    (eq_profile.j_parallel,                                         r"fytok",     r"$j_{\parallel}$"),
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

                # [
                #     (eq.coordinate_system.surface_integrate2(lambda r, z:1.0/r**2), \
                #      r"$\left<\frac{1}{R^2}\right>$"),
                #     (eq.coordinate_system.surface_integrate(1/eq.coordinate_system.r**2), \
                #      r"$\left<\frac{1}{R^2}\right>$"),
                # ]
            ],
            x_axis=(eq_profile._coord.psi_norm,                                                r"$\psi/\psi_{bdry}$"),
            # x_axis=([0, 1.0],                                                r"$\psi/\psi_{bdry}$"),

            title="Equlibrium",
            grid=True, fontsize=16) .savefig("/home/salmon/workspace/output/equilibrium.svg", transparent=True)

    if True:  # CoreProfile initialize value

        core_profile = tok.core_profiles.profiles_1d

        plot_profiles(
            [
                [
                    (b_ne,  "electron astra",    r"$n_{e} [m \cdot s^{-3}]$", {"marker": '.', "linestyle": ''}),
                    (b_nDT*0.5,  "D astra",    r"$n_{D} [m \cdot s^{-3}]$", {"marker": '.', "linestyle": ''}),
                    (b_nHe,  "He astra",    r"$n_{He} [m \cdot s^{-3}]$", {"marker": '.', "linestyle": ''}),

                    (core_profile.electrons.density,             r"$electron$", ),
                    *[(ion.density,          f"${ion.label}$") for ion in core_profile.ion],

                ],
                [
                    (b_Te,    r"astra $T_i$",       r"T $[eV]$", {"marker": '.', "linestyle": ''}),
                    (b_Ti,    r"astra $T_i$",       r"$T_{i}$", {"marker": '.', "linestyle": ''}),
                    (core_profile.electrons.temperature,       r"$e$", r"T $[eV]$"),
                    *[(ion.temperature,      f"${ion.label}$") for ion in core_profile.ion],
                ],

                [
                    (Function(bs_r_norm, baseline["Zeff"].values),       r"astra",
                     r"$Z_{eff}  [-]$", {"marker": '.', "linestyle": ''}),
                    (core_profile.zeff,                                   r"$fytok$"),
                ],
                (core_profile.grid.psi,                                    r"$\psi$"),
            ],
            x_axis=([0, 1.0],                                  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_profiles_initialize.svg", transparent=True)

    if True:  # CoreTransport

        tok.core_transport.refresh(equilibrium=tok.equilibrium, core_profiles=tok.core_profiles)

        core_transport_model = tok.core_transport.model_combiner

        core_transport = core_transport_model.profiles_1d

        nc_profiles_1d = tok.core_transport.model[{"code.name": "neoclassical"}].profiles_1d

        plot_profiles(
            [
                [
                    (Function(bs_r_norm, baseline["Xi"].values),          r"astra",
                     r"$\chi_{i}$", {"marker": '.', "linestyle": ''}),
                    *[(core_transport.ion[{"label": ion.label}].energy.d,
                       f"{ion.label}", r"$\chi_{i}$") for ion in core_profile.ion if not ion.is_impurity],
                ],

                [
                    (Function(bs_r_norm,  np.log(baseline["XiNC"].values)),
                     "astra", r"$ln \chi_{i,nc}$", {"marker": '.', "linestyle": ''}),
                    # * [(np.log(core_transport1d_nc.ion[{"label": label}].energy.d),   f"${label}$", r"$ln \chi_{i,nc}$")
                    #     for label in ("H", "D", "He")],
                ],

                [
                    (Function(bs_r_norm, baseline["He"].values), "astra",
                     r"$\chi_{e}$", {"marker": '.', "linestyle": ''}),
                    (core_transport.electrons.energy.d,  "fytok", r"$\chi_{e}$"),
                ],

                # [
                #     (core_transport1d.electrons.energy.v,           r"$electron$", r"V pinch $[m\cdot s^{-1}]$"),
                #     *[(core_transport1d.ion[{"label": ion.label}].energy.v,  f"${ion.label}$",)
                #       for ion in core_profile.ion],
                # ],

                [
                    (Function(bs_r_norm, baseline["Joh"].values*1.0e6 / baseline["U"].values *
                              (2.0*constants.pi * tok.equilibrium.time_slice.vacuum_toroidal_field.r0)),
                     r"astra", r"$\sigma_{\parallel}$", {"marker": '.', "linestyle": ''}),

                    (core_transport.conductivity_parallel,  r"fytok", r"$\sigma_{\parallel}$"),
                ],
                # [
                #     (Function(bs_r_norm, baseline["XiNC"].values), "astra",
                #      "neoclassical  $\\chi_{NC}$ \n ion heat conductivity", {"marker": '.', "linestyle": ''}),

                #     *[(ion.energy.d,  f"{ion.label}", r"Neoclassical $\chi_{NC}$")
                #       for ion in nc_profiles_1d.ion if not ion.is_impurity],
                # ],
                # [(ion.particles.d,  f"{ion.label}", r"Neoclassical $D_{NC}$")
                #  for ion in nc_profiles_1d.ion if not ion.is_impurity],

                # (core_transport1d.e_field_radial,                                             r"$E_{radial}$"),

            ],
            x_axis=([0, 1.0],   r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title="combine",
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_transport.svg", transparent=True)

    if True:  # CoreSources
        tok.core_sources.refresh(equilibrium=tok.equilibrium, core_profiles=tok.core_profiles)

        core_source = tok.core_sources.source_combiner.profiles_1d

        plot_profiles(
            [
                [
                    (Function(bs_r_norm, baseline["Jtot"].values),  "astra",
                     r"$J_{\parallel} [MA\cdot m^{-2}]$", {"marker": '.', "linestyle": ''}),
                    (core_source.j_parallel*1e-6,     "fytok", r"$J_{\parallel} [A\cdot m^{-2}]$"),
                ],

                # [
                #     (Function(bs_r_norm, baseline["Joh"].values), "astra",    r"$j_{ohmic} [MA\cdot m^{-2}]$"),
                #     (core_profile.j_ohmic,                        "fytok",    r"$j_{ohmic} [MA\cdot m^{-2}]$"),
                # ],

                [
                    (Function(bs_r_norm, baseline["Jbs"].values),
                     r"astra", r"bootstrap current $[MA\cdot m^{-2}]$", {"marker": '.', "linestyle": ''}),
                    (tok.core_sources.source[{"code.name": "bootstrap_current"}
                                             ].profiles_1d.j_parallel*1e-6, r"fytok",),
                ],
                [
                    (rms_residual(Function(bs_r_norm, baseline["Jbs"].values*1e6),
                                  tok.core_sources.source[{"code.name": "bootstrap_current"}].profiles_1d.j_parallel),
                     r"bootstrap current", r"  rms residual $[\%]$"),
                    (rms_residual(Function(bs_r_norm, baseline["Jtot"].values), core_source.j_parallel*1e-6),
                     r"total current", r"  rms residual $[\%]$"),
                ],

                # [
                (core_source.electrons.energy,   "electron",    r"$Q [eV\cdot m^{-3} s^{-1}]$"),

                *[(ion.energy*1e-6,  f"{ion.label}", f"$Q_{ion.label}$")
                  for ion in core_source.ion if not ion.is_impurity],
                # ],


            ],
            x_axis=([0, 1.0], r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_sources.svg", transparent=True)

    ###################################################################################################
    # TransportSolver
    if True:
        tok.solve(enable_ion_particle_solver=False,
                  max_nodes=500,
                  tolerance=1.0e-4,
                  verbose=2,
                  bvp_rms_mask=[r_ped])

        core_profile = tok.core_profiles.profiles_1d

        plot_profiles(
            [

                ######################################################################
                # psi ,current
                [
                    (Function(bs_r_norm, (bs_psi_norm*(psi_boundary-psi_axis)+psi_axis)),
                     r"astra", r"$\psi [Wb]$", {"marker": '.', "linestyle": ''}),
                    (core_profile["psi"],  r"fytok", r"$\psi  [Wb]$"),
                ],

                ######################################################################
                # electron particles
                [
                    (b_ne, r"astra", r"$n_e [m^{-3}]$",  {"marker": '.', "linestyle": ''}),
                    (core_profile.electrons.density, r"fytok", r"$n_e [ m^{-3}]$"),
                    # (core_profile.electrons["density_error"], r"rms residuals ",
                    #     r"$n_e [ m^{-3}]$",  {"color": "red", "linestyle": "dashed"}),
                ],

                [
                    (b_nDT/2,    r"astra", r"$n_i [m^-3]$", {"marker": '.', "linestyle": ''}),
                    * [(ion.density,   f"${ion.label}$") for ion in core_profile.ion],
                ],
                # [
                #     (core_profile.electrons["density_flux"], r"Source",
                #      r"$\Gamma_e$ Particle flux", {"color": "green", }),
                #     (core_profile.electrons["diff_flux"],    r"Diffusive ",  "", {"color": "black", }),
                #     (core_profile.electrons["conv_flux"],    r"Convective ", "",  {"color": "blue", }),
                #     (
                #         core_profile.electrons["density_flux"]
                #         - core_profile.electrons["diff_flux"]
                #         - core_profile.electrons["conv_flux"],
                #         r"residual",  "", {"color": "red", }),
                # ],
                ######################################################################
                # electron temperature
                [
                    (b_Te*1e-3, r"astra", r"$T_e [KeV]$",  {"marker": '.', "linestyle": ''}),
                    (core_profile.electrons.temperature*1e-3, r"fytok", r"$ [KeV]$"),
                ],
                # (core_profile.electrons["temperature_error"], r"rms_residuals",
                #  r"$[eV]$",  {"color": "red", "linestyle": "dashed"}),


                ######################################################################
                # ion temperature
                # [
                #     (core_profile.electrons.get("density_flux"), r"$\Gamma_e$", r"$\Gamma_i$"),
                #     * [(ion.get("density_flux"),          f"$\\Gamma_{ion.label}$", r"$\Gamma_i [eV]$")
                #         for ion in core_profile.ion if  not ion.is_impurity],
                # ],

                [
                    (b_Ti*1e-3,    r"astra $T_i$",       r"$T_{i} [KeV]$", {"marker": '.', "linestyle": ''}),
                    * [(ion.temperature*1e-3,          f"${ion.label}$", r"$T_i [KeV]$")
                        for ion in core_profile.ion if not ion.is_impurity],
                ],

                [
                    (rms_residual(Function(bs_r_norm, (bs_psi_norm*(psi_boundary-psi_axis)+psi_axis)),
                                  core_profile["psi"]), r"$\psi$", " rms residual [%]"),

                    (rms_residual(b_ne, core_profile.electrons.density), r"$n_e$"),
                    *[(rms_residual(b_nDT/2, ion.density), f"$n_{ion.label}$")
                      for ion in core_profile.ion if not ion.is_impurity],
                    (rms_residual(b_Te, core_profile.electrons.temperature), r"$T_e$"),
                    *[(rms_residual(b_Ti, ion.temperature), f"$T_{ion.label}$")
                      for ion in core_profile.ion if not ion.is_impurity],

                ],
            ],
            # x_axis=(rho_tor_norm,                             r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            x_axis=([0, 1.0],  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title="Result of TransportSolver",
            # index_slice=slice(0, 200, 1),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_profiles_result.svg", transparent=True)

    logger.info("====== DONE ========")
