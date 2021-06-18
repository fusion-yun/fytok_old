from math import log
from operator import eq
import pandas as pd

from fytok.Tokamak import TWOPI, Tokamak
from spdm.data.File import File
from spdm.data.Function import Function, PiecewiseFunction
from spdm.numlib import constants, np
from spdm.numlib.smooth import smooth
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles, sp_figure
from fytok.common.Atoms import atoms

if __name__ == "__main__":
    logger.info("====== START ========")

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml")

    baseline = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')
    bs_psi_norm = baseline["Fp"].values
    bs_r_nrom = baseline["x"].values

    eqdsk = File(
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        format="geqdsk").entry

    ###################################################################################################
    # Configure

    configure = {
        "wall": device.entry.find("wall"),
        "pf_active": device.entry.find("pf_active"),
        "tf": device.entry.find("tf"),
        "magnetics": device.entry.find("magnetics"),
    }

    # Equilibrium

    R0 = eqdsk.find("vacuum_toroidal_field.r0", None)
    psi_axis = eqdsk.find("global_quantities.psi_axis", None)
    psi_boundary = eqdsk.find("global_quantities.psi_boundary", None)
    noise = 0.2  # np.random.random(bs_r_nrom.shape)*0.1
    configure["equilibrium"] = {
        "code": {"name": "dummy"},
        "time_slice": {
            "vacuum_toroidal_field": eqdsk.find("vacuum_toroidal_field", {}),
            "profiles_1d": eqdsk.find("profiles_1d"),
            "profiles_2d": {
                "psi": eqdsk.find("profiles_2d.psi")*TWOPI,
                "grid_type": "rectangular",
                "grid_index": 1,
                "grid": {
                    "dim1": eqdsk.find("profiles_2d.grid.dim1"),
                    "dim2": eqdsk.find("profiles_2d.grid.dim2"),
                }
            },
            "boundary_separatrix": eqdsk.find("boundary"),
            "coordinate_system": {"psi_norm": {"axis": 0.0, "boundary": 0.995, "npoints": 128}}
            # "coordinate_system": {"psi_norm": baseline["Fp"].values[:-1]}
        }}

    # Core profile

    b_Te = Function(bs_r_nrom, baseline["TE"].values*1000)
    b_Ti = Function(bs_r_nrom, baseline["TI"].values*1000)

    b_ne = Function(bs_r_nrom, baseline["NE"].values*1.0e19)
    b_nHe = Function(bs_r_nrom, baseline["Nalf"].values*1.0e19)
    # nDT = Function(bs_r_nrom, baseline["Nd+t"].values*1.0e19)
    b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_nrom, baseline["Zeff"].values)

    configure["core_profiles"] = {
        "profiles_1d": {
            "electrons": {**atoms["e"], "density":        b_ne*noise,   "temperature":  b_Te*noise, },
            "ion": [
                {**atoms["D"],          "density":   0.5*b_nDT*noise,   "temperature":  b_Ti*noise, },
                {**atoms["T"],          "density":   0.5*b_nDT*noise,   "temperature":  b_Ti*noise, },
                {**atoms["He"],         "density":             b_nHe,   "temperature":        b_Ti, },
                {**atoms["Be"],         "density":         0.02*b_ne,   "temperature":        b_Ti, },
                {**atoms["Ar"],         "density":       0.0012*b_ne,   "temperature":        b_Ti, },
            ]}}

    # Core Transport
    conductivity_parallel = Function(bs_r_nrom, baseline["Joh"].values*1.0e6 / baseline["U"].values *
                                     (2.0*constants.pi * R0))
    r_ped = 0.96  # np.sqrt(0.88)
    Cped = 0.2
    Ccore = 0.4
    chi = PiecewiseFunction([0, r_ped, 1.0],  [lambda x: 1.1*Ccore*(1.0 + 3*(x**2)), lambda x: Cped])
    chi_e = PiecewiseFunction([0, r_ped, 1.0], [lambda x:0.6 * Ccore*(1.0 + 3*(x**2)), lambda x: 1.5 * Cped])

    # D = Function(
    #     [lambda r:r < r_ped, lambda r:r >= r_ped],
    #     [lambda x:  0.5+(x**4), lambda x: 0.11])

    D = PiecewiseFunction([0, r_ped, 1.0],  [lambda x:0.1 * Ccore*(1.0 + 3*(x**2)), lambda x: 0.2*Cped])
    v_pinch = Function([0, r_ped, 1.0], lambda x: 0.4 * D(x) * x / R0)   # FIXME: The coefficient 0.4 is a guess.
    v_pinch_T = Function([0, r_ped, 1.0], lambda x: 0.1*chi_e(x) * x / R0)

    configure["core_transport"] = {
        "model": [
            {
                "code": {"name": "dummy"},
                "profiles_1d": {
                    "conductivity_parallel": conductivity_parallel,
                    "electrons": {
                        **atoms["e"],
                        "particles":   {"d": D, "v": -v_pinch},
                        "energy":      {"d": chi_e, "v": -v_pinch_T},
                    },
                    "ion": [
                        {
                            **atoms["D"],
                            "particles":{"d":  D, "v": v_pinch},
                            "energy": {"d": chi, "v": v_pinch_T},
                        },
                        {
                            **atoms["T"],
                            "particles":{"d":  D, "v": v_pinch},
                            "energy": {"d": chi, "v": v_pinch_T},
                        },
                        {
                            **atoms["He"],
                            "particles":{"d": 0.1 * (chi+chi_e), "v": 0},
                            "energy": {"d": 0.5*chi, "v": 0}, }
                    ]}
            },

            # {"code": {"name": "neoclassical"}},
            # {"code": {"name": "spitzer"}},
        ]}

    S = Function(lambda x: 1e21 * np.exp(15.0*(x**2-1.0)))

    Qe = Function(bs_r_nrom,
                  (baseline["Poh"].values
                   + baseline["Pdte"].values
                   + baseline["Paux"].values
                   - baseline["Peic"].values
                   - baseline["Prad"].values
                   - baseline["Pneu"].values
                   )*1e6/constants.electron_volt)

    Q_DT = Function(bs_r_nrom,
                    (baseline["Peic"].values
                     + baseline["Pdti"].values
                     + baseline["Pibm"].values
                     )*1e6/constants.electron_volt)

    Q_He = Function(bs_r_nrom,
                    (- baseline["Pdti"].values
                     - baseline["Pdte"].values
                     )*1e6/constants.electron_volt)

    impurities = ['He', 'Be', 'Ar']
    # Core Source
    configure["core_sources"] = {
        "source": [
            {
                "code": {"name": "dummy"},
                "profiles_1d": {
                    "j_parallel": Function(bs_r_nrom, baseline["Jtot"].values*1e6),
                    "electrons":{**atoms["e"],  "particles": S,         "energy": Qe},
                    "ion": [
                        {**atoms["D"],          "particles":S*0.5,      "energy":Q_DT*0.5},
                        {**atoms["T"],          "particles":S*0.5,      "energy":Q_DT*0.5},
                        {**atoms["He"],         "particles":0,          "energy":Q_He}
                    ]
                }},
            # {"code": {"name": "q_ei"}, }
            # {"code": {"name": "bootstrap_current"}},
        ]}

    #  TransportSolver
    configure["transport_solver"] = {
        "code": {"name": "bvp_solver2"},
        "boundary_conditions_1d": {
            "current": {"identifier": {"index": 1}, "value": [0.995*(psi_boundary-psi_axis)+psi_axis]},
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
        }}

    ###################################################################################################
    # Initialize
    tok = Tokamak(configure)

    # logger.debug(configure)

    ###################################################################################################
    # Plot profiles

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

        fpol = Function(eqdsk.find('profiles_1d.psi_norm', None), eqdsk.get('profiles_1d.f', None))

        # ffprime = fpol*fpol.derivative()/(psi_boundary-psi_axis)

        plot_profiles(
            [
                [
                    (Function(bs_psi_norm, baseline["q"].values), r"astra",  r"$q [-]$", {"marker": "+"}),
                    (Function(eqdsk.get('profiles_1d.psi_norm', None), eqdsk.get('profiles_1d.q', None)), "eqdsk"),
                    (magnetic_surface.q,  r"$fytok$", r"$[Wb]$"),
                    (magnetic_surface.dphi_dpsi,  r"$\frac{d\phi}{d\psi}$", r"$[Wb]$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["rho"].values), r"astra", r"$\rho_{tor}[m]$",  {"marker": "+"}),
                    (magnetic_surface.rho_tor,  r"$\rho$", r"$[m]$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["x"].values),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", {"marker": "+"}),
                    (magnetic_surface.rho_tor_norm,  r"$\bar{\rho}$", r"$[-]$"),
                ],

                [
                    (Function(bs_psi_norm, 4*(constants.pi**2)*magnetic_surface.vacuum_toroidal_field.r0 * baseline["rho"].values),
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho$",  {"marker": "+"}),
                    (magnetic_surface.dvolume_drho_tor, r"$dV/d\rho_{tor}$", r"$[m^2]$"),
                ],

                (magnetic_surface.dvolume_dpsi, r"$\frac{dV}{d\psi}$"),

                [
                    (magnetic_surface.volume, r"$V$  from $\psi$"),
                    # (magnetic_surface.volume1, r"$V$ from $\rho_{tor}$"),
                ],


                [
                    (np.abs(fpol), "eqdsk"),
                    (magnetic_surface.fpol,  r"$F_{pol}$", r"$[Wb]$"),
                ],


                (magnetic_surface.psi,  r"$\psi$", r"$[Wb]$"),
                (magnetic_surface.phi,  r"$\phi$", r"$[Wb]$"),
                (magnetic_surface.psi_norm,  r"$\bar{\psi}$", r"$[-]$"),

                (magnetic_surface.dpsi_drho_tor, r"$\frac{d\psi}{d\rho_{tor}}$", "", {"marker": "."}),

                # (magnetic_surface.drho_tor_dpsi, r"$\frac{d\rho_{tor}}{d\psi}$", "", {"marker": "."}),
                [
                    (magnetic_surface.dpsi_drho_tor,  r"$\frac{d\psi}{d\rho_{tor}}$"),
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
                    (Function(bs_psi_norm, baseline["q"].values),            r"astra",  r"$q [-]$", {"marker": "+"}),
                    (eq_profile.q,                                      r"fytok",  r"$q [-]$"),
                    (eq_profile.dphi_dpsi,                        r"$\frac{d\phi}{d\psi}$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["rho"].values), r"astra", r"$\rho_{tor}[m]$",  {"marker": "+"}),
                    (eq_profile.rho_tor,                                      r"fytok",    r"$\rho_{tor}[m]$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["x"].values),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", {"marker": "+"}),
                    (eq_profile.rho_tor_norm,                        r"fytok"),
                ],

                [
                    (Function(bs_psi_norm, baseline["shif"].values),
                     r"astra", "$\Delta$ shafranov \n shift $[m]$ ", {"marker": "+"}),
                    (eq_profile.geometric_axis.r-tok.equilibrium.time_slice.vacuum_toroidal_field.r0,
                     r"fytok", "shafranov \n shift $\Delta [m]$ "),
                ],
                [
                    (Function(bs_psi_norm, baseline["k"].values),
                     r"astra", r"$elongation[-]$", {"marker": "+"}),
                    (eq_profile.elongation,                                 r"fytok", r"$elongation[-]$"),
                ],
                [
                    (4*(constants.pi**2)*tok.equilibrium.time_slice.vacuum_toroidal_field.r0*tok.equilibrium.time_slice.profiles_1d.rho_tor,
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho , dV/d\rho$"),
                    (tok.equilibrium.time_slice.profiles_1d.dvolume_drho_tor,   r"$V^{\prime}$", r"$dV/d\rho$"),
                ],
                [
                    (Function(bs_psi_norm, baseline["Jtot"].values*1e6),   r"astra",
                     r"$j_{\parallel} [A\cdot m^{-2}]$", {"marker": "+"}),
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

    if True:  # CoreProfile

        core_profile = tok.core_profiles.profiles_1d

        plot_profiles(
            [
                [
                    (b_ne,  "astra",    r"$n_{e} [m \cdot s^{-3}]$"),
                    (core_profile.electrons.density,             r"$e$", ),
                    *[(ion.density,          f"${ion.label}$") for ion in core_profile.ion],

                ],
                [
                    (b_Te,    r"astra $T_i$",       r"$T [eV]$"),
                    (b_Ti,    r"astra $T_i$",       r"$T_{i} [eV]$"),
                    (core_profile.electrons.temperature,       r"$e$", r"T $[eV]$"),
                    *[(ion.temperature,      f"${ion.label}$") for ion in core_profile.ion],
                ],

                [
                    (Function(bs_r_nrom, baseline["Zeff"].values),       r"astra", r"$Z_{eff}  [-]$"),
                    (core_profile.zeff,                                   r"$fytok$"),
                ],
                (core_profile.grid.psi,                                    r"$\psi$"),
            ],
            x_axis=([0, 1.0],                                  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_profiles.svg", transparent=True)

    if True:  # CoreTransport
        # core_transport1d_nc = tok.core_transport.model[{"code.name": "neoclassical"}].profiles_1d
        # core_transport1d_dummy = tok.core_transport.model[{"code.name": "dummy"}].profiles_1d

        core_transport1d = tok.core_transport.model.combine.profiles_1d

        plot_profiles(
            [
                # [
                #     (core_transport1d.electrons.particles.flux,                            "e",  r"$\Gamma$"),
                #     *[(core_transport1d.ion[{"label": ion.label}].particles.flux,
                #        f"{ion.label}", f"$\Gamma$") for ion in core_profile.ion],
                # ],
                # [
                #     (core_transport1d.electrons.particles.d,                                  "e",     r"$D$"),
                #     *[(core_transport1d.ion[{"label": ion.label}].particles.d,     f"{ion.label}", r"$D$")
                #       for ion in core_profile.ion],
                # ],
                # [
                #     (core_transport1d.electrons.particles.v,                        "e",              r"$v$"),
                #     *[(core_transport1d.ion[{"label": ion.label}].particles.v, f"{ion.label}",  f"$v$")
                #       for ion in core_profile.ion],
                # ],

                [
                    (Function(bs_r_nrom, baseline["Xi"].values),          r"astra", r"$\chi_{i}$", {"marker": "+"}),
                    *[(ion.energy.d,  f"{ion.label}", r"$\chi_{i}$") for ion in core_transport1d.ion],
                ],

                # [
                #     (Function(bs_r_nrom,  np.log(baseline["XiNC"].values)),  "astra", r"$ln \chi_{i,nc}$", {"marker": "+"}),
                #     # * [(np.log(core_transport1d_nc.ion[{"label": label}].energy.d),   f"${label}$", r"$ln \chi_{i,nc}$")
                #     #     for label in ("H", "D", "He")],
                # ],
                [
                    (Function(bs_r_nrom, baseline["He"].values), "astra", r"$\chi_{e}$"),
                    (core_transport1d.electrons.energy.d,  "fytok", r"$\chi_{e}$"),
                ],

                # [
                #     (core_transport1d.electrons.energy.v,           r"$electron$", r"V pinch $[m\cdot s^{-1}]$"),
                #     *[(core_transport1d.ion[{"label": ion.label}].energy.v,  f"${ion.label}$",)
                #       for ion in core_profile.ion],
                # ],

                [
                    (Function(bs_r_nrom, baseline["Joh"].values*1.0e6 / baseline["U"].values *
                              (2.0*constants.pi * tok.equilibrium.time_slice.vacuum_toroidal_field.r0)),     r"astra", r"$\sigma_{\parallel}$", {"marker": "+"}),
                    # (tok.core_transport.model[{"code.name": "spitzer"}].profiles_1d.conductivity_parallel,
                    #  "spitzer", r"$\sigma_{\parallel}$"),
                    (core_transport1d.conductivity_parallel,  r"fytok"),
                ],

                # (core_transport1d.e_field_radial,                                             r"$E_{radial}$"),
                # (tok.equilibrium.time_slice[-1].profiles_1d.trapped_fraction(
                # core_transport.model[0].profiles_1d[-1].grid_v.psi_norm),      r"trapped"),
                # (core_profile.electrons.pressure,                                                  r"$p_{e}$"),
            ],
            x_axis=([0, 1.0],   r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            # index_slice=slice(10, 110, 1),
            title=tok.core_transport.model[0].identifier.name,
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_transport.svg", transparent=True)

    if True:  # CoreSources
        core_source_1d = tok.core_sources.source.combine.profiles_1d

        plot_profiles(
            [
                [
                    (Function(bs_r_nrom, baseline["Jtot"].values*1e6),  "astra",
                     r"$J_{\parallel} [A\cdot m^{-2}]$", {"marker": "+"}),
                    (tok.core_sources.source[{"code.name": "dummy"}].profiles_1d.j_parallel,
                     "dummy", r"$J_{\parallel} [A\cdot m^{-2}]$"),
                    (core_source_1d.j_parallel,                     "fytok", r"$J_{\parallel} [MA\cdot m^{-2}]$"),
                ],
                (core_source_1d.electrons.particles,       "fytok", r"$S_{e} [ m^{-3} s^-1]$"),
                (core_source_1d.electrons.energy,          "fytok", r"$Q_{e}$"),
                [
                    * [(ion.density,   f"${ion.label}$") for ion in core_profile.ion if ion.label not in impurities],
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
    # TransportSolver
    if True:

        core_profile = tok.core_profiles.profiles_1d

        tok.update(enable_ion_particle_solver=False,
                   max_nodes=500, tolerance=1.0e-4,
                   impurities=impurities,
                   bvp_rms_mask=[1.0/128, r_ped])

        plot_profiles(
            [

                ######################################################################
                # psi ,current
                [
                    (Function(bs_r_nrom, (bs_psi_norm*(psi_boundary-psi_axis)+psi_axis)),
                     r"astra", r"$\psi [Wb]$", {"marker": "+"}),
                    (core_profile["psi"],  r"fytok", r"$\psi  [Wb]$"),
                    # (core_profile["psi_error"], r"residual", r"",  {"color": "red", "linestyle": "dashed"}),
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
                # [
                #     (b_ne, r"astra", r"$n_e [m^{-3}]$",  {"marker": "+"}),
                #     (core_profile.electrons.density, r"fytok", r"$n_e [ m^{-3}]$"),
                #     (core_profile.electrons["density_error"], r"rms residuals ",
                #         r"$n_e [ m^{-3}]$",  {"color": "red", "linestyle": "dashed"}),
                # ],
                # [
                #     (b_nDT/2,    r"astra $T_D$", r"$n_i [m^-3]$", {"marker": '+'}),
                #     * [(ion.density,   f"${ion.label}$") for ion in core_profile.ion],
                # ],
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
                # electron energy
                # [
                #     (b_Te, r" (astra)", r"$T_e [eV]$",  {"marker": "+"}),
                #     (core_profile.electrons.temperature, r" (fytok)  ", r"$ [eV]$"),
                # ],
                # (core_profile.electrons["temperature_error"], r"rms_residuals",
                #  r"$[eV]$",  {"color": "red", "linestyle": "dashed"}),


                # (core_source_1d.electrons.energy,          r"$Q_{e}$"),


                # (core_profile.electrons["temperature_prime"], r"$T^{\prime}$", r"$T_e [eV m^-1]$"),
                # (core_profile.electrons["heat_flux_prime"], r"$q^{\prime}$"),

                # [
                #     (core_profile.electrons["heat_flux"],      r"Total",  "Heat flux", {"color": "green", }),
                #     (core_profile.electrons["diff_flux_T"],    r"Diffusive",  "", {"color": "black", }),
                #     (core_profile.electrons["conv_flux_T"],    r"Convective", "",  {"color": "blue", }),

                #     (core_profile.electrons["heat_flux"]
                #      - core_profile.electrons["diff_flux_T"]
                #      - core_profile.electrons["conv_flux_T"],     r"residual",  "", {"color": "red", "linestyle": "dashed"}),
                # ],


                ######################################################################
                # [
                #     (b_Ti,    r"astra $T_i$",       r"$T_{i} [eV]$", {"marker": '+'}),
                #     * [(ion.temperature,          f"${ion.label}$", r"$T_i [eV]$")
                #         for ion in core_profile.ion if ion.label not in impurities],
                # ]
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
                # (core_profile.electrons["g"],  r"electron.f", r"$g$"),

                # (core_profile.e_field.parallel,                    r"fytok",   r"$E_{\parallel} [V\cdot m^{-1}]$ "),
            ],
            # x_axis=(rho_tor_norm,                             r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            x_axis=(core_profile.electrons.temperature.x_axis,  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
            title="Result of TransportSolver",
            # index_slice=slice(0, 200, 1),
            grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_profile_result.svg", transparent=True)

    logger.info("====== DONE ========")
