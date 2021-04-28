import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants
from fytok.Tokamak import Tokamak
from spdm.data.Collection import Collection
from spdm.data.File import File
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles
import sys


if __name__ == "__main__":
    sys.path.append("/home/salmon/workspace/fytok/phys_modules/")
    import transport.nclass as nclass

    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml").entry

    equilibrium = File(
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/15MA inductive - burn/Increased domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16VVHR.txt",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        format="geqdsk").entry

    profile = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')

    eq_slice = equilibrium.time_slice[-1]
    logger.debug(type(eq_slice))
    tok = Tokamak({
        "radial_grid": {"axis": 64},
        "wall":  device.wall,
        "equilibrium": {
            "vacuum_toroidal_field": equilibrium.vacuum_toroidal_field,
            "time_slice": {
                "global_quantities": eq_slice.global_quantities,
                "profiles_1d": eq_slice.profiles_1d,
                "profiles_2d": eq_slice.profiles_2d,
                "coordinate_system": {"grid": {"dim1": 64, "dim2": 512}}
            }
        },
        "core_profiles": {
            "profiles_1d": {
                "electrons": {
                    "label": "electrons",
                    "density":     1e19,
                    "temperature": lambda x: (1-x**2)**2
                },
                "ion": [
                    {

                        "label": "H^+",
                        "z_ion": 1,
                        "neutral_index": 1,
                        "element": [{"a": 1, "z_n": 1, "atoms_n": 1}],
                        "density":     0.5e19,
                        "temperature": lambda x: (1-x**2)**2
                    },
                    {
                        "label": "D^+",
                        "z_ion": 1,
                        "element": [{"a": 2, "z_n": 1, "atoms_n": 1}],
                        "neutral_index": 2,
                        "density":     0.5e19,
                        "temperature": lambda x: (1-x**2)**2
                    }
                ],
                "conductivity_parallel": 1.0,
                "psi":   1.0,
            }
        }
    })

    eq = tok.equilibrium.time_slice[-1]

    logger.debug(type(eq))

    plot_profiles(
        [
            (eq.profiles_1d.dpressure_dpsi,         r"$dP/d\psi$"),
            [
                (eq.profiles_1d.ffprime,            r"$ff^{\prime}$"),
                (Function(equilibrium.profiles_1d.psi_norm,
                          equilibrium.profiles_1d.f_df_dpsi),    r"$ff^{\prime}_{0}$"),
            ],
            [
                (eq.profiles_1d.fpol,               r"$fpol$"),
                (Function(equilibrium.profiles_1d.psi_norm,
                          np.abs(equilibrium.profiles_1d.f)),    r"$\left|f_{pol0}\right|$"),
            ],
            [
                (eq.profiles_1d.q,                  r"$q$"),
                # (eq.profiles_1d.dphi_dpsi,                    r"$\frac{d\phi}{d\psi}$"),
                # (Function(equilibrium.profiles_1d.psi_norm, equilibrium.profiles_1d.q), r"$q_0$"),
                (Function(profile["Fp"].values, profile["q"].values),             r"$q^{\star}$"),
            ],
            [
                (eq.profiles_1d.rho_tor,           r"$\rho_{tor}$"),
                (Function(profile["Fp"].values, profile["rho"].values),             r"$\rho_{tor}^{\star}$"),
                #     # (eq.profiles_1d.dvolume_drho_tor / ((scipy.constants.pi**2) * 4.0 * eq.vacuum_toroidal_field.r0),
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
            #     (Function(equilibrium.profiles_1d.psi_norm, equilibrium.profiles_1d.q), r"$q_0$"),
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
            # (tok.core_profiles.electrons.temperature, r"$T_{e}$"),
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
        grid=True, fontsize=16
    ) .savefig("/home/salmon/workspace/output/equilibrium.svg", transparent=True)
if False:
    plot_profiles(
        [
            (tok.core_profiles.profiles_1d.electrons.density,       r"$n_e$"),
            (tok.core_profiles.profiles_1d.electrons.temperature,   r"$T_e$"),
            (tok.core_profiles.profiles_1d.ion[0].density,
             f"$n_{{{tok.core_profiles.profiles_1d.ion[0].label}}}$"),
            (tok.core_profiles.profiles_1d.ion[0].temperature,
             f"$T_{{{tok.core_profiles.profiles_1d.ion[0].label}}}$"),
        ],
        x_axis=(tok.core_profiles.profiles_1d.grid.rho_tor_norm,    r"$\sqrt{\Phi/\Phi_{bdry}}$"),  # x axis,
        grid=True, fontsize=10
    ) .savefig("/home/salmon/workspace/output/core_profile.svg", transparent=True)

if False:
    core_transport = nclass.transport_nclass(tok.equilibrium, tok.core_profiles)

    logger.debug(core_transport.identifier)

    plot_profiles(
        [
            [
                (core_transport.profiles_1d.electrons.particles.flux, r"$\Gamma_e$"),
                *[(ion.particles.flux,        f"$\Gamma_{{{ion.label}}}$") for ion in core_transport.profiles_1d.ion],
            ],
            [
                (core_transport.profiles_1d.electrons.particles.d,   r"$D_e$"),
                *[(ion.particles.d,           f"$D_{{{ion.label}}}$") for ion in core_transport.profiles_1d.ion],
            ],
            [
                (core_transport.profiles_1d.electrons.particles.v,   r"$v_e$"),
                * [(ion.particles.v,           f"$v_{{{ion.label}}}$") for ion in core_transport.profiles_1d.ion],
            ],
            [
                (core_transport.profiles_1d.electrons.energy.flux,   r"$q_e$"),
                *[(ion.energy.flux,        f"$q_{{{ion.label}}}$") for ion in core_transport.profiles_1d.ion],
            ],
            [
                (core_transport.profiles_1d.electrons.energy.d,      r"$\chi_e$"),
                *[(ion.energy.d,           f"$\chi_{{{ion.label}}}$") for ion in core_transport.profiles_1d.ion],
            ],
            [
                (core_transport.profiles_1d.electrons.energy.v,      r"$v_{Te}$"),
                *[(ion.energy.v,           f"$v_{{T,{ion.label}}}$") for ion in core_transport.profiles_1d.ion],
            ]
        ],
        x_axis=(core_transport.profiles_1d.grid_v.rho_tor_norm,   r"$\sqrt{\Phi/\Phi_{bdry}}$"),  # x axis,
        annotation=core_transport.identifier.name,
        grid=True, fontsize=10) .savefig("/home/salmon/workspace/output/core_transport.svg", transparent=True)
