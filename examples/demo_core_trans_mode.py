import sys
from math import log
from operator import contains, eq
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreTransport import CoreTransport

import numpy as np
import pandas as pd
import scipy.constants
from fytok.constants.Atoms import atoms
from fytok.Tokamak import TWOPI, Tokamak
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.File import File
from spdm.data import Function, PiecewiseFunction
from scipy import constants
from spdm.numlib.smooth import rms_residual, smooth_1d
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles, sp_figure

if __name__ == "__main__":
    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml")

    baseline = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')
    bs_psi_norm = baseline["Fp"].values
    bs_r_norm = baseline["x"].values

    eqdsk = File(
        # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt",
        # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
        format="geqdsk").entry
    # Equilibrium

    R0 = eqdsk.get("vacuum_toroidal_field.r0")
    B0 = eqdsk.get("vacuum_toroidal_field.b0")
    psi_axis = eqdsk.get("global_quantities.psi_axis")
    psi_boundary = eqdsk.get("global_quantities.psi_boundary")
    noise = 1  # np.random.random(bs_r_norm.shape)*0.1

    c_equilibrium = {
        "code": {"name": "dummy"},
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
    }

    equilibrium = Equilibrium(c_equilibrium)

    radial_grid = equilibrium.radial_grid.remesh(np.linspace(0.04, 0.9, 101), "rho_tor_norm")

    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    b_Te = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["TE"].values, i_end=i_ped-10, window_len=21)*1000)
    b_Ti = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["TI"].values, i_end=i_ped-10, window_len=21)*1000)
    b_ne = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["NE"].values, i_end=i_ped-10, window_len=21)*1.0e19)
    b_nDT = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["Nd+t"].values, i_end=i_ped-10, window_len=21)*1.0e19)
    b_nHe = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["Nalf"].values, i_end=i_ped-10, window_len=21)*1.0e19)

    # b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_norm, baseline["Zeff"].values)

    c_core_profiles = {
        "profiles_1d": {
            "electrons": {**atoms["e"], "density":              b_ne,   "temperature":        b_Te, },
            "ion": [
                {**atoms["D"],          "density":         0.5*b_nDT,   "temperature":        b_Ti, },
                {**atoms["T"],          "density":         0.5*b_nDT,   "temperature":        b_Ti, },
                # {**atoms["He"],         "density":             b_nHe,   "temperature":        b_Ti, "is_impurity":False},
                # {**atoms["Be"],         "density":         0.02*b_ne,   "temperature":        b_Ti, "is_impurity":True},
                # {**atoms["Ar"],         "density":       0.0012*b_ne,   "temperature":        b_Ti, "is_impurity":True},
            ]}}

    core_profiles = CoreProfiles(c_core_profiles, grid=radial_grid)

    c_core_transport = {
        "model": [

            {"code": {"name": "neoclassical"}},
            {"code": {"name": "spitzer"}},
            # {"code": {"name": "glf23"}},
            # {"code": {"name": "nclass"}},
        ]}

    core_transport = CoreTransport(c_core_transport, grid=radial_grid)

    core_transport.refresh(equilibrium=equilibrium, core_profiles=core_profiles)

    plot_profiles(
        [
            (core_transport.model[0].profiles_1d.electrons.particles.d, "electron", r"$D_e$"),
            (core_transport.model[0].profiles_1d.electrons.energy.d, "electron", r"$\chi_e$"),

            (core_transport.model[0].profiles_1d.ion[0].particles.d, "ion", r"$D_i$"),
            (core_transport.model[0].profiles_1d.ion[0].energy.d, "ion", r"$\chi_i$"),

        ],
        # x_axis=(rho_tor_norm,                             r"$\sqrt{\Phi/\Phi_{bdry}}$"),
        x_axis=([0.05, 0.95],  r"$\sqrt{\Phi/\Phi_{bdry}}$"),
        title="Result of GLF23",
        grid=True, fontsize=10) \
        .savefig("/home/salmon/workspace/output/core_transport_glf23.svg", transparent=True)

    # rlti = np.asarray(core_transport.model[0].profiles_1d["debug_rlti"])
    # plot_profiles(
    #     [
    #         (np.asarray(core_transport.model[0].profiles_1d["debug_gamma"]), r"$\gamma$", r"$[-]$"),
    #         (np.asarray(core_transport.model[0].profiles_1d["debug_freq"]), "freq", r"$[-]$"),
    #         (np.asarray(core_transport.model[0].profiles_1d.electrons.energy.d),  r"$\chi_e$"),
    #         (np.asarray(core_transport.model[0].profiles_1d.ion[0].energy.d), r"$\chi_i$"),
    #         (np.asarray(core_transport.model[0].profiles_1d.ion[0].particles.d), r"$D_i$"),
    #         # (core_transport.model[0].profiles_1d["debug_kyf"], "debug_kyf", r"$[-]$"),
    #     ],
    #     x_axis=(rlti,  r"$a/L_T$"),
    #     title="Result of GLF23",
    #     grid=True, fontsize=10) \
    #     .savefig("/home/salmon/workspace/output/core_transport_rlti.svg", transparent=True)

    # xkyf_k = core_transport.model[0].profiles_1d.get("debug_xkyf_k")
    # gamma_k = core_transport.model[0].profiles_1d.get("debug_gamma_k")
    # freq_k = core_transport.model[0].profiles_1d.get("debug_freq_k")
    # diff_k = core_transport.model[0].profiles_1d.get("debug_diff_k")
    # chi_e_k = core_transport.model[0].profiles_1d.get("debug_chi_e_k")
    # chi_i_k = core_transport.model[0].profiles_1d.get("debug_chi_i_k")
    # num = xkyf_k.shape[0]
    # plot_profiles(
    #     [
    #         [(Function(xkyf_k[idx], gamma_k[idx]),
    #           f"$q={q[idx]:.2f}$", r"${\gamma}/{\left(c_{s}/a\right)}$") for idx in range(0, num, 20)],
    #         [(Function(xkyf_k[idx], diff_k[idx]),
    #           f"$q={q[idx]:.2f}$", r"$   {D_i}/{\left(c_{s}\rho_{s}^2/a\right)}$") for idx in range(0, num, 20)],
    #         [(Function(xkyf_k[idx], chi_e_k[idx]),
    #           f"$q={q[idx]:.2f}$", r"${\chi_e}/{\left(c_{s}\rho_{s}^2/a\right)}$") for idx in range(0, num, 20)],
    #         [(Function(xkyf_k[idx], chi_i_k[idx]),
    #           f"$q={q[idx]:.2f}$", r"${\chi_i}/{\left(c_{s}\rho_{s}^2/a\right)}$") for idx in range(0, num, 20)],
    #     ],
    #     x_axis=([0.02, 0.8],  r"$k_y \rho_s$"),
    #     title="Result of GLF23",
    #     grid=True, fontsize=10) \
    #     .savefig("/home/salmon/workspace/output/core_transport_k.svg", transparent=True)

    logger.debug("DONE")
