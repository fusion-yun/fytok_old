import sys
from math import log
from operator import contains, eq
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreTransport import CoreTransport

import numpy as np
import pandas as pd
import scipy.constants
from fytok.common.Atoms import atoms
from fytok.Tokamak import TWOPI, Tokamak
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.File import File
from spdm.data.Function import Function, PiecewiseFunction
from spdm.numlib import constants, np
from spdm.numlib.smooth import rms_residual, smooth_1d
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles, sp_figure

sys.path.append("/home/salmon/workspace/fytok/phys_modules/transport/core_transport/")

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

    equilibrium = Equilibrium(c_equilibrium)

    radial_grid = equilibrium.time_slice.radial_grid.remesh(np.linspace(0, 1.0, 128), "rho_tor_norm")

    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    b_Te = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["TE"].values, i_end=i_ped-10, window_len=21)*1000)
    b_Ti = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["TI"].values, i_end=i_ped-10, window_len=21)*1000)
    b_ne = Function(bs_r_norm, smooth_1d(bs_r_norm, baseline["NE"].values, i_end=i_ped-10, window_len=21)*1.0e19)

    b_nHe = Function(bs_r_norm, baseline["Nalf"].values*1.0e19)
    # nDT = Function(bs_r_norm, baseline["Nd+t"].values*1.0e19)
    b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_norm, baseline["Zeff"].values)

    c_core_profiles = {
        "profiles_1d": {
            "electrons": {**atoms["e"], "density":              b_ne,   "temperature":        b_Te, },
            "ion": [
                {**atoms["D"],          "density":         0.5*b_nDT,   "temperature":        b_Ti, },
                {**atoms["T"],          "density":         0.5*b_nDT,   "temperature":        b_Ti, },
                {**atoms["He"],         "density":             b_nHe,   "temperature":        b_Ti, "is_impurity":True},
                {**atoms["Be"],         "density":         0.02*b_ne,   "temperature":        b_Ti, "is_impurity":True},
                {**atoms["Ar"],         "density":       0.0012*b_ne,   "temperature":        b_Ti, "is_impurity":True},
            ]}}

    core_profiles = CoreProfiles(c_core_profiles, grid=radial_grid)

    c_core_transport = {
        "model": [

            # {"code": {"name": "neoclassical"}},
            {"code": {"name": "glf23"}},
            # {"code": {"name": "nclass"}},
            # {"code": {"name": "spitzer"}},
        ]}

    core_transport = CoreTransport(c_core_transport, grid=radial_grid)

    core_transport.refresh(equilibrium=equilibrium, core_profiles=core_profiles)

    logger.debug(np.asarray(core_transport.model[0].profiles_1d.electrons.energy.d))
    logger.debug(np.asarray(core_transport.model[0].profiles_1d.electrons.particles.d))
