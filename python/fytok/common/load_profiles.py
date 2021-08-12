import collections
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.constants
from spdm.data.File import File
from spdm.data.Function import Function, function_like
from spdm.numlib import constants, np
from spdm.numlib.smooth import rms_residual, smooth_1d
from spdm.util.logger import logger

from .Atoms import atoms


def load_core_profiles(profiles='/home/salmon/workspace/data/15MA inductive - burn/profile.txt'):
    if isinstance(profiles, (str, pathlib.Path)):
        profiles = pd.read_csv(profiles, sep='\t')
    else:  # if not isinstance(profiles, collections.abc.Mapping):
        raise TypeError(type(profiles))

    bs_psi_norm = profiles["Fp"].values
    bs_r_norm = profiles["x"].values
    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    b_Te = Function(bs_r_norm, smooth_1d(bs_r_norm, profiles["TE"].values, i_end=i_ped-10, window_len=21)*1000)
    b_Ti = Function(bs_r_norm, smooth_1d(bs_r_norm, profiles["TI"].values, i_end=i_ped-10, window_len=21)*1000)
    b_ne = Function(bs_r_norm, smooth_1d(bs_r_norm, profiles["NE"].values, i_end=i_ped-10, window_len=21)*1.0e19)

    b_nHe = Function(bs_r_norm, smooth_1d(bs_r_norm, profiles["Nalf"].values, i_end=i_ped-10, window_len=21)*1.0e19)
    b_nDT = Function(bs_r_norm, smooth_1d(bs_r_norm, profiles["Nd+t"].values, i_end=i_ped-10, window_len=21)*1.0e19)
    b_nImp = Function(bs_r_norm, smooth_1d(bs_r_norm, profiles["Nz"].values, i_end=i_ped-10, window_len=21)*1.0e19)
    b_zeff = Function(bs_r_norm,   profiles["Zeff"].values)

    z_eff_star = b_zeff-(b_nDT+4*b_nHe)/b_ne
    z_imp = 1-(b_nDT+2*b_nHe)/b_ne
    b = -2*z_imp/(0.02+0.0012)
    c = (z_imp**2-0.02*z_eff_star)/0.0012/(0.02+0.0012)

    z_Ar = np.asarray((-b+np.sqrt(b**2-4*c))/2)
    z_Be = np.asarray((z_imp-0.0012*z_Ar)/0.02)
    # b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_norm, baseline["Zeff"].values)
    # e_parallel = baseline["U"].values / (2.0*constants.pi * R0)

    return {  # core profiles
        "profiles_1d": {
            "electrons": {**atoms["e"], "density":       b_ne,   "temperature": b_Te, },
            "ion": [
                {**atoms["D"],  "density":  0.5*b_nDT,  "temperature": b_Ti, },
                {**atoms["T"],  "density":  0.5*b_nDT,  "temperature": b_Ti, },
                {**atoms["He"], "density":      b_nHe,  "temperature": b_Ti},
                {**atoms["Be"], "density":  0.02*b_ne,  "temperature": b_Ti,
                 "z_ion_1d":Function(bs_r_norm, z_Be),  "is_impurity":True},
                {**atoms["Ar"], "density":0.0012*b_ne,  "temperature": b_Ti,
                 "z_ion_1d":Function(bs_r_norm, z_Ar),  "is_impurity":True},
            ],
            # "e_field": {"parallel": Function(bs_r_norm, e_parallel)},
            # "conductivity_parallel": Function(bs_r_norm, baseline["Joh"].values*1.0e6 / baseline["U"].values * (2.0*constants.pi * R0)),
            "zeff": Function(bs_r_norm, profiles["Zeff"].values),
            "vloop": Function(bs_r_norm, profiles["U"].values),
            "j_ohmic": Function(bs_r_norm, profiles["Joh"].values*1.0e6),
            "j_non_inductive": Function(bs_r_norm, profiles["Jnoh"].values*1.0e6),
            "j_bootstrap": Function(bs_r_norm, profiles["Jbs"].values*1.0e6),
            "j_total": Function(bs_r_norm, profiles["Jtot"].values*1.0e6),

            "XiNC": Function(bs_r_norm, profiles["XiNC"].values),

        }}


def load_equilibrium(eqdsk_path):
    device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml")
    # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
    # "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt",
    # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",

    eqdsk = File(eqdsk_path, format="geqdsk").entry

    R0 = eqdsk.get("vacuum_toroidal_field.r0")
    B0 = eqdsk.get("vacuum_toroidal_field.b0")
    psi_axis = eqdsk.get("global_quantities.psi_axis")
    psi_boundary = eqdsk.get("global_quantities.psi_boundary")

    return {
        "vacuum_toroidal_field": {
            "b0": eqdsk.get("vacuum_toroidal_field.b0"),
            "r0": eqdsk.get("vacuum_toroidal_field.r0"),
        },
        "global_quantities": eqdsk.get("global_quantities"),
        "profiles_1d":   eqdsk.get("profiles_1d"),
        "profiles_2d": {
            "psi": eqdsk.get("profiles_2d.psi"),
            "grid_type": "rectangular",
            "grid_index": 1,
            "grid": {
                "dim1": eqdsk.get("profiles_2d.grid.dim1"),
                "dim2": eqdsk.get("profiles_2d.grid.dim2"),
            }
        },
        "boundary_separatrix": eqdsk.get("boundary"),

        # "coordinate_system": {"psi_norm": baseline["Fp"].values[:-1]}
    }
