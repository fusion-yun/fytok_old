import collections
import pathlib
import sys
from typing import Union

import numpy as np
import pandas as pd
import scipy.constants
from spdm.data.Entry import Entry
from spdm.data.File import File
from spdm.data.Function import Function, PiecewiseFunction, function_like
from spdm.numlib import constants, np
from spdm.numlib.smooth import rms_residual, smooth_1d
from spdm.util.logger import logger

from .Atoms import atoms


def load_core_profiles(profiles: Union[str, pathlib.Path]):
    if isinstance(profiles, (str, pathlib.Path)):
        profiles = pd.read_csv(profiles, sep='\t')
    # else:  # if not isinstance(profiles, collections.abc.Mapping):
    #     raise TypeError(type(profiles))

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

    return {
        "rho_tor": profiles["rho"].values,
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
        # "j_bootstrap": Function(bs_r_norm, profiles["Jbs"].values*1.0e6),
        "j_total": Function(bs_r_norm, profiles["Jtot"].values*1.0e6),

        "XiNC": Function(bs_r_norm, profiles["XiNC"].values),

    }


def load_core_transport(profiles: Union[str, pathlib.Path], R0: float, B0: float):
    if isinstance(profiles, (str, pathlib.Path)):
        profiles = pd.read_csv(profiles, sep='\t')

    bs_psi_norm = profiles["Fp"].values

    bs_r_norm = profiles["x"].values

    # Core profiles
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    # Core Transport
    conductivity_parallel = Function(bs_r_norm, profiles["Joh"].values*1.0e6 / profiles["U"].values *
                                     (2.0*constants.pi * R0))

    Cped = 0.17
    Ccore = 0.4
    # Function(bs_r_norm, profiles["Xi"].values)  Cped = 0.2
    chi = PiecewiseFunction([0, r_ped, 1.0],  [lambda x: Ccore*(1.0 + 3*(x**2)), lambda x: Cped])
    chi_e = PiecewiseFunction([0, r_ped, 1.0],  [lambda x: 0.5 * Ccore*(1.0 + 3*(x**2)), lambda x: Cped])

    D = 0.1*(chi+chi_e)

    v_pinch_ne = Function([0, r_ped, 1.0], lambda x: -0.6 * D(x) * x / R0)
    v_pinch_Te = Function([0, r_ped, 1.0], lambda x:  2.5 * chi_e(x) * x / R0)

    v_pinch_ni = Function([0, r_ped, 1.0], lambda x:  D(x) * x / R0)
    v_pinch_Ti = Function([0, r_ped, 1.0], lambda x:  chi(x) * x / R0)

    conductivity_parallel = Function(
        bs_r_norm, profiles["Joh"].values*1.0e6 / profiles["U"].values * (2.0*constants.pi * R0))

    return {
        "conductivity_parallel": conductivity_parallel,
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


def load_core_source(profiles: Union[str, pathlib.Path], R0: float, B0: float, **kwargs):
    if isinstance(profiles, (str, pathlib.Path)):
        profiles = pd.read_csv(profiles, sep='\t', **kwargs)

    bs_r_norm = profiles["x"].values

    S = Function(lambda x: 9e20 * np.exp(15.0*(x**2-1.0)))

    Q_e = Function(bs_r_norm,
                   (profiles["Poh"].values
                    + profiles["Pdte"].values
                    + profiles["Paux"].values
                    - profiles["Peic"].values
                    - profiles["Prad"].values
                    # - profiles["Pneu"].values
                    )*1e6/constants.electron_volt)

    Q_DT = Function(bs_r_norm,
                    (profiles["Peic"].values
                     + profiles["Pdti"].values
                     + profiles["Pibm"].values
                     )*1e6/constants.electron_volt)

    Q_He = Function(bs_r_norm,
                    (- profiles["Pdti"].values
                     - profiles["Pdte"].values
                     )*1e6/constants.electron_volt)

    # Core Source
    return {
        "j_parallel": Function(
            bs_r_norm,
            (
                # profiles["Jtot"].values
                profiles["Joh"].values
                # + profiles["Jbs"].values
                + profiles["Jnb"].values
                + profiles["Jrf"].values
            ) * 1e6),
        "electrons": {**atoms["e"],  "particles": S, "energy": Q_e},
        "ion": [
            {**atoms["D"],          "particles":S*0.5,      "energy":Q_DT*0.5},
            {**atoms["T"],          "particles":S*0.5,      "energy":Q_DT*0.5},
            {**atoms["He"],         "particles":0,          "energy":Q_He}
        ]}


def load_equilibrium(eqdsk):
    if not isinstance(eqdsk, Entry):
        eqdsk = File(eqdsk, format="geqdsk").entry

    R0 = eqdsk.get("vacuum_toroidal_field.r0")
    B0 = eqdsk.get("vacuum_toroidal_field.b0")

    return {
        "vacuum_toroidal_field": {"b0": B0, "r0": R0, },
        "global_quantities": eqdsk.get("global_quantities"),
        "profiles_1d": eqdsk.get("profiles_1d"),
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
