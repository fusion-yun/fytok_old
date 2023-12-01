import pathlib

import pathlib
import re
import scipy.constants
import numpy as np
import pandas as pd
from spdm.data.Expression import Piecewise, Variable
from spdm.data.File import File
from spdm.data.Entry import Entry
from spdm.numlib.smooth import smooth_1d

PI = scipy.constants.pi
TWOPI = scipy.constants.pi * 2.0


def load_core_profiles(profiles, grid):
    bs_r_norm = profiles["x"].values

    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm - r_ped))
    # fmt:off
    bs_psi_norm = profiles["Fp"].values
    # bs_psi = bs_psi_norm*(psi_boundary-psi_axis)+psi_axis

    b_Te =    smooth_1d( bs_r_norm, profiles["TE"].values,     i_end=i_ped-10, window_len=21)*1000
    b_Ti =    smooth_1d( bs_r_norm, profiles["TI"].values,     i_end=i_ped-10, window_len=21)*1000
    b_ne =    smooth_1d( bs_r_norm, profiles["NE"].values,     i_end=i_ped-10, window_len=21)*1.0e19
    b_nDT =   smooth_1d( bs_r_norm, profiles["Nd+t"].values,   i_end=i_ped-10, window_len=21)*1.0e19*0.5
    b_nHe =   smooth_1d( bs_r_norm, profiles["Nath"].values,   i_end=i_ped-10, window_len=21)*1.0e19
    b_nImp =  smooth_1d( bs_r_norm, profiles["Nz"].values,     i_end=i_ped-10, window_len=21)*1.0e19
    b_zeff = profiles["Zeff"].values
    # fmt:on

    z_eff_star = b_zeff - (b_nDT * 2.0 + 4 * b_nHe) / b_ne
    z_imp = 1 - (b_nDT * 2.0 + 2 * b_nHe) / b_ne
    b = -2 * z_imp / (0.02 + 0.0012)
    c = (z_imp**2 - 0.02 * z_eff_star) / 0.0012 / (0.02 + 0.0012)

    z_Ar = np.asarray((-b + np.sqrt(b**2 - 4 * c)) / 2)
    z_Be = np.asarray((z_imp - 0.0012 * z_Ar) / 0.02)
    # b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_norm, baseline["Zeff"].values)
    # e_parallel = baseline["U"].values / (TWOPI * R0)

    return {
        "time": 0.0,
        "grid": grid,
        "electrons": {
            "label": "e",
            "density_thermal": b_ne,
            "temperature": b_Te,
        },
        "ion": [
            {"label": "D", "density_thermal": b_nDT, "temperature": b_Ti},
            {"label": "T", "density_thermal": b_nDT, "temperature": b_Ti},
            {"label": "He", "density_thermal": b_nHe, "temperature": b_Ti},
            {"label": "Be", "density_thermal": 0.02 * b_ne, "temperature": b_Ti, "z_ion_1d": z_Be, "is_impurity": True},
            {"label": "Ar", "density_thermal": 0.0012 * b_ne, "temperature": b_Ti, "z_ion_1d": z_Ar, "is_impurity": True},
        ],
        # "e_field": {"parallel":  Function(bs_r_norm,e_parallel,)},
        # "conductivity_parallel": Function(bs_r_norm,baseline["Joh"].values*1.0e6 / baseline["U"].values * (TWOPI * grid.r0),),
        "rho_tor": profiles["rho"].values,
        "zeff": profiles["Zeff"].values,
        "vloop": profiles["U"].values,
        "j_ohmic": profiles["Joh"].values * 1.0e6,
        "j_non_inductive": profiles["Jnoh"].values * 1.0e6,
        "j_bootstrap": profiles["Jbs"].values * 1.0e6,
        "j_total": profiles["Jtot"].values * 1.0e6,
        "XiNC": profiles["XiNC"].values,
        "ffprime": profiles["EQFF"].values * 1.0e6,
        "pprime": profiles["EQPF"].values * 1.0e6,
    }


def load_core_transport(profiles, grid, R0: float, B0: float = None):
    bs_r_norm = profiles["x"].values
    bs_psi_norm = profiles["Fp"].values

    _x = Variable(0, "rho_tor_norm", label=r"\bar{\rho}_{tor}")

    # Core profiles
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm - r_ped))

    # Core Transport

    Cped = 0.17
    Ccore = 0.4
    # Function( profiles["Xi"].values,bs_r_norm)  Cped = 0.2
    chi = Piecewise([Ccore * (1.0 + 3 * (_x**2)), Cped], [(_x < r_ped), (_x >= r_ped)], label=r"\chi")
    chi_e = Piecewise([0.5 * Ccore * (1.0 + 3 * (_x**2)), Cped], [(_x < r_ped), (_x >= r_ped)], label=r"\chi_e")

    D = 0.1 * (chi + chi_e)

    v_pinch_ne = -0.6 * D * _x / R0
    v_pinch_Te = 2.5 * chi_e * _x / R0
    v_pinch_ni = D * _x / R0
    v_pinch_Ti = chi * _x / R0

    return {
        "grid_d": grid,
        "conductivity_parallel": profiles["Joh"].values * 1.0e6 / profiles["U"].values * (TWOPI * R0),
        "electrons": {
            "label": "e",
            "particles": {"d": D, "v": v_pinch_ne},
            "energy": {"d": chi_e, "v": v_pinch_Te},
        },
        "ion": [
            {
                "label": "D",
                "particles": {"d": D, "v": v_pinch_ni},
                "energy": {"d": chi, "v": v_pinch_Ti},
            },
            {
                "label": "T",
                "particles": {"d": D, "v": v_pinch_ni},
                "energy": {"d": chi, "v": v_pinch_Ti},
            },
            {
                "label": "He",
                "particles": {"d": D, "v": v_pinch_ni},
                "energy": {"d": chi, "v": v_pinch_Ti},
            },
        ],
    }


def load_core_source(profiles, grid, R0: float, B0: float = None):
    bs_r_norm = profiles["x"].values
    bs_psi_norm = profiles["Fp"].values

    _x = Variable(0, "rho_tor_norm", label=r"\bar{\rho}_{tor}")

    S = 9e20 * np.exp(15.0 * (_x**2 - 1.0))

    Q_e = (
        (
            profiles["Poh"].values
            + profiles["Pdte"].values
            + profiles["Paux"].values
            - profiles["Peic"].values
            - profiles["Prad"].values
            # - profiles["Pneu"].values
        )
        * 1e6
        / scipy.constants.electron_volt
    )

    Q_DT = (
        (profiles["Peic"].values + profiles["Pdti"].values + profiles["Pibm"].values)
        * 1e6
        / scipy.constants.electron_volt
    )

    Q_He =(profiles["Pdti"].values + profiles["Pdte"].values) * 1e6 / scipy.constants.electron_volt

    # Core Source
    return {
        "grid": grid,
        "j_parallel": (
            # profiles["Jtot"].values
            profiles["Joh"].values
            # + profiles["Jbs"].values
            + profiles["Jnb"].values
            + profiles["Jrf"].values
        )
        * 1e6,  # A/m^2
        "electrons": {"label": "e", "particles": S, "energy": Q_e},
        "ion": [
            {"label": "D", "particles": S * 0.5, "energy": Q_DT * 0.5},
            {"label": "T", "particles": S * 0.5, "energy": Q_DT * 0.5},
            {"label": "He", "particles": S * 0.1, "energy": Q_He},
        ],
    }


def read_iter_profiles(path):
    path = pathlib.Path(path)

    excel_file = pd.read_excel(path, sheet_name=1)

    entry = Entry(
        {
            "dataset_fair": {
                "identifier": "15MA Inductive at burn-ASTRA",
                "provenance": {"node": [{"path": "core_profiles", "sources": f"{path.as_posix()}"}]},
            }
        }
    )

    profiles_0D = {}

    for s in excel_file.iloc[0, 3:7]:
        res = re.match(r"(\w+)=(\d+\.?\d*)(\D+)", s)
        profiles_0D[res.group(1)] = (float(res.group(2)), str(res.group(3)))

    profiles_1D = pd.read_excel(path, sheet_name=1, header=10, usecols="B:BN")

    time = 0.0

    R0 = profiles_0D["R"][0]
    B0 = profiles_0D["B"][0]

    vacuum_toroidal_field = {"r0": R0, "b0": B0}

    rho_tor_norm = profiles_1D["x"].values
    rho_tor = profiles_1D["rho"].values
    psi_norm = profiles_1D["Fp"].values

    grid = {
        "rho_tor_norm": rho_tor_norm,
        "rho_tor_boundary": rho_tor[-1],
        "psi_norm": psi_norm,
        "psi_boundary": None,
        "psi_axis": None,
    }

    entry["core_profiles"] = {"time_slice": [{"time": time, "vacuum_toroidal_field": vacuum_toroidal_field}]}

    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(rho_tor_norm - r_ped))
    # fmt:off
    psi_norm = profiles_1D["Fp"].values
    # bs_psi = bs_psi_norm*(psi_boundary-psi_axis)+psi_axis

    b_Te =    smooth_1d(rho_tor_norm, profiles_1D["TE"].values,      i_end=i_ped-10, window_len=21)*1000
    b_Ti =    smooth_1d(rho_tor_norm, profiles_1D["TI"].values,      i_end=i_ped-10, window_len=21)*1000
    b_ne =    smooth_1d(rho_tor_norm, profiles_1D["NE"].values,      i_end=i_ped-10, window_len=21)*1.0e19
    b_nDT =   smooth_1d(rho_tor_norm, profiles_1D["Nd+t"].values,    i_end=i_ped-10, window_len=21)*1.0e19*0.5
    b_nHe =   smooth_1d(rho_tor_norm, profiles_1D["Nath"].values,    i_end=i_ped-10, window_len=21)*1.0e19
    b_nImp =  smooth_1d(rho_tor_norm, profiles_1D["Nz"].values,      i_end=i_ped-10, window_len=21)*1.0e19
    b_zeff = profiles_1D["Zeff"].values
    # fmt:on

    z_eff_star = b_zeff - (b_nDT * 2.0 + 4 * b_nHe) / b_ne
    z_imp = 1 - (b_nDT * 2.0 + 2 * b_nHe) / b_ne
    b = -2 * z_imp / (0.02 + 0.0012)
    c = (z_imp**2 - 0.02 * z_eff_star) / 0.0012 / (0.02 + 0.0012)

    z_Ar = np.asarray((-b + np.sqrt(b**2 - 4 * c)) / 2)
    z_Be = np.asarray((z_imp - 0.0012 * z_Ar) / 0.02)
    # b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_norm, baseline["Zeff"].values)
    # e_parallel = baseline["U"].values / (TWOPI * R0)

    entry["core_profiles/time_slice/0/profiles_1d"] = {
        "time": 0.0,
        "grid": grid,
        "electrons": {
            "label": "e",
            "density_thermal": b_ne,
            "temperature": b_Te,
        },
        "ion": [
            {"label": "D", "density_thermal": b_nDT, "temperature": b_Ti},
            {"label": "T", "density_thermal": b_nDT, "temperature": b_Ti},
            {"label": "He", "density_thermal": b_nHe, "temperature": b_Ti, "density_fast": True},
            {"label": "Be", "density_thermal": 0.02 * b_ne, "temperature": b_Ti, "z_ion_1d": z_Be, "is_impurity": True},
            {
                "label": "Ar",
                "density_thermal": 0.0012 * b_ne,
                "temperature": b_Ti,
                "z_ion_1d": z_Ar,
                "is_impurity": True,
            },
        ],
        # "e_field": {"parallel":  Function(e_parallel,bs_r_norm)},
        # "conductivity_parallel": Function(baseline["Joh"].values*1.0e6 / baseline["U"].values * (TWOPI * grid.r0),bs_r_norm),
        "rho_tor": profiles_1D["rho"].values,
        "zeff": profiles_1D["Zeff"].values,
        "vloop": profiles_1D["U"].values,
        "j_ohmic": profiles_1D["Joh"].values * 1.0e6,
        "j_non_inductive": profiles_1D["Jnoh"].values * 1.0e6,
        "j_bootstrap": profiles_1D["Jbs"].values * 1.0e6,
        "j_total": profiles_1D["Jtot"].values * 1.0e6,
        "XiNC": profiles_1D["XiNC"].values,
        "ffprime": profiles_1D["EQFF"].values * 1.0e6,
        "pprime": profiles_1D["EQPF"].values * 1.0e6,
    }

    entry["core_transport"] = {
        "model": [
            {
                "code": {"name": "dummy"},
                "time_slice": [
                    {
                        "time": time,
                        "vacuum_toroidal_field": vacuum_toroidal_field,
                    }
                ],
            }
        ]
    }

    _x = Variable(0, "rho_tor_norm")

    # Core profiles
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(rho_tor_norm - r_ped))

    # Core Transport

    Cped = 0.17
    Ccore = 0.4
    # Function( profiles["Xi"].values,bs_r_norm)  Cped = 0.2
    chi = Piecewise([Ccore * (1.0 + 3 * (_x**2)), Cped], [(_x < r_ped), (_x >= r_ped)], label=r"\chi")
    chi_e = Piecewise([0.5 * Ccore * (1.0 + 3 * (_x**2)), Cped], [(_x < r_ped), (_x >= r_ped)], label=r"\chi_e")

    D = 0.1 * (chi + chi_e)

    v_pinch_ne = 0.6 * D * _x / R0
    v_pinch_Te = 2.5 * chi_e * _x / R0
    v_pinch_ni = D * _x / R0
    v_pinch_Ti = chi * _x / R0

    entry["core_transport/model/0/time_slice/0/flux_multiplier"] = 3 / 2

    entry["core_transport/model/0/time_slice/0/profiles_1d"] = {
        "grid_d": grid,
        "conductivity_parallel": profiles_1D["Joh"].values * 1.0e6 / profiles_1D["U"].values * (TWOPI * R0),
        "electrons": {
            "label": "e",
            "particles": {"d": D, "v": v_pinch_ne},
            "energy": {"d": chi_e, "v": v_pinch_Te},
        },
        "ion": [
            {
                "label": "D",
                "particles": {"d": D, "v": v_pinch_ni},
                "energy": {"d": chi, "v": v_pinch_Ti},
            },
            {
                "label": "T",
                "particles": {"d": D, "v": v_pinch_ni},
                "energy": {"d": chi, "v": v_pinch_Ti},
            },
            {
                "label": "He",
                "particles": {"d": D, "v": v_pinch_ni},
                "energy": {"d": chi, "v": v_pinch_Ti},
            },
        ],
    }

    entry["core_sources"] = {
        "source": [
            {
                "code": {"name": "dummy"},
                "time_slice": [
                    {
                        "time": time,
                        "vacuum_toroidal_field": vacuum_toroidal_field,
                    }
                ],
            }
        ]
    }

    S = 9e20 * np.exp(15.0 * (_x**2 - 1.0))

    Q_e = (
        (
            profiles_1D["Poh"].values
            + profiles_1D["Pdte"].values
            + profiles_1D["Paux"].values
            - profiles_1D["Peic"].values
            - profiles_1D["Prad"].values
            # - profiles_1D["Pneu"].values
        )
        * 1e6
        / scipy.constants.electron_volt
    )

    Q_DT = (
        (profiles_1D["Peic"].values + profiles_1D["Pdti"].values + profiles_1D["Pibm"].values)
        * 1e6
        / scipy.constants.electron_volt
    )

    Q_He = (profiles_1D["Pdti"].values + profiles_1D["Pdte"].values) * 1e6 / scipy.constants.electron_volt

    # Core Source
    entry["core_sources/source/0/time_slice/0/profiles_1d"] = {
        "grid": grid,
        "j_parallel": (
            # profiles_1D["Jtot"].values
            profiles_1D["Joh"].values
            # + profiles_1D["Jbs"].values
            + profiles_1D["Jnb"].values
            + profiles_1D["Jrf"].values
        )
        * 1e6,  # A/m^2
        "electrons": {"label": "e", "particles": S, "energy": Q_e},
        "ion": [
            {"label": "D", "particles": S * 0.5, "energy": Q_DT * 0.5},
            {"label": "T", "particles": S * 0.5, "energy": Q_DT * 0.5},
            {"label": "He", "particles": S * 0.01, "energy": Q_DT*0.01},
        ],
    }

    return entry


@File.register(["iterprofiles"])
class ITERProfiles(File):
    """Read iter_profiles.xslx file"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self) -> Entry:
        if self.url.authority:
            raise NotImplementedError(f"{self.url}")

        return read_iter_profiles(pathlib.Path(self.url.path))

    def write(self, d, *args, **kwargs):
        raise NotImplementedError(f"TODO: write ITERProfiles {self.url}")
