
import numpy as np
from scipy import constants
from spdm.data.Entry import Entry
from spdm.data.File import File
from spdm.data.Function import Function, PiecewiseFunction
from spdm.numlib.smooth import smooth_1d

from .modules.Utilities import CoreRadialGrid


def load_core_profiles(profiles, grid: CoreRadialGrid):

    bs_psi_norm = profiles["Fp"].values
    bs_r_norm = profiles["x"].values

    grid = grid.remesh("rho_tor_norm", bs_r_norm)

    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    b_Te = Function(smooth_1d(profiles["TE"].values,  bs_r_norm, i_end=i_ped-10, window_len=21)*1000, bs_r_norm)
    b_Ti = Function(smooth_1d(profiles["TI"].values,  bs_r_norm, i_end=i_ped-10, window_len=21)*1000, bs_r_norm)
    b_ne = Function(smooth_1d(profiles["NE"].values,  bs_r_norm, i_end=i_ped-10, window_len=21)*1.0e19,   bs_r_norm)
    b_nDT = Function(smooth_1d(profiles["Nd+t"].values, bs_r_norm, i_end=i_ped-10, window_len=21)*1.0e19*0.5, bs_r_norm)
    b_nHe = Function(smooth_1d(profiles["Nath"].values, bs_r_norm, i_end=i_ped-10, window_len=21)*1.0e19, bs_r_norm)
    b_nImp = Function(smooth_1d(profiles["Nz"].values,  bs_r_norm, i_end=i_ped-10, window_len=21)*1.0e19,   bs_r_norm)
    b_zeff = Function(profiles["Zeff"].values,    bs_r_norm)

    z_eff_star = b_zeff-(b_nDT*2.0+4*b_nHe)/b_ne
    z_imp = 1-(b_nDT*2.0+2*b_nHe)/b_ne
    b = -2*z_imp/(0.02+0.0012)
    c = (z_imp**2-0.02*z_eff_star)/0.0012/(0.02+0.0012)

    z_Ar = np.asarray((-b+np.sqrt(b**2-4*c))/2)
    z_Be = np.asarray((z_imp-0.0012*z_Ar)/0.02)
    # b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_norm, baseline["Zeff"].values)
    # e_parallel = baseline["U"].values / (2.0*constants.pi * grid.r0)

    return {
        "grid": grid,
        "rho_tor": profiles["rho"].values,
        "electrons": {"label": "e", "density":       b_ne,   "temperature": b_Te, },
        "ion": [
            {"label": "D",  "density":      b_nDT,  "temperature": b_Ti},
            {"label": "T",  "density":      b_nDT,  "temperature": b_Ti},
            {"label": "He", "density_thermal":      b_nHe,  "temperature": b_Ti, "has_fast_particle": True},
            {"label": "Be", "density":  0.02*b_ne,  "temperature": b_Ti,
             "z_ion_1d": Function(z_Be, bs_r_norm),  "is_impurity": True},
            {"label": "Ar", "density": 0.0012*b_ne,  "temperature": b_Ti,
             "z_ion_1d": Function(z_Ar, bs_r_norm, ),  "is_impurity": True},
        ],
        # "e_field": {"parallel":  Function(e_parallel,bs_r_norm)},
        # "conductivity_parallel": Function(baseline["Joh"].values*1.0e6 / baseline["U"].values * (2.0*constants.pi * grid.r0),bs_r_norm),
        "zeff":             Function(profiles["Zeff"].values, bs_r_norm),
        "vloop":            Function(profiles["U"].values, bs_r_norm),
        "j_ohmic":          Function(profiles["Joh"].values*1.0e6, bs_r_norm),
        "j_non_inductive":  Function(profiles["Jnoh"].values*1.0e6, bs_r_norm),
        "j_bootstrap":      Function(profiles["Jbs"].values*1.0e6, bs_r_norm),
        "j_total":          Function(profiles["Jtot"].values*1.0e6, bs_r_norm),
        "XiNC":             Function(profiles["XiNC"].values, bs_r_norm),
    }


def load_core_transport(profiles, grid: CoreRadialGrid):

    bs_psi_norm = profiles["Fp"].values

    bs_r_norm = profiles["x"].values

    # Core profiles
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    # Core Transport
    conductivity_parallel = Function(profiles["Joh"].values*1.0e6 /
                                     profiles["U"].values * (2.0*constants.pi * grid.r0), bs_r_norm)

    Cped = 0.17
    Ccore = 0.4
    # Function( profiles["Xi"].values,bs_r_norm)  Cped = 0.2
    chi = PiecewiseFunction([lambda x: Ccore*(1.0 + 3*(x**2)), lambda x: Cped], [0, r_ped, 1.0])
    chi_e = PiecewiseFunction([lambda x: 0.5 * Ccore*(1.0 + 3*(x**2)), lambda x: Cped], [0, r_ped, 1.0])

    D = 0.1*(chi+chi_e)

    v_pinch_ne = Function(lambda x: -0.6 * D(x) * x / grid.r0,      [0, r_ped, 1.0])
    v_pinch_Te = Function(lambda x:  2.5 * chi_e(x) * x / grid.r0,  [0, r_ped, 1.0])
    v_pinch_ni = Function(lambda x:  D(x) * x / grid.r0,            [0, r_ped, 1.0])
    v_pinch_Ti = Function(lambda x:  chi(x) * x / grid.r0,          [0, r_ped, 1.0])

    conductivity_parallel = Function(profiles["Joh"].values*1.0e6 /
                                     profiles["U"].values * (2.0*constants.pi * grid.r0),  bs_r_norm)

    return {
        "grid": grid,
        "conductivity_parallel": conductivity_parallel,
        "electrons": {
            "label": "e",
            "particles":   {"d": D,     "v": v_pinch_ne},
            "energy":      {"d": chi_e, "v": v_pinch_Te},
        },
        "ion": [
            {
                "label": "D",
                "particles": {"d":  D, "v": v_pinch_ni},
                "energy": {"d":  chi, "v": v_pinch_Ti},
            },
            {
                "label": "T",
                "particles": {"d":  D, "v": v_pinch_ni},
                "energy": {"d":  chi, "v": v_pinch_Ti},
            },
            {
                "label": "He",
                "particles": {"d": D, "v": v_pinch_ni},
                "energy": {"d": chi, "v": v_pinch_Ti},
            }
        ]}


def load_core_source(profiles, grid: CoreRadialGrid):

    bs_r_norm = profiles["x"].values

    S = Function(lambda x: 9e20 * np.exp(15.0*(x**2-1.0)))

    Q_e = Function((profiles["Poh"].values
                    + profiles["Pdte"].values
                    + profiles["Paux"].values
                    - profiles["Peic"].values
                    - profiles["Prad"].values
                    # - profiles["Pneu"].values
                    )*1e6/constants.electron_volt, bs_r_norm)

    Q_DT = Function((profiles["Peic"].values
                     + profiles["Pdti"].values
                     + profiles["Pibm"].values
                     )*1e6/constants.electron_volt, bs_r_norm)

    Q_He = Function((- profiles["Pdti"].values
                     - profiles["Pdte"].values
                     )*1e6/constants.electron_volt, bs_r_norm)

    # Core Source
    return {
        "grid": grid,
        "j_parallel": Function((
            # profiles["Jtot"].values
            profiles["Joh"].values
            # + profiles["Jbs"].values
            + profiles["Jnb"].values
            + profiles["Jrf"].values
        ) * 1e6,            bs_r_norm),
        "electrons": {"label": "e",  "particles": S, "energy": Q_e},
        "ion": [
            {"label": "D",          "particles": S*0.5,      "energy": Q_DT*0.5},
            {"label": "T",          "particles": S*0.5,      "energy": Q_DT*0.5},
            {"label": "He",         "particles": S*0.01,      "energy": Q_He}
        ]}


def load_equilibrium(eqdsk,  **kwargs):
    if not isinstance(eqdsk, Entry):
        eqdsk = File(eqdsk, format="GEQdsk").read()

    # R0 = eqdsk.get("vacuum_toroidal_field/r0")
    # B0 = eqdsk.get("vacuum_toroidal_field/b0")

    return {**eqdsk.dump(), ** kwargs
            # "vacuum_toroidal_field": {"b0": B0, "r0": R0, },
            # "time_slice": [{
            #     "global_quantities": eqdsk.get("global_quantities"),
            #     "profiles_1d": eqdsk.get("profiles_1d"),
            #     "profiles_2d": {
            #         "psi": eqdsk.get("profiles_2d/psi"),
            #         "grid_type": {
            #             "name": "rectangular",
            #             "index": 1},
            #         "grid": {
            #             "dim1": eqdsk.get("profiles_2d/grid/dim1"),
            #             "dim2": eqdsk.get("profiles_2d/grid/dim2"),
            #         }
            #     },
            #     "boundary_separatrix": eqdsk.get("boundary"),

            }
