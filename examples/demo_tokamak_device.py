import pathlib
import pandas as pd
import numpy as np
from fytok.Tokamak import Tokamak
from fytok.utils.plot_profiles import sp_figure, plot_profiles
from spdm.data.Function import function_like
from spdm.data.File import File
from spdm.utils.logger import logger

if __name__ == "__main__":

    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    ###################################################################################################
    # baseline
    device_desc = File("/home/salmon/workspace/fytok_data/mapping/ITER/imas/3/static/config.xml", format="XML").read()
    # Equilibrium
    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="GEQdsk").read()

    R0 = eqdsk_file.get("vacuum_toroidal_field/r0")
    B0 = eqdsk_file.get("vacuum_toroidal_field/b0")

    psi_axis = eqdsk_file.get("time_slice/0/global_quantities/psi_axis")
    psi_boundary = eqdsk_file.get("time_slice/0/global_quantities/psi_boundary")
    bs_eq_psi = eqdsk_file.get("time_slice/0/profiles_1d/psi")
    bs_eq_psi_norm = (bs_eq_psi-psi_axis)/(psi_boundary-psi_axis)
    bs_eq_fpol = function_like(bs_eq_psi_norm, eqdsk_file.get("time_slice/0/profiles_1d/f"))

    profiles = pd.read_excel('/home/salmon/workspace/data/15MA inductive - burn/15MA Inductive at burn-ASTRA.xls',
                             sheet_name='15MA plasma', header=10, usecols="B:BN")

    bs_r_norm = profiles["x"].values

    b_Te = function_like(profiles["TE"].values*1000, bs_r_norm)
    b_Ti = function_like(profiles["TI"].values*1000, bs_r_norm)
    b_ne = function_like(profiles["NE"].values*1.0e19, bs_r_norm)

    b_nHe = function_like(profiles["Nalf"].values * 1.0e19, bs_r_norm)
    b_ni = function_like(profiles["Nd+t"].values * 1.0e19*0.5, bs_r_norm)
    b_nImp = function_like(profiles["Nz"].values * 1.0e19, bs_r_norm)
    b_zeff = function_like(profiles["Zeff"].values, bs_r_norm)
    bs_psi_norm = profiles["Fp"].values
    bs_psi = bs_psi_norm*(psi_boundary-psi_axis)+psi_axis

    bs_r_norm = profiles["x"].values
    ###################################################################################################
    # Initialize Tokamak

    tok = Tokamak(device_desc[{"wall", "pf_active", "tf", "magnetics"}])
    tok["equilibrium"] = {**eqdsk_file.dump(),
                          "code": {"name":  "eq_analyze",
                                   "parameters": {
                                       "boundary": {"psi_norm": 0.995},
                                       "coordinate_system": {"psi_norm": np.linspace(0.001, 0.995, 64), "theta": 64}}
                                   }
                          }
    if True:
        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},
                        "vessel": {"edgecolor": "blue"}},
                  pf_active={"facecolor": 'red'},
                  equilibrium={  # "contours": [0, 2],
                      "boundary": True,
                      "separatrix": True,
                  }
                  ) .savefig(output_path/"tokamak.svg", transparent=True)

        time_slice = -1
        eq_profiles_1d = tok.equilibrium.time_slice[time_slice].profiles_1d
        
        bs_line_style = {"marker": '.', "linestyle": ''}

        plot_profiles(
            [
                [
                    (bs_eq_fpol, "astra", r"$F_{pol} [Wb\cdot m]$", bs_line_style),
                    (eq_profiles_1d.f,  r"fytok", r"$[Wb]$"),
                ],

                [
                    (function_like(profiles["q"].values, bs_psi_norm), r"astra", r"$q [-]$", bs_line_style),
                    # (function_like(eqdsk.get('profiles_1d.psi_norm'), eqdsk.get('profiles_1d.q')), "eqdsk"),
                    (eq_profiles_1d.q,  r"$fytok$", r"$[Wb]$"),
                    # (magnetic_surface.dphi_dpsi,  r"$\frac{d\phi}{d\psi}$", r"$[Wb]$"),
                ],
                [
                    (function_like(profiles["rho"].values, bs_psi_norm), r"astra", r"$\rho_{tor}[m]$",  bs_line_style),
                    (eq_profiles_1d.rho_tor,  r"$\rho$", r"$[m]$"),
                ],
                [
                    (function_like(profiles["x"].values, bs_psi_norm),           r"astra",
                     r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$", bs_line_style),
                    (eq_profiles_1d.rho_tor_norm,
                     r"$\bar{\rho}$", r"$[-]$"),
                ],

                [
                    (function_like(4*(np.constants.pi**2) * R0 * profiles["rho"].values, bs_psi_norm),
                     r"$4\pi^2 R_0 \rho$", r"$4\pi^2 R_0 \rho$",  bs_line_style),
                    (eq_profiles_1d.dvolume_drho_tor,
                     r"$dV/d\rho_{tor}$", r"$[m^2]$"),
                ],

                # (magnetic_surface.dvolume_dpsi, r"$\frac{dV}{d\psi}$"),

                # [
                #     (magnetic_surface.volume, r"$V$  from $\psi$"),
                #     # (magnetic_surface.volume1, r"$V$ from $\rho_{tor}$"),
                # ],



                # (magnetic_surface.psi,  r"$\psi$", r"$[Wb]$"),
                # (magnetic_surface.phi,  r"$\phi$", r"$[Wb]$"),
                # (magnetic_surface.psi_norm,  r"$\bar{\psi}$", r"$[-]$"),


                # [
                #     (magnetic_surface.dpsi_drho_tor,
                #      r"$\frac{d\psi}{d\rho_{tor}}$", "", {"marker": '.'}),
                #     (magnetic_surface.dvolume_drho_tor/magnetic_surface.dvolume_dpsi,
                #      r"$\frac{dV}{d\rho_{tor}} / \frac{dV}{d\psi}$")
                # ],
                # (magnetic_surface.drho_tor_dpsi*magnetic_surface.dpsi_drho_tor,
                #  r"$\frac{d\rho_{tor}}{d\psi} \cdot \frac{d\psi}{d\rho_{tor}}$"),
                # (magnetic_surface.gm2_,
                #  r"$gm2_=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                # (magnetic_surface.dpsi_drho_tor, r"$\frac{d\rho_{tor}}{d\psi}$", "", {"marker": "."}),


                (eq_profiles_1d.gm1, r"$gm1=\left<\frac{1}{R^2}\right>$"),
                (eq_profiles_1d.gm2, r"$gm2=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"),
                (eq_profiles_1d.gm3, r"$gm3=\left<\left|\nabla \rho\right|^2\right>$"),
                (eq_profiles_1d.gm7, r"$gm7=\left<\left|\nabla \rho\right|\right>$"),
                (eq_profiles_1d.gm8, r"$gm8=\left<R\right>$"),

                # (magnetic_surface.dphi_dpsi,                                                  r"$\frac{d\phi}{d\psi}$"),
                # (magnetic_surface.dpsi_drho_tor,                                        r"$\frac{d\psi}{d\rho_{tor}}$"),
            ],
            # x_axis=(magnetic_surface.rho_tor_norm,      r"$\bar{\rho}_{tor}$"),
            x_axis=(eq_profiles_1d.psi_norm,      r"$\bar{\psi}$"),
            title="Equilibrium",
            grid=True, fontsize=16) .savefig(output_path/"equilibrium_coord.svg", transparent=True)
