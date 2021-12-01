
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants
from spdm.common.logger import logger
from spdm.util.plot_profiles import plot_profiles
from fytok.numlib.smooth import rms_residual, smooth_1d

if __name__ == "__main__":
    # profile = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')
    profiles = pd.read_excel('/home/salmon/workspace/data/15MA inductive - burn/15MA Inductive at burn-ASTRA.xls',
                             sheet_name='15MA plasma', header=10, usecols="B:BN")
    plot_profiles(
        [
            [
                (profiles["NE"].values-profiles["Nd+t"].values-profiles["Nalf"].values*2, r"$N_{z0}$"),
                # (profile["NE"].values*(0.02*4 + 0.0012*18), r"$N_{e}*(0.02*4 + 0.0012*18)$"),
                (profiles["Nz"].values,                        r"impurity density"),
            ],
            ((profiles["NE"].values-profiles["Nd+t"].values-profiles["Nalf"].values*2)/profiles["Nz"].values, r"$N_{z0}$"),
            ((profiles["NE"].values*profiles["Zeff"].values
              - profiles["Nd+t"].values
              - profiles["Nalf"].values * 4) /
             profiles["Nz"].values, r"$Z_{imp}$"),
            (profiles["Zeff"].values, r"$Z_{eff}$"),
            (rms_residual(profiles["Nd+t"].values, profiles["NE"].values*(1 - 0.02*4 - 0.0012*18) -
                          profiles["Nalf"].values * 2),
             r"NDT", r"  rms residual $[\%]$", {"color": "red", "linestyle": "dashed"}),

            #  - (0.02*4 - 0.0012*18)*profile["NE"].values,   r"$N_{e}-N_{DT}-N_{\alpha}*2-N_{imp}*Z$", r"$10^{19}[m^{-3}]$",
            #  {"color": "red", "linestyle": "dashed"}),
            # (profile["Nz"].values/profile["NE"].values-0.02,                                   r"$N_{z}$"),
            # ((0.02*4 - 0.0012*18)*profile["NE"].values, "N_imp"),
            # ((0.02 - 0.0012)*profile["NE"].values, "N_imp2"),

            # (profile["Nb"].values,                                       r"fast NBI deuterium density"),

            # (profile["NE"].values
            #  - profile["Nd+t"].values
            #  - profile["Nz"].values
            #  - profile["Nalf"].values*2,
            #  r"rms",  {"color": "red", "linestyle": "dashed"}),


            [
                (profiles["Nalf"].values,                   r"$He$ alpha density", r"$10^{19}[m^{-3}]$"),
                (profiles["Nz"].values,                       r"impurity density", r"$10^{19}[m^{-3}]$"),
                (profiles["Nb"].values,             r"fast NBI deuterium density", r"$10^{19}[m^{-3}]$"),
            ],
            [
                (profiles["Nalf"].values,                      r"alpha density", r"$10^{19}[m^{-3}]$"),
                (profiles["Nath"].values,                      r"thermal He density", r"$10^{19}[m^{-3}]$"),
                (profiles["Naff"].values,      r"alpha prtcl. density (thin orbits)", r"$10^{19}[m^{-3}]$"),
                (profiles["Nalf"].values
                 - profiles["Nath"].values
                 - profiles["Naff"].values,   r"rms", r"$10^{19}[m^{-3}]$",
                 {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profiles["TE"].values,                                   r"$T_{e}$", r"$T[eV]$"),
                (profiles["TI"].values,                                   r"$T_{i}$", r"$T[eV]$"),

            ],
            [
                (profiles["Jext"].values,                                r"ext", r"$J [MA\cdot m^{-2}]$"),
                (profiles["Jnb"].values,                                 r"nb",  r"$J [MA\cdot m^{-2}]$"),
                (profiles["Jrf"].values,                                 r"rf",  r"$J [MA\cdot m^{-2}]$"),
                (profiles["Jext"].values
                 - profiles["Jnb"].values
                 - profiles["Jrf"].values,                  r"$J_{ext}-J_{nb}-J_{rf}$", r"$J [MA\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),
            ],

            [
                (profiles["Jnoh"].values,                                r"$j_{noh}$", r"$J [MA\cdot m^{-2}]$"),
                (profiles["Jbs"].values,                           r"$j_{bootstrap}$", r"$J [MA\cdot m^{-2}]$"),
                (profiles["Jnb"].values,                                  r"$j_{nb}$", r"$J [MA\cdot m^{-2}]$"),
                (profiles["Jrf"].values,                                  r"$j_{rf}$", r"$J [MA\cdot m^{-2}]$"),

                (profiles["Jnoh"].values
                 - profiles["Jbs"].values
                 - profiles["Jnb"].values
                 - profiles["Jrf"].values,           r"$J_{noh}-J_{bs}-J_{nb}-J_{rf}$", r"$J [MA\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),
            ],

            [

                (profiles["Jtot"].values,                          r" parallel ",  r"$J [MA\cdot m^{-2}]$"),
                (profiles["Joh"].values,                                  r"oh  ", r"$J [MA\cdot m^{-2}]$"),
                (profiles["Jnoh"].values,                                r"noh ",  r"$J [MA\cdot m^{-2}]$"),
                (profiles["Jtot"].values
                 - profiles["Jnoh"].values
                 - profiles["Joh"].values,           r"$j_{\parallel}-j_{oh}-j_{noh}$", r"$J [MA\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),

            ],

            [
                (profiles["Paux"].values,                 r"auxilliary power density", r"P $[MW\cdot m^{-2}]$"),
                (profiles["PeEX"].values,               r"RF+NB heating of electrons", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Pex"].values,                   r"RF heating of electrons", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Pnbe"].values,                  r"NB heating of electrons", r"P $[MW\cdot m^{-2}]$"),

                (profiles["Paux"].values
                 - profiles["Pex"].values
                 - profiles["Pnbe"].values,                                   r"rms", r"$[MW\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),

            ],


            [
                (profiles["Pdt"].values,                          r"$\alpha$-heating", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Pdti"].values,                r"heating of ions by alphas", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Pdte"].values,            r"heating of elecrons by alphas", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Pdt"].values
                 - profiles["Pdti"].values
                 - profiles["Pdte"].values,                                    r"rms", r"$[MW\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profiles["Prad"].values,                     r"total radiative loss", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Plin"].values,                                     r"lin",  r"P $[MW\cdot m^{-2}]$"),
                (profiles["Psyn"].values,           r"electron synchrotoron radiaion", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Pbrm"].values,                                     r"brm",  r"P $[MW\cdot m^{-2}]$"),

                (profiles["Prad"].values
                    - profiles["Psyn"].values
                    - profiles["Pbrm"].values
                    - profiles["Plin"].values,                                 r"rms", r"P $[MW\cdot m^{-2}]$",
                    {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profiles["Pibm"].values,              r"Beam power absorbed by ions", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Peic"].values,
                 r"?electron-ion heat exchange $\frac{3}{\tau_{ei}}n_{e}\left(T_{e}-T_{i}\right)\frac{m}{M}$", r"$[MW\cdot m^{-2}]$"),
                (profiles["Pix"].values,                         r"auxiliary heating", r"P $[MW\cdot m^{-2}]$"),
                (profiles["Pneu"].values,
                 r"?electron thermal losses due to ionzation of cold neutrals", r"P $[MW\cdot m^{-2}]$"),
            ],
            (profiles["Poh"].values,
             r"Joul heating power density $\sigma_{\parallel}\cdot E^2$", r"P $[MW\cdot m^{-2}]$"),
            (profiles["Xi"].values,                                   r"ion heat conductivity $\chi_i$"),
            (profiles["XiNC"].values,                 r"neoclassical ion heat conductivity $\chi_{NC}$"),
            (profiles["U"].values,                                                        r"$V_{loop}$"),
            (profiles["shif"].values,                                                r"shafranov shift"),
            (profiles["k"].values,                                                        r"elongation"),
            (profiles["He"].values,                                       r"electron heat conductivity"),
        ],
        x_axis=(profiles["x"].values,                                   r"$\rho_{N}$"),
        # index_slice=slice(-100,None, 1),
        grid=True) .savefig("/home/salmon/workspace/output/profiles_exp.svg", transparent=True)
