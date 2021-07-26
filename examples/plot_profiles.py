
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spdm.numlib import constants
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles
from spdm.numlib.smooth import rms_residual, smooth_1d

if __name__ == "__main__":
    profile = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')

    plot_profiles(
        [
            [
                (profile["NE"].values-profile["Nd+t"].values-profile["Nalf"].values*2, r"$N_{z0}$"),
                # (profile["NE"].values*(0.02*4 + 0.0012*18), r"$N_{e}*(0.02*4 + 0.0012*18)$"),
                (profile["Nz"].values,                        r"impurity density"),
            ],
            ((profile["NE"].values-profile["Nd+t"].values-profile["Nalf"].values*2)/profile["Nz"].values, r"$N_{z0}$"),
            ((profile["NE"].values*profile["Zeff"].values
              - profile["Nd+t"].values
              - profile["Nalf"].values * 4) /
             profile["Nz"].values, r"$Z_{imp}$"),
            (profile["Zeff"].values, r"$Z_{eff}$"),
            (rms_residual(profile["Nd+t"].values, profile["NE"].values*(1 - 0.02*4 - 0.0012*18) -
                          profile["Nalf"].values * 2),
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
                (profile["Nalf"].values,                   r"$He$ alpha density", r"$10^{19}[m^{-3}]$"),
                (profile["Nz"].values,                       r"impurity density", r"$10^{19}[m^{-3}]$"),
                (profile["Nb"].values,             r"fast NBI deuterium density", r"$10^{19}[m^{-3}]$"),
            ],
            [
                (profile["Nalf"].values,                      r"alpha density", r"$10^{19}[m^{-3}]$"),
                (profile["Nath"].values,                      r"thermal He density", r"$10^{19}[m^{-3}]$"),
                (profile["Naff"].values,      r"alpha prtcl. density (thin orbits)", r"$10^{19}[m^{-3}]$"),
                (profile["Nalf"].values
                 - profile["Nath"].values
                 - profile["Naff"].values,   r"rms", r"$10^{19}[m^{-3}]$",
                 {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profile["TE"].values,                                   r"$T_{e}$", r"$T[eV]$"),
                (profile["TI"].values,                                   r"$T_{i}$", r"$T[eV]$"),

            ],
            [
                (profile["Jext"].values,                                r"ext", r"$J [MA\cdot m^{-2}]$"),
                (profile["Jnb"].values,                                 r"nb",  r"$J [MA\cdot m^{-2}]$"),
                (profile["Jrf"].values,                                 r"rf",  r"$J [MA\cdot m^{-2}]$"),
                (profile["Jext"].values
                 - profile["Jnb"].values
                 - profile["Jrf"].values,                  r"$J_{ext}-J_{nb}-J_{rf}$", r"$J [MA\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),
            ],

            [
                (profile["Jnoh"].values,                                r"$j_{noh}$", r"$J [MA\cdot m^{-2}]$"),
                (profile["Jbs"].values,                           r"$j_{bootstrap}$", r"$J [MA\cdot m^{-2}]$"),
                (profile["Jnb"].values,                                  r"$j_{nb}$", r"$J [MA\cdot m^{-2}]$"),
                (profile["Jrf"].values,                                  r"$j_{rf}$", r"$J [MA\cdot m^{-2}]$"),

                (profile["Jnoh"].values
                 - profile["Jbs"].values
                 - profile["Jnb"].values
                 - profile["Jrf"].values,           r"$J_{noh}-J_{bs}-J_{nb}-J_{rf}$", r"$J [MA\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),
            ],

            [

                (profile["Jtot"].values,                          r" parallel ",  r"$J [MA\cdot m^{-2}]$"),
                (profile["Joh"].values,                                  r"oh  ", r"$J [MA\cdot m^{-2}]$"),
                (profile["Jnoh"].values,                                r"noh ",  r"$J [MA\cdot m^{-2}]$"),
                (profile["Jtot"].values
                 - profile["Jnoh"].values
                 - profile["Joh"].values,           r"$j_{\parallel}-j_{oh}-j_{noh}$", r"$J [MA\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),

            ],

            [
                (profile["Paux"].values,                 r"auxilliary power density", r"P $[MW\cdot m^{-2}]$"),
                (profile["PeEX"].values,               r"RF+NB heating of electrons", r"P $[MW\cdot m^{-2}]$"),
                (profile["Pex"].values,                   r"RF heating of electrons", r"P $[MW\cdot m^{-2}]$"),
                (profile["Pnbe"].values,                  r"NB heating of electrons", r"P $[MW\cdot m^{-2}]$"),

                (profile["Paux"].values
                 - profile["Pex"].values
                 - profile["Pnbe"].values,                                   r"rms", r"$[MW\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),

            ],


            [
                (profile["Pdt"].values,                          r"$\alpha$-heating", r"P $[MW\cdot m^{-2}]$"),
                (profile["Pdti"].values,                r"heating of ions by alphas", r"P $[MW\cdot m^{-2}]$"),
                (profile["Pdte"].values,            r"heating of elecrons by alphas", r"P $[MW\cdot m^{-2}]$"),
                (profile["Pdt"].values
                 - profile["Pdti"].values
                 - profile["Pdte"].values,                                    r"rms", r"$[MW\cdot m^{-2}]$",
                 {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profile["Prad"].values,                     r"total radiative loss", r"P $[MW\cdot m^{-2}]$"),
                (profile["Plin"].values,                                     r"lin",  r"P $[MW\cdot m^{-2}]$"),
                (profile["Psyn"].values,           r"electron synchrotoron radiaion", r"P $[MW\cdot m^{-2}]$"),
                (profile["Pbrm"].values,                                     r"brm",  r"P $[MW\cdot m^{-2}]$"),

                (profile["Prad"].values
                    - profile["Psyn"].values
                    - profile["Pbrm"].values
                    - profile["Plin"].values,                                 r"rms", r"P $[MW\cdot m^{-2}]$",
                    {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profile["Pibm"].values,              r"Beam power absorbed by ions", r"P $[MW\cdot m^{-2}]$"),
                (profile["Peic"].values,
                 r"?electron-ion heat exchange $\frac{3}{\tau_{ei}}n_{e}\left(T_{e}-T_{i}\right)\frac{m}{M}$", r"$[MW\cdot m^{-2}]$"),
                (profile["Pix"].values,                         r"auxiliary heating", r"P $[MW\cdot m^{-2}]$"),
                (profile["Pneu"].values,
                 r"?electron thermal losses due to ionzation of cold neutrals", r"P $[MW\cdot m^{-2}]$"),
            ],
            (profile["Poh"].values,
             r"Joul heating power density $\sigma_{\parallel}\cdot E^2$", r"P $[MW\cdot m^{-2}]$"),
            (profile["Xi"].values,                                   r"ion heat conductivity $\chi_i$"),
            (profile["XiNC"].values,                 r"neoclassical ion heat conductivity $\chi_{NC}$"),
            (profile["U"].values,                                                        r"$V_{loop}$"),
            (profile["shif"].values,                                                r"shafranov shift"),
            (profile["k"].values,                                                        r"elongation"),
            (profile["He"].values,                                       r"electron heat conductivity"),
        ],
        x_axis=(profile["x"].values,                                   r"$\rho_{N}$"),
        # index_slice=slice(-100,None, 1),
        grid=True) .savefig("/home/salmon/workspace/output/profiles_exp.svg", transparent=True)
