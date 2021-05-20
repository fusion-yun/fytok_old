
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants
from spdm.util.logger import logger
from spdm.util.plot_profiles import plot_profiles

if __name__ == "__main__":
    profile = pd.read_csv('/home/salmon/workspace/data/15MA inductive - burn/profile.txt', sep='\t')

    plot_profiles(
        [
            [
                # (profile["NE"].values,                                                           r"$N_{e}$"),
                # (profile["Nd+t"].values,                        r"Nd,thermal + Nt thermalised fuel density"),
                (profile["NE"].values
                 - profile["Nd+t"].values
                 - profile["Nalf"].values*2,                                r"$N_{e}-N_{DT}-N_{\alpha}*2$"),
                (profile["Nz"].values*profile["Zeff"].values,                                   r"$N_{z}$"),
                (profile["Nb"].values,                                       r"fast NBI deuterium density"),

                # (profile["NE"].values
                #  - profile["Nd+t"].values
                #  - profile["Nz"].values*profile["Zeff"].values
                #  - profile["Nalf"].values*2,
                #  r"rms",  {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profile["Nalf"].values,                                              r"He + alpha density"),
                (profile["Nz"].values,                                                  r"impurity density"),
                (profile["Nb"].values,                                        r"fast NBI deuterium density"),
            ],
            [
                (profile["Nalf"].values,                                              r"He + alpha density"),
                (profile["Nath"].values,                                              r"thermal He density"),
                (profile["Naff"].values,                              r"alpha prtcl. density (thin orbits)"),
                (profile["Nalf"].values
                 - profile["Nath"].values
                 - profile["Naff"].values,   r"rms",
                 {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profile["TE"].values,                                   r"$T_{e}$"),
                (profile["TI"].values,                                   r"$T_{i}$"),

            ],
            [
                (profile["Jext"].values,                               r"$j_{ext}$"),
                (profile["Jnb"].values,                                 r"$j_{nb}$"),
                (profile["Jrf"].values,                                 r"$j_{rf}$"),
                (profile["Jext"].values
                 - profile["Jnb"].values
                 - profile["Jrf"].values,                  r"$J_{ext}-J_{nb}-J_{rf}$",
                 {"color": "red", "linestyle": "dashed"}),
            ],

            [
                (profile["Jnoh"].values,                                r"$j_{noh}$"),
                (profile["Jbs"].values,                           r"$j_{bootstrap}$"),
                (profile["Jnb"].values,                                  r"$j_{nb}$"),
                (profile["Jrf"].values,                                  r"$j_{rf}$"),

                (profile["Jnoh"].values
                 - profile["Jbs"].values
                 - profile["Jnb"].values
                 - profile["Jrf"].values,           r"$J_{noh}-J_{bs}-J_{nb}-J_{rf}$",
                 {"color": "red", "linestyle": "dashed"}),
            ],

            [

                (profile["Jtot"].values,                          r"$j_{\parallel}$"),
                (profile["Joh"].values,                                  r"$j_{oh}$"),
                (profile["Jnoh"].values,                                r"$j_{noh}$"),
                (profile["Jtot"].values
                 - profile["Jnoh"].values
                 - profile["Joh"].values,           r"$j_{\parallel}-j_{oh}-j_{noh}$",
                 {"color": "red", "linestyle": "dashed"}),

            ],

            [
                (profile["Paux"].values,                 r"auxilliary power density"),
                (profile["PeEX"].values,               r"RF+NB heating of electrons"),
                (profile["Pex"].values,                   r"RF heating of electrons"),
                (profile["Pnbe"].values,                  r"NB heating of electrons"),

                (profile["Paux"].values
                 - profile["Pex"].values
                 - profile["Pnbe"].values,                                   r"rms",
                 {"color": "red", "linestyle": "dashed"}),

            ],


            [
                (profile["Pdt"].values,                          r"$\alpha$-heating"),
                (profile["Pdti"].values,                r"heating of ions by alphas"),
                (profile["Pdte"].values,            r"heating of elecrons by alphas"),
                (profile["Pdt"].values
                 - profile["Pdti"].values
                 - profile["Pdte"].values,                                    r"rms",
                 {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profile["Prad"].values,                     r"total radiative loss"),
                (profile["Plin"].values,                                     r"Plin"),
                (profile["Psyn"].values,           r"electron synchrotoron radiaion"),
                (profile["Pbrm"].values,                                     r"Pbrm"),

                (profile["Prad"].values
                    - profile["Psyn"].values
                    - profile["Pbrm"].values
                    - profile["Plin"].values,                                 r"rms",
                    {"color": "red", "linestyle": "dashed"}),

            ],
            [
                (profile["Pibm"].values,              r"Beam power absorbed by ions"),
                (profile["Peic"].values,
                 r"?electron-ion heat exchange $\frac{3}{\tau_{ei}}n_{e}\left(T_{e}-T_{i}\right)\frac{m}{M}$"),
                (profile["Pix"].values,                         r"auxiliary heating"),
                (profile["Pneu"].values,  r"?electron thermal losses due to ionzation of cold neutrals"),
            ],
            (profile["Poh"].values,        r"Joul heating power density $\sigma_{\parallel}\cdot E^2$"),
            (profile["Xi"].values,                                   r"ion heat conductivity $\chi_i$"),
            (profile["XiNC"].values,                 r"neoclassical ion heat conductivity $\chi_{NC}$"),
            (profile["U"].values,                                                        r"$V_{loop}$"),
            (profile["shif"].values,                                                r"shafranov shift"),
            (profile["k"].values,                                                        r"elongation"),

        ],
        x_axis=(profile["x"].values,                                   r"$\rho_{N}$"),
        # index_slice=slice(-100,None, 1),
        grid=True) .savefig("/home/salmon/workspace/output/profiles_exp.svg", transparent=True)
