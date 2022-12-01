
import collections

import numpy as np
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.Equilibrium import Equilibrium
from spdm.numlib.misc import array_like
from scipy import constants
from spdm.data import Function
from spdm.util.logger import logger


class BootstrapCurrent(CoreSources.Source):
    def __init__(self, d: collections.abc.Mapping = None,  *args, **kwargs):
        super().__init__(collections.ChainMap(
            {"identifier": {"name": f"bootstrap_current",  "index": 13,
                            "description": f"{self.__class__.__name__} Bootstrap current, based on  Tokamaks, 3ed, sec 14.12 J.A.Wesson 2003"},
             "code": {"name": "bootstrap_current"}}, d or {}),
            *args, **kwargs)

    def refresh(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs) -> float:

        residual = super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        equilibrium_1d = equilibrium.profiles_1d
        core_profiles_1d = core_profiles.profiles_1d
        profiles_1d = self.profiles_1d
        eV = constants.electron_volt
        B0 = equilibrium.vacuum_toroidal_field.b0
        R0 = equilibrium.vacuum_toroidal_field.r0

        # rho_tor_norm = (core_profile.grid.rho_tor_norm[:-1]+core_profile.grid.rho_tor_norm[1:])*0.5
        # rho_tor = (core_profile.grid.rho_tor[:-1]+core_profile.grid.rho_tor[1:])*0.5
        # psi_norm = (core_profile.grid.psi_norm[:-1]+core_profile.grid.psi_norm[1:])*0.5
        rho_tor_norm = profiles_1d.grid.rho_tor_norm[1:]
        rho_tor = profiles_1d.grid.rho_tor[1:]
        psi_norm = profiles_1d.grid.psi_norm[1:]

        q = equilibrium_1d.q(psi_norm)

        # max(np.asarray(1.07e-4*((Te[0]/1000)**(1/2))/B0), rho_tor[0])   # Larmor radius,   eq 14.7.2

        Te = core_profiles_1d.electrons.temperature(rho_tor_norm)
        Ne = core_profiles_1d.electrons.density(rho_tor_norm)
        Pe = core_profiles_1d.electrons.pressure(rho_tor_norm)
        dlnTe = core_profiles_1d.electrons.temperature.derivative(rho_tor_norm)/Te
        dlnNe = core_profiles_1d.electrons.density.derivative(rho_tor_norm)/Ne
        dlnPe = dlnNe+dlnTe

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003
        # lnCoul = (14.9 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te < 10) +\
        #     (15.2 - 0.5*np.log(Ne/1e20) + np.log(Te/1000))*(Te >= 10)
        # (17.3 - 0.5*np.log(Ne/1e20) + 1.5*np.log(Te/1000))*(Te >= 10)
        # lnCoul = 14
        lnCoul = core_profiles_1d.coulomb_logarithm(rho_tor_norm)

        # electron collision time , eq 14.6.1
        tau_e = 1.09e16*((Te/1000.0)**(3/2))/Ne/lnCoul

        vTe = np.sqrt(Te*eV/constants.electron_mass)

        epsilon = rho_tor/R0
        epsilon12 = np.sqrt(epsilon)
        epsilon32 = epsilon12**3

        nu_e = R0*q/vTe/tau_e/epsilon32
        # Zeff = core_profile.zeff

        x = equilibrium_1d.trapped_fraction(psi_norm)  # np.sqrt(2*epsilon)  #
        c1 = np.array((4.0+2.6*x)/(1.0+1.02*np.sqrt(nu_e)+1.07*nu_e)/(1.0 + 1.07 * epsilon32*nu_e))
        c3 = np.array((7.0+6.5*x)/(1.0+0.57*np.sqrt(nu_e)+0.61*nu_e)/(1.0 + 0.61 * epsilon32*nu_e) - c1*5/2)

        j_bootstrap = np.asarray(c1 * dlnPe + c3 * dlnTe)

        for sp in core_profiles_1d.ion:

            Ti = sp.temperature(rho_tor_norm)
            Ni = sp.density(rho_tor_norm)

            dlnTi = sp.temperature.derivative(rho_tor_norm)/Ti
            dlnNi = sp.density.derivative(rho_tor_norm)/Ni
            dlnPi = dlnNi + dlnTi
            mi = sp.a

            # ion collision time Tokamaks 3ed, eq 14.6.2 p730
            tau_i = 6.6e17*np.sqrt(mi)*((Ti/1000)**(3/2))/Ni/(1.1*lnCoul)

            # thermal velocity
            v_Ti = np.sqrt(Ti*(eV/constants.m_p/mi))

            nu_i = R0*q/epsilon32/v_Ti/tau_i

            #########################################################################
            #  Sec 14.12 Bootstrap current

            c2 = c1*Ti/Te

            e3n2 = (epsilon ** 3)*(nu_i**2)

            c4 = ((-1.17/(1.0+0.46*x) + 0.35*np.sqrt(nu_i)) / (1 + 0.7*np.sqrt(nu_i)) + 2.1*e3n2) \
                / (1 - e3n2) / (1 + e3n2)*c2

            j_bootstrap += np.asarray(c2*dlnPi + c4*dlnTi)
            # eq 4.9.2
            # j_bootstrap = j_bootstrap + Ni*Ti*eV*(2.44*dlnNe - 0.42*dlnTi)
            #########################################################################

        # eq 4.9.2
        # src.j_bootstrap = (-(q/B0/epsilon12))*j_bootstrap
        fpol = equilibrium_1d.fpol(psi_norm)
        j_bootstrap = array_like(rho_tor_norm,
                                 - j_bootstrap * x/(2.4+5.4*x+2.6*x**2) * Pe
                                 * fpol * q / rho_tor_norm / (rho_tor[-1])**2 / (2.0*constants.pi*B0))

        self["profiles_1d.j_parallel"] = Function(
            profiles_1d.grid.rho_tor_norm, np.hstack([j_bootstrap[0], j_bootstrap]))

        return residual


__SP_EXPORT__ = BootstrapCurrent
