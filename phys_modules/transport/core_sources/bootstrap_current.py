
import collections

from spdm.numlib import np
from spdm.numlib import constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreSources import CoreSources
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, Node
from spdm.util.logger import logger


class BootstrapCurrent(CoreSources.Source):
    def __init__(self, d=None, *args,  **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {
                "name": f"bootstrap_current",
                "index": 13,
                "description": f"{self.__class__.__name__} Bootstrap current, based on  Tokamaks, 3ed, sec 14.12 J.A.Wesson 2003"
            }}, d or {}), *args, **kwargs)

    def update(self, *args,
               equilibrium: Equilibrium,
               core_profiles: CoreProfiles,
               **kwargs):

        super().update(*args, core_profiles=core_profiles, **kwargs)

        eV = constants.electron_volt
        B0 = abs(equilibrium.vacuum_toroidal_field.b0)
        R0 = equilibrium.vacuum_toroidal_field.r0

        core_profile = core_profiles.profiles_1d

        rho_tor_norm = np.asarray(core_profile.grid.rho_tor_norm)
        rho_tor = np.asarray(core_profile.grid.rho_tor)
        psi_norm = np.asarray(core_profile.grid.psi_norm)
        psi = np.asarray(core_profile.grid.psi)
        q = np.asarray(equilibrium.time_slice.profiles_1d.q(core_profile.grid.psi_norm))

        # Tavg = np.sum([ion.density*ion.temperature for ion in core_profile.ion]) / \
        #     np.sum([ion.density for ion in core_profile.ion])

        Te = core_profile.electrons.temperature
        Ne = core_profile.electrons.density
        Pe = core_profile.electrons.pressure
        dlnTe = Te.derivative/Te
        dlnPe = Pe.derivative/Pe
        Te = np.asarray(Te)
        Ne = np.asarray(Ne)
        Pe = np.asarray(Pe)

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003
        # lnCoul = (14.9 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te < 10) +\
        #     (15.2 - 0.5*np.log(Ne/1e20) + np.log(Te/1000))*(Te >= 10)
        # (17.3 - 0.5*np.log(Ne/1e20) + 1.5*np.log(Te/1000))*(Te >= 10)
        # lnCoul = 14
        lnCoul = core_profile.coulomb_logarithm

        # electron collision time , eq 14.6.1
        tau_e = np.asarray(1.09e16*((Te/1000)**(3/2))/Ne/lnCoul)

        vTe = np.asarray(np.sqrt(Te*eV/constants.electron_mass))

        rho_tor[0] = max(np.asarray(1.07e-4*((Te[0]/1000)**(1/2))/B0), rho_tor[0])   # Larmor radius,   eq 14.7.2

        epsilon = np.asarray(rho_tor/R0)
        epsilon12 = np.sqrt(epsilon)
        epsilon32 = epsilon**(3/2)

        nu_e = np.asarray(R0*q/vTe/tau_e/epsilon32)
        Zeff = np.asarray(core_profile.zeff)

        x = equilibrium.time_slice.profiles_1d.trapped_fraction(psi_norm)  # np.sqrt(2*epsilon)  #
        c1 = (4.0+2.6*x)/(1.0+1.02*np.sqrt(nu_e)+1.07*nu_e)/(1.0 + 1.07 * epsilon32*nu_e)
        c3 = (7.0+6.5*x)/(1.0+0.57*np.sqrt(nu_e)+0.61*nu_e)/(1.0 + 0.61 * epsilon32*nu_e) - c1*5/2

        j_bootstrap = c1 * dlnPe + c3 * dlnTe

        for sp in core_profile.ion:
            Ti = sp.temperature
            Ni = sp.density
            Pi = sp.pressure
            dlnTi = Ti.derivative/Ti
            dlnPi = Pi.derivative/Pi
            mi = sp.a

            # ion collision time Tokamaks 3ed, eq 14.6.2 p730
            tau_i = np.asarray(6.6e17*np.sqrt(mi)*((Ti/1000)**(3/2))/Ni/(1.1*lnCoul))

            # thermal velocity
            v_Ti = np.sqrt(Ti*(eV/constants.m_p/mi))

            nu_i = np.asarray(R0*q/epsilon32/v_Ti/tau_i)

            #########################################################################
            #  Sec 14.12 Bootstrap current

            c2 = c1*Ti/Te

            e3n2 = (epsilon ** 3)*(nu_i**2)

            c4 = ((-1.17/(1.0+0.46*x) + 0.35*np.sqrt(nu_i)) / (1 + 0.7*np.sqrt(nu_i)) + 2.1*e3n2) \
                / (1 - e3n2) / (1 + e3n2)*c2

            j_bootstrap = j_bootstrap + c2*dlnPi + c4*dlnTi

            # eq 4.9.2
            # j_bootstrap = j_bootstrap + Ni*Ti*eV*(2.44*dlnNe - 0.42*dlnTi)
            #########################################################################

        # eq 4.9.2
        # src.j_bootstrap = (-(q/B0/epsilon12))*j_bootstrap

        j_bootstrap = - j_bootstrap * x/(2.4+5.4*x+2.6*x**2) * Pe * \
            equilibrium.time_slice.profiles_1d.fpol(psi_norm) * q / rho_tor / rho_tor[-1] / (2.0*constants.pi*B0)
        self.profiles_1d.j_parallel = j_bootstrap
        return 0.0


__SP_EXPORT__ = BootstrapCurrent
