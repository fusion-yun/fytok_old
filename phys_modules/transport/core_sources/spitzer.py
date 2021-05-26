
import collections

from spdm.util.numlib import np
from spdm.util.numlib import constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreSources import CoreSources
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, Node
from spdm.util.logger import logger


class Spitzer(CoreSources.Source):
    def __init__(self, d=None, *args,  **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {
                "name": f"neoclassical",
                "index": 5,
                "description": f"{self.__class__.__name__}  Neoclassical model, based on  Tokamaks, 3ed, J.A.Wesson 2003"
            }}, d or {}), *args, **kwargs)

    def update(self, *args,
               equilibrium: Equilibrium,
               core_profiles: CoreProfiles,
               **kwargs):

        super().update(*args, core_profiles=core_profiles, **kwargs)

        eV = constants.electron_volt
        B0 = equilibrium.vacuum_toroidal_field.b0
        R0 = equilibrium.vacuum_toroidal_field.r0

        core_profile = core_profiles.profiles_1d
        trans = self.profiles_1d

        rho_tor_norm = np.asarray(core_profile.grid.rho_tor_norm)
        rho_tor = np.asarray(core_profile.grid.rho_tor)
        psi_norm = np.asarray(core_profile.grid.psi_norm)
        psi = np.asarray(core_profile.grid.psi)
        q = np.asarray(equilibrium.profiles_1d.q(core_profile.grid.psi_norm))

        # Tavg = np.sum([ion.density*ion.temperature for ion in core_profile.ion]) / \
        #     np.sum([ion.density for ion in core_profile.ion])

        Te = np.asarray(core_profile.electrons.temperature)
        Ne = np.asarray(core_profile.electrons.density)
        Pe = np.asarray(core_profile.electrons.pressure)

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003
        # lnCoul = (14.9 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te < 10) +\
        #     (15.2 - 0.5*np.log(Ne/1e20) + np.log(Te/1000))*(Te >= 10)
        # (17.3 - 0.5*np.log(Ne/1e20) + 1.5*np.log(Te/1000))*(Te >= 10)

        # lnCoul = 14
        lnCoul=core_profile.coulomb_logarithm
        
        # electron collision time , eq 14.6.1
        tau_e = np.asarray(1.09e16*((Te/1000)**(3/2))/Ne/lnCoul)

        vTe = np.asarray(np.sqrt(Te*eV/constants.electron_mass))

        # Larmor radius,   eq 14.7.2
        rho_e = np.asarray(1.07e-4*((Te/1000)**(1/2))/B0)

        rho_tor[0] = max(rho_e[0], rho_tor[0])

        epsilon = np.asarray(rho_tor/R0)
        epsilon12 = np.sqrt(epsilon)
        epsilon32 = epsilon**(3/2)
        ###########################################################################################
        #  Sec 14.10 Resistivity
        #
        eta_s = np.asarray(1.65e-9*lnCoul*(Te/1000)**(-3/2))
        nu_e = np.asarray(R0*q/vTe/tau_e/epsilon32)
        Zeff = np.asarray(core_profile.zeff)
        fT = 1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon))
        phi = np.asarray(fT/(1.0+(0.58+0.20*Zeff)*nu_e))
        C = np.asarray(0.56/Zeff*(3.0-Zeff)/(3.0+Zeff))

        eta = eta_s*Zeff/(1-phi)/(1.0-C*phi)*(1.0+0.27*(Zeff-1.0))/(1.0+0.47*(Zeff-1.0))
        trans.conductivity_parallel = 1.0/eta

        return 0.02


__SP_EXPORT__ = Spitzer
