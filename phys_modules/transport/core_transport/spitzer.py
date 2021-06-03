
import collections

from spdm.numlib import np
from spdm.numlib import constants
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreTransport import CoreTransport
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.util.logger import logger


class Spitzer(CoreTransport.Model):
    """
        Spitzer Resistivity
        ===============================

        References:
        =============
        - Tokamaks, Third Edition, Chapter 14  ,p727,  J.A.Wesson 2003
    """

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

        super().update(*args, **kwargs)

        eV = constants.electron_volt
        B0 = equilibrium.vacuum_toroidal_field.b0
        R0 = equilibrium.vacuum_toroidal_field.r0

        core_profile = core_profiles.profiles_1d

        rho_tor_norm = core_profile.grid.rho_tor_norm
        rho_tor = core_profile.grid.rho_tor
        psi_norm = core_profile.grid.psi_norm
        psi = core_profile.grid.psi
        q = equilibrium.time_slice.profiles_1d.q(psi_norm)
        rho_tor_norm[0] = 0.001
        rho_tor[0] = rho_tor_norm[0]*rho_tor[-1]
        # Tavg = np.sum([ion.density*ion.temperature for ion in core_profile.ion]) / \
        #     np.sum([ion.density for ion in core_profile.ion])

        Te = core_profile.electrons.temperature(rho_tor_norm)
        Ne = core_profile.electrons.density(rho_tor_norm)
        Pe = core_profile.electrons.pressure(rho_tor_norm)

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003
        # lnCoul = (14.9 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te < 10) +\
        #     (15.2 - 0.5*np.log(Ne/1e20) + np.log(Te/1000))*(Te >= 10)
        # (17.3 - 0.5*np.log(Ne/1e20) + 1.5*np.log(Te/1000))*(Te >= 10)

        # lnCoul = 14
        lnCoul = core_profile.coulomb_logarithm(rho_tor_norm)

        # electron collision time , eq 14.6.1
        tau_e = 1.09e16*((Te/1000)**(3/2))/Ne/lnCoul

        vTe = np.sqrt(Te*eV/constants.electron_mass)

        # Larmor radius,   eq 14.7.2
        # rho_e = 1.07e-4*((Te/1000)**(1/2))/B0

        # rho_tor[0] = max(rho_e[0], rho_tor[0])

        epsilon = rho_tor/R0
        epsilon12 = np.sqrt(epsilon)
        epsilon32 = epsilon**(3/2)
        ###########################################################################################
        #  Sec 14.10 Resistivity
        #
        eta_s = 1.65e-9*lnCoul*(Te/1000)**(-3/2)
        nu_e = R0*q/vTe/tau_e/epsilon32
        Zeff = core_profile.zeff(rho_tor_norm)
        fT = 1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon))
        phi = fT/(1.0+(0.58+0.20*Zeff)*nu_e)
        C = 0.56/Zeff*(3.0-Zeff)/(3.0+Zeff)

        eta = eta_s*Zeff/(1-phi)/(1.0-C*phi)*(1.0+0.27*(Zeff-1.0))/(1.0+0.47*(Zeff-1.0))

        self.profiles_1d["conductivity_parallel"] = Function(rho_tor_norm, np.asarray(1.0/eta))

        return 0.00


__SP_EXPORT__ = Spitzer
