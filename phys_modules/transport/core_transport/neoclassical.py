import collections

from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreTransport import CoreTransport
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Entry import _next_
from spdm.data.Function import Function
from spdm.numlib import constants, np
from spdm.util.logger import logger
from spdm.util.utilities import _not_found_


class NeoClassical(CoreTransport.Model):
    """
        Neoclassical Transport Model
        ===============================
        Neoclassical model, based on  Tokamaks, 3ed, J.A.Wesson 2003
        References:
        =============
        - Tokamaks, 3ed,  J.A.Wesson 2003
    """

    def __init__(self, d, *args, **kwargs):
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
        B0 = abs(equilibrium.vacuum_toroidal_field.b0)
        R0 = equilibrium.vacuum_toroidal_field.r0

        core_profile = core_profiles.profiles_1d

        rho_tor_norm = core_profile.grid.rho_tor_norm
        rho_tor = core_profile.grid.rho_tor
        psi_norm = core_profile.grid.psi_norm
        psi = core_profile.grid.psi
        q = equilibrium.time_slice.profiles_1d.q(psi_norm)

        # Tavg = np.sum([ion.density*ion.temperature for ion in core_profile.ion]) / \
        #     np.sum([ion.density for ion in core_profile.ion])

        sum1 = 0
        sum2 = 0

        Te = core_profile.electrons.temperature(rho_tor_norm)
        Ne = core_profile.electrons.density(rho_tor_norm)

        lnCoul = core_profile.coulomb_logarithm(rho_tor_norm)
        # electron collision time , eq 14.6.1

        rho_tor[0] = max(1.07e-4*((Te[0]/1000)**(1/2))/B0, rho_tor[0])  # Larmor radius,   eq 14.7.2

        epsilon = (rho_tor/R0)
        epsilon12 = np.sqrt(epsilon)
        epsilon32 = epsilon**(3/2)

        ###########################################################################################
        #  Sec 14.11 Chang-Hinton formula for \Chi_i

        # Shafranov shift
        delta_ = Function(rho_tor, np.array(
            equilibrium.time_slice.profiles_1d.geometric_axis.r(psi_norm)-R0)).derivative(rho_tor_norm)

        # impurity ions
        nZI = 0.0

        f1 = (1.0 + (epsilon**2+epsilon*delta_)*3/2 + 3/8*(epsilon**3)*delta_)/(1.0 + epsilon*delta_/2)
        f2 = np.sqrt(1-epsilon**2)*(1 + epsilon*delta_/2)/(1+delta_*(np.sqrt(1-epsilon**2)-1)/epsilon)

        sum1 = 0.0
        sum2 = 0.0
        for sp in core_profile.ion:
            Ti = sp.temperature(rho_tor_norm)
            Ni = sp.density(rho_tor_norm)

            mi = sp.a
            Zi = sp.z_ion_1d
            Zi2 = sp.z_ion_square_1d
            alpha = (nZI/(Ni*Zi*Zi))

            # Larmor radius, Tokamaks 3ed, eq 14.7.2
            rho_i = (4.57e-3 * np.sqrt(mi*Ti/1000) / B0)

            # ion collision time Tokamaks 3ed, eq 14.6.2 p730
            tau_i = (6.6e17*np.sqrt(mi)*((Ti/1000)**(3/2))/Ni/(1.1*lnCoul))

            # thermal velocity
            v_Ti = np.sqrt(Ti*(eV/constants.m_p/mi))

            nu_i = (R0*q/epsilon32/v_Ti/tau_i)
            mu_i = (nu_i*(1.0+1.54*alpha))

            chi_i = (0.66*(1.0+1.54*alpha)+(1.88*np.sqrt(epsilon)-1.54*epsilon)
                     * (1.0+3.75*epsilon))/(1.0+1.03*np.sqrt(mu_i)+0.31*mu_i)

            chi_i = chi_i * f1 + 0.59*mu_i*epsilon/(1.0+0.74*mu_i*epsilon32)\
                * (1.0 + 1.33*alpha*(1.0+0.60*alpha)/(1.0+1.79*alpha))*(f1-f2)

            chi_i = chi_i/epsilon32*(q**2)*(rho_i**2)/(1.0+0.74*mu_i*epsilon32)

            sp_trans = self.profiles_1d.ion.find({"label": sp.label}, only_first=True, default_value=_not_found_)
            if sp_trans is _not_found_:
                self.profiles_1d.ion[_next_] = {
                    "label": sp.label,
                    "z_ion": sp.z_ion,
                    "neutral_index": sp.neutral_index,
                    "element": sp.element._as_list(),
                }
            sp_trans = self.profiles_1d.ion.find({"label": sp.label}, only_first=True, default_value=_not_found_)

            # TODO: Need node to support conditional insertion
            # sp_trans = self.profiles_1d.ion.insert({"label": sp.label},
            #                                        {_next_: {
            #                                            "label": sp.label,
            #                                            "z_ion": sp.z_ion,
            #                                            "neutral_index": sp.neutral_index,
            #                                            "element": sp.element._as_list(),
            #                                        }},
            #                                        only_first=True)

            if sp_trans is _not_found_:
                logger.error(f"Can not add ion {sp.label}!")
            else:
                sp_trans.energy.d = chi_i
                sp_trans.particles.d = chi_i/3.0

            #########################################################################

            sum1 = sum1 + chi_i/3.0*sp.pressure.derivative(rho_tor_norm)*Zi/Ti
            sum2 = sum2 + chi_i/3.0*Ni*Zi2 / Ti

        self.profiles_1d.e_field_radial = sum1/sum2

        return 0.0


__SP_EXPORT__ = NeoClassical
