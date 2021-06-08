
import collections

from spdm.numlib import np
from spdm.numlib import constants
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, Node, _next_
from spdm.util.logger import logger
from fytok.common.Atoms import atoms


class CoreSourceQei(CoreSources.Source):
    def __init__(self, d=None, *args,  **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {
                "name": f"Qei Source",
                "index": -1,
                "description": f"{self.__class__.__name__} Qei Source "
            }}, d or {}), *args, **kwargs)

    def update(self, *args,
               equilibrium: Equilibrium,
               core_profiles: CoreProfiles,
               **kwargs):

        super().update(*args, **kwargs)

        Te = core_profiles.profiles_1d.electrons.temperature
        ne = core_profiles.profiles_1d.electrons.density

        gamma_ei = 15.2 - np.log(ne)/np.log(1.0e20) + np.log(Te)/np.log(1.0e3)
        epsilon = constants.epsilon_0
        e = constants.elementary_charge
        me = constants.electron_mass
        mp = constants.proton_mass
        PI = constants.pi
        tau_e = 12*(PI**(3/2))*(epsilon**2)/(e**4)*np.sqrt(me/2)*((e*Te)**(3/2))/ne/gamma_ei

        def qei_f(ion: CoreProfiles.Profiles1D.Ion):
            return ion.density*(ion.z_ion**2)/sum(ele.atoms_n*ele.a for ele in ion.element)*(Te-ion.temperature)

        coeff = (3/2) * e/(mp/me/2)/tau_e
        q_ie = 0
        for ion in core_profiles.profiles_1d.ion:
            q_ei = ion.density*(ion.z_ion**2)/sum(ele.atoms_n*ele.a for ele in ion.element)*(Te-ion.temperature)*coeff
            self.profiles_1d.ion[_next_] = {
                **atoms[ion.label],
                "label": ion.label,
                "energy": q_ei
            }
            q_ie = q_ie + q_ei

        self.profiles_1d.electrons["energy"] = -q_ie

        return 0.0


__SP_EXPORT__ = CoreSourceQei
