
import collections

import numpy as np
from fytok.constants.Atoms import atoms
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.Equilibrium import Equilibrium
from scipy import constants
from spdm.common.logger import logger
from spdm.common.tags import _next_
from spdm.data import Dict, Function, List, Node


class CollisionalEquipartition(CoreSources.Source):
    def __init__(self, d=None, /,  **kwargs):
        super().__init__(d,
                         identifier={
                             "name": f"collisional_equipartition",
                             "index": 11,
                             "description": f"{self.__class__.__name__} Collisional Energy Tansport "
                         },   **kwargs)

    def refresh(self, *args,   equilibrium: Equilibrium,  core_profiles: CoreProfiles,     **kwargs) -> float:

        residual = super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

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

        return residual


__SP_EXPORT__ = CollisionalEquipartition
