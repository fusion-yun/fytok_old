import collections

import numpy as np
from fytok.numlib.misc import array_like
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium
from scipy import constants
from spdm.logger import logger
from spdm.tags import _next_, _not_found_
from spdm.data import Function, function_like


class FastAlpha(CoreTransport.Model):
    """
        FastAlpha   Model

        Reference:

        [1] Angioni, C., Peeters, A. G., Pereverzev, G. V., Bottino, A., Candy, J., Dux, R., Fable, E., Hein, T., & Waltz, R. E. (2009). 
            Gyrokinetic simulations of impurity, He ash and α particle transport and consequences on ITER transport modelling. 
            Nuclear Fusion, 49(5), 055013. https://doi.org/10.1088/0029-5515/49/5/055013
        [2] Waltz, R. E., & Bass, E. M. (2014). 
            Prediction of the fusion alpha density profile in ITER from local marginal stability to Alfvén eigenmodes. 
            Nuclear Fusion, 54(10), 104006. https://doi.org/10.1088/0029-5515/54/10/104006

    """

    def __init__(self, d=None,  *args, **kwargs):
        super().__init__(d, *args,
                         identifier={"name": "fusion", "index": 5,
                                     "description": f"Fast alpha"},
                         code={"name": "FastAlpha"},
                         ** kwargs)

    def refresh(self, *args,  equilibrium: Equilibrium,  core_profiles: CoreProfiles, **kwargs) -> None:

        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        core_profiles_1d = core_profiles.profiles_1d

        rho_tor_norm = core_profiles_1d.grid.rho_tor_norm

        Te = core_profiles_1d.electrons.temperature(rho_tor_norm)

        L_Te = Te / core_profiles_1d.electrons.temperature.derivative(rho_tor_norm)

        Te_Ea = Te/3.5e6
        Ec_Ea = 33.05 * Te_Ea

        fast_factor_d = 0.02 + 4.5*(Te_Ea) + 8.0*(Te_Ea**2)+350*(Te_Ea**3)

        fast_factor_v = fast_factor_d*1.5*(1.0/np.log((Ec_Ea**(-1.5)+1)*(Ec_Ea**1.5+1))-1)/L_Te

        self.profiles_1d["ion"] = [
            {"label": "He",
             "particles": {"d_fast_factor": fast_factor_d, "v_fast_factor": fast_factor_v},
             #  "energy": {"d_fast": diff, "v_fast": vconv}
             },
        ]


__SP_EXPORT__ = FastAlpha
