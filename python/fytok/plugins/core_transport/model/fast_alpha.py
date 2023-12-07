import collections

import numpy as np
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.Equilibrium import Equilibrium
from scipy import constants
from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_
from spdm.data.Expression import Variable, Expression
from spdm.data.sp_property import sp_tree


@CoreTransport.Model.register(["fast_alpha"])
@sp_tree
class FastAlpha(CoreTransport.Model):
    """
    FastAlpha   Model

    Reference:

    [1] Angioni, C., Peeters, A. G., Pereverzev, G. V., Bottino, A., Candy, J., Dux, R., Fable, E., Hein, T., & Waltz, R. E. (2009).
        Gyrokinetic simulations of impurity, He ash and \\alpha particle transport and consequences on ITER transport modelling.
        Nuclear Fusion, 49(5), 055013. https://doi.org/10.1088/0029-5515/49/5/055013
    [2] Waltz, R. E., & Bass, E. M. (2014).
        Prediction of the fusion alpha density profile in ITER from local marginal stability to Alfvén eigenmodes.
        Nuclear Fusion, 54(10), 104006. https://doi.org/10.1088/0029-5515/54/10/104006

    """

    identifier = "slowing_down"
    code = {"name": "fast_alpha", "description": f" Fast alpha", "copyright": "fytok"}

    def fetch(self, x: Variable, **vars: Expression) -> CoreTransport.Model.TimeSlice:
        current: CoreTransport.Model.TimeSlice = super().fetch(x, **vars)

        Te = vars.get("electrons/temperature")
        # ne = vars.get("electrons/density")
        inv_L_Te = Te.dln

        Te_Ea = Te / 3.5e6

        Ec_Ea = 33.05 * Te_Ea

        fast_factor_d = 0.02 + 4.5 * (Te_Ea) + 8.0 * (Te_Ea**2) + 350 * (Te_Ea**3)

        fast_factor_v = fast_factor_d * 1.5 * (1.0 / np.log((Ec_Ea ** (-1.5) + 1) * (Ec_Ea**1.5 + 1)) - 1) * inv_L_Te

        core_trans_1d = current.profiles_1d

        logger.debug(fast_factor_d._repr_latex_())

        core_trans_1d["ion"] = [
            {
                "label": "alpha",
                "particles": {"d": fast_factor_d, "v": fast_factor_v},
                #  "energy": {"d_fast": diff, "v_fast": vconv}
            },
        ]
        return current
