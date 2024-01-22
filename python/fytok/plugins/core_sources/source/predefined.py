import collections

import numpy as np
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.Equilibrium import Equilibrium
from spdm.utils.logger import logger
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_
from spdm.data.Expression import Variable, Expression, Piecewise, derivative
from spdm.data.sp_property import sp_tree
from spdm.numlib.misc import step_function_approx
import typing
import scipy.constants
from spdm.data.Expression import Variable, Expression, zero, antiderivative
from spdm.data.sp_property import sp_tree
from spdm.numlib.misc import step_function_approx
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_

from fytok.utils.atoms import nuclear_reaction, atoms
from fytok.utils.logger import logger
from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.Utilities import *

PI = scipy.constants.pi


@sp_tree
class PredefinedSource(CoreSources.Source):
    identifier = "predefined"
    code = {"name": "predefined", "description": f"predefined"}

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D, *args, **kwargs) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(*args, **kwargs)

        rho_tor_norm = profiles_1d.rho_tor_norm

        _x = rho_tor_norm

        S = 9e20 * np.exp(15.0 * (_x**2 - 1.0))

        current.profiles_1d.grid = profiles_1d.grid
        current.profiles_1d.electrons.particles = S

        current.profiles_1d.ion.extend(
            [
                {"@name": "D", "particles": S * 0.48},
                {"@name": "T", "particles": S * 0.48},
                {"@name": "He", "particles": S * 0.02},
            ]
        )

        return current


CoreTransport.Model.register(["predefined"], PredefinedSource)
