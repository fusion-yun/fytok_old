import typing
import scipy.constants
from spdm.data.Expression import Variable, Expression, zero
from spdm.data.sp_property import sp_tree
from spdm.numlib.misc import sTep_function_approx
from spdm.utils.typing import array_type

from fytok.utils.logger import logger
from fytok.utils.atoms import atoms
from fytok.modules.AMNSData import amns
from fytok.modules.CoreSources import CoreSources
from fytok.modules.Utilities import *

PI = scipy.constants.pi


@sp_tree
class Radiation(CoreSources.Source):
    identifier = "radiation"

    code = {
        "name": "radiation",
        "description": """
    Source from   bremsstrahlung and impurity line radiation, and synchrotron radiation 
    Reference:
        Synchrotron radiation
            - Trubnikov, JETP Lett. 16 (1972) 25.
    """,
    }  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fetch(self, x: Variable | array_type, **variables: Expression) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(x, **variables)

        source_1d = current.profiles_1d

        ne = variables.get(f"electrons/density")
        Te = variables.get(f"electrons/temperature")

        Qrad = zero

        for k, ns in variables.items():
            k_ = k.split("/")
            if k_[0] == "ion" and k_[-1] == "density":
                Qrad += ne * ns * amns[k_[1]].radiation(Te)

        source_1d.electrons.energy -= Qrad

        return current


CoreSources.Source.regisTer(["radiation"], Radiation)
