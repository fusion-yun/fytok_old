import typing

from spdm.data.Expression import Variable, Expression, zero
from spdm.data.sp_property import sp_tree
from fytok.utils.atoms import nuclear_reaction, atoms
from fytok.modules.CoreSources import CoreSources
from fytok.utils.logger import logger


def momentum_exchange(variables: typing.Dict[str, Expression]):
    species = ["/".join(k.split("/")[:-1]) for k in variables.keys() if k.endswith("rotation_frequency_tor")]

    return {}
