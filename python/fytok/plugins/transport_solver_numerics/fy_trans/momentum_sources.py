import typing
import numpy as np
import typing
import scipy.constants
from spdm.data.Expression import Variable, Expression, zero
from spdm.utils.tags import _not_found_

from fytok.utils.atoms import atoms
from fytok.utils.logger import logger


def momentum_sources(Uij: typing.Dict[str, Expression], variables: typing.Dict[str, Expression]):
    species = ["/".join(k.split("/")[:-1]) for k in variables.keys() if k.endswith("momentum_tor")]

    for idx, i in enumerate(species):
        ni = variables.get(f"{i}/density", _not_found_)
        ui = variables.get(f"{i}/momentum_tor", _not_found_)

        if ni is _not_found_:
            raise RuntimeError(f"Density {i} is not defined!")

        zi = atoms[i].z

        for j in species[idx + 1 :]:
            nj = variables.get(f"{j}/density", _not_found_)
            uj = variables.get(f"{j}/momentum_tor", _not_found_)
            if nj is _not_found_:
                raise RuntimeError(f"Density {j} is not defined!")

            zj = atoms[i].z
            tau_ij = zero

            U = tau_ij * (ui - uj)

            Uij[i] = Uij.get(i, zero) + U
            Uij[j] = Uij.get(j, zero) - U

    return Uij
