import numpy as np
import scipy.constants
import typing

from fytok.utils.atoms import atoms

from spdm.utils.tags import _not_found_
from spdm.data.Expression import Expression, Variable, zero


def collisional_sources(Qij: typing.Dict[str, Expression], variables: typing.Dict[str, Expression]):
    # 粒子组份，包含离子和电子，如 electrons, ion/D,ion/T, ...
    return {}
    species = ["/".join(k.split("/")[:-1]) for k in variables.keys() if k.endswith("temperature")]

    epsilon = scipy.constants.epsilon_0
    e = scipy.constants.elementary_charge
    me = scipy.constants.electron_mass
    mp = scipy.constants.proton_mass
    PI = scipy.constants.pi

    for idx, i in enumerate(species):
        ni = variables.get(f"{i}/density", _not_found_)
        Ti = variables.get(f"{i}/temperature", _not_found_)
        if ni is _not_found_:
            raise RuntimeError(f"Density {i} is not defined!")

        zi = atoms[i].z
        ai = atoms[i].a

        for j in species[idx + 1 :]:
            nj = variables.get(f"{j}/density", _not_found_)
            Tj = variables.get(f"{j}/temperature", _not_found_)
            if nj is _not_found_:
                raise RuntimeError(f"Density {j} is not defined!")

            zj = atoms[i].z
            aj = atoms[i].a

            gamma_ij = 15.2 - np.log(ni) + np.log(1.0e20) + np.log(Ti) - np.log(1.0e3)

            tau_i = (
                12
                * (PI ** (3 / 2))
                * (epsilon**2)
                / (e**4)
                * np.sqrt(me / 2)
                * ((e * Ti) ** (3 / 2))
                / ni
                / gamma_ij
            )

            coeff = (3 / 2) * zi / (aj / ai / 2) / tau_i

            nu_ij = ni * nj * (zj**2) / aj * coeff

            Q = nu_ij * (Ti - Tj)

            Qij[i] = Qij.get(i, zero) + Q
            Qij[j] = Qij.get(j, zero) - Q

    return Qij
