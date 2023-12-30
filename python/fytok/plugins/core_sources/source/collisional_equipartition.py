import numpy as np
import scipy.constants
import typing
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.Equilibrium import Equilibrium

from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_
from spdm.data.Expression import Expression, Variable, Piecewise, piecewise, smooth, zero
from spdm.data.sp_property import sp_tree

from fytok.modules.CoreSources import CoreSources
from fytok.utils.atoms import atoms
from fytok.utils.logger import logger


@sp_tree
class CollisionalEquipartition(CoreSources.Source):
    identifier = "collisional_equipartition"

    code = {"name": "collisional_equipartition", "description": "Fusion reaction"}  # type: ignore

    def fetch(self, x: Variable, **variables: Expression) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(x, **variables)

        source_1d = current.profiles_1d

        e = scipy.constants.elementary_charge
        me = scipy.constants.electron_mass
        ae = scipy.constants.electron_mass / scipy.constants.atomic_mass

        ne = variables.get(f"electrons/density")
        Te = variables.get(f"electrons/temperature")

        conductivity_parallel = zero

        species = ["electrons"] + [
            "/".join(k.split("/")[:-1])
            for k, v in variables.items()
            if k.endswith("/temperature")
            and k.startswith("ion/")
            and v is not zero
            and not (not isinstance(v, (array_type, Expression)) and v == 0)
        ]

        for idx, i in enumerate(species):
            zi = atoms[i].z
            ai = atoms[i].a
            mi = atoms[i].mass

            ni = variables.get(f"{i}/density")
            Ti = variables.get(f"{i}/temperature")
            vi = variables.get(f"{i}/velocity/toroidal", zero)

            # collisions frequency and energy exchange terms:
            # @ref NRL 2019 p.34
            for j in species[idx + 1 :]:
                zj = atoms[j].z
                aj = atoms[j].a
                mj = atoms[j].mass

                nj = variables.get(f"{j}/density")
                Tj = variables.get(f"{j}/temperature")
                vj = variables.get(f"{j}/velocity/toroidal", zero)

                # Coulomb logarithm:
                if i == "electrons":  # electron-Ion collisions:
                    clog = piecewise(
                        [
                            (
                                16 - np.log(zj * zj * aj) - 0.5 * np.log(nj * 1.0e-6) + 1.5 * np.log(Tj),
                                Te <= (Ti * me / mj),
                            ),
                            (
                                23 - np.log(zj) - 0.5 * np.log(ne * 1.0e-6) + 1.5 * np.log(Te),
                                ((Ti * me / mj) < Te) & (Te <= (10 * zj * zj)),
                            ),
                            (
                                24 - 0.5 * np.log(ne * 1.0e-6) + np.log(Te),
                                ((10 * zj * zj) < Te),  # & (Ti * me / mj) < (10 * zj * zj))
                            ),
                        ],
                        name="clog",
                        label=r"\Lambda_{ei}",
                    )
                    nv_ij = 3.2e-9 * zj * zj * clog / aj * Te**1.5

                    conductivity_parallel += 1.96e-09 * e**2 / me * ne * nj / nv_ij
                else:  # ion-Ion collisions:
                    clog = (
                        23
                        - np.log(zi * zj * (ai + aj) / (ai * aj) / (Ti / ai + Tj / aj))
                        - 0.5 * np.log((ni * zi * zi / Ti + nj * zj * zj / Tj) * 1.0e-6)
                    )

                    nv_ij = (
                        1.8e-19
                        * (mi * mj * 1.0e-6)
                        * (zi * zi * zj * zj)
                        * (((Ti * mj + Tj * mi) * 1.0e-3) ** (-1.5))
                        * clog
                    )

                # nv_ij = smooth(nv_ij, window_length=3, polyorder=2)

                nv_ij = ni * nj * nv_ij * 1.0e-12

                Tij = Ti - Tj
                # Tij = Expression(deburr, Ti - Tj)

                ##############################
                # 增加阻尼，消除震荡
                # epsilon = 1.0e-10

                # c = (1.5 - 1.0 / (1.0 + np.exp(-np.abs(Ti - Tj) / (Ti + Tj) / epsilon))) * 2
                # Tij = (Ti - Tj) * c
                # if isinstance(c, array_type):
                #     logger.debug((c,))
                ##############################

                # energy exchange term

                source_1d[i].energy -= Tij * nv_ij
                source_1d[j].energy += Tij * nv_ij

                # momentum exchange term
                source_1d[i].momentum.toroidal += (vi - vj) * nv_ij
                source_1d[j].momentum.toroidal += (vj - vi) * nv_ij

        # Plasma electrical conductivity:
        source_1d.conductivity_parallel = conductivity_parallel
        return current


CoreSources.Source.register(["collisional_equipartition"], CollisionalEquipartition)
