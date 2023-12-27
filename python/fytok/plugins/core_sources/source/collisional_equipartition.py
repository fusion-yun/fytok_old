import numpy as np
import scipy.constants
import typing
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.Equilibrium import Equilibrium

from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_
from spdm.data.Expression import Expression, Variable, Piecewise, piecewise, zero
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

        ne = variables.get(f"electrons/density")
        Te = variables.get(f"electrons/temperature")
        ve = variables.get(f"electrons/velocity/toroidal")

        clog = piecewise(
            [
                (30.9 - 1.15 * np.log10(ne) + 2.30 * np.log10(Te), Te >= 10),
                (29.9 - 1.15 * np.log10(ne) + 3.45 * np.log10(Te), Te < 10),
            ],
            name="clog",
            label=r"\Lambda_{e}",
        )

        # electron collision time:
        tau_e = np.sqrt(2.0 * me) / 1.8e-25 * (Te**1.5) / ne / clog

        # Plasma electrical conductivity:
        source_1d.conductivity_parallel = 1.96e-09 * e**2 / me * ne * tau_e

        species = [
            "/".join(k.split("/")[:-1])
            for k, v in variables.items()
            if k.endswith("temperature")
            and v is not zero
            and not (not isinstance(v, (array_type, Expression)) and v == 0)
        ]
        logger.debug(species)
        for idx, i in enumerate(species):
            zi = atoms[i].z
            mi = atoms[i].mass

            ni = variables.get(f"{i}/density", zero)
            Ti = variables.get(f"{i}/temperature", zero)
            vi = variables.get(f"{i}/velocity/toroidal", zero)

            if Ti is zero:
                continue

            # ion-ion collisions:
            for j in species[idx + 1 :]:
                zj = atoms[j].z
                mj = atoms[j].mass

                nj = variables.get(f"{j}/density", zero)
                Tj = variables.get(f"{j}/temperature", zero)
                vj = variables.get(f"{j}/velocity/toroidal", zero)

                if Tj is _not_found_ or Tj is zero:
                    continue

                #   Coulomb logarithm:
                if i == "electrons":  # electron-Ion collisions:
                    clog = piecewise(
                        [
                            (30.9 - 1.15 * np.log10(ni) + 2.30 * np.log10(Ti), Ti >= 10 * zi**2),
                            (29.9 - 1.15 * np.log10(ni) + 3.45 * np.log10(Ti), Ti < 10 * zi**2),
                        ],
                        name="clog",
                        label=r"\Lambda_{ei}",
                    )

                else:  # ion-Ion collisions:
                    clog = (
                        29.9
                        - np.log10(zi * zj * (mi + mj) / (mi * Tj + mj * Ti))
                        - np.log10(np.sqrt(ni * zi**2.0 / Ti + nj * zj**2.0 / Tj))
                    )

                # collision time :
                tau_ij = (Ti * mj + Tj * mi) ** 1.5 / 1.8e-25 / (np.sqrt(mi * mj)) / (zi * zj) ** 2 / clog

                # energy exchange term
                source_1d[i].energy += (Ti - Tj) / tau_ij
                source_1d[j].energy += (Tj - Ti) / tau_ij

                # momentum exchange term
                source_1d[i].momentum.toroidal += mj * nj * (vi - vj) / tau_ij
                source_1d[j].momentum.toroidal += mi * ni * (vj - vi) / tau_ij

        return current


CoreSources.Source.register(["collisional_equipartition"], CollisionalEquipartition)
