import numpy as np
import scipy.constants
import typing


from spdm.utils.tags import _not_found_
from spdm.data.Expression import Expression, Variable, zero, Piecewise
from spdm.data.sp_property import sp_tree

from fytok.modules.CoreSources import CoreSources
from fytok.utils.atoms import nuclear_reaction, atoms
from fytok.utils.logger import logger


@sp_tree
class CollisionalEquipartition(CoreSources.Source):
    identifier = "collisional_equipartition"

    code = {"name": "collisional_equipartition", "description": "Fusion reaction"}  # type: ignore

    def fetch(self, x: Variable, variables: typing.Dict[str, Expression]):
        # 粒子组份，包含离子和电子，如 electrons, ion/D,ion/T, ...
        current: CoreSources.Source.TimeSlice = self.fetch()

        source_1d = current.profiles_1d

        epsilon = scipy.constants.epsilon_0
        e = scipy.constants.elementary_charge
        me = scipy.constants.electron_mass

        ne = variables.get(f"electrons/density")
        Te = variables.get(f"electrons/temperature")
        ve = variables.get(f"electrons/velocity/toroidal")

        clog = Piecewise(
            [
                24.0e0 - 1.15 * np.log(ne * 1.0e-6) + 2.30e0 * np.log(Te),
                23.0e0 - 1.15 * np.log(ne * 1.0e-6) + 3.45e0 * np.log(Te),
            ],
            [Te >= 10, Te < 10],
        )

        # electron collision time:
        tau_e = (np.sqrt(2.0 * me) * (Te) ** 1.5) / 1.8e-19 / (ne * 1.0e-6) / clog

        # Plasma electrical conductivity:
        source_1d.conductivity_parallel = 1.96e0 * e**2 * ne * 1.0e-6 * tau_e / me / 9.0e9

        species = [k.split("/")[1] for k in variables.keys() if k.endswith("temperature") and k.startswith("ion")]

        for idx, i in enumerate(species):
            zi = atoms[i].z
            mi = atoms[i].mass

            ni = variables.get(f"ion/{i}/density", zero)
            Ti = variables.get(f"ion/{i}/temperature", zero)
            vi = variables.get(f"ion/{i}/velocity/toroidal", zero)

            if Ti is zero:
                continue

            # electron-Ion collisions:
            #   Coulomb logarithm:
            clog = Piecewise(
                [
                    24.0e0 - 1.15 * np.log(1.0e-6) - 1.15 * np.log(ne) + 2.30 * np.log(Te),
                    23.0e0 - 1.15 * np.log(1.0e-6) - 1.15 * np.log(ne) + 3.45 * np.log(Te),
                ],
                [Te >= 10 * zi**2, Te < 10 * zi**2],
            )

            # electron-ion collision time and energy exchange term:
            tau_ie = (Te * mi + Ti * me) ** 1.5 / 1.8e-25 / (np.sqrt(mi * me)) / ni / zi**2 / clog

            source_1d.ion[i].energy += (Ti - Te) / tau_ie
            source_1d.ion[i].momentum.toroidal += (vi - ve) / tau_ie

            source_1d.electrons.energy += (Te - Ti) / tau_ie
            source_1d.electrons.momentum.toroidal += (ve - vi) / tau_ie

            # ion-ion collisions:
            for j in species[idx + 1 :]:
                zj = atoms[j].z
                mj = atoms[j].mass
                if j != "electrons":
                    j = f"ion/{j}"

                nj = variables.get(f"ion/{j}/density", zero)
                Tj = variables.get(f"ion/{j}/temperature", zero)
                vj = variables.get(f"ion/{j}/velocity/toroidal", zero)

                if Tj is zero:
                    continue

                # Coulomb logarithm:
                clog = (
                    23.0
                    - np.log(1.0e-3)
                    - np.log(zi * zj * (mi + mj) / (mi * Tj + mj * Ti))
                    - np.log(np.sqrt(ni * zi**2.0 / Ti + nj * zj**2.0 / Tj))
                )

                # ion-ion collision time and energy exchange term:
                tau_ij = (Ti * mj + Tj * mi) ** 1.5 / 1.8e-25 / (np.sqrt(mi * mj)) / ni / zi**2 / zj**2 / clog

                source_1d.ion[i].energy += (Ti - Tj) / tau_ij
                source_1d.ion[j].energy += (Tj - Ti) / tau_ij
                source_1d.ion[i].momentum.toroidal += (mi * ni * vi - mj * nj * vj) / tau_ij
                source_1d.ion[j].momentum.toroidal += (mj * nj * vj - mi * ni * vi) / tau_ij

        return current


CoreSources.Source.register(["collisional_equipartition"], CollisionalEquipartition)
