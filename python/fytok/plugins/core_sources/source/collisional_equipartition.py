import numpy as np
import scipy.constants
import typing
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.Equilibrium import Equilibrium


from spdm.utils.tags import _not_found_
from spdm.data.Expression import Expression, Variable, Piecewise, zero
from spdm.data.sp_property import sp_tree

from fytok.modules.CoreSources import CoreSources
from fytok.utils.atoms import atoms
from fytok.utils.logger import logger


@sp_tree
class CollisionalEquipartition(CoreSources.Source):
    identifier = "collisional_equipartition"

    code = {"name": "collisional_equipartition", "description": "Fusion reaction"}  # type: ignore

    def fetch(self, x: Variable, **variables: Expression) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch()

        source_1d = current.profiles_1d

        e = scipy.constants.elementary_charge
        me = scipy.constants.electron_mass

        ne = variables.get(f"electrons/density")
        Te = variables.get(f"electrons/temperature")
        ve = variables.get(f"electrons/velocity/toroidal")

        clog = Piecewise(
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
                    (30.9 - 1.15 * np.log10(ne) + 2.30 * np.log10(Te), Te >= 10 * zi**2),
                    (29.9 - 1.15 * np.log10(ne) + 3.45 * np.log10(Te), Te < 10 * zi**2),
                ],
                name="clog",
                label=r"\Lambda_{ei}",
            )

            # clog = 24.0e0 - 1.15 * np.log(1.0e-6) - 1.15 * np.log(ne) + 2.30 * np.log(Te)

            # electron-ion collision time and energy exchange term:
            tau_ie = (Te * mi + Ti * me) ** 1.5 / 1.8e-25 / (np.sqrt(mi * me)) / zi**2 / clog

            source_1d.ion[i].energy += ni * (Ti - Te) / tau_ie
            source_1d.ion[i].momentum.toroidal += me * ne * (vi - ve) / tau_ie

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
                    29.9
                    - np.log10(zi * zj * (mi + mj) / (mi * Tj + mj * Ti))
                    - np.log10(np.sqrt(ni * zi**2.0 / Ti + nj * zj**2.0 / Tj))
                )

                # ion-ion collision time and energy exchange term:
                tau_ij = (Ti * mj + Tj * mi) ** 1.5 / 1.8e-25 / (np.sqrt(mi * mj)) / (zi * zj) ** 2 / clog

                source_1d.ion[i].energy += (Ti - Tj) / tau_ij
                source_1d.ion[j].energy += (Tj - Ti) / tau_ij
                source_1d.ion[i].momentum.toroidal += (mi * ni * vi - mj * nj * vj) / tau_ij
                source_1d.ion[j].momentum.toroidal += (mj * nj * vj - mi * ni * vi) / tau_ij

        return current


CoreSources.Source.register(["collisional_equipartition"], CollisionalEquipartition)
