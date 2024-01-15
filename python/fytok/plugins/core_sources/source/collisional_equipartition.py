import numpy as np
import scipy.constants
import typing


from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_
from spdm.data.Expression import Expression, Variable, Piecewise, piecewise, smooth, zero
from spdm.data.sp_property import sp_tree
from fytok.modules.CoreProfiles import CoreProfiles, CoreProfilesSpecies
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.CoreSources import CoreSources
from fytok.utils.atoms import atoms
from fytok.utils.logger import logger


@sp_tree
class CollisionalEquipartition(CoreSources.Source):
    identifier = "collisional_equipartition"

    code = {"name": "collisional_equipartition", "description": "Fusion reaction"}  # type: ignore

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(profiles_1d)

        source_1d = current.profiles_1d

        e = scipy.constants.elementary_charge
        ze = -1.0
        me = scipy.constants.electron_mass
        ae = scipy.constants.electron_mass / scipy.constants.atomic_mass

        ne = profiles_1d.electrons.density
        Te = profiles_1d.electrons.temperature
        ve = profiles_1d.electrons.velocity.toroidal

        conductivity_parallel = zero

        species: typing.List[CoreProfilesSpecies] = [*profiles_1d.ion]

        for i, ion_i in enumerate(species[:-1]):
            zi = ion_i.z
            ai = ion_i.a
            mi = ion_i.a * scipy.constants.atomic_mass

            ni = ion_i.density
            Ti = ion_i.temperature
            vi = ion_i.velocity.toroidal

            if Ti is _not_found_:
                continue

            clog_ei = piecewise(
                [
                    (
                        16 - np.log(ze * ze * ae) - 0.5 * np.log(ne * 1.0e-6) + 1.5 * np.log(Te),
                        Te <= (Ti * me / me),
                    ),
                    (
                        23 - np.log(ze) - 0.5 * np.log(ne * 1.0e-6) + 1.5 * np.log(Te),
                        ((Ti * me / me) < Te) & (Te <= (10 * ze * ze)),
                    ),
                    (
                        24 - 0.5 * np.log(ne * 1.0e-6) + np.log(Te),
                        ((10 * ze * ze) < Te),  # & (Ti * me / mj) < (10 * zj * zj))
                    ),
                ],
                name="clog",
                label=r"\Lambda_{ei}",
            )
            nv_ei = 3.2e-9 * ze * ze * clog_ei / ae * Te**1.5

            conductivity_parallel += 1.96e-09 * e**2 / me * ne * ne / nv_ei

            nv_ei = ni * ne * nv_ei * 1.0e-12

            Tei = Ti - Te

            # energy exchange term

            source_1d.ion[ion_i.label].energy -= Tei * nv_ei
            source_1d.electrons.energy += Tei * nv_ei

            # momentum exchange term
            source_1d.ion[ion_i.label].momentum.toroidal += (vi - ve) * nv_ei
            source_1d.electrons.momentum.toroidal += (ve - vi) * nv_ei

            # collisions frequency and energy exchange terms:
            # @ref NRL 2019 p.34
            for ion_j in species[i + 1 :]:
                # ion-Ion collisions:

                zj = ion_j.z
                aj = ion_j.a
                mj = ion_j.a * scipy.constants.atomic_mass

                nj = ion_j.density
                Tj = ion_j.temperature
                vj = ion_j.velocity.toroidal

                if Tj is _not_found_:
                    continue

                # Coulomb logarithm:
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

                # energy exchange term

                source_1d.ion[ion_i.label].energy -= Tij * nv_ij
                source_1d.ion[ion_j.label].energy += Tij * nv_ij

                # momentum exchange term
                source_1d.ion[ion_i.label].momentum.toroidal += (vi - vj) * nv_ij
                source_1d.ion[ion_j.label].momentum.toroidal += (vj - vi) * nv_ij

                # Tij = Expression(deburr, Ti - Tj)

                ##############################
                # 增加阻尼，消除震荡
                # epsilon = 1.0e-10

                # c = (1.5 - 1.0 / (1.0 + np.exp(-np.abs(Ti - Tj) / (Ti + Tj) / epsilon))) * 2
                # Tij = (Ti - Tj) * c
                # if isinstance(c, array_type):
                #     logger.debug((c,))
                ##############################
        # Plasma electrical conductivity:
        source_1d.conductivity_parallel = conductivity_parallel
        return current


CoreSources.Source.register(["collisional_equipartition"], CollisionalEquipartition)
