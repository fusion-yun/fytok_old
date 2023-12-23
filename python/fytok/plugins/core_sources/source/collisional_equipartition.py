import numpy as np
import scipy.constants
import typing
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.atoms import atoms
from spdm.utils.tags import _next_, _not_found_
from spdm.data.sp_property import sp_tree
from spdm.data.Expression import Expression, Variable, zero


@sp_tree
class CollisionalEquipartition(CoreSources.Source):
    identifier = "collisional_equipartition"
    code = {"name": "collisional_equipartition", "description": "Collisional Energy Tansport "}

    def fetch_ij(self, x: Variable, **variables: typing.Dict[str, Expression]) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch()
        core_source_1d = current.profiles_1d

        epsilon = scipy.constants.epsilon_0
        e = scipy.constants.elementary_charge
        me = scipy.constants.electron_mass
        mp = scipy.constants.proton_mass
        PI = scipy.constants.pi

        species = [
            "/".join(identifier.split("/")[:-1])
            for identifier in variables.keys()
            if identifier.endswith("temperature")
        ]

        Qij = {}
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

        core_source_1d.update({s: {"label": s.split("/")[-1], "energy": v} for s, v in Qij.items()})

        return current

    def fetch(self, x: Variable, **variables: typing.Dict[str, Expression]) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch()

        core_source_1d = current.profiles_1d

        Te = variables.get("electrons/temperature")
        ne = variables.get("electrons/density")

        gamma_ei = 15.2 - (np.log(ne) - np.log(1.0e20)) + (np.log(Te) - np.log(1.0e3))

        epsilon = scipy.constants.epsilon_0
        e = scipy.constants.elementary_charge
        me = scipy.constants.electron_mass
        mp = scipy.constants.proton_mass
        PI = scipy.constants.pi

        tau_e = (
            12 * (PI ** (3 / 2)) * (epsilon**2) / (e**4) * np.sqrt(me / 2) * ((e * Te) ** (3 / 2)) / ne / gamma_ei
        )

        coeff = (3 / 2) * e / (mp / me / 2) / tau_e

        Qei = zero

        core_source_ion = []
        for identifier, ns in variables.items():
            if not identifier.endswith("density") or identifier.startswith("electrons"):
                continue

            label = identifier.split("/")[-2]

            Ts = variables.get(f"ion/{label}/temperature", _not_found_)

            if Ts is _not_found_:
                continue

            ion = atoms[label]

            Qie = ns * (ion.z**2) / ion.a * (Te - Ts) * coeff

            core_source_ion.append({"label": label, "energy": Qie})

            Qei -= Qie

        core_source_1d["ion"] = core_source_ion

        core_source_1d.electrons["energy"] = Qei

        return current


CoreSources.Source.register(["collisional_equipartition"], CollisionalEquipartition)
