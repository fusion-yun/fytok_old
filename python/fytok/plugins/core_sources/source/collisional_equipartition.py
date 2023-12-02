import numpy as np
import scipy.constants
import typing
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.atoms import atoms
from spdm.utils.tags import _next_
from spdm.data.sp_property import sp_tree
from spdm.data.Expression import Expression, Variable


@CoreSources.Source.register(["collisional_equipartition"])
@sp_tree
class CollisionalEquipartition(CoreSources.Source):
    identifier = "collisional_equipartition"
    code = {"name": "collisional_equipartition", "description": "Collisional Energy Tansport "}

    def fetch(self, x: Variable, **vars: typing.Dict[str, Expression]) -> CoreSources.Source.TimeSlice:
        res: CoreSources.Source.TimeSlice = super().fetch(x, **vars)

        core_source_1d = res.profiles_1d

        # equilibrium: Equilibrium = self.inputs.get_source("equilibrium")

        core_profiles: CoreProfiles = self.inputs.get_source("core_profiles")

        core_profiles_1d = core_profiles.time_slice.current.profiles_1d

        Te = core_profiles_1d.electrons.temperature
        ne = core_profiles_1d.electrons.density

        gamma_ei = 15.2 - np.log(ne) / np.log(1.0e20) + np.log(Te) / np.log(1.0e3)
        epsilon = scipy.constants.epsilon_0
        e = scipy.constants.elementary_charge
        me = scipy.constants.electron_mass
        mp = scipy.constants.proton_mass
        PI = scipy.constants.pi

        tau_e = (
            12 * (PI ** (3 / 2)) * (epsilon**2) / (e**4) * np.sqrt(me / 2) * ((e * Te) ** (3 / 2)) / ne / gamma_ei
        )

        coeff = (3 / 2) * e / (mp / me / 2) / tau_e

        core_source_1d["ion"] = [
            {"label": ion.label, "energy": (ion.density * (ion.z**2) / ion.a * (Te - ion.temperature) * coeff)}
            for ion in core_profiles_1d.ion
        ]

        core_source_1d.electrons["energy"] = -sum([ion.energy for ion in core_source_1d.ion], 0)

        return res

    def fetch_old(self, x: Variable, **vars: typing.Dict[str, Expression]) -> CoreSources.Source.TimeSlice:
        res: CoreSourcesTimeSlice = super().fetch(x, **vars)

        ns = vars[f"{spec}/density"]

        Ts = vars[f"{spec}/temperature"]

        gamma_ei = 15.2 - np.log(ns) / np.log(1.0e20) + np.log(Ts) / np.log(1.0e3)

        epsilon = scipy.constants.epsilon_0
        e = scipy.constants.elementary_charge
        me = scipy.constants.electron_mass
        mp = scipy.constants.proton_mass
        PI = scipy.constants.pi

        tau_e = (
            12 * (PI ** (3 / 2)) * (epsilon**2) / (e**4) * np.sqrt(me / 2) * ((e * Ts) ** (3 / 2)) / ns / gamma_ei
        )

        coeff = (3 / 2) * e / (mp / me / 2) / tau_e
        q_explicit = 0
        q_implicit = 0

        for k, v in vars.items():
            if not k.endswith("density"):
                continue
            identifier = vars.get(k.removesuffix("/density"))
            nj: Expression = v
            Tj = vars.get(f"{identifier}/temperature", 0)
            spec = k.split("/")[-2]
            zj = atoms[spec].z
            aj = atoms[spec].a
            q_explicit += nj * Tj * (zj**2) / aj * coeff
            q_implicit += nj * (zj**2) / aj * coeff

        return q_explicit, q_implicit
