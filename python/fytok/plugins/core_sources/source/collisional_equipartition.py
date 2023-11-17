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

    def execute(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs) -> float:
        residual = super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        Te = core_profiles.profiles_1d.electrons.temperature
        ne = core_profiles.profiles_1d.electrons.density

        gamma_ei = 15.2 - np.log(ne) / np.log(1.0e20) + np.log(Te) / np.log(1.0e3)
        epsilon = scipy.constants.epsilon_0
        e = scipy.constants.elementary_charge
        me = scipy.constants.electron_mass
        mp = scipy.constants.proton_mass
        PI = scipy.constants.pi
        tau_e = (
            12 * (PI ** (3 / 2)) * (epsilon**2) / (e**4) * np.sqrt(me / 2) * ((e * Te) ** (3 / 2)) / ne / gamma_ei
        )

        def qei_f(ion: CoreProfiles.Profiles1D.Ion):
            return (
                ion.density
                * (ion.z_ion**2)
                / sum(ele.atoms_n * ele.a for ele in ion.element)
                * (Te - ion.temperature)
            )

        coeff = (3 / 2) * e / (mp / me / 2) / tau_e
        q_ie = 0
        for ion in core_profiles.profiles_1d.ion:
            q_ei = (
                ion.density
                * (ion.z_ion**2)
                / sum(ele.atoms_n * ele.a for ele in ion.element)
                * (Te - ion.temperature)
                * coeff
            )
            self.profiles_1d.ion[_next_] = {**atoms[ion.label], "label": ion.label, "energy": q_ei}
            q_ie = q_ie + q_ei

        self.profiles_1d.electrons["energy"] = -q_ie

        return residual

    def fetch(self, x: Variable, vars: typing.Dict[str, Expression]) -> CoreSources.Source.TimeSlice:
        ns = vars[f"{spec}/density_thermal"]

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
            if not k.endswith("density_thermal"):
                continue
            identifier = vars.get(k.removesuffix("/density_thermal"))
            nj: Expression = v
            Tj = vars.get(f"{identifier}/temperature", 0)
            spec = k.split("/")[-2]
            zj = atoms[spec].z
            aj = atoms[spec].a
            q_explicit += nj * Tj * (zj**2) / aj * coeff
            q_implicit += nj * (zj**2) / aj * coeff

        return q_explicit, q_implicit
