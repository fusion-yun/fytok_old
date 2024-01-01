import typing
import scipy.constants
from spdm.data.Expression import Variable, Expression, zero
from spdm.data.sp_property import sp_tree
from spdm.numlib.misc import sTep_function_approx
from spdm.utils.typing import array_type

from fytok.utils.logger import logger
from fytok.utils.atoms import atoms

from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreProfiles import CoreProfiles

from fytok.modules.Utilities import *

PI = scipy.constants.pi


@sp_tree
class SynchrotronRadiation(CoreSources.Source):
    identifier = "synchrotron"

    code = {
        "name": "synchrotron_radiation",
        "description": """
    Source from Synchrotron radition 
    
    """,
    }  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(profiles_1d)

        source_1d = current.profiles_1d

        ne = profiles_1d.electrons.density
        Te = profiles_1d.electrons.temperature

        equilibrium: Equilibrium.TimeSlice = self.inputs.get_source("equilibrium").time_slice.current

        B0 = np.abs(equilibrium.vacuum_toroidal_field.b0)
        R0 = equilibrium.vacuum_toroidal_field.r0
        if False:
            #   Reference: (GACODE)
            #    Synchrotron synchrotron
            #        - Trubnikov, JETP Lett. 16 (1972) 25.

            eq_1d = equilibrium.profiles_1d

            psi_norm = Function(eq_1d.grid.rho_tor_norm, eq_1d.grid.psi_norm, label=r"\bar{\psi}")(x)

            r_min = eq_1d.major_radius(psi_norm)

            aspect_rat = R0 / r_min

            r_coeff = 0.8  # Reflection coefficient (Rosenbluth)

            me = scipy.constants.electron_mass
            e = scipy.constants.elementary_charge
            PI = scipy.constants.pi
            c = scipy.constants.speed_of_light

            wpe = np.sqrt(4 * PI * ne * e**2 / me)
            wce = e * B0 / (me * c)
            g = Te / scipy.constants.electron_volt / (me * c**2)
            phi = (
                60
                * g**1.5
                * np.sqrt((1.0 - r_coeff) * (1 + 1 / aspect_rat / np.sqrt(g)) / (r_min * wpe**2 / c / wce))
            )

            qsync = me / (3 * PI * c) * g * (wpe * wce) ** 2 * phi

        else:
            qsync = 6.2e-22 * B0**2.0 * ne * Te

        source_1d.electrons.energy -= qsync

        return current


CoreSources.Source.regisTer(["synchrotron_radiation"], SynchrotronRadiation)
