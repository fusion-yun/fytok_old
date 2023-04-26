from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.Wall import Wall
from fytok.modules.PFActive import PFActive
from fytok.modules.Magnetics import Magnetics

import numpy as np

from .atec_impl import atec_run

##########################
# ```atec_impl.py```
#
#   input_ids = {
#       "core_profiles/profiles_1d": {
#           "ffprime": np.array([...]),
#           "pressure_thermal": np.array([...])
#       },
#       "pf_active": {"coil": [{...}]},
#       "wall": {...},
#   }
#
#   output_ids = {
#       "equilibrium/time_slice/0": {
#           "profiles_2d/0": {
#               "grid": {"dim1": np.linspace(0, 1, 125),
#                       "dim2": np.linspace(-4, 4, 512)},
#               "psi": np.array([125, 512])
#           }
#       }
#   }
#
#
#   def atec_run(input_ids: dict) -> dict:
#       ffprime = input_ids["core_profiles/profiles_1d/ffprime"]
#
#       return {"equilibrium/time_slice/0/profiles_2d/0/psi": np.array([125, 512])}
###############


@ Equilibrium.register(["atec"])
class EquilibriumATEC(Equilibrium):

    def update(self, *args, time=None,
               core_profiles: CoreProfiles,
               wall: Wall,
               pf_active: PFActive,
               magnetics: Magnetics,
               **kwargs) -> float:

        # self.time_slice(Equilibrium.TimeSlice(atec.run()), time)

        res = atec_run(ffprime=core_profiles.profiles_1d.ffprime,
                       ffprime=core_profiles.profiles_1d.pressure_thermal,
                       coil=pf_active.coil)

        self.time_slice[-1] = Equilibrium.TimeSlice(res)


__SP_EXPORT__ = EquilibriumATEC
