from functools import cached_property

from _imas.core_transport import (_T_core_transport, _T_core_transport_model,
                                  _T_core_transport_model_profiles_1d)
from spdm.data.List import List
from spdm.data.sp_property import sp_property

from .CoreProfiles import CoreProfiles
from .MagneticCoordSystem import RadialGrid

class CoreTransportProfiles1D(_T_core_transport_model_profiles_1d):

    @property
    def grid(self) -> RadialGrid:
        return self._parent.grid

    # grid_d: RadialGrid = sp_property(
    #     lambda self: self.grid.remesh("rho_tor_norm", 0.5*(self.grid.rho_tor_norm[:-1]+self.grid.rho_tor_norm[1:])),
    #     doc="""Grid for effective diffusivity and parallel conductivity""")

    @cached_property
    def grid_d(self) -> RadialGrid:
        rho_tor_norm = self.grid.rho_tor_norm
        return self.grid.remesh("rho_tor_norm", 0.5*(rho_tor_norm[:-1]+rho_tor_norm[1:]))

    @cached_property
    def grid_v(self) -> RadialGrid:
        """ Grid for effective convections  """
        return self.grid.remesh("rho_tor_norm", self.grid.rho_tor_norm)

    @cached_property
    def grid_flux(self) -> RadialGrid:
        """ Grid for fluxes  """
        return self.grid.remesh("rho_tor_norm", 0.5*(self.grid.rho_tor_norm[:-1]+self.grid.rho_tor_norm[1:]))


class CoreTransportModel(_T_core_transport_model):

    _IDS = "core_transport/model"

    @ property
    def grid(self) -> RadialGrid:
        return self._parent.grid

    profiles_1d: CoreTransportProfiles1D = sp_property()

    def refresh(self, *args, core_profiles: CoreProfiles, **kwargs) -> None:
        super().refresh(*args, core_profiles=core_profiles, **kwargs)
        # self.profiles_1d["grid"] = core_profiles.profiles_1d.grid


class CoreTransport(_T_core_transport):

    Model = CoreTransportModel
    
    grid: RadialGrid = sp_property()

    model: List[Model] = sp_property()

    @property
    def model_combiner(self) -> Model:
        return self.model.combine(
            common_data={
                "identifier": {"name": "combined", "index": 1,
                               "description": """Combination of data from available transport models.
                                Representation of the total transport in the system"""},
                "code": {"name": None},
                "profiles_1d": {"grid": self.grid}
            })

    def refresh(self, *args, core_profiles=None, **kwargs) -> None:
        if "model_combiner" in self.__dict__:
            del self.__dict__["model_combiner"]

        if core_profiles is not None:
            self["grid"] = core_profiles.profiles_1d.grid

        for model in self.model:
            model.refresh(*args, core_profiles=core_profiles, **kwargs)
