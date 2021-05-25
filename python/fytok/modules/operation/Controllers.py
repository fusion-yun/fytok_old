
from spdm.data.sp_property import sp_property

from spdm.data.Node import Dict, _not_found_, List
from ..common.IDS import IDS,


class ControllersLineController(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)


class ControllersNonLinearController(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Controllers(IDS):
    r"""Feedback and feedforward controllers

        Note: Controllers is an ids
    """
    _IDS = "controllers"
    LineController = ControllersLineController
    NonLinearController = ControllersNonLinearController

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def linear_controller(self) -> List[LineController]:
        return self["linear_controller"]

    @sp_property
    def nonlinear_controller(self) -> List[NonLinearController]:
        return self["nonlinear_controller"]
