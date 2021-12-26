
from ..common.IDS import IDS
from spdm.data import Dict, List,  sp_property


class ControllersLineController(Dict):
    pass


class ControllersNonLinearController(Dict):
    pass


class Controllers(IDS):
    r"""Feedback and feedforward controllers

        Note: Controllers is an ids
    """
    _IDS = "controllers"
    LineController = ControllersLineController
    NonLinearController = ControllersNonLinearController

    linear_controller: List[LineController] = sp_property()

    nonlinear_controller: List[NonLinearController] = sp_property()
