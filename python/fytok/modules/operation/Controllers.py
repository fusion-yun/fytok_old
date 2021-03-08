
from functools import cached_property

import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph, _next_

from fytok.Misc import IDSProperties, Signal


class Controllers(PhysicalGraph):
    r"""Feedback and feedforward controllers

        Note: Controllers is an ids
    """
    IDS = "controllers"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def ids_properties(self):
        return IDSProperties(self._cache.ids_properties)

    class LinearController(PhysicalGraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    @cached_property
    def linear_controller(self):
        res = PhysicalGraph(
            default_factory_array=lambda _holder=self: Controllers.LinearController(None, parend=_holder))

        for lin_contr in self._cache.linear_controller:
            res[_next_] = lin_contr

        return res

    class NonLinearController(PhysicalGraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    @cached_property
    def nonlinear_controller(self):
        res = PhysicalGraph(
            default_factory_array=lambda _holder=self: Controllers.NonLinearController(None, parend=_holder))

        for nonlin_contr in self._cache.nonlinear_controller:
            res[_next_] = nonlin_contr
        return res
