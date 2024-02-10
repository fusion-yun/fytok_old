from functools import cached_property

from spdm.core.Actor import Actor
from spdm.core.sp_property import sp_tree
from .modules.PulseSchedule import PulseSchedule
from .modules.TransportSolverNumerics import TransportSolverNumerics

from .utils.logger import logger
from .Tokamak import Tokamak


@sp_tree
class Scenario(Actor):
    """
    Scenario

    """

    tokamak: Tokamak

    pulse_schedule: PulseSchedule
