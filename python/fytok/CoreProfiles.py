import numpy as np
import scipy
import functools
import collections

core_profiles = collections.namedtuple("core_profiles", [

    "global_quantity",
    "vacuum_toroidal_field",

])


class CoreProfiles:
    """
        imas dd version 3.28

        ids = core_profiles.profiles_1d
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return NotImplemented
