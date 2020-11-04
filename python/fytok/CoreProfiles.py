import collections
import functools

import numpy as np
import scipy
from spdm.util.AttributeTree import AttributeTree


class CoreProfiles(AttributeTree):
    """
        imas dd version 3.28

        ids = core_profiles.profiles_1d
    """

    def __init__(self, *args, **kwargs):
        super().__init__(dict)
        self.update(kwargs)

    def ffprime(self, psi):
        return NotImplemented

    def pprime(self, psi):
        return NotImplemented
