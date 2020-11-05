import collections
import functools

import numpy as np
import scipy
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles


class CoreProfiles(AttributeTree):
    """
        imas dd version 3.28

        ids = core_profiles.profiles_1d
    """

    def __init__(self, *args, dims=None, **kwargs):
        super().__init__()
        # if len(args)+len(kwargs) > 0:
        self.load(*args, **kwargs)

    def load(self, entry=None, *args, dims=None, itime=0, **kwargs):
        if dims is None:
            dims = 129
        self.vacuum_toroidal_field.b0 = 1.0
        self.vacuum_toroidal_field.r0 = 1.0
        self.profiles_1d = Profiles(np.linspace(1.0/(dims+1), 1, dims))
        self.profiles_1d |= {"grid": {}, "electron": {}, "ion": [],
                             "neutral": [],
                             "efield": {}
                             }
        self.profiles_1d.conductivity_parallel = np.linspace(1.0/(dims+1), 1, dims)

    def ffprime(self, psi_norm):
        return self.profiles_1d.ffprime or []

    def pprime(self, psi_norm):
        return self.profiles_1d.pprime or []
