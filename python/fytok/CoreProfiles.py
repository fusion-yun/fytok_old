import collections
import functools

import numpy as np
import scipy
from spdm.util.AttributeTree import AttributeTree
from spdm.util.Profiles import Profiles1D


class CoreProfiles(AttributeTree):
    """
        imas dd version 3.28

        ids = core_profiles.profiles_1d
    """

    def __init__(self, *args, dims=None, **kwargs):
        super().__init__(dims or 129)
        # if len(args)+len(kwargs) > 0:
        self.load(*args, **kwargs)

    def load(self, entry=None, *args, dims=None, itime=0, **kwargs):
        self.entry.profiles_1d.grid.rho = np.linspace(0, 1, dims or 129)
        self.entry.profiles_1d.grid.psi = np.linspace(0, 1, dims or 129)
        self.entry.profiles_1d.grid.dpsi = np.linspace(0, 1, dims or 129)
        self.entry.profiles_1d.grid.psi_norm = np.linspace(0, 1, dims or 129)
        self.entry.profiles_1d.conductivity_parallel= np.linspace(0, 1, dims or 129)
