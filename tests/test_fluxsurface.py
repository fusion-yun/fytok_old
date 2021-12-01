import unittest

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Coordinates import Coordinates
from spdm.data.Field import Field
from spdm.common.logger import logger


class TestFluxSurface(unittest.TestCase):

    def test_flux_surface(self):
