import unittest

import matplotlib.pyplot as plt
from spdm.util.numlib import np
from spdm.data.Coordinates import Coordinates
from spdm.data.Field import Field
from spdm.util.logger import logger


class TestFluxSurface(unittest.TestCase):

    def test_flux_surface(self):
