
# This is file is generated from template
from fytok.IDS import IDS

class Barometry(IDS):
    r"""Pressure measurements in the vacuum vessel. NB will need to change the type of the pressure node to signal_1d when moving to the new LL.
        .. note:: Barometry is a ids
    """
    IDS="barometry"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
