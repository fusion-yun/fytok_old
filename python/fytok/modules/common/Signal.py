import numpy as np
from spdm.data import Function


class Signal(Function):

    def __init__(self, time: np.ndarray, data: np.ndarray = None):
        if data is None:
            data = time["data"]
            time = time["time"]
        super().__init__(time, data)

    @property
    def data(self) -> np.ndarray:
        return self.__array__()

    @property
    def time(self) -> np.ndarray:
        return self._x_axis
