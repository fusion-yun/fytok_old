import typing

import scipy.interpolate as interpolate
from skimage import measure

from spdm.utils.logger import deprecated, logger
import numpy as  np

# d: np.ndarray, x: typing.Optional[np.ndarray] = None, y: typing.Optional[np.ndarray] = None

# import matplotlib.pyplot as plt

# @deprecated
# def find_countours_matplotlib(z: np.ndarray, x: np.ndarray = None, y: np.ndarray = None, /, *args, levels=None, ** kwargs) -> typing.List[typing.List[np.ndarray]]:
#     """
#         args:X: np.ndarray, Y: np.ndarray, Z: np.ndarray
#         TODO: need improvement
#     """
#     fig = plt.figure()
#     contour_set = fig.gca().contour(x, y, z, *args, levels=levels, ** kwargs)
#     return [(contour_set.levels[idx], col.get_segments()) for idx, col in enumerate(contour_set.collections)]


def find_countours_skimage(z: np.ndarray, x: np.ndarray = None, y: np.ndarray = None, /, levels=128,  ** kwargs) -> typing.List[typing.List[np.ndarray]]:
    if z.shape == x.shape and z.shape == y.shape:
        pass
    else:
        raise ValueError(f"Array shape does not match! x:{x.shape} , y:{y.shape}, z:{z.shape} ")
    shape = z.shape
    dim0 = np.linspace(0, shape[0]-1, shape[0])
    dim1 = np.linspace(0, shape[1]-1, shape[1])
    x_inter = interpolate.RectBivariateSpline(dim0, dim1, x)
    y_inter = interpolate.RectBivariateSpline(dim0, dim1, y)

    surfs = []

    def coord_map(c):
        return np.asarray([[x_inter(p[0], p[1], grid=False), y_inter(p[0], p[1], grid=False)] for p in c])

    if isinstance(levels, int):
        levels = range(levels)

    for val in levels:
        countours = measure.find_contours(z, val)
        surfs.append((val, [coord_map(c) for c in countours]))

    return surfs


def find_countours(z: np.ndarray, x: np.ndarray = None, y: np.ndarray = None, /, levels=128,  ** kwargs) -> typing.List[typing.List[np.ndarray]]:
    return find_countours_skimage(z, x, y, levels=levels)
