import collections
import os
import pprint
from typing import Callable

import numpy as np
from scipy import optimize
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.optimize import fsolve, minimize, root_scalar
from spdm.utils.logger import logger

SP_EXPERIMENTAL = os.environ.get("SP_EXPERIMENTAL", False)

# logger.info(f"SP_EXPERIMENTAL \t: {SP_EXPERIMENTAL}")

EPSILON = 1.0e-2


def _minimize_filter_2d_image(Z):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    # local_extremum = (maximum_filter(Z, footprint=neighborhood) == Z)\
    local_extremum = (minimum_filter(Z, footprint=neighborhood) == Z)

    # local_extremum is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (Z == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_extremum, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1)

    # local_extremum = binary_erosion(local_extremum, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_extremum mask (xor operation)
    detected_peaks = local_extremum ^ eroded_background
    idxs = np.asarray(np.where(detected_peaks)).T

    for ix, iy in idxs:
        yield ix, iy


def minimize_filter(func: Callable[..., float], xmin: float, ymin: float, xmax: float, ymax: float, tolerance=EPSILON):

    if isinstance(tolerance, collections.abc.Sequence) and len(tolerance) == 2:
        dx, dy = tolerance
    else:
        dx = tolerance
        dy = tolerance

    nx = int((xmax-xmin)/dx)+1
    ny = int((ymax-ymin)/dy)+1

    X, Y = np.meshgrid(np.linspace(xmin, xmax, nx),
                       np.linspace(ymin, ymax, ny), indexing='ij')

    for ix, iy in _minimize_filter_2d_image(func([X.ravel(), Y.ravel()]).reshape([nx, ny])):

        if ix == 0 or iy == 0 or ix == nx-1 or iy == ny-1:
            continue

        xmin = X[ix-1, iy]
        xmax = X[ix+1, iy]
        ymin = Y[ix, iy-1]
        ymax = Y[ix, iy+1]

        xsol = X[ix, iy]
        ysol = Y[ix, iy]

        if True:
            sol = minimize(func, np.asarray([xsol, ysol]),   bounds=[(xmin, xmax), (ymin, ymax)])
            xsol, ysol = sol.x
            if sol.success and abs(sol.fun) < EPSILON and (xsol >= xmin or xsol <= xmax or ysol >= ymin or ysol <= ymax):
                yield xsol, ysol
        else:  # obsolete
            def f(r):
                if r[0] < xmin or r[0] > xmax or r[1] < ymin or r[1] > ymax:
                    raise LookupError(r)
                fx = func(r[0], r[1], dx=1, grid=False)
                fy = func(r[0], r[1], dy=1, grid=False)
                return fx, fy

            def fprime(r):
                fxx = func(r[0], r[1], dx=2, grid=False)
                fyy = func(r[0], r[1], dy=2, grid=False)
                fxy = func(r[0], r[1], dy=1, dx=1, grid=False)

                return [[fxx, fxy], [fxy, fyy]]

            try:
                x1, y1 = fsolve(f, [xsol, ysol],   fprime=fprime)
            except LookupError as error:
                # TODO: need handle exception
                # logger.debug(error)
                continue
            else:
                xsol = x1
                ysol = y1


def minimize_filter_2d_experimental(func: Callable[..., float], x0, y0, x1, y1, p0=None, tolerance=[0.1, 0.1]):
    if abs((x0-x1)) < tolerance[0] or abs(y0-y1) < tolerance[1]:
        yield from []
        return

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)
    xc = 0.5*(x0 + x1)
    yc = 0.5*(y0 + y1)

    if p0 is None or not (xmin < p0[0] and xmax > p0[0] and ymin < p0[1] and ymax > p0[1]):
        # ,   bounds=[(xmin, xmax), (ymin, ymax)]
        sol = minimize(func, np.asarray([xc, yc]), method="BFGS")
        p0 = sol.x
        if sol.success and (xmin < p0[0] and xmax > p0[0] and ymin < p0[1] and ymax > p0[1]):
            yield p0[0], p0[1]
            logger.debug((p0, (xmin, ymin), (xmax, ymax)))
        else:
            p0 = None

    yield from minimize_filter_2d_experimental(func, xc, yc, xmin, ymax, p0=p0, tolerance=tolerance)
    yield from minimize_filter_2d_experimental(func, xc, yc, xmin, ymin, p0=p0, tolerance=tolerance)
    yield from minimize_filter_2d_experimental(func, xc, yc, xmax, ymin, p0=p0, tolerance=tolerance)
    yield from minimize_filter_2d_experimental(func, xc, yc, xmax, ymax, p0=p0, tolerance=tolerance)


def find_critical_points_2d_experimental(func: Callable[..., float], xmin, ymin, xmax, ymax, tolerance=EPSILON):

    # def grad_func(p): return func(*p, dx=1, grid=False)**2 + func(*p, dy=1, grid=False)**2
    def grad_func(p): return func(*p, grid=False)

    for x, y in minimize_filter_2d_experimental(grad_func, xmin, ymin, xmax, ymax, tolerance=tolerance):
        D = func(x, y, dx=2, grid=False) * func(x, y, dy=2, grid=False) - (func(x, y,  dx=1, dy=1, grid=False))**2
        yield x, y, func(x, y, grid=False), D


def find_critical_points(func: Callable[..., float], xmin: float, ymin: float, xmax: float, ymax: float, tolerance=EPSILON):

    def grad2_func(p): return func(*p, dx=1, grid=False)**2 + func(*p, dy=1, grid=False)**2

    for xsol, ysol in minimize_filter(grad2_func, xmin, ymin, xmax, ymax, tolerance=tolerance):
        D = func(xsol, ysol, dx=2, grid=False) * func(xsol, ysol, dy=2, grid=False) - \
            (func(xsol, ysol,  dx=1, dy=1, grid=False))**2

        yield xsol, ysol, func(xsol, ysol, grid=False), D
