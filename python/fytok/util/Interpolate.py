import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.optimize import root_scalar, fsolve
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module


class Interpolate(object):

    @classmethod
    def __new__(cls, *args, **kwargs):
        if hasattr(Interpolate, "_create_interpolate"):
            return Interpolate._create_interpolate(cls, *args, **kwargs)
        else:
            return object.__new__(cls)

    def __init__(self,   *args,   **kwargs):
        pass


class Interpolate1D(Interpolate):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fun = UnivariateSpline(x,  y, *args,  **kwargs)

    def __call__(self, x=None, *args, **kwargs):
        if x is None:
            return self._fun
        else:
            return self._fun(x)

    def derivate(self, *args, n=1, **kwargs):
        return self._fun.derivative(n)

    def integral(self, *args,  **kwargs):
        return self._fun.integral(*args, **kwargs)


class Interpolate2D(Interpolate):
    def __init__(self,  x, y, z, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_x = x
        self.dim_y = y
        self.value = z
        if (isinstance(x, list) or len(x.shape) == 1) and \
            (isinstance(y, list) or len(y.shape) == 1) and \
                (hasattr(z, 'shape') and len(z.shape) == 2):

            self._fun = RectBivariateSpline(x,  y,  z, *args, **kwargs)
        else:
            raise NotImplementedError()

    def __call__(self, x, y, **kwargs):
        return self._fun(x, y, **kwargs)

    @property
    def mesh_coordinates(self):
        return np.meshgrid(self.dim_x, self.dim_y)

    def derivate(self, x, y, * args, dx=1, dy=1, **kwargs):
        return self._fun(x, y, * args, dx=dx, dy=dy, **kwargs)

    def dx(self, x, y, dx=1,  **kwargs):
        return self._fun(x, y,   dx=dx, **kwargs)

    def dy(self, x, y, dy=1, **kwargs):
        return self._fun(x, y,   dy=dy, **kwargs)

    def _find_critical(self, *args, **kwargs):
        pass


class InterpolateND(Interpolate):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


def _create_interpolate(cls, y,  x=None, *args, backend=None, ** kwargs):
    if cls is not Interpolate:
        return super(Interpolate, cls).__new__(cls)

    if backend is not None:
        plugin_name = f"{__package__}.plugins.interpolate.Plugin{backend}"

        n_cls = sp_find_module(plugin_name, fragment=f"Interpolate{backend}")

        if n_cls is None:
            raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#Interpolate{backend}")
    elif len(x.shape) == 1:
        n_cls = Interpolate1D
    elif len(x.shape) == 2:
        n_cls = Interpolate2D
    else:
        n_cls = InterpolateND

    return object.__new__(n_cls)


Interpolate._create_interpolate = _create_interpolate


def interpolate(y, x, *args, **kwargs):
    if isinstance(y, Interpolate1D):
        return y.reset(x, *args, **kwargs)
    else:
        return Interpolate(y, x, *args, **kwargs)


def derivate(y, x=None, *args, **kwargs):
    if isinstance(y, Interpolate1D):
        pass
    elif x is not None:
        y = Interpolate1D(x, y)
    else:
        raise TypeError(f"y={type(y)} x={type(x)}")

    return y.derivate(*args, **kwargs)


def integral(y, x, *args, **kwargs):
    return NotImplemented


def find_root1d(fun, x0=None, x1=None,  bracket=[0, 1], fprime=None):
    if isinstance(fun, Interpolate1D):
        sol = root_scalar(fun, bracket[0], bracket=bracket, fprime=lambda x: fun.derivate(x), method='newton')
    elif fprime is None:
        logger.debug((fun(bracket[0]), fun(bracket[1])))
        sol = root_scalar(fun, bracket=bracket, method='brentq')
    else:
        sol = root_scalar(fun, x0=x0 or bracket[0], x1=x1 or (bracket[0]+bracket[1])
                          * 0.5, bracket=bracket, fprime=fprime, method='newton')

    return sol


def find_root2d(fun,   line=None, box=None):
    if line is not None:
        p0, p1 = line
        # vx = p1[0]-p0[0]
        # vy = p1[1]-p0[1]
        # vr = np.sqrt(vx**2+vy**2)
        # vx /= vr
        # vy /= vr
        # v = [vx, vy]

        def f(r, x0=p0, x1=p1, fun=fun):
            return fun((1.0-r)*x0[0]+r*x1[0], (1.0-r)*x0[1]+r*x1[1])

        def fprime(r, x0=p0, x1=p1, fun=fun):
            x = (1.0-r)*x0[0]+r*x1[0]
            y = (1.0-r)*x0[1]+r*x1[1]
            df = fun(x, y, dx=1)*(x1[0]-x0[0])+fun(x, y, dy=1)*(x1[1]-x0[1])
            return df

        sol = find_root1d(f, x0=0.01,
                          fprime=fprime,
                          bracket=[0.01, 1.0])

        r = sol.root
        return r*p1[0]+(1.0-r)*p0[0], r*p1[1]+(1.0-r)*p0[1]
    else:
        raise NotImplementedError()


def find_root(fun,   *args, **kwargs):
    if isinstance(fun, Interpolate1D):
        return find_root1d(fun,  *args, **kwargs)
    elif isinstance(fun, Interpolate2D):
        return find_root2d(fun,  *args, **kwargs)
    else:
        raise NotImplementedError()


def find_peaks_2d_image(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    idxs = np.where(detected_peaks)

    for n in range(len(idxs[0])):
        yield idxs[0][n], idxs[1][n]


def find_critical(fun2d, X, Y, * args, **kwargs):

    fxy2 = fun2d(X, Y, dx=1)**2+fun2d(X, Y, dy=1)**2

    span = 1
    for ix, iy in find_peaks_2d_image(-fxy2[span:-span, span:-span]):
        ix += span
        iy += span
        x = float(X[ix, iy])
        y = float(Y[ix, iy])

        if abs(fxy2[ix+span, iy+span]) > 1.0e-5:  # FIXME: replace magnetic number
            xmin = X[ix-1, iy]
            xmax = X[ix+1, iy]
            ymin = Y[ix, iy-1]
            ymax = Y[ix, iy+1]

            def f(r):
                if r[0] < xmin or r[0] > xmax or r[1] < ymin or r[1] > ymax:
                    raise LookupError(r)
                fx = fun2d(r[0], r[1], dx=1)
                fy = fun2d(r[0], r[1], dy=1)

                return fx, fy

            def fprime(r):
                fxx = fun2d(r[0], r[1], dx=2)
                fyy = fun2d(r[0], r[1], dy=2)
                fxy = fun2d(r[0], r[1], dy=1, dx=1)

                return [[fxx, fxy], [fxy, fyy]]  # FIXME: not sure, need double check

            try:
                x1, y1 = fsolve(f, [x, y],   fprime=fprime)
            except LookupError:
                continue
            else:
                x = x1
                y = y1

        # D = fxx * fyy - (fxy)^2
        D = fun2d(x, y, dx=2) * fun2d(x, y, dy=2) - (fun2d(x, y,  dx=1, dy=1))**2

        yield x, y, D
