import collections

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger


def fetch_profile(holder, desc, prefix=[]):
    if isinstance(desc, str):
        path = desc
        opts = {"label": desc}
    elif isinstance(desc, collections.abc.Mapping):
        path = desc.get("name", None)
        opts = desc.get("opts", {})
    elif isinstance(desc, tuple):
        path, opts = desc
    elif isinstance(desc, AttributeTree):
        path = desc.data
        opts = desc.opts
    else:
        raise TypeError(f"Illegal profile type! {desc}")

    if isinstance(opts, str):
        opts = {"label": opts}

    if prefix is None:
        prefix = []
    elif isinstance(prefix, str):
        prefix = prefix.split(".")

    if isinstance(path, str):
        path = path.split(".")

    path = prefix+path

    if isinstance(path, np.ndarray):
        data = path
    else:
        data = holder[path]

    # else:
    #     raise TypeError(f"Illegal data type! {prefix} {type(data)}")

    return data, opts


def plot_profiles(holder, profiles, axis=None, x_axis=None, prefix=None):
    if isinstance(profiles, str):
        profiles = profiles.split(",")
    elif not isinstance(profiles, collections.abc.Sequence):
        profiles = [profiles]

    if prefix is None:
        prefix = []
    elif isinstance(prefix, str):
        prefix = prefix.split(".")
    elif not isinstance(prefix, collections.abc.Sequence):
        prefix = [prefix]

    x_axis, x_axis_opts = fetch_profile(holder, x_axis, prefix=prefix)

    fig = None
    if isinstance(axis, collections.abc.Sequence):
        pass
    elif axis is None:
        fig, axis = plt.subplots(ncols=1, nrows=len(profiles), sharex=True)
    elif len(profiles) == 1:
        axis = [axis]
    else:
        raise RuntimeError(f"Too much profiles!")

    for idx, data in enumerate(profiles):
        ylabel = None
        opts = {}
        if isinstance(data, tuple):
            data, ylabel = data

        if not isinstance(data, list):
            data = [data]

        for d in data:
            value, opts = fetch_profile(holder, d,  prefix=prefix)

            if value is not NotImplemented and value is not None and len(value) > 0:
                axis[idx].plot(x_axis, value, **opts)
            else:
                logger.error(f"Can not find profile '{d}'")

        axis[idx].legend(fontsize=6)

        if ylabel:
            axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
        axis[idx].labelsize = "media"
        axis[idx].tick_params(labelsize=6)

    axis[-1].set_xlabel(x_axis_opts.get("label", ""),  fontsize=6)

    return axis, fig
