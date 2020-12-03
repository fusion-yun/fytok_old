import collections

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profile


def fetch_profile(holder, desc, prefix=[]):
    x_axis = None
    if isinstance(desc, str):
        path = desc
        opts = {"label": desc}
    elif isinstance(desc, collections.abc.Mapping):
        path = desc.get("name", None)
        opts = desc.get("opts", {})
        x_axis = desc.get("x_axis", None)
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

    if isinstance(x_axis, str):
        x_axis = holder[x_axis]
    # else:
    #     raise TypeError(f"Illegal data type! {prefix} {type(data)}")

    return data, opts, x_axis


def plot_profiles(holder, profiles, axis=None, x_axis=None, prefix=None, grid=False):
    if isinstance(profiles, str):
        profiles = [s.strip() for s in profiles.split(",")]
    elif not isinstance(profiles, collections.abc.Sequence):
        profiles = [profiles]

    if prefix is None:
        prefix = []
    elif isinstance(prefix, str):
        prefix = prefix.split(".")
    elif not isinstance(prefix, collections.abc.Sequence):
        prefix = [prefix]

    x_axis, x_axis_opts, _ = fetch_profile(holder, x_axis, prefix=prefix)

    fig = None
    if isinstance(axis, collections.abc.Sequence):
        pass
    elif axis is None:
        nprofiles = len(profiles)
        fig, axis = plt.subplots(ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2*nprofiles))

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
            profile, opts, x_axis_alt = fetch_profile(holder, d,  prefix=prefix)

            if x_axis_alt is None:
                x_axis_alt = x_axis
            if isinstance(profile, Profile):
                axis[idx].plot(profile.x_axis, profile, **opts)
            elif isinstance(profile, np.ndarray):
                axis[idx].plot(x_axis_alt, profile, **opts)
            else:
                logger.error(f"Can not find profile '{d}'")

        axis[idx].legend(fontsize=6)

        if grid:
            axis[idx].grid()

        if ylabel:
            axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
        axis[idx].labelsize = "media"
        axis[idx].tick_params(labelsize=6)

    axis[-1].set_xlabel(x_axis_opts.get("label", ""),  fontsize=6)

    return fig
