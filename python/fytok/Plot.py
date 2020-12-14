import collections

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profile


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
    if 'label' not in opts:
        opts["label"] = path

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

    return data, opts


def plot_profiles(holder, profiles, fig_axis=None, axis=None, prefix=None, grid=False):
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

    if not isinstance(axis, np.ndarray):
        axis, axis_opts = fetch_profile(holder, axis, prefix=prefix)
    else:
        axis = None
        axis_opts = {}

    fig = None
    if isinstance(fig_axis, collections.abc.Sequence):
        pass
    elif fig_axis is None:
        nprofiles = len(profiles)
        fig, fig_axis = plt.subplots(ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2*nprofiles))

    elif len(profiles) == 1:
        fig_axis = [fig_axis]
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
            profile, opts = fetch_profile(holder, d,  prefix=prefix)
            if isinstance(profile, Profile) and hasattr(profile, "axis"):
                fig_axis[idx].plot(profile.axis, profile, **opts)
            elif isinstance(profile, np.ndarray):
                try:
                    if axis is not None:
                        axis[idx].plot(axis, profile, **opts)
                    else:
                        axis[idx].plot(profile, **opts)

                except Exception as error:
                    logger.error(f"Can not plot profile! [{error}]")
            else:
                logger.error(f"Can not find profile '{d}'")

        fig_axis[idx].legend(fontsize=6)

        if grid:
            fig_axis[idx].grid()

        if ylabel:
            fig_axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
        fig_axis[idx].labelsize = "media"
        fig_axis[idx].tick_params(labelsize=6)

    fig_axis[-1].set_xlabel(axis_opts.get("label", ""),  fontsize=6)

    return fig
