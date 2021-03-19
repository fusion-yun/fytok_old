import collections

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.util.utilities import try_get


def parse_profile(desc, holder=None, **kwargs):
    if isinstance(desc, str):
        data = desc
        opts = {"label": desc}
    elif isinstance(desc, collections.abc.Mapping):
        data = desc.get("name", None)
        if data is None:
            data = desc.get("data", None)
        opts = desc.get("opts", {})
    elif isinstance(desc, tuple):
        data, opts = desc
    elif isinstance(desc, PhysicalGraph):
        data = desc.data
        opts = desc.opts
    else:
        raise TypeError(f"Illegal profile type! {desc}")

    if isinstance(opts, str):
        opts = {"label": opts}

    opts.setdefault("label", "unnamed")

    if not isinstance(data, np.ndarray):
        data = try_get(holder, data, None)

    return data, opts


def plot_profiles(profile_list, *args, sub_plot=None, x_axis=None, fontsize=6,  grid=False, **kwargs):
    if not isinstance(profile_list, collections.abc.Sequence):
        profile_list = [profile_list]

    profile_list += args

    if not isinstance(x_axis, np.ndarray):
        x_axis, x_axis_opts = parse_profile(x_axis, **kwargs)
    else:
        x_axis = None
        x_axis_opts = {}

    nprofiles = len(profile_list)

    fig, sub_plot = plt.subplots(ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2*nprofiles))

    if not isinstance(sub_plot,  collections.abc.Sequence):
        sub_plot = [sub_plot]

    for idx, profile_grp in enumerate(profile_list):
        grp_opts = {}

        if isinstance(tuple, profile_grp):
            profile_grp, grp_opts = profile_grp

        if not isinstance(profile_grp, list):
            profile_grp = [profile_grp]

        for p_desc in profile_grp:
            profile, opts = parse_profile(p_desc, **kwargs)
            if isinstance(profile, (Field, Function)):
                sub_plot[idx].plot(x_axis, profile(x_axis), **opts)
            elif isinstance(profile, np.ndarray):
                try:
                    if x_axis is not None and x_axis.shape == profile.shape:
                        sub_plot[idx].plot(x_axis, profile, **opts)
                    else:
                        sub_plot[idx].plot(profile, **opts)
                except Exception as error:
                    logger.error(f"Can not plot profile! [{error}]")
            else:
                logger.error(f"Can not plot profile '{p_desc}'")

        sub_plot[idx].legend(fontsize=fontsize)

        if grid:
            sub_plot[idx].grid()

        if "ylabel" in grp_opts:
            sub_plot[idx].set_ylabel(grp_opts["ylabel"], fontsize=fontsize).set_rotation(0)
        sub_plot[idx].labelsize = "media"
        sub_plot[idx].tick_params(labelsize=fontsize)

    sub_plot[-1].set_xlabel(x_axis_opts.get("label", ""),  fontsize=fontsize)

    return fig
