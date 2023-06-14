import collections
import collections.abc
import datetime
import getpass

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Dict import Dict
from spdm.data.Function import Function
from spdm.utils.logger import logger
from spdm.utils.misc import try_get

# def signaturebar(fig, text, fontsize=10, pad=5, xpos=20, ypos=7.5,
#                  rect_kw={"facecolor": "grey", "edgecolor": None},
#                  text_kw={"color": "w"}):
#     w, h = fig.get_size_inches()
#     height = ((fontsize+2*pad)/72.)/h
#     rect = plt.Rectangle((0, 0), 1, height, transform=fig.transFigure, clip_on=False, **rect_kw)
#     fig.axes[0].add_patch(rect)
#     fig.text(xpos/72./h, ypos/72./h, text, fontsize=fontsize, **text_kw)
#     fig.subplots_adjust(bottom=fig.subplotpars.bottom+height)


def sp_figure_signature(fig: plt.Figure, signature=None, x=1.0, y=0.1):
    if signature is False:
        return fig
    elif not isinstance(signature, str):
        signature = f"author: {getpass.getuser().capitalize()}. Create by SpDM at {datetime.datetime.now().isoformat()}."

    fig.text(x, y, signature, va='bottom', ha='left', fontsize='small', alpha=0.5, rotation='vertical')
    return fig


def parse_profile(desc, holder=None, **kwargs):
    opts = {}
    if desc is None:
        return None, {}
    elif isinstance(desc, str):
        data = desc
        opts = {"label": desc}
    elif isinstance(desc, collections.abc.Mapping):
        data = desc.get("name", None)
        if data is None:
            data = desc.get("data", None)
        opts = desc.get("opts", {})
    elif isinstance(desc, tuple):
        data, opts = desc
    elif isinstance(desc, Dict):
        data = desc.data
        opts = desc.opts
    elif isinstance(desc, np.ndarray):
        data = desc
        opts = {}
    else:
        raise TypeError(f"Illegal profile type! {desc}")

    if isinstance(opts, str):
        opts = {"label": opts}

    if opts is None:
        opts = {}

    opts.setdefault("label", "")

    if isinstance(data, str):
        data = try_get(holder, data, None)
    elif isinstance(data, np.ndarray):
        pass
    elif data == None:
        logger.error(f"Value error { (data)}")
    else:
        logger.error(f"Type error {data}")
    return data, opts


def plot_profiles(profile_list, *args, x_axis=None, x=None, default_num_of_points=128, fontsize=10,  grid=True, signature=None, title=None, **kwargs):
    if not isinstance(profile_list, collections.abc.Sequence):
        profile_list = [profile_list]

    if isinstance(x_axis, collections.abc.Sequence) and not isinstance(x_axis, np.ndarray):
        x_axis, x_label,  *x_opts = x_axis
        x_opts = (x_opts or [{}])[0]
    else:
        x_axis = [0, 1]
        x_label = ""
        x_opts = {}

    if isinstance(x_axis, Function) and x is not None:
        x_axis = x_axis(x)
    elif x is None and isinstance(x_axis, np.ndarray):
        x = x_axis

    if isinstance(x_axis, np.ndarray):
        x_min = x_axis[0]
        x_max = x_axis[-1]
    elif isinstance(x_axis, collections.abc.Sequence) and len(x_axis) == 2:
        x_min, x_max = x_axis
        x_axis = np.linspace(x_min, x_max, default_num_of_points)
    else:
        raise TypeError(x_axis)

    if x is None and isinstance(x_axis, np.ndarray):
        x = x_axis
    elif callable(x_axis) or isinstance(x_axis, Function):
        x_axis = x_axis(x)

    nprofiles = len(profile_list)

    fig, sub_plot = plt.subplots(ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2*nprofiles))

    if not isinstance(sub_plot,  (collections.abc.Sequence, np.ndarray)):
        sub_plot = [sub_plot]

    for idx, profile_grp in enumerate(profile_list):

        if not isinstance(profile_grp, list):
            profile_grp = [profile_grp]
        ylabel = None
        for jdx, p_desc in enumerate(profile_grp):
            profile, label, *o_args = p_desc
            opts = {}
            if len(o_args) > 0 and ylabel is None:
                ylabel = o_args[0]
            if len(o_args) > 1:
                opts = o_args[1]

            y = None

            if isinstance(profile, Function) or callable(profile):
                try:
                    y = profile(x)
                except Exception as error:
                    raise RuntimeError(
                        f"Can not get profile [idx={idx} jdx={jdx}]! name={getattr(profile,'_name',profile)}\n {error} ") from error

            elif isinstance(profile, np.ndarray) and len(profile) == len(x):
                y = profile
            elif np.isscalar(profile):
                y = np.full_like(x, profile, dtype=float)
            else:
                raise RuntimeError(f"Illegal profile! {profile}!={x}  idx={idx} jdx={jdx}")

            if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
                logger.warning(f"Illegal profile! {(type(x) ,type(y), label, o_args)}  idx={idx} jdx={jdx}")
                continue
            elif x.shape != y.shape:
                logger.warning(f"Illegal profile! {x.shape} !={y.shape}  idx={idx} jdx={jdx}")
                continue
            else:
                # 删除 y 中的 nan
                mark = np.isnan(y)
                # if np.any(mark):
                #     logger.warning(f"Found NaN in array  {np.argwhere(mark)}! {profile}  idx={idx} jdx={jdx}  ")
                sub_plot[idx].plot(x_axis[~mark], y[~mark], label=label, **opts)

        sub_plot[idx].legend(fontsize=fontsize)

        if grid:
            sub_plot[idx].grid()

        if ylabel is not None:
            sub_plot[idx].set_ylabel(ylabel, fontsize=fontsize)
        sub_plot[idx].labelsize = "media"
        sub_plot[idx].tick_params(labelsize=fontsize)

    if len(sub_plot) <= 1:
        sub_plot[0].set_xlabel(x_label,  fontsize=fontsize)
    else:
        sub_plot[-1].set_xlabel(x_label,  fontsize=fontsize)
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)
    fig = sp_figure_signature(fig, signature=signature)
    fig.align_ylabels()
    fig.tight_layout()
    return fig


def sp_figure(obj, *args, signature=None, title=None, fontsize=10, **kwargs):
    fig = plt.figure()

    if not hasattr(obj, 'plot'):
        raise NotImplementedError(type(obj))

    axis = obj.plot(fig.gca(), *args, **kwargs)

    if title is None:
        title = getattr(obj, "name", None)

    if title is not None:
        fig.suptitle(title, fontsize=fontsize)

    fig.align_ylabels()

    fig.tight_layout()

    fig = sp_figure_signature(fig, signature=signature, x=axis.get_position().xmax+0.01)
    # fig.tight_layout()
    # fig.gca().axis('scaled')
    return fig
