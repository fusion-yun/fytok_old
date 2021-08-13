import collections
import os
import pathlib
import re
from functools import cached_property

from spdm.numlib import np
from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

import MDSplus as mds

from ..Collection import Collection
from ..Document import Document
from ..Entry import Entry
from ...util.dict_util import format_string_recursive


class MDSplusEntry(Entry):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self,  *args, **kwargs):
        return self._parent.fetch(*args, **kwargs)

    def put(self,  path, value, *args, **kwargs):
        return self._parent.update({path: value}, *args, **kwargs)

    def iter(self,  path, *args, **kwargs):
        return self._parent.iter(path, *args, **kwargs)


def open_mdstree(tree_name, shot,  mode="NORMAL", path=None):
    if tree_name is None:
        raise ValueError(f"Treename is empty!")
    try:
        shot = int(shot)
        logger.info(f"Open MDSTree: tree_name={tree_name} shot={shot} mode=\"{mode}\" path='{path}'")
        tree = mds.Tree(tree_name, shot, mode=mode, path=path)
    except mds.mdsExceptions.TreeFOPENR as error:
        # tree_path = os.environ.get(f"{tree_name}_path", None)
        raise FileNotFoundError(
            f"Can not open mdsplus tree! tree_name={tree_name} shot={shot} tree_path={path} mode={mode} \n {error}")
    except mds.mdsExceptions.TreeNOPATH as error:
        raise FileNotFoundError(
            f"{tree_name}_path is not defined! tree_name={tree_name} shot={shot}  \n {error}")
    return tree


class MDSplusDocument(Document):
    MDS_MODE = {
        "r": "ReadOnly",
        "rw": "Normal",
        "w": "Normal",
        "w+": "Edit",
        "x": "New"
    }

    def __init__(self, *args, tree_name=None,   **kwargs):
        super().__init__(*args,  ** kwargs)

        self._envs = {k: (v if not isinstance(v, slice) else f"{v.start}:{v.stop}:{v.step}")
                      for k, v in self._envs.items()}

        self._mds_mode = MDSplusDocument.MDS_MODE.get(self.mode, "NORMAL")
        self._tree_name = tree_name or self.metadata.get("query", {}).get("tree_name", None)

    def __del__(self):
        pass

    @property
    def entry(self):
        return MDSplusEntry(parent=self)

    def fetch(self, request, *args,   **kwargs):
        tree_name = self._tree_name
        if not request:
            return self
        elif isinstance(request, collections.abc.Mapping):
            tree_name = request.get("@tree", None)
            tdi = request.get("@text", None)
        elif isinstance(request, str):
            tdi = request
        else:
            raise ValueError(request)

        if not tdi:
            return self.entry

        tdi = tdi.format_map(self._envs)

        mode = self._mds_mode
        shot = self.fid
        path = self.metadata.get("path", None)

        res = None
        try:
            with mds.Tree(tree_name, int(shot), mode=mode, path=path) as tree:
                res = tree.tdiExecute(tdi).data()

        except mds.mdsExceptions.TdiException as error:
            raise RuntimeError(f"MDSplus TDI error [{tdi}]! {error}")
        except mds.mdsExceptions.TreeFOPENR as error:
            raise FileNotFoundError(
                f"Can not open mdsplus tree! tree_name={tree_name} shot={shot} tree_path={path} mode={mode} \n {error}")
        except mds.mdsExceptions.TreeNOPATH as error:
            raise FileNotFoundError(
                f"{tree_name}_path is not defined! tree_name={tree_name} shot={shot}  \n {error}")
        except Exception as error:
            raise error

        if not isinstance(res, np.ndarray):
            pass
        elif len(res.shape) == 2:
            if res.shape[1] == 1:
                res = res[:, 0]
            elif res.shape[0] == 1:
                res = res[0]
            else:
                res = res.transpose(1, 0)
        return res

    def update(self, request, *args,  envs=None, **kwargs):
        raise NotImplementedError()


class MDSplusCollection(Collection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)

    def insert_one(self, fid=None, *args,  query=None, mode=None, **kwargs):
        fid = fid or self.guess_id(*args, **collections.ChainMap((query or {}), kwargs)) or self.next_id
        return MDSplusDocument(self.metadata, fid=fid, mode=mode or "w", **kwargs)

    def find_one(self, fid=None, *args, query=None, projection=None, mode=None, **kwargs):
        fid = fid or self.guess_id(*args, **collections.ChainMap((query or {}), kwargs))
        return MDSplusDocument(self.metadata, fid=fid, mode=mode or "w", **kwargs).fetch(projection)

    def count(self, predicate=None, *args, **kwargs) -> int:
        return NotImplemented()

    # def find_one(self, predicate: Document,  projection: Document = None, *args, **kwargs):
    #     shot = getitem(predicate, "shot", None) or getitem(predicate, "_id", None)
    #     if shot is not None:
    #         return MDSplusEntry(self._tree_name, shot, mode="r") .fetch(projection)
    #     else:
    #         for shot in self._foreach_shot():
    #             res = MDSplusEntry(self._tree_name, shot, mode="r").fetch_if(
    #                 projection, predicate)
    #             if res is not None:
    #                 return res
    #     return None

    # def _foreach_shot(self):
    #     f_prefix = f"{self._tree_name.lower()}_"
    #     f_prefix_l = len(f_prefix)
    #     glob = f"{f_prefix}*.tree"
    #     for fp in self._path.glob(glob):
    #         yield fp.stem[f_prefix_l:]

    # def find(self, predicate: Document = None, projection: Document = None, *args, **kwargs):

    #     for shot in self._foreach_shot():
    #         res = MDSplusEntry(self._tree_name, shot, mode="r").fetch_if(projection, predicate)
    #         logger.debug(res)

    #         if res is not None:
    #             yield res

    # def insert_one(self, document: Document, *args, **kwargs):
    #     self._count += 1

    #     shot = int(document.get("shot", self._count))

    #     MDSplusEntry(self._tree_name, shot, mode="x").update(document)

    #     return shot


__SP_EXPORT__ = MDSplusDocument
